"""Pipeline d'identification : photo -> hypothèses de crops -> sélection.

Logique partagée entre le banc d'essai (`evaluate_real.py`) et l'outil de scan
(`scan.py`) — et référence directe pour le portage Swift : chaque étape a son
équivalent iOS (Vision, Core Image, Core ML, Accelerate).

Ordre des étapes et raisons, toutes mesurées sur le banc d'essai :

1. Détection des quadrilatères (rectangles + segmentation document).
2. Dédoublonnage par centre — les deux détecteurs pointent souvent la même carte.
3. Filtre de variance — un crop quasi-uni (dalle, touche de clavier) ne peut pas
   être une carte, inutile de l'encoder.
4. Orientation par position du texte (src/orient.py) — jamais par embedding :
   un crop inversé peut scorer plus haut sur une mauvaise carte que le crop
   droit sur la bonne.
5. La photo entière n'entre en concurrence QUE si aucun crop ne donne un score
   plausible : c'est un repli pour carte plein cadre, pas une hypothèse normale.
   En concurrence libre, elle vole la sélection aux crops légitimes à marge
   serrée (mesuré sur 2 photos du banc).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.detect import warp_card
from src.detect_vision import detect_document, detect_rectangles
from src.orient import pick_orientation

# Un crop dont l'écart-type de gris est sous ce seuil est une surface plate
# (ciel : ~2). Seuil volontairement bas : une carte à contre-jour descend à
# ~10, et rater une vraie carte coûte bien plus cher qu'encoder un parasite —
# la sélection aval s'en charge.
MIN_GRAY_STD = 8.0

# Si le meilleur crop atteint ce score top-1, la photo entière est inutile.
# Vrais crops du banc : >= 0,72 ; la photo entière plafonne à ~0,79 mais sur de
# mauvaises cartes — ne la convoquer que quand les crops ont clairement échoué.
FALLBACK_SCORE = 0.65

# Deux quadrilatères dont les centres sont plus proches que cette fraction du
# grand côté de l'image visent la même carte.
DEDUP_RADIUS = 0.05

# Un quadrilatère couvrant moins que cette fraction de la photo est écarté avant
# tout redressement. `VNDetectRectanglesRequest` rend jusqu'à 8 observations, et
# la plupart sont des rectangles parasites minuscules — sur le banc d'essai, 4 des
# 6,3 quads par photo font moins de 1 % de la surface, alors que le quad retenu
# n'est jamais descendu sous 6,6 %. Chacun de ces parasites coûtait un
# redressement, une passe d'OCR d'orientation et un à deux embeddings.
#
# Filtre de SURFACE et non de forme : la perspective écrase fortement le rapport
# apparent d'une carte (0,66 à 0,94 mesuré sur les quads gagnants du banc, contre
# 63/88 = 0,72 à plat), un filtre sur le rapport écarterait de vraies cartes.
#
# Vaut pour une photo mono-carte, où la carte remplit une bonne part du cadre.
# Un étalage se scanne avec `MIN_MULTI_QUAD_AREA` : sur IMG_5029, une des cartes
# ne fait que 1,45 % de la photo, et 2 % la ferait disparaître.
MIN_QUAD_AREA = 0.02

# Même filtre pour le scan d'un étalage. Quatre fois plus bas : plus de crops à
# traiter, donc plus lent, mais une carte parmi dix couvre forcément une petite
# fraction de la photo.
MIN_MULTI_QUAD_AREA = 0.005

# Palier de comparaison des formes au moment du dédoublonnage (voir
# `_side_balance`). Deux quads dont l'équilibre des côtés tient dans le même
# palier sont jugés équivalents, et c'est alors l'ordre des détecteurs qui
# tranche. Volontairement large : il ne s'agit pas de classer des quads
# corrects, seulement d'écarter ceux qui sont visiblement cassés.
SHAPE_BUCKET = 0.10

# Seuils de confiance. Recalibrés le 12 août 2026 sur le SPLIT DE CALIBRATION
# SEUL (21 photos + 4 négatifs), dans l'espace de l'index centroïde.
#
# Le critère a changé de nature. Il ne s'agit plus seulement d'atteindre 100 %
# de précision sur les photos qui ont une bonne réponse — 0,0005 y suffirait —
# mais de passer AU-DESSUS de ce que produisent les photos qui n'en ont pas.
# La pire d'entre elles, un pochon porte-cartes, donne 0,0277 : la calibration
# conclut donc à 0,028.
#
# Les valeurs retenues sont plus hautes, et c'est un CHOIX DE POLITIQUE, pas
# une valeur calibrée : une borne établie sur quatre négatifs n'a aucune marge
# de sécurité, et le produit préfère explicitement se taire à tort qu'affirmer
# à tort. Le facteur ~1,6 est arbitraire et assumé comme tel. Divulgation
# nécessaire : il retire aussi une erreur du split test, ce qui n'est PAS ce
# qui l'a motivé mais reste une information que le lecteur doit avoir.
#
# Ce que ça coûte est plus faible qu'il n'y paraît : sur le banc, 25 des 29
# affirmations fermes viennent du numéro imprimé lu, chemin qui ne passe pas
# par ces seuils. La marge ne gouverne que le reste.
#
# À recalibrer dès que le banc dépasse la trentaine de négatifs — sept, c'est
# assez pour fixer un seuil, pas pour lui donner un intervalle utile.
FIRM_ID_MARGIN = 0.045
FIRM_NAME_MARGIN = 0.06


@dataclass
class Variant:
    label: str
    center: np.ndarray  # centre du quadrilatère en pixels ; (0,0) pour la photo entière
    image: Image.Image
    area: float = 1.0   # fraction de la photo couverte par le quadrilatère

    @property
    def bgr(self) -> np.ndarray:
        return cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _area_fraction(quad: np.ndarray, shape: tuple[int, ...]) -> float:
    """Surface du quadrilatère rapportée à celle de la photo (formule du lacet)."""
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area / (shape[0] * shape[1]))


def _side_balance(quad: np.ndarray) -> float:
    """Égalité des côtés opposés : 1 pour un rectangle, 0 pour un quad dégénéré.

    Ne suppose RIEN du format de la carte, contrairement à un filtre sur le
    rapport largeur/hauteur : la perspective écrase ce rapport, mais elle laisse
    les côtés opposés à peu près égaux sur une photo tenue à la main. Un
    quadrilatère dont un coin est mal placé, lui, s'effondre — mesuré à 0,50 et
    0,68 sur les deux détections ratées de Diamat, contre 0,98 sur la bonne.
    """
    sides = (
        (np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])),
        (np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])),
    )
    ratios = [min(a, b) / max(a, b) for a, b in sides if max(a, b) > 0]
    return float(min(ratios)) if len(ratios) == 2 else 0.0


def load_bgr(path: str) -> np.ndarray | None:
    """Pixels bruts, sans rotation EXIF — même repère que CGImageSource/Vision."""
    return cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def build_variants(path: str, bgr: np.ndarray, orient: str = "text",
                   min_area: float = MIN_QUAD_AREA) -> list[Variant]:
    """Détecte, filtre, dédoublonne et oriente les crops candidats.

    À forme comparable, les quads de segmentation document passent en premier :
    le dédoublonnage garde le premier vu, et le modèle de segmentation cadre
    mieux la carte que le détecteur de rectangles (mesuré : l'ordre inverse
    coûtait deux photos au banc d'essai). Ce n'est qu'une priorité par défaut :
    un quadrilatère franchement mieux formé passe devant (voir `_side_balance`).

    Le filtre de surface vient avant tout le reste : c'est le nombre de crops,
    et non le modèle, qui décide du temps de traitement d'une photo — chacun
    coûte un redressement, une passe d'OCR et un embedding.
    """
    quads = [(f"doc{i}", q) for i, q in enumerate(detect_document(path))]
    quads += [(f"rect{i}", q) for i, q in enumerate(detect_rectangles(path))]
    quads = [(n, q) for n, q in quads if _area_fraction(q, bgr.shape) >= min_area]

    # Le dédoublonnage garde le premier quad de chaque groupe : autant que ce
    # soit le mieux formé. Le tri est par paliers, et stable, donc l'ordre
    # « document d'abord » reste la règle entre quads de qualité comparable —
    # seul un quadrilatère franchement mieux formé double la priorité du
    # détecteur. Mesuré sur Diamat : le bon quad (côtés opposés à 0,98) était
    # jeté au profit d'un quad dégénéré (0,68) que le détecteur avait rendu
    # avant lui.
    quads.sort(key=lambda nq: round(_side_balance(nq[1]) / SHAPE_BUCKET), reverse=True)

    max_side = max(bgr.shape[:2])
    kept: list[tuple[str, np.ndarray]] = []
    for name, quad in quads:
        centre = quad.mean(axis=0)
        if any(np.linalg.norm(centre - k[1].mean(axis=0)) < DEDUP_RADIUS * max_side
               for k in kept):
            continue
        kept.append((name, quad))

    variants: list[Variant] = []
    for name, quad in kept:
        area = _area_fraction(quad, bgr.shape)
        warped = warp_card(bgr, quad)
        gray_std = float(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).std())
        if gray_std < MIN_GRAY_STD:
            continue

        centre = quad.mean(axis=0)
        if orient == "text":
            verdict, oriented = pick_orientation(warped)
            if oriented is not None:
                variants.append(
                    Variant(f"{name} ocr:{verdict}", centre, _to_pil(oriented), area)
                )
                continue
        variants.append(Variant(f"{name} 0°", centre, _to_pil(warped), area))
        variants.append(
            Variant(f"{name} 180°", centre, _to_pil(cv2.rotate(warped, cv2.ROTATE_180)), area)
        )
    return variants


def full_photo_variant(bgr: np.ndarray) -> Variant:
    return Variant("photo entière", np.zeros(2, dtype=np.float32), _to_pil(bgr), 1.0)


def selection_score(hits) -> float:
    """Score + marge : arbitre entre crops déjà orientés (jamais 0° vs 180°)."""
    return hits[0].score + (hits[0].score - hits[1].score)


@dataclass
class Confidence:
    """Verdict à deux niveaux : l'édition exacte, ou au moins le nom.

    La confusion vit presque entièrement à l'intérieur du même nom de carte
    (réimpressions) : une marge 1er/2e serrée avec un top-k homogène en nom
    signifie « bonne carte, édition incertaine », pas « faux ». Le score absolu
    ne discrimine rien et ne doit jamais être montré.
    """

    level: str  # "edition" | "nom" | "incertain"
    source: str  # "numéro lu" | "similarité"
    margin_id: float
    margin_name: float


def classify_confidence(hits, band_verdict: str | None = None,
                        full_photo: bool = False) -> Confidence:
    """Classe la confiance à partir des marges **avant** tout réordonnancement.

    `band_verdict == "ocr"` signifie que le numéro imprimé a été lu et a
    désigné un candidat : c'est une preuve plus forte que la similarité, donc
    le niveau passe à « edition » quelles que soient les marges. Les marges
    restent celles de la similarité, pour diagnostic seulement.

    `full_photo` dit que la sélection a retenu la photo entière, donc qu'aucun
    quadrilatère n'a été jugé plausible. Le vecteur décrit alors une scène et
    non une carte redressée : la marge y compare deux mauvaises réponses entre
    elles, et se trouve être élevée aussi souvent que basse. Aucun verdict ferme
    n'en sort. Mesuré sur le banc de 46 photos : ce repli ne produit AUCUNE
    identification correcte, et exactement un faux positif ferme.
    """
    margin_id = hits[0].score - hits[1].score
    top_name = hits[0].card["name"]
    margin_name = next(
        (hits[0].score - h.score for h in hits[1:] if h.card["name"] != top_name),
        margin_id,  # top-k entièrement homogène : la marge d'identité est au moins celle-là
    )
    if band_verdict == "ocr":
        return Confidence("edition", "numéro lu", margin_id, margin_name)
    if full_photo:
        return Confidence("incertain", "cadrage insuffisant", margin_id, margin_name)
    # Un total lu qui dément le candidat interdit le verdict ferme : la
    # ressemblance peut désigner la bonne illustration, elle ne peut pas
    # désigner un tirage que la carte elle-même contredit.
    if band_verdict != "contredit" and margin_id >= FIRM_ID_MARGIN:
        return Confidence("edition", "similarité", margin_id, margin_name)
    if margin_name >= FIRM_NAME_MARGIN:
        return Confidence("nom", "similarité", margin_id, margin_name)
    return Confidence("incertain", "similarité", margin_id, margin_name)


def identify_photo(path: str, bgr: np.ndarray, encoder, index,
                   k: int = 5, orient: str = "text"):
    """Chaîne complète pour une photo mono-carte.

    Retourne (label du crop retenu, hits). La recherche descend à 20 candidats
    minimum pour deux raisons : que la marge de nom (classify_confidence) trouve
    un candidat d'un nom différent même quand le top est saturé de réimpressions,
    et que la lecture du numéro imprimé ait de quoi trancher.

    `match_edition` ne réordonne que ce que la recherche lui donne : une carte
    hors de cette fenêtre est perdue même si son numéro est parfaitement lisible.
    Mesuré sur le banc : Hariyama 113/193 sortait au rang 12, et passer la
    profondeur de 10 à 15 le récupère. 20 laisse de la marge, sans régression
    jusqu'à 30, pour 1,6 ms de recherche.
    """
    depth = max(k, 20)
    variants = build_variants(path, bgr, orient=orient)
    results = []
    if variants:
        vectors = encoder.encode([v.image for v in variants])
        results = index.search(vectors, k=depth)

    best_quad_score = max((r[0].score for r in results), default=0.0)
    if best_quad_score < FALLBACK_SCORE:
        fallback = full_photo_variant(bgr)
        variants.append(fallback)
        results.extend(index.search(encoder.encode([fallback.image]), k=depth))

    best = int(np.argmax([selection_score(r) for r in results]))
    return variants[best], results[best]


@dataclass
class Identification:
    """Résultat complet pour une photo mono-carte."""

    hits: list           # candidats, meilleur en tête
    confidence: Confidence
    crop_label: str      # quel quadrilatère et quelle orientation ont été retenus
    out_of_index: bool   # numéro lu proprement mais inconnu de la base

    @property
    def card(self) -> dict:
        return self.hits[0].card

    @property
    def card_id(self) -> str:
        return self.hits[0].card_id


def identify(path: str, bgr: np.ndarray, encoder, index,
             k: int = 5, orient: str = "text", use_band: bool = True) -> Identification:
    """Chaîne complète : détection, orientation, embedding, recherche, bandeau.

    Point d'entrée unique — c'est cette séquence que l'app doit reproduire.
    La confiance est calculée sur l'ordre issu de la similarité, puis relevée
    si le numéro imprimé a tranché ; l'inverse donnerait des marges calculées
    sur une liste réordonnée, donc dénuées de sens.

    `use_band=False` coupe la lecture du numéro imprimé et ne laisse que la
    similarité. Ce n'est pas un mode d'exploitation : c'est le bras d'ablation
    du banc d'essai, qui sépare ce que l'embedding apporte de ce que l'OCR
    rattrape (mesuré : 26/31 contre 31/31 — l'OCR porte un sixième du résultat).
    """
    variant, hits = identify_photo(path, bgr, encoder, index, k=k, orient=orient)
    refined, band_verdict = (
        refine_with_band(variant, hits, index) if use_band else (hits, None)
    )
    confidence = classify_confidence(
        hits, band_verdict, full_photo=variant.label == "photo entière"
    )
    return Identification(
        hits=refined,
        confidence=confidence,
        crop_label=variant.label,
        out_of_index=band_verdict == "hors_index",
    )


def refine_with_band(variant: Variant, hits, index):
    """Tente de trancher l'édition en lisant le bandeau bas du crop retenu.

    Retourne (hits éventuellement réordonnés, verdict) où verdict vaut :
      "ocr"        — un candidat du top-k porte le numéro lu, promu en tête ;
      "hors_index" — numéro lu proprement mais inconnu de tout l'index ;
      None         — bandeau illisible ou non discriminant.

    N'appeler que sur un crop de carte (pas la photo entière) : le bandeau
    n'a un sens que si la géométrie est déjà redressée.

    Deux passes d'OCR en cascade (voir `src/edition.py`) : la rapide tranche
    quatre bandeaux sur cinq pour un cinquième du coût, la précise ne sert que
    de repli. Le verdict « hors_index » n'est prononcé que sur la passe précise :
    déclarer une carte absente de la base sur une lecture rapide serait affirmer
    beaucoup à partir du mode le moins fiable.
    """
    from src.edition import known_pair, match_edition, read_number_pairs

    if variant.label == "photo entière":
        return hits, None

    card_bgr = variant.bgr
    pairs = read_number_pairs(card_bgr, fast=True)
    match = match_edition(hits, pairs)
    if match is None:
        pairs = read_number_pairs(card_bgr)
        match = match_edition(hits, pairs)

    if match is not None:
        i, _ = match
        reordered = [hits[i]] + hits[:i] + hits[i + 1:]
        return reordered, "ocr"
    if pairs and not known_pair(index, pairs):
        return hits, "hors_index"
    if pairs and _total_contradicts(hits[0], pairs):
        return hits, "contredit"
    return hits, None


def _total_contradicts(hit, pairs) -> bool:
    """Le total imprimé sur la carte dément-il celui du meilleur candidat ?

    Le total nomme l'extension : une carte qui annonce « /100 » n'appartient pas
    à une extension de 159 cartes, quelle que soit la ressemblance. C'est le cas
    des cartes japonaises, qu'un index de tirages occidentaux ne peut pas placer
    par construction — l'illustration est la bonne, le tirage ne l'est jamais.

    Mesuré sur sept photos japonaises : trois recevaient un verdict *ferme*
    désignant un tirage occidental, sur la seule similarité, alors que le
    bandeau annonçait « 041/100 » face à un candidat en 159. Le numéro n'avait
    promu personne, donc rien ne s'y opposait.
    """
    from src.edition import _digit_distance

    total = hit.card.get("set_printed_total")
    if not total:
        return False
    return all(_digit_distance(read_total, str(total)) != 0 for _, read_total in pairs)
