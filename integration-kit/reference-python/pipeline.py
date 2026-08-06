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

# Seuils de confiance, calibrés sur le banc d'essai (18 photos) : marge
# 1er/2e >= 0,03 donne 100 % de précision sur l'édition exacte ; marge au
# premier candidat d'un NOM différent >= 0,04 donne 100 % sur l'identité.
# À reconfirmer sur 40+ photos avant de figer côté app.
FIRM_ID_MARGIN = 0.03
FIRM_NAME_MARGIN = 0.04


@dataclass
class Variant:
    label: str
    center: np.ndarray  # centre du quadrilatère en pixels ; (0,0) pour la photo entière
    image: Image.Image

    @property
    def bgr(self) -> np.ndarray:
        return cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_bgr(path: str) -> np.ndarray | None:
    """Pixels bruts, sans rotation EXIF — même repère que CGImageSource/Vision."""
    return cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def build_variants(path: str, bgr: np.ndarray, orient: str = "text") -> list[Variant]:
    """Détecte, dédoublonne, filtre et oriente les crops candidats.

    Les quads de segmentation document passent en premier : à centre égal, le
    dédoublonnage garde le premier vu, et le modèle de segmentation cadre mieux
    la carte que le détecteur de rectangles (mesuré : l'ordre inverse coûtait
    deux photos au banc d'essai).
    """
    quads = [(f"doc{i}", q) for i, q in enumerate(detect_document(path))]
    quads += [(f"rect{i}", q) for i, q in enumerate(detect_rectangles(path))]

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
        warped = warp_card(bgr, quad)
        gray_std = float(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).std())
        if gray_std < MIN_GRAY_STD:
            continue

        centre = quad.mean(axis=0)
        if orient == "text":
            verdict, oriented = pick_orientation(warped)
            if oriented is not None:
                variants.append(Variant(f"{name} ocr:{verdict}", centre, _to_pil(oriented)))
                continue
        variants.append(Variant(f"{name} 0°", centre, _to_pil(warped)))
        variants.append(
            Variant(f"{name} 180°", centre, _to_pil(cv2.rotate(warped, cv2.ROTATE_180)))
        )
    return variants


def full_photo_variant(bgr: np.ndarray) -> Variant:
    return Variant("photo entière", np.zeros(2, dtype=np.float32), _to_pil(bgr))


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


def classify_confidence(hits, band_verdict: str | None = None) -> Confidence:
    """Classe la confiance à partir des marges **avant** tout réordonnancement.

    `band_verdict == "ocr"` signifie que le numéro imprimé a été lu et a
    désigné un candidat : c'est une preuve plus forte que la similarité, donc
    le niveau passe à « edition » quelles que soient les marges. Les marges
    restent celles de la similarité, pour diagnostic seulement.
    """
    margin_id = hits[0].score - hits[1].score
    top_name = hits[0].card["name"]
    margin_name = next(
        (hits[0].score - h.score for h in hits[1:] if h.card["name"] != top_name),
        margin_id,  # top-k entièrement homogène : la marge d'identité est au moins celle-là
    )
    if band_verdict == "ocr":
        return Confidence("edition", "numéro lu", margin_id, margin_name)
    if margin_id >= FIRM_ID_MARGIN:
        return Confidence("edition", "similarité", margin_id, margin_name)
    if margin_name >= FIRM_NAME_MARGIN:
        return Confidence("nom", "similarité", margin_id, margin_name)
    return Confidence("incertain", "similarité", margin_id, margin_name)


def identify_photo(path: str, bgr: np.ndarray, encoder, index,
                   k: int = 5, orient: str = "text"):
    """Chaîne complète pour une photo mono-carte.

    Retourne (label du crop retenu, hits). La recherche descend à 10 candidats
    minimum pour que la marge de nom (classify_confidence) trouve un candidat
    d'un nom différent même quand le top est saturé de réimpressions.
    """
    depth = max(k, 10)
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
             k: int = 5, orient: str = "text") -> Identification:
    """Chaîne complète : détection, orientation, embedding, recherche, bandeau.

    Point d'entrée unique — c'est cette séquence que l'app doit reproduire.
    La confiance est calculée sur l'ordre issu de la similarité, puis relevée
    si le numéro imprimé a tranché ; l'inverse donnerait des marges calculées
    sur une liste réordonnée, donc dénuées de sens.
    """
    variant, hits = identify_photo(path, bgr, encoder, index, k=k, orient=orient)
    refined, band_verdict = refine_with_band(variant, hits, index)
    confidence = classify_confidence(hits, band_verdict)
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
    """
    from src.edition import known_pair, match_edition, read_number_pairs

    if variant.label == "photo entière":
        return hits, None
    pairs = read_number_pairs(variant.bgr)
    if not pairs:
        return hits, None

    match = match_edition(hits, pairs)
    if match is not None:
        i, _ = match
        reordered = [hits[i]] + hits[:i] + hits[i + 1:]
        return reordered, "ocr"
    if not known_pair(index, pairs):
        return hits, "hors_index"
    return hits, None
