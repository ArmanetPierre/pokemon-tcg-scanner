"""Évalue la chaîne complète sur les photos réelles listées dans truth.json.

    python scripts/evaluate_real.py
    python scripts/evaluate_real.py --ablation        # ce qu'apporte chaque étage
    python scripts/evaluate_real.py --coreml          # avec le modèle embarqué
    python scripts/evaluate_real.py --json out.json   # sortie exploitable

Trois règles de protocole, appliquées par ce script et non laissées à la
discipline du lecteur :

1. **Tout taux sort avec son intervalle.** 31/31 se lit comme 100 % ; l'IC de
   Wilson à 95 % descend à 89 %. Sur 31 photos, c'est ce dernier chiffre qui
   est défendable.
2. **Calibration et test sont séparés et rapportés séparément.** Le split est
   inscrit dans truth.json, avec la liste précise de ce dont le split test est
   — et n'est pas — du hold-out.
3. **La confiance se mesure par une courbe risque/couverture, pas par un
   seuil.** Un seuil n'est comparable ni entre deux modèles ni entre deux
   espaces métriques ; une courbe l'est toujours.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detect import find_card_quads, warp_card  # noqa: E402
from src.detect_vision import detect_document, detect_rectangles  # noqa: E402
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.orient import pick_orientation  # noqa: E402
from src.pipeline import full_photo_variant, identify  # noqa: E402
from src.search import CardIndex  # noqa: E402
from src.stats import fmt_ratio, render_risk_coverage, risk_coverage  # noqa: E402

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
TRUTH = ML_ROOT / "data" / "eval" / "truth.json"
CROP_DIR = ML_ROOT / "data" / "eval" / "crops"

SPLITS = ("calibration", "test")


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_bgr(path: str) -> np.ndarray | None:
    """Charge les pixels bruts, sans appliquer la rotation EXIF.

    `cv2.imread` redresse par défaut selon l'EXIF, alors que
    `CGImageSourceCreateImageAtIndex` renvoie les pixels tels quels. Les
    quadrilatères de Vision seraient alors exprimés dans un repère différent de
    l'image découpée — et les crops sortiraient hors cadre.
    """
    return cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


class ProjectedEncoder:
    """Encodeur enveloppé d'une projection apprise (`scripts/train_projection.py`).

    La projection s'applique symétriquement aux requêtes et à l'index. En
    l'insérant ici plutôt que dans `src/`, le banc peut comparer avec et sans
    sans qu'aucun code de production ne dépende d'un artefact expérimental.
    """

    def __init__(self, encoder, matrix: np.ndarray):
        self._encoder = encoder
        self._matrix = matrix
        self.name = f"{getattr(encoder, 'name', 'encoder')}+projection"

    def encode(self, images, **kwargs) -> np.ndarray:
        vectors = self._encoder.encode(images, **kwargs) @ self._matrix
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def apply_projection(index, matrix: np.ndarray) -> None:
    """Projette l'index sur place, pour que `index.search` reste inchangé."""
    projected = index.embeddings.astype(np.float32) @ matrix
    index.embeddings = projected / np.linalg.norm(projected, axis=1, keepdims=True)


@dataclass
class Record:
    """Une photo passée dans la chaîne, sous une configuration donnée."""

    path: str
    split: str
    expected: str | None
    label_fr: str
    conditions: list[str] = field(default_factory=list)
    predicted: str | None = None
    top_ids: list[str] = field(default_factory=list)
    top1_score: float = 0.0   # score absolu du meilleur candidat (pour FALLBACK_SCORE)
    margin_id: float = 0.0
    margin_name: float = 0.0
    level: str = "incertain"
    source: str = ""
    out_of_index: bool = False
    crop_label: str = ""

    @property
    def answerable(self) -> bool:
        return self.expected is not None

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected

    @property
    def in_top_k(self) -> bool:
        return self.expected in self.top_ids

    @property
    def firm(self) -> bool:
        """Verdict affiché fermement à l'utilisateur (édition annoncée)."""
        return self.level == "edition"

    @property
    def refused(self) -> bool:
        return self.out_of_index or self.level == "incertain"


def run_arm(truth: dict, encoder, index, top_k: int, *,
            detect: bool = True, use_band: bool = True,
            verbose: bool = False) -> list[Record]:
    """Passe toutes les photos dans une configuration donnée.

    `detect=False` court-circuite détection et orientation : la photo entière
    part directement à l'embedding. C'est le bras de référence basse, celui qui
    dit ce que vaut le modèle sans le travail de cadrage.
    """
    records: list[Record] = []
    for rel_path, info in truth.items():
        bgr = load_bgr(str(PROJECT_ROOT / rel_path))
        if bgr is None:
            print(f"illisible : {rel_path}", file=sys.stderr)
            continue

        rec = Record(path=rel_path, split=info.get("split", "calibration"),
                     expected=info["card_id"], label_fr=info["fr"],
                     conditions=info.get("conditions", []))

        if detect:
            result = identify(str(PROJECT_ROOT / rel_path), bgr, encoder, index,
                              k=top_k, use_band=use_band)
            hits = result.hits
            rec.level = result.confidence.level
            rec.source = result.confidence.source
            rec.margin_id = result.confidence.margin_id
            rec.margin_name = result.confidence.margin_name
            rec.out_of_index = result.out_of_index
            rec.crop_label = result.crop_label
        else:
            variant = full_photo_variant(bgr)
            hits = index.search(encoder.encode([variant.image]), k=max(top_k, 20))[0]
            rec.margin_id = hits[0].score - hits[1].score
            rec.crop_label = variant.label

        rec.predicted = hits[0].card_id
        rec.top_ids = [h.card_id for h in hits[:top_k]]
        rec.top1_score = float(hits[0].score)
        records.append(rec)

        if verbose and rec.answerable:
            status = ("TOP-1" if rec.correct else
                      f"rang {rec.top_ids.index(rec.expected) + 1}" if rec.in_top_k
                      else f"hors top-{top_k}")
            print(f"\n{Path(rel_path).name}  {info['fr']} ({rec.expected})  "
                  f"[{rec.crop_label} · {rec.level}/{rec.source} · {rec.split}]  -> {status}")
            for i, hit in enumerate(hits[:top_k], 1):
                mark = " <<<" if hit.card_id == rec.expected else ""
                print(f"   {i}. {hit.score:.4f} {hit.card['name']:<22} "
                      f"{hit.card['set_name'][:26]:<27}{mark}")
    return records


def report(records: list[Record], top_k: int, title: str) -> dict:
    """Tableau par split, avec intervalles, courbe risque/couverture et refus."""
    answerable = [r for r in records if r.answerable]
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")

    print(f"\n{'split':<14} {'top-1':<30} {'top-' + str(top_k):<30}")
    print("-" * 74)
    summary: dict = {}
    for split in SPLITS + ("TOTAL",):
        rows = answerable if split == "TOTAL" else [r for r in answerable if r.split == split]
        if not rows:
            continue
        t1 = sum(r.correct for r in rows)
        tk = sum(r.in_top_k for r in rows)
        print(f"{split:<14} {fmt_ratio(t1, len(rows)):<30} {fmt_ratio(tk, len(rows)):<30}")
        summary[split] = {"n": len(rows), "top1": t1, f"top{top_k}": tk}
    print("-" * 74)
    print(f"  IC95 de Wilson. Même sans une seule erreur, {len(answerable)} photos ne peuvent pas")
    print("  démontrer mieux que leur borne basse — c'est la taille du banc qui parle.")

    # --- décision réellement affichée à l'utilisateur ---
    firm = [r for r in answerable if r.firm]
    wrong_firm = [r for r in firm if not r.correct]
    print(f"\nverdicts fermes (« edition ») : {fmt_ratio(len(firm), len(answerable))} du banc")
    print(f"  dont justes : {fmt_ratio(sum(r.correct for r in firm), len(firm))}")
    if wrong_firm:
        print("  FAUX POSITIFS FERMES — le pire résultat possible :")
        for r in wrong_firm:
            print(f"    {Path(r.path).name}  attendu {r.expected}, annoncé {r.predicted} "
                  f"(marge {r.margin_id:.4f}, source {r.source})")
    summary["firm"] = {"n": len(firm), "correct": len(firm) - len(wrong_firm)}

    # --- courbe risque/couverture sur la marge de similarité ---
    rc = risk_coverage([r.margin_id for r in answerable], [r.correct for r in answerable])
    cov_100, thr_100 = rc.coverage_at(1.0)
    cov_95, thr_95 = rc.coverage_at(0.95)
    print("\nrisque/couverture, tri par marge 1er/2e (échelle-invariante)")
    print(f"  couverture      {'100%':>7} {'75%':>7} {'50%':>7} {'25%':>7}")
    print(f"  précision      {render_risk_coverage(rc)}")
    print(f"  AURC {rc.aurc:.4f}  (0 = tri parfait ; plus bas = mieux)")
    print(f"  précision 100 % jusqu'à {cov_100:.0%} de couverture (marge >= {thr_100:.4f})")
    print(f"  précision  95 % jusqu'à {cov_95:.0%} de couverture (marge >= {thr_95:.4f})")
    summary["aurc"] = rc.aurc
    summary["coverage_at_100"] = cov_100
    summary["threshold_at_100"] = thr_100

    # --- photos sans réponse juste possible ---
    no_answer = [r for r in records if not r.answerable]
    if no_answer:
        print(f"\n{len(no_answer)} photo(s) sans réponse juste possible — "
              "le seul verdict acceptable est un aveu d'ignorance")
        for r in no_answer:
            verdict = "hors index" if r.out_of_index else r.level
            flag = "correct : rien affirmé" if r.refused else "FAUX POSITIF"
            print(f"  {Path(r.path).name:<20} [{r.split}] -> {verdict}  ({flag})")
        summary["refusals"] = {"n": len(no_answer), "ok": sum(r.refused for r in no_answer)}

    return summary


def calibrate(records: list[Record]) -> None:
    """Propose des seuils à partir du SPLIT DE CALIBRATION seul.

    À lancer après tout changement qui déplace l'échelle des scores — un autre
    encodeur, une projection apprise, une autre métrique. Les trois constantes
    de `src/pipeline.py` sont des seuils sur des grandeurs absolues : elles ne
    survivent pas à un changement d'espace, et les garder telles quelles produit
    des symptômes trompeurs (le dos de carte annoncé fermement, par exemple)
    qu'on impute à tort au nouveau modèle.

    Le split test n'est jamais regardé ici : c'est ce qui lui permet de rester
    une mesure.
    """
    cal = [r for r in records if r.split == "calibration" and r.answerable]
    negatifs = [r for r in records if r.split == "calibration" and not r.answerable]
    if not cal:
        print("\npas de photos de calibration : rien à proposer")
        return

    print(f"\n{'=' * 74}\nSeuils proposés — calibration seule "
          f"({len(cal)} photos + {len(negatifs)} négatifs)\n{'=' * 74}")

    rc = risk_coverage([r.margin_id for r in cal], [r.correct for r in cal])
    cov, thr_pos = rc.coverage_at(1.0)
    print(f"\nsur les positifs   : marge >= {thr_pos:.4f} donne 100 % de précision "
          f"({cov:.0%} de couverture)")

    # Un seuil calibré sur les seuls positifs ne dit rien du cas le plus grave :
    # une photo SANS bonne réponse à laquelle le système en donne une quand
    # même. Le seuil doit donc aussi passer au-dessus de ce que les négatifs
    # obtiennent — un dos de carte, un flou illisible, une carte d'un autre jeu
    # produisent eux aussi une marge, et rien ne garantit qu'elle soit petite.
    thr = thr_pos
    if negatifs:
        pire = max(negatifs, key=lambda r: r.margin_id)
        print(f"sur les négatifs   : la pire marge est {pire.margin_id:.4f} "
              f"({Path(pire.path).name}, {', '.join(pire.conditions)})")
        if pire.margin_id >= thr_pos:
            thr = pire.margin_id + 0.0001
            perdu = sum(1 for r in cal if thr_pos <= r.margin_id < thr)
            print(f"  -> elle DÉPASSE le seuil des positifs : il faut monter à "
                  f"{thr:.4f}, au prix de {perdu} identification(s) fermes perdue(s).")
        else:
            print("  -> elle reste sous le seuil des positifs : pas de contrainte.")
    else:
        print("sur les négatifs   : AUCUN en calibration — le comportement de refus")
        print("                     n'est donc calibré sur rien, quel que soit le seuil.")

    couverture = sum(1 for r in cal if r.margin_id >= thr) / len(cal)
    print(f"\nFIRM_ID_MARGIN   >= {thr:.4f}   ({couverture:.0%} des positifs restent fermes)")

    # Marge de nom : plus petite marge observée parmi les photos dont le premier
    # candidat porte le BON nom. En dessous, le niveau « nom » cesse d'être sûr.
    justes = [r for r in cal if r.correct]
    if justes:
        plancher = min(r.margin_name for r in justes)
        if negatifs:
            plancher = max(plancher, max(r.margin_name for r in negatifs) + 0.0001)
        print(f"FIRM_NAME_MARGIN >= {plancher:.4f}   "
              f"(plancher des {len(justes)} réponses justes, relevé au-dessus des négatifs)")

    # FALLBACK_SCORE sépare « un crop a marché » de « aucun crop n'a marché ».
    # Le repli photo-entière ne doit se déclencher que sous le pire vrai crop.
    scores = sorted(r.top1_score for r in justes)
    if scores:
        print(f"FALLBACK_SCORE   <  {scores[0]:.4f}   "
              f"(plus bas score top-1 d'un crop juste ; médiane {np.median(scores):.4f})")
    print("\nÀ reporter dans src/pipeline.py, puis à revérifier sur le split test —")
    print("qui n'a servi à rien de ce qui précède.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument(
        "--coreml", action="store_true",
        help="encoder les requêtes avec le modèle Core ML exporté (index inchangé)",
    )
    parser.add_argument("--detector", choices=("pipeline", "vision", "opencv", "none"),
                        default="pipeline")
    parser.add_argument(
        "--orient", choices=("text", "both"), default="both",
        help="text = trancher 0°/180° par OCR ; both = garder les deux en concurrence",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--ablation", action="store_true",
        help="mesurer ce que chaque étage apporte, du modèle nu à la chaîne complète",
    )
    parser.add_argument("--json", type=Path, help="écrire le résumé en JSON")
    parser.add_argument("--quiet", action="store_true", help="tableaux seuls, sans le détail")
    parser.add_argument(
        "--projection", type=Path,
        help="projection apprise, appliquée aux requêtes ET à l'index",
    )
    parser.add_argument(
        "--embeddings", type=Path,
        help="index alternatif (ex. centroïde multi-vues), aligné sur card_ids",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="proposer des seuils à partir du split de calibration seul",
    )
    # Les trois constantes de src/pipeline.py sont des seuils sur des grandeurs
    # absolues. Les surcharger ici permet de vérifier une recalibration sans
    # toucher au code de production tant que le gain n'est pas démontré.
    parser.add_argument("--firm-id-margin", type=float)
    parser.add_argument("--firm-name-margin", type=float)
    parser.add_argument("--fallback-score", type=float)
    parser.add_argument(
        "--selection-margin-weight", type=float,
        help="poids du terme de marge dans selection_score (1 = comportement actuel)",
    )
    args = parser.parse_args()

    from src import pipeline as pipeline_module

    if args.selection_margin_weight is not None:
        # selection_score additionne un score et une marge, ce qui suppose
        # implicitement qu'ils vivent sur la même échelle. Changer d'espace
        # rompt cette hypothèse sans rien signaler : une projection qui
        # multiplie les marges par 5 transforme « score + marge » en « marge
        # seule » — un régime que le projet a mesuré comme défaillant (un crop
        # plat se détache nettement sur une carte Énergie).
        weight = args.selection_margin_weight
        pipeline_module.selection_score = (
            lambda hits: hits[0].score + weight * (hits[0].score - hits[1].score)
        )
        print(f"poids de la marge dans la sélection : {weight}")

    for attr, value in (("FIRM_ID_MARGIN", args.firm_id_margin),
                        ("FIRM_NAME_MARGIN", args.firm_name_margin),
                        ("FALLBACK_SCORE", args.fallback_score)):
        if value is not None:
            print(f"seuil surchargé : {attr} {getattr(pipeline_module, attr)} -> {value}")
            setattr(pipeline_module, attr, value)

    truth = json.loads(TRUTH.read_text())["photos"]
    if args.coreml:
        from src.coreml_encoder import CoreMLEncoder

        encoder = CoreMLEncoder()
        print(f"encodeur : {encoder.name} (index construit avec {args.model})")
    else:
        encoder = load_encoder(args.model)
    index = CardIndex(args.model)

    if args.embeddings:
        alt = np.load(args.embeddings).astype(np.float32)
        alt /= np.linalg.norm(alt, axis=1, keepdims=True)
        index.embeddings = alt
        print(f"index alternatif : {args.embeddings.name}")

    if args.projection:
        matrix = np.load(args.projection).astype(np.float32)
        encoder = ProjectedEncoder(encoder, matrix)
        apply_projection(index, matrix)
        print(f"projection {matrix.shape} appliquée aux requêtes et à l'index")
        # FALLBACK_SCORE compare un score de similarité ABSOLU à 0,65. Une
        # projection change l'échelle de ces scores, donc ce seuil ne veut plus
        # rien dire : c'est exactement le piège « un seuil n'est pas comparable
        # entre deux espaces » que le protocole documente. Tant qu'il n'est pas
        # recalibré, le repli photo-entière se déclenche presque toujours et la
        # comparaison avec/sans projection n'est pas à configuration égale.
        print("  ATTENTION : FALLBACK_SCORE (0,65) est un seuil sur un score absolu,")
        print("  non transposable. À recalibrer avant de conclure sur cette mesure.")

    if args.detector == "pipeline":
        # Chaque bras ajoute un étage au précédent : l'écart entre deux lignes
        # est exactement ce que cet étage apporte, mesuré et non supposé.
        arms = [("chaîne complète", dict(detect=True, use_band=True))]
        if args.ablation:
            arms = [
                ("1. photo entière, sans détection ni orientation",
                 dict(detect=False, use_band=False)),
                ("2. + détection, filtrage, orientation (similarité seule)",
                 dict(detect=True, use_band=False)),
                ("3. + lecture du numéro imprimé (chaîne complète)",
                 dict(detect=True, use_band=True)),
            ]

        out = {}
        for title, kwargs in arms:
            records = run_arm(truth, encoder, index, args.top_k,
                              verbose=not args.quiet and not args.ablation, **kwargs)
            out[title] = report(records, args.top_k, title)
            if args.calibrate and kwargs.get("use_band"):
                calibrate(records)

        if args.ablation:
            print(f"\n{'=' * 74}\nCe que chaque étage apporte\n{'=' * 74}")
            print(f"\n{'configuration':<52} {'top-1':<22}")
            print("-" * 74)
            prev = None
            for title in out:
                s = out[title]["TOTAL"]
                delta = "" if prev is None else f"  {s['top1'] - prev:+d}"
                print(f"{title:<52} {fmt_ratio(s['top1'], s['n'])}{delta}")
                prev = s["top1"]
            print("-" * 74)

        if args.json:
            args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(f"\nrésumé écrit : {args.json}")
        return 0

    return run_manual_detector(truth, encoder, index, args)


def run_manual_detector(truth: dict, encoder, index, args) -> int:
    """Bras historique : choisir le détecteur à la main, sans la logique du pipeline.

    Conservé pour comparer des détecteurs entre eux (`--detector vision|opencv|none`)
    et pour produire les crops de diagnostic. La sélection y est refaite à la
    main plutôt qu'appelée depuis `src.pipeline`.
    """
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    top1 = top5 = n = 0
    for rel_path, info in truth.items():
        expected = info["card_id"]
        if expected is None:
            continue
        bgr = load_bgr(str(PROJECT_ROOT / rel_path))
        if bgr is None:
            print(f"illisible : {rel_path}")
            continue

        variants: list[tuple[str, Image.Image]] = []
        quads: list[tuple[str, np.ndarray]] = []
        if args.detector == "opencv":
            quads = [(f"cv{i}", q) for i, q in enumerate(find_card_quads(bgr))]
        elif args.detector == "vision":
            full_path = str(PROJECT_ROOT / rel_path)
            quads = [(f"rect{i}", q) for i, q in enumerate(detect_rectangles(full_path))]
            quads += [(f"doc{i}", q) for i, q in enumerate(detect_document(full_path))]

        for name, quad in quads:
            warped = warp_card(bgr, quad)
            if args.orient == "text":
                verdict, oriented = pick_orientation(warped)
                if oriented is not None:
                    variants.append((f"{name} ocr:{verdict}", to_pil(oriented)))
                    continue
            variants.append((f"{name} 0°", to_pil(warped)))
            variants.append((f"{name} 180°", to_pil(cv2.rotate(warped, cv2.ROTATE_180))))
        # La photo entière reste toujours en lice : quand la carte remplit le
        # cadre, ses bords sortent de l'image et aucun détecteur ne peut fermer
        # un quadrilatère.
        variants.append(("photo entière", to_pil(bgr)))

        vectors = encoder.encode([img for _, img in variants])
        results = index.search(vectors, k=max(args.top_k, 2))
        best_i = int(np.argmax([r[0].score + (r[0].score - r[1].score) for r in results]))
        label, image = variants[best_i]
        hits = results[best_i]
        ids = [h.card_id for h in hits]

        n += 1
        top1 += ids[0] == expected
        hitk = expected in ids
        top5 += hitk
        image.save(CROP_DIR / f"{Path(rel_path).stem}.jpg", quality=90)

        rank = ids.index(expected) + 1 if hitk else None
        status = "TOP-1" if ids[0] == expected else (f"rang {rank}" if rank else "hors top-5")
        print(f"\n{Path(rel_path).name}  {info['fr']} ({expected})  [{label}]  -> {status}")
        for i, hit in enumerate(hits, 1):
            mark = " <<<" if hit.card_id == expected else ""
            print(f"   {i}. {hit.score:.4f} {hit.card['name']:<22} "
                  f"{hit.card['set_name'][:26]:<27}{mark}")

    print(f"\ndétecteur {args.detector} — top-1 {fmt_ratio(top1, n)}   "
          f"top-{args.top_k} {fmt_ratio(top5, n)}")
    print(f"crops écrits dans {CROP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
