"""Évalue la chaîne complète sur les photos réelles listées dans truth.json.

    python scripts/evaluate_real.py
    python scripts/evaluate_real.py --no-detect   # photo brute, sans détection

Compare l'identification sur la photo entière et sur le crop redressé. C'est la
mesure qui dit si l'effort doit porter sur le modèle d'embedding ou sur l'étape
de détection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detect import find_card_quads, warp_card  # noqa: E402
from src.detect_vision import detect_document, detect_rectangles  # noqa: E402
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.orient import pick_orientation  # noqa: E402
from src.pipeline import identify  # noqa: E402
from src.search import CardIndex  # noqa: E402

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
TRUTH = ML_ROOT / "data" / "eval" / "truth.json"
CROP_DIR = ML_ROOT / "data" / "eval" / "crops"


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_bgr(path: str) -> np.ndarray | None:
    """Charge les pixels bruts, sans appliquer la rotation EXIF.

    `cv2.imread` redresse par défaut selon l'EXIF, alors que
    `CGImageSourceCreateImageAtIndex` renvoie les pixels tels quels. Les
    quadrilatères de Vision seraient alors exprimés dans un repère différent de
    l'image découpée — et les crops sortiraient hors cadre. L'orientation finale
    n'a pas d'importance ici : `warp_card` remet la carte en portrait et les
    deux variantes 0°/180° couvrent le reste.
    """
    return cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


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
    args = parser.parse_args()

    truth = json.loads(TRUTH.read_text())["photos"]
    if args.coreml:
        from src.coreml_encoder import CoreMLEncoder

        encoder = CoreMLEncoder()
        print(f"encodeur : {encoder.name} (index construit avec {args.model})")
    else:
        encoder = load_encoder(args.model)
    index = CardIndex(args.model)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    top1 = top5 = 0
    no_answer: list[tuple[str, dict]] = []
    for rel_path, info in truth.items():
        expected = info["card_id"]
        if expected is None:
            # Aucune réponse juste n'existe : carte hors index, ou pas de carte
            # du tout. Ces photos ne comptent pas dans le top-1, mais elles sont
            # passées dans la chaîne à part (voir plus bas) — c'est là qu'on
            # vérifie que le pipeline se tait au lieu d'inventer.
            no_answer.append((rel_path, info))
            continue
        bgr = load_bgr(str(PROJECT_ROOT / rel_path))
        if bgr is None:
            print(f"illisible : {rel_path}")
            continue

        if args.detector == "pipeline":
            result = identify(
                str(PROJECT_ROOT / rel_path), bgr, encoder, index, k=args.top_k
            )
            hits = result.hits
            label = f"{result.crop_label} [{result.confidence.level}/{result.confidence.source}]"
            ids = [h.card_id for h in hits]
            hit1 = ids[0] == expected
            hitk = expected in ids[: args.top_k]
            top1 += hit1
            top5 += hitk
            rank = ids.index(expected) + 1 if hitk else None
            status = "TOP-1" if hit1 else (f"rang {rank}" if rank else "hors top-5")
            print(f"\n{Path(rel_path).name}  {info['fr']} ({expected})  [{label}]  -> {status}")
            for i, hit in enumerate(hits[: args.top_k], 1):
                mark = " <<<" if hit.card_id == expected else ""
                print(
                    f"   {i}. {hit.score:.4f} {hit.card['name']:<22} "
                    f"{hit.card['set_name'][:26]:<27}{mark}"
                )
            continue

        # Chaque variante est une hypothèse (quel quadrilatère, quelle
        # orientation) ; on garde celle qui obtient le meilleur score, ce qui
        # est exactement la logique que l'app appliquera.
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
                # OCR non décisif : retomber sur les deux hypothèses.
            variants.append((f"{name} 0°", to_pil(warped)))
            variants.append((f"{name} 180°", to_pil(cv2.rotate(warped, cv2.ROTATE_180))))
        # Toujours garder la photo entière comme hypothèse : quand la carte
        # remplit le cadre, ses bords sortent de l'image et aucun détecteur ne
        # peut fermer un quadrilatère.
        variants.append(("photo entière", to_pil(bgr)))

        vectors = encoder.encode([img for _, img in variants])
        results = index.search(vectors, k=max(args.top_k, 2))

        # Sélection par score + marge, mesuré meilleur que l'un ou l'autre seul
        # (4/6 contre 3/6 sur le jeu de photos réelles).
        # Le score seul se fait piéger par une carte à l'envers, qui ressemble
        # encore à une carte ; la marge seule se fait piéger par un crop plat
        # (bout de table, écran) qui se détache nettement sur une carte Énergie,
        # elle aussi presque unie.
        best_i = int(np.argmax([r[0].score + (r[0].score - r[1].score) for r in results]))
        label, image = variants[best_i]
        hits = results[best_i]
        ids = [h.card_id for h in hits]

        hit1 = ids[0] == expected
        hitk = expected in ids
        top1 += hit1
        top5 += hitk
        rank = ids.index(expected) + 1 if hitk else None

        image.save(CROP_DIR / f"{Path(rel_path).stem}.jpg", quality=90)

        status = "TOP-1" if hit1 else (f"rang {rank}" if rank else "hors top-5")
        print(f"\n{Path(rel_path).name}  {info['fr']} ({expected})  [{label}]  -> {status}")
        for i, hit in enumerate(hits, 1):
            mark = " <<<" if hit.card_id == expected else ""
            print(
                f"   {i}. {hit.score:.4f} {hit.card['name']:<22} "
                f"{hit.card['set_name'][:26]:<27}{mark}"
            )

    n = len(truth) - len(no_answer)
    mode = f"détecteur {args.detector}"
    print(f"\n{mode} — top-1 {top1}/{n}   top-{args.top_k} {top5}/{n}")

    # Photos sans réponse juste. Le seul verdict acceptable est un aveu
    # d'ignorance : « incertain », ou « hors index » quand le numéro imprimé a
    # été lu proprement sans correspondre à rien. Une attribution ferme est un
    # faux positif — le pire résultat possible pour l'utilisateur, qui n'a aucun
    # moyen de savoir que la réponse est inventée.
    if no_answer and args.detector == "pipeline":
        print(f"\n--- {len(no_answer)} photos sans réponse juste possible ---")
        refused = 0
        for rel_path, info in no_answer:
            bgr = load_bgr(str(PROJECT_ROOT / rel_path))
            if bgr is None:
                print(f"illisible : {rel_path}")
                continue
            result = identify(
                str(PROJECT_ROOT / rel_path), bgr, encoder, index, k=args.top_k
            )
            verdict = "hors index" if result.out_of_index else result.confidence.level
            ok = result.out_of_index or result.confidence.level == "incertain"
            refused += ok
            print(
                f"\n{Path(rel_path).name}  {info['fr']}  [{result.crop_label}]"
                f"  -> {verdict} {'(correct : rien affirmé)' if ok else '(FAUX POSITIF)'}"
            )
            print(
                f"   plus proche : {result.card['name']} "
                f"({result.card['set_name']} {result.card['number']})"
                f"   marges : édition {result.confidence.margin_id:.4f} · "
                f"nom {result.confidence.margin_name:.4f}"
            )
        print(f"\nrefus corrects : {refused}/{len(no_answer)}")

    print(f"\ncrops écrits dans {CROP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
