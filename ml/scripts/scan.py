"""Identifie les cartes présentes sur une ou plusieurs photos.

    python scripts/scan.py ../sample/IMG_5023.jpeg
    python scripts/scan.py ../sample/*.jpeg --multi

Version ligne de commande de la logique de l'app : pipeline partagé
(src/pipeline.py) et verdict à deux niveaux — édition exacte, nom seul, ou
incertain. Le score absolu n'est jamais affiché : il ne discrimine rien.

En mode `--multi`, chaque quadrilatère détecté est traité comme une carte
distincte au lieu de ne garder que la meilleure hypothèse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.pipeline import (  # noqa: E402
    MIN_MULTI_QUAD_AREA,
    build_variants,
    classify_confidence,
    identify,
    load_bgr,
    selection_score,
)
from src.search import CardIndex  # noqa: E402

# En mode multi, un crop sous ce score top-1 est un parasite, pas une carte.
MIN_MULTI_SCORE = 0.55


def describe(hits, confidence) -> str:
    card = hits[0].card
    ref = f"{card['set_name']} {card['number']}/{card['set_printed_total']} [{hits[0].card_id}]"
    if confidence.level == "edition":
        return f"{card['name']} — {ref}"
    if confidence.level == "nom":
        return f"{card['name']} — édition à confirmer (probablement {ref})"
    return f"incertain — meilleur candidat {card['name']} ({ref})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--multi", action="store_true", help="plusieurs cartes par photo")
    args = parser.parse_args()

    encoder = load_encoder(args.model)
    index = CardIndex(args.model)

    for image_path in args.images:
        bgr = load_bgr(str(image_path))
        if bgr is None:
            print(f"{image_path.name} : illisible")
            continue

        print(f"\n=== {image_path.name} ===")

        if not args.multi:
            result = identify(str(image_path), bgr, encoder, index, k=args.top_k)
            hits, confidence, variant_label = result.hits, result.confidence, result.crop_label
            if confidence.source == "numéro lu":
                card = hits[0].card
                print(
                    f"  [{variant_label}] {card['name']} — {card['set_name']} "
                    f"{card['number']}/{card['set_printed_total']} [{hits[0].card_id}]"
                    f"  (numéro lu sur la carte)"
                )
                continue
            if result.out_of_index:
                print(
                    f"  [{variant_label}] numéro lu sur la carte inconnu de l'index — "
                    f"carte probablement absente de la base. "
                    f"Plus proche visuellement : {describe(hits, confidence)}"
                )
                continue
            print(f"  [{variant_label}] {describe(hits, confidence)}")
            print(f"      marges : édition {confidence.margin_id:.4f} · "
                  f"nom {confidence.margin_name:.4f}")
            if confidence.level != "edition":
                for rank, hit in enumerate(hits[1:args.top_k], 2):
                    print(
                        f"      {rank}. {hit.card['name']} "
                        f"({hit.card['set_name']} {hit.card['number']})"
                    )
            continue

        variants = build_variants(str(image_path), bgr, min_area=MIN_MULTI_QUAD_AREA)
        if not variants:
            print("  aucune carte détectée")
            continue
        results = index.search(encoder.encode([v.image for v in variants]), k=10)

        # Une carte par centre de quadrilatère : les orientations ambiguës et
        # les détecteurs redondants pointent souvent le même objet.
        order: list[int] = []
        claimed: list[np.ndarray] = []
        for i in np.argsort([selection_score(r) for r in results])[::-1]:
            if results[i][0].score < MIN_MULTI_SCORE:
                continue
            centre = variants[i].center
            if any(np.linalg.norm(centre - c) < 0.08 * max(bgr.shape[:2]) for c in claimed):
                continue
            claimed.append(centre)
            order.append(int(i))

        if not order:
            print("  aucune carte reconnue avec confiance")
        for i in order:
            confidence = classify_confidence(results[i])
            print(f"  [{variants[i].label}] {describe(results[i], confidence)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
