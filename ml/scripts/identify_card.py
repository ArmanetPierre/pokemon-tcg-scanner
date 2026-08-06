"""Identifie une carte à partir d'une image.

    python scripts/identify_card.py photo.jpg
    python scripts/identify_card.py photo.jpg --model mobileclip2-s2 --top-k 5

L'écart entre le 1er et le 2e score compte plus que le score absolu : toutes les
cartes Pokémon partagent une mise en page commune, donc même deux cartes sans
rapport se ressemblent à ~0,79. Un top-1 à 0,95 avec un top-2 à 0,949 n'est pas
une identification fiable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.search import CardIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image introuvable : {args.image}", file=sys.stderr)
        return 1

    encoder = load_encoder(args.model)
    index = CardIndex(args.model)

    img = Image.open(args.image)
    t0 = time.time()
    vector = encoder.encode([img])
    t_embed = time.time() - t0

    t0 = time.time()
    hits = index.search(vector, k=args.top_k)[0]
    t_search = time.time() - t0

    print(f"{args.image.name} — {len(index)} cartes indexées ({args.model})\n")
    for rank, hit in enumerate(hits, 1):
        card = hit.card
        print(
            f"{rank}. {hit.score:.4f}  {card['name']}\n"
            f"          {card['set_name']} — {card['number']}/{card['set_printed_total']}"
            f"  [{card['id']}]"
        )

    margin = hits[0].score - hits[1].score if len(hits) > 1 else float("nan")
    print(f"\nmarge 1er/2e : {margin:.4f}")
    print(f"embedding {t_embed * 1000:.0f} ms · recherche {t_search * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
