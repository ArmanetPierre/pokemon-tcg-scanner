"""Écrit l'index d'embeddings et les métadonnées au format attendu par l'app.

    python scripts/export_index.py

Produit dans `data/export/` :
  index.bin    embeddings float16 contigus, N x dim, sans en-tête
  index.json   forme et ordre des cartes, pour lire index.bin sans deviner
  cards.json   métadonnées d'affichage, allégées

Le float16 divise la taille par deux et se lit directement en Swift : la
recherche est un produit matriciel via Accelerate, pas une bibliothèque
vectorielle. À 20 000 cartes, c'est exact et instantané — l'index Python fait
exactement la même chose, ce qui garantit des résultats identiques entre le
banc d'essai et l'app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search import CardIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "export"

# Champs nécessaires à l'affichage et au départage par numéro (chantier D).
# Le reste (attaques, texte, légalité) n'a rien à faire dans le bundle.
CARD_FIELDS = (
    "id", "name", "number", "rarity",
    "set_id", "set_name", "set_printed_total",
    "image_small",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2")
    args = parser.parse_args()

    index = CardIndex(args.model)
    embeddings = index.embeddings.astype(np.float32)
    print(f"index source : {embeddings.shape} ({embeddings.nbytes / 1e6:.0f} Mo en float32)")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    half = embeddings.astype(np.float16)
    (EXPORT_DIR / "index.bin").write_bytes(half.tobytes())

    (EXPORT_DIR / "index.json").write_text(json.dumps({
        "model": index.meta["model"],
        "count": int(half.shape[0]),
        "dim": int(half.shape[1]),
        "dtype": "float16",
        "layout": "row_major",
        "normalized": True,
        "note": "vecteurs L2-normalisés : la similarité cosinus est un simple "
                "produit scalaire",
        "card_ids": index.card_ids,
    }))

    cards = [
        {k: index.cards_by_id[card_id].get(k) for k in CARD_FIELDS}
        for card_id in index.card_ids
    ]
    (EXPORT_DIR / "cards.json").write_text(json.dumps(cards, ensure_ascii=False))

    # Vérifier que la perte de précision ne réordonne aucun voisinage : on
    # rejoue une recherche sur un échantillon et on compare les top-5.
    rng = np.random.default_rng(0)
    sample = rng.choice(half.shape[0], size=200, replace=False)
    reference = embeddings[sample] @ embeddings.T
    degraded = half[sample].astype(np.float32) @ half.astype(np.float32).T
    top_ref = np.argsort(-reference, axis=1)[:, :5]
    top_deg = np.argsort(-degraded, axis=1)[:, :5]
    identical = int(np.sum(np.all(top_ref == top_deg, axis=1)))
    top1_same = int(np.sum(top_ref[:, 0] == top_deg[:, 0]))

    for name in ("index.bin", "index.json", "cards.json"):
        size = (EXPORT_DIR / name).stat().st_size
        print(f"  {name:<12} {size / 1e6:7.1f} Mo")
    total = sum((EXPORT_DIR / n).stat().st_size for n in ("index.bin", "index.json", "cards.json"))
    print(f"  {'total':<12} {total / 1e6:7.1f} Mo")

    print(f"\nfloat32 -> float16 sur {len(sample)} requêtes :")
    print(f"  top-1 identique : {top1_same}/{len(sample)}")
    print(f"  top-5 identique : {identical}/{len(sample)}")
    if top1_same < len(sample):
        print("  (des réordonnancements en tête : vérifier avant d'embarquer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
