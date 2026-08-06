"""Encode toutes les images de référence et enregistre la banque d'embeddings.

Sortie dans data/embeddings/<modele>/ :
  embeddings.npy  float32 (N, dim), L2-normalisés, alignés sur card_ids.json
  card_ids.json   liste des IDs de cartes, dans l'ordre des lignes
  meta.json       modèle, dimension, prétraitement (à répliquer en Swift)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.encoder import REGISTRY, load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARDS_JSON = ROOT / "data" / "processed" / "cards.json"
EMB_DIR = ROOT / "data" / "embeddings"


class CardImages(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.preprocess(Image.open(self.paths[i]).convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--image-dir", default="data/raw/images_large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cards = json.loads(CARDS_JSON.read_text())
    image_root = ROOT / args.image_dir

    # Aligner strictement cartes et fichiers présents : une ligne d'embedding
    # doit correspondre à un card_id, sans décalage possible.
    usable, missing = [], []
    for card in cards:
        path = image_root / card["set_id"] / f"{card['id']}.jpg"
        (usable if path.exists() else missing).append((card, path))

    if args.limit:
        usable = usable[: args.limit]

    print(f"{len(usable)} cartes à encoder, {len(missing)} sans image")

    encoder = load_encoder(args.model)
    print(f"modèle {encoder.name} — dim {encoder.dim} — entrée {encoder.preprocess_spec.size}px")

    loader = DataLoader(
        CardImages([p for _, p in usable], encoder.transform),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,  # l'ordre garantit l'alignement avec card_ids
    )

    chunks: list[np.ndarray] = []
    start = time.time()
    for i, batch in enumerate(loader, 1):
        chunks.append(encoder.encode_batch(batch))
        done = min(i * args.batch_size, len(usable))
        if i % 20 == 0 or done == len(usable):
            rate = done / (time.time() - start)
            print(
                f"  {done}/{len(usable)}  {rate:.0f} img/s  "
                f"ETA {(len(usable) - done) / rate / 60:.1f} min",
                flush=True,
            )

    embeddings = np.concatenate(chunks)
    assert embeddings.shape[0] == len(usable), "désalignement embeddings/cartes"

    out_dir = EMB_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "card_ids.json").write_text(json.dumps([c["id"] for c, _ in usable]))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": encoder.name,
                "dim": encoder.dim,
                "count": len(usable),
                "preprocess": asdict(encoder.preprocess_spec),
                "image_dir": args.image_dir,
                "missing_card_ids": [c["id"] for c, _ in missing],
            },
            indent=2,
        )
    )

    elapsed = (time.time() - start) / 60
    size_mb = embeddings.nbytes / 1e6
    print(f"\n{embeddings.shape} en {elapsed:.1f} min -> {out_dir}")
    print(f"  {size_mb:.0f} Mo en float32, {size_mb / 2:.0f} Mo en float16 (cible iOS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
