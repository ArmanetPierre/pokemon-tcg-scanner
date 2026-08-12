"""Banc synthétique : évalue la recherche sur tout l'index, pas sur 31 photos.

    python scripts/evaluate_synthetic.py                    # 150 cartes par ère
    python scripts/evaluate_synthetic.py --per-era 400      # plus serré
    python scripts/evaluate_synthetic.py --projection data/export/projection.npy

Le banc de photos réelles mesure 31 cartes, toutes récentes, alors que 47 % de
l'index précède Sun & Moon. Il ne peut donc pas voir un décrochage par ère.
Celui-ci le peut : la requête est un scan de référence dégradé (`src/augment.py`),
donc la vérité terrain est gratuite et l'échantillon peut couvrir l'index entier.

Ce que ce banc mesure et ne mesure pas :

  MESURE     comparer deux modèles, deux prétraitements, deux projections, et
             voir sur quelles ères et quels sets un changement paie ou coûte.
  NE MESURE  la précision attendue sur de vraies photos. Les augmentations ne
   PAS       reproduisent ni l'optique, ni le traitement d'image du téléphone,
             ni les erreurs de la détection en amont — et le pipeline complet
             (orientation, lecture du numéro) n'est pas dans la boucle.

ATTENTION, limite découverte en s'en servant (chantier D) : ce banc **surestime
massivement** tout enrôlement multi-vues. L'index centroïde y monte à 99,9 %
contre 91,1 % pour l'index de référence, alors que sur de vraies photos il fait
légèrement MOINS bien en top-1. La raison est structurelle : enrôlement et
requête sont produits par le même modèle de dégradation, donc l'index a été
déplacé vers exactement la famille d'images qu'on lui présente ensuite. Changer
le sel ne corrige pas cela — le sel change le tirage, pas la famille.

En clair, ce banc compare honnêtement deux ESPACES (projection, encodeur,
prétraitement) mais pas deux stratégies d'ENRÔLEMENT. Pour celles-là, seul
evaluate_real.py tranche.

Le juge de paix reste `evaluate_real.py`. Celui-ci sert à ne pas naviguer à
l'aveugle entre deux mesures sur 31 photos.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augment import degrade, view_rng  # noqa: E402
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.search import CardIndex  # noqa: E402
from src.stats import fmt_ratio, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "raw" / "images_large"

# Regroupement par ère : les séries de l'API sont trop fines (176 sets) pour
# qu'un taux par set ait un n exploitable, et trop hétérogènes pour être
# moyennées d'un bloc. Les frontières suivent les changements de gabarit de
# carte, qui sont ce qui compte pour un modèle visuel.
SERIES_TO_ERA = {
    "Base": "1 WotC", "Gym": "1 WotC", "Neo": "1 WotC", "Legendary Collection": "1 WotC",
    "EX": "2 EX",
    "Diamond & Pearl": "3 DP/Pt", "Platinum": "3 DP/Pt",
    "HeartGold & SoulSilver": "4 HGSS",
    "Black & White": "5 BW",
    "XY": "6 XY",
    "Sun & Moon": "7 SM",
    "Sword & Shield": "8 SWSH",
    "Scarlet & Violet": "9 SV",
}


def era_of(card: dict) -> str:
    """Ère d'une carte, l'ère WotC étant coupée par décennie.

    Base, Jungle, Fossil et Legendary Collection (1999-2002) se réimpriment les
    uns les autres massivement — jusqu'à 90 % des cartes de Legendary Collection
    ont un voisin à 0,97 dans l'index. Les fondre avec Neo et Gym masquerait
    précisément le groupe le plus difficile de tout l'index.
    """
    era = SERIES_TO_ERA.get(card["set_series"], "A autre/promo")
    if era == "1 WotC":
        decade = int(card["release_date"][:4]) // 10 * 10
        return f"1 WotC {decade}s"
    return era


class DegradedCards(Dataset):
    """Charge un scan, le dégrade, puis applique le prétraitement de l'encodeur.

    L'augmentation vit dans le Dataset et non dans la boucle : elle est
    coûteuse en CPU (homographie, flou, JPEG) et sature un cœur unique pendant
    que le GPU attend.
    """

    def __init__(self, items: list[tuple[str, Path, int]], preprocess, salt: int):
        self.items = items
        self.preprocess = preprocess
        self.salt = salt

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> torch.Tensor:
        card_id, path, view = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.preprocess(degrade(img, view_rng(card_id, view, self.salt)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--per-era", type=int, default=150,
                        help="cartes tirées par ère (0 = tout l'index)")
    parser.add_argument("--views", type=int, default=1, help="dégradations par carte")
    parser.add_argument("--projection", type=Path,
                        help="matrice apprise à appliquer aux requêtes ET à l'index")
    parser.add_argument("--embeddings", type=Path,
                        help="index alternatif (ex. centroïde multi-vues), "
                             "aligné ligne à ligne sur card_ids")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--salt", type=int, default=0,
                        help="change le tirage des dégradations sans changer les cartes")
    parser.add_argument("--only-cards", type=Path,
                        help="JSON de card_ids : restreint le tirage (cartes tenues à l'écart)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    index = CardIndex(args.model)
    gallery = index.embeddings.astype(np.float32)

    if args.embeddings:
        gallery = np.load(args.embeddings).astype(np.float32)
        if gallery.shape != index.embeddings.shape:
            print(f"forme {gallery.shape} incompatible avec l'index "
                  f"{index.embeddings.shape}", file=sys.stderr)
            return 1
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        print(f"index alternatif : {args.embeddings.name}")

    if args.projection:
        W = np.load(args.projection).astype(np.float32)
        gallery = gallery @ W
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        print(f"projection {W.shape} appliquée à l'index et aux requêtes")

    allowed = None
    if args.only_cards:
        allowed = set(json.loads(args.only_cards.read_text()))
        print(f"tirage restreint à {len(allowed)} cartes")

    by_era: dict[str, list[str]] = defaultdict(list)
    for cid in index.card_ids:
        if allowed is None or cid in allowed:
            by_era[era_of(index.cards_by_id[cid])].append(cid)

    # Tirage déterministe : trier puis mélanger avec une graine fixe, plutôt que
    # de dépendre de l'ordre du fichier d'index.
    items: list[tuple[str, Path, int]] = []
    rng = np.random.default_rng(0)
    for era in sorted(by_era):
        pool = sorted(by_era[era])
        rng.shuffle(pool)
        chosen = pool if args.per_era == 0 else pool[: args.per_era]
        for cid in chosen:
            path = IMAGES / index.cards_by_id[cid]["set_id"] / f"{cid}.jpg"
            if path.exists():
                items += [(cid, path, v) for v in range(args.views)]

    if not items:
        print(f"aucune image de référence sous {IMAGES}", file=sys.stderr)
        return 1
    print(f"{len(items)} requêtes ({len(by_era)} ères, {args.views} vue(s) par carte)",
          flush=True)

    encoder = load_encoder(args.model)
    loader = DataLoader(DegradedCards(items, encoder.transform, args.salt),
                        batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False)

    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "margins": []})
    ids = np.array(index.card_ids)
    done = 0
    for batch in loader:
        vectors = encoder.encode_batch(batch)
        if args.projection:
            vectors = vectors @ W
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        sims = vectors @ gallery.T
        top = np.argpartition(-sims, 5, axis=1)[:, :5]
        for row in range(len(batch)):
            card_id, _, _ = items[done + row]
            order = top[row][np.argsort(-sims[row, top[row]])]
            era = era_of(index.cards_by_id[card_id])
            s = stats[era]
            s["n"] += 1
            s["top1"] += ids[order[0]] == card_id
            s["top5"] += card_id in ids[order]
            s["margins"].append(float(sims[row, order[0]] - sims[row, order[1]]))
        done += len(batch)
        if done % (args.batch_size * 16) == 0:
            print(f"  {done}/{len(items)}", flush=True)

    print(f"\n{'ère':<16} {'top-1':<28} {'top-5':<28} {'marge méd.':>10}")
    print("-" * 86)
    total = {"n": 0, "top1": 0, "top5": 0}
    out = {}
    for era in sorted(stats):
        s = stats[era]
        for key in ("n", "top1", "top5"):
            total[key] += s[key]
        print(f"{era:<16} {fmt_ratio(s['top1'], s['n']):<28} "
              f"{fmt_ratio(s['top5'], s['n']):<28} {np.median(s['margins']):>10.4f}")
        low, high = wilson(s["top1"], s["n"])
        out[era] = {"n": s["n"], "top1": s["top1"], "top5": s["top5"],
                    "top1_rate": s["top1"] / s["n"], "ci95": [low, high],
                    "margin_median": float(np.median(s["margins"]))}
    print("-" * 86)
    print(f"{'TOTAL':<16} {fmt_ratio(total['top1'], total['n']):<28} "
          f"{fmt_ratio(total['top5'], total['n']):<28}")

    worst = min(out.items(), key=lambda kv: kv[1]["top1_rate"])
    best = max(out.items(), key=lambda kv: kv[1]["top1_rate"])
    print(f"\nécart entre ères : {worst[0]} à {worst[1]['top1_rate']:.1%} "
          f"contre {best[0]} à {best[1]['top1_rate']:.1%} "
          f"({100 * (best[1]['top1_rate'] - worst[1]['top1_rate']):.0f} points)")

    out["TOTAL"] = {"n": total["n"], "top1": total["top1"], "top5": total["top5"],
                    "top1_rate": total["top1"] / total["n"]}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"résumé écrit : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
