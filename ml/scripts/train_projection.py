"""Apprend une projection linéaire qui rapproche photo dégradée et scan.

    python scripts/train_projection.py --cache          # encode les vues (long)
    python scripts/train_projection.py                  # entraîne et évalue

Pourquoi une projection, et pourquoi linéaire
---------------------------------------------

L'audit (`docs/audit-ml.md` §4) a mesuré que l'espace de MobileCLIP2 est très
anisotrope : la norme du vecteur moyen vaut 0,81, donc l'essentiel de la
similarité entre deux cartes est une constante partagée — d'où le « plancher à
0,79 ». Les deux corrections non supervisées classiques ont été essayées et
**dégradent** la précision (centrage 26→23, CSLS 26→22 sur le banc réel). Il
reste la correction supervisée, et c'est ce script.

Linéaire, pour trois raisons :

- elle se replie dans le graphe Core ML comme une couche finale, donc coût
  d'inférence nul sur le téléphone ;
- elle s'applique aussi à l'index, une fois, hors ligne : la recherche reste un
  produit matriciel ;
- avec ~20 000 exemples, une tête non linéaire mémoriserait les augmentations.

Ce que l'entraînement voit, et ce qu'il ne doit jamais voir
-----------------------------------------------------------

Supervision : des paires (scan dégradé, scan) fabriquées par `src/augment.py`.
Aucune photo réelle n'entre dans l'entraînement — les 31 photos du banc restent
le juge de paix, et elles ne peuvent le rester que si le modèle ne les a jamais
vues. Les cartes elles-mêmes sont en outre séparées en train/val : la projection
doit marcher sur des cartes qu'elle n'a pas vues, sans quoi elle ne serait qu'un
sur-apprentissage de l'index.

Le risque propre à cette approche, à surveiller : le modèle peut apprendre à
inverser *mes* augmentations plutôt qu'à combler le vrai décalage de domaine.
C'est exactement ce que le banc réel mesure, et pourquoi il décide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augment import degrade, view_rng  # noqa: E402
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.search import CardIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "raw" / "images_large"
CACHE_DIR = ROOT / "data" / "embeddings"
EXPORT_DIR = ROOT / "data" / "export"


class DegradedViews(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        card_id, path, view = self.items[i]
        return self.preprocess(degrade(Image.open(path).convert("RGB"),
                                       view_rng(card_id, view)))


def build_cache(model: str, views: int, batch_size: int, workers: int) -> Path:
    """Encode `views` dégradations de chaque carte et met le résultat sur disque.

    C'est le seul poste coûteux (~20 min pour 2 vues sur 20 512 cartes). Une
    fois en cache, entraîner devient une affaire de secondes, ce qui permet
    d'explorer les hyperparamètres sans réencoder d'images.
    """
    index = CardIndex(model)
    items = []
    for cid in index.card_ids:
        path = IMAGES / index.cards_by_id[cid]["set_id"] / f"{cid}.jpg"
        if path.exists():
            items += [(cid, path, v) for v in range(views)]

    encoder = load_encoder(model)
    loader = DataLoader(DegradedViews(items, encoder.transform),
                        batch_size=batch_size, num_workers=workers, shuffle=False)

    chunks, start = [], time.time()
    for i, batch in enumerate(loader, 1):
        chunks.append(encoder.encode_batch(batch))
        done = min(i * batch_size, len(items))
        if i % 40 == 0 or done == len(items):
            rate = done / (time.time() - start)
            print(f"  {done}/{len(items)}  {rate:.0f} img/s  "
                  f"ETA {(len(items) - done) / rate / 60:.1f} min", flush=True)

    out = CACHE_DIR / model / f"augmented_v{views}.npz"
    np.savez(out, vectors=np.concatenate(chunks),
             card_ids=np.array([c for c, _, _ in items]),
             views=np.array([v for _, _, v in items]))
    print(f"\ncache écrit : {out}")
    return out


def train(queries: torch.Tensor, query_rows: torch.Tensor, refs: torch.Tensor,
          hard: torch.Tensor, dim_out: int, epochs: int, batch_size: int, lr: float,
          temperature: float, n_hard: int, device: str,
          evaluate=None, seed: int = 0) -> torch.Tensor:
    """InfoNCE sur les embeddings, avec négatifs durs tirés de l'index.

    `queries[i]` est une vue dégradée de la carte dont la référence est
    `refs[query_rows[i]]`. Cette indirection est indispensable : il y a
    plusieurs vues par carte, et `refs` couvre l'index entier alors que
    `queries` n'en couvre que la partie entraînement. Confondre indice de
    requête et ligne de carte apparierait chaque vue à une carte au hasard.

    `hard[r]` liste les plus proches voisins de la carte r dans l'espace
    d'origine : ce sont les confusions réelles du système — réimpressions,
    cartes de même illustration — et non des négatifs tirés au hasard, qui sont
    déjà tous à 0,70 et n'apprennent rien.

    Initialisation à l'identité (complétée de zéros si l'on réduit la dimension)
    pour que l'entraînement parte exactement du système actuel : tout écart
    mesuré ensuite est un effet de l'apprentissage, pas du point de départ.
    """
    torch.manual_seed(seed)
    dim_in = queries.shape[1]
    W = torch.zeros(dim_in, dim_out, device=device)
    W[:dim_out, :dim_out] = torch.eye(dim_out, device=device)
    W.requires_grad_(True)

    opt = torch.optim.AdamW([W], lr=lr, weight_decay=0.0)
    n = len(queries)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (n // batch_size + 1))

    # Sélection sur la validation, pas sur la dernière époque : la perte
    # d'entraînement baisse toujours, y compris quand la projection commence à
    # mémoriser les augmentations au lieu de généraliser.
    best = (float("-inf"), W.detach().clone())

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total, steps = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 8:
                continue

            # Galerie du lot : les références des requêtes, plus les négatifs
            # durs de chacune. Les doublons sont écartés pour ne pas qu'une même
            # carte soit à la fois cible et distracteur.
            pos = query_rows[idx]
            neg = hard[pos, :n_hard].reshape(-1)
            gallery_ids = torch.cat([pos, neg]).unique()
            # position de chaque cible dans la galerie dédoublonnée
            target = torch.searchsorted(gallery_ids, pos)

            q = F.normalize(queries[idx] @ W, dim=-1)
            g = F.normalize(refs[gallery_ids] @ W, dim=-1)
            loss = F.cross_entropy(q @ g.T / temperature, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            steps += 1

        line = f"  époque {epoch + 1}/{epochs}  perte {total / max(steps, 1):.4f}"
        if evaluate is not None:
            score = evaluate(W.detach().cpu().numpy())
            line += f"  rappel@1 val {score:.1%}"
            if score > best[0]:
                best = (score, W.detach().clone())
                line += "  <- meilleur"
        print(line, flush=True)

    return best[1] if evaluate is not None else W.detach()


def recall(queries: np.ndarray, gallery: np.ndarray, target_rows: np.ndarray,
           W: np.ndarray | None = None) -> tuple[float, float, float]:
    """Rappel@1, rappel@5 et marge médiane, galerie = index entier."""
    q, g = queries, gallery
    if W is not None:
        q, g = q @ W, g @ W
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)

    top1 = top5 = 0
    margins = []
    for s in range(0, len(q), 512):
        sims = q[s:s + 512] @ g.T
        top = np.argpartition(-sims, 5, axis=1)[:, :5]
        for r in range(sims.shape[0]):
            order = top[r][np.argsort(-sims[r, top[r]])]
            tgt = target_rows[s + r]
            top1 += order[0] == tgt
            top5 += tgt in order
            margins.append(sims[r, order[0]] - sims[r, order[1]])
    n = len(q)
    return top1 / n, top5 / n, float(np.median(margins))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--cache", action="store_true", help="(ré)encoder les vues dégradées")
    p.add_argument("--dim-out", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--n-hard", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=EXPORT_DIR / "projection.npy")
    args = p.parse_args()

    cache = CACHE_DIR / args.model / f"augmented_v{args.views}.npz"
    if args.cache or not cache.exists():
        build_cache(args.model, args.views, args.encode_batch, args.workers)

    index = CardIndex(args.model)
    E = index.embeddings.astype(np.float32)
    row_of = {cid: i for i, cid in enumerate(index.card_ids)}

    data = np.load(cache, allow_pickle=False)
    aug = data["vectors"].astype(np.float32)
    aug_rows = np.array([row_of[c] for c in data["card_ids"]])
    print(f"{len(aug)} vues dégradées en cache, {len(E)} références")

    # Séparation par CARTE, pas par vue : les deux vues d'une même carte doivent
    # tomber du même côté, sinon la validation mesure de la mémorisation.
    rng = np.random.default_rng(0)
    cards = np.arange(len(E))
    rng.shuffle(cards)
    n_val = int(len(cards) * args.val_frac)
    val_cards = np.sort(cards[:n_val])
    is_val = np.zeros(len(E), bool)
    is_val[val_cards] = True
    train_mask = ~is_val[aug_rows]
    print(f"{train_mask.sum()} vues d'entraînement, {(~train_mask).sum()} de validation "
          f"({len(val_cards)} cartes tenues à l'écart)")

    # Négatifs durs : plus proches voisins dans l'espace d'origine.
    print("recherche des négatifs durs...", flush=True)
    K = max(args.n_hard, 8)
    hard = np.zeros((len(E), K), np.int64)
    for s in range(0, len(E), 2048):
        sims = E[s:s + 2048] @ E.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -2.0
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        for r in range(sims.shape[0]):
            hard[s + r] = top[r][np.argsort(-sims[r, top[r]])]

    # Sonde de validation : un sous-échantillon de vues de cartes jamais vues,
    # cherchées dans l'index ENTIER — la difficulté réelle, pas une recherche
    # restreinte aux cartes de validation qui serait bien plus facile.
    val_idx = np.flatnonzero(~train_mask)
    probe = val_idx[np.random.default_rng(1).permutation(len(val_idx))[:2000]]
    probe_q, probe_rows = aug[probe], aug_rows[probe]

    def val_recall(matrix: np.ndarray) -> float:
        return recall(probe_q, E, probe_rows, matrix)[0]

    print(f"sonde de validation : {len(probe)} vues, galerie = {len(E)} cartes")
    depart = val_recall(np.eye(512, dtype=np.float32))
    print(f"  point de départ (identité) : rappel@1 {depart:.1%}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    W = train(
        torch.tensor(aug[train_mask], device=device),
        torch.tensor(aug_rows[train_mask], device=device),
        torch.tensor(E, device=device),
        torch.tensor(hard, device=device),
        dim_out=args.dim_out, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, temperature=args.temperature, n_hard=args.n_hard, device=device,
        evaluate=val_recall,
    ).cpu().numpy().astype(np.float32)

    print(f"\n{'configuration':<34} {'rappel@1':>10} {'rappel@5':>10} {'marge méd.':>12}")
    print("-" * 70)
    results = {}
    for split, mask in (("validation (cartes jamais vues)", ~train_mask),
                        ("entraînement", train_mask)):
        for name, matrix in (("sans projection", None), ("avec projection", W)):
            r1, r5, mg = recall(aug[mask], E, aug_rows[mask], matrix)
            print(f"{split + ', ' + name:<34} {r1:>9.1%} {r5:>9.1%} {mg:>12.4f}")
            results[f"{split} / {name}"] = {"recall1": r1, "recall5": r5, "margin": mg}
        print("-" * 70)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, W)
    meta = {
        "model": args.model, "shape": list(W.shape), "views": args.views,
        "epochs": args.epochs, "temperature": args.temperature,
        "n_hard": args.n_hard, "val_cards": int(len(val_cards)),
        "results": results,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nprojection écrite : {args.out}")
    print("Valider maintenant sur les deux bancs — c'est le banc réel qui décide :")
    print(f"    python scripts/evaluate_synthetic.py --projection {args.out}")
    print(f"    python scripts/evaluate_real.py --ablation --projection {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
