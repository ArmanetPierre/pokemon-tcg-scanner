"""Enrôle chaque carte sous plusieurs vues, au lieu d'une seule.

    python scripts/build_multiview_index.py
    python scripts/evaluate_synthetic.py \
        --embeddings data/embeddings/mobileclip2-s2/centroid.npy --salt 7

L'index tient un vecteur par carte : celui du scan de référence, propre, à plat,
en lumière studio. Les requêtes, elles, sont des photos. C'est le décalage de
domaine que `docs/audit-ml.md` §6 relève, et l'enrôlement multi-vues est la
façon la moins chère de l'attaquer — aucun réentraînement, aucun changement
d'architecture, et pour la variante centroïde, **pas un octet de plus** dans le
bundle iOS.

Le principe : encoder chaque référence sous plusieurs dégradations, puis
résumer. Deux résumés possibles, et ce script produit le premier :

- **centroïde** — moyenne des vues, renormalisée. L'index garde sa forme, donc
  la recherche, le format de fichier et le portage Swift ne bougent pas.
- **multi-vecteurs** — k vecteurs par carte, score = maximum sur les vues.
  Plus expressif, mais l'index est multiplié par k et `card_ids` n'est plus en
  correspondance ligne à ligne. À n'envisager que si le centroïde paie.

**Le piège à éviter :** enrôler avec les mêmes vues que celles servant à
évaluer. Le score serait excellent et ne voudrait rien dire. Les vues d'enrôlement
viennent du cache (`train_projection.py --cache`, sel 0) ; l'évaluation doit
donc employer un autre sel — d'où `--salt 7` dans la commande ci-dessus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search import CardIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2")
    parser.add_argument("--views", type=int, default=2, help="vues en cache à utiliser")
    parser.add_argument("--weight-reference", type=float, default=1.0,
                        help="poids du scan de référence face aux vues dégradées")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    index = CardIndex(args.model)
    reference = index.embeddings.astype(np.float32)
    row_of = {cid: i for i, cid in enumerate(index.card_ids)}

    cache = ROOT / "data" / "embeddings" / args.model / f"augmented_v{args.views}.npz"
    if not cache.exists():
        print(f"cache absent : {cache}\n"
              f"le produire avec : python scripts/train_projection.py --cache "
              f"--views {args.views}", file=sys.stderr)
        return 1

    data = np.load(cache, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    rows = np.array([row_of[c] for c in data["card_ids"]])

    # Somme pondérée : la référence garde un poids propre, parce qu'elle est la
    # seule vue non dégradée et que la diluer dans k vues abîmées ferait dériver
    # l'index loin des scans dont il est censé décrire les cartes.
    total = reference * args.weight_reference
    counts = np.full(len(reference), args.weight_reference, np.float32)
    np.add.at(total, rows, vectors)
    np.add.at(counts, rows, 1.0)

    centroid = total / counts[:, None]
    norms = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid /= norms

    out = args.out or (ROOT / "data" / "embeddings" / args.model / "centroid.npy")
    np.save(out, centroid.astype(np.float32))

    # Combien le centroïde s'éloigne-t-il de la référence ? S'il n'en bouge
    # presque pas, il ne peut rien changer ; s'il en part très loin, il ne décrit
    # plus la carte. C'est le premier diagnostic à lire.
    drift = (centroid * reference).sum(axis=1)
    orphelines = int((counts == args.weight_reference).sum())
    print(f"{len(centroid)} cartes, {len(vectors)} vues dégradées "
          f"({orphelines} cartes sans vue, laissées telles quelles)")
    print(f"cos(centroïde, référence) : min {drift.min():.4f}  "
          f"méd {np.median(drift):.4f}  max {drift.max():.4f}")
    print(f"écrit : {out}")
    print("\nÉvaluer avec un AUTRE sel que celui de l'enrôlement :")
    print(f"    python scripts/evaluate_synthetic.py --embeddings {out} --salt 7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
