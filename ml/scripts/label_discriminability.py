"""Pour chaque carte : l'image seule peut-elle trancher, ou faut-il lire le numéro ?

    python scripts/label_discriminability.py

L'audit a mesuré un plafond que le pipeline subit sans le savoir : 4,3 % des
cartes ont dans l'index un voisin à 0,99 ou plus, et la marge intrinsèque
médiane entre une carte et sa plus proche voisine vaut 0,0099 — sous le seuil de
décision de 0,03. Pour une partie de l'index, la similarité **ne peut pas**
produire un verdict ferme, même sur une photo parfaite. Ce n'est pas un défaut
du modèle : ce sont des réimpressions de la même illustration.

Aujourd'hui l'app le découvre carte par carte, à l'exécution, sous la forme
d'une marge serrée. Elle pourrait le savoir d'avance.

Ce script étiquette chaque carte, **par la mesure et non par une heuristique** :
il rejoue les vues dégradées en cache (`train_projection.py --cache`) contre
l'index entier et regarde ce que la recherche rend réellement. Une carte est
marquée `needs_printed_number` si aucune de ses vues ne la retrouve en tête avec
une marge exploitable.

Limites, à garder en tête avant de s'appuyer dessus :

- les requêtes sont des dégradations synthétiques, pas des photos ; l'étiquette
  est un indicateur de difficulté, pas une prédiction d'échec ;
- deux vues par carte, donc l'étiquette individuelle est bruitée ; c'est la
  distribution qui est solide, pas le verdict sur une carte donnée ;
- l'étiquette est calculée dans l'espace de l'index courant. Changer d'encodeur
  ou appliquer une projection la périme — d'où l'empreinte écrite avec elle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import FIRM_ID_MARGIN  # noqa: E402
from src.search import CardIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "discriminability.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2")
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--margin", type=float, default=FIRM_ID_MARGIN,
                        help="marge en deçà de laquelle un verdict ferme est exclu")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    index = CardIndex(args.model)
    E = index.embeddings.astype(np.float32)
    ids = np.array(index.card_ids)
    row_of = {cid: i for i, cid in enumerate(index.card_ids)}

    cache = ROOT / "data" / "embeddings" / args.model / f"augmented_v{args.views}.npz"
    if not cache.exists():
        print(f"cache absent : {cache}\n"
              f"le produire avec : python scripts/train_projection.py --cache "
              f"--views {args.views}", file=sys.stderr)
        return 1

    data = np.load(cache, allow_pickle=False)
    queries = data["vectors"].astype(np.float32)
    query_rows = np.array([row_of[c] for c in data["card_ids"]])
    print(f"{len(queries)} vues dégradées contre {len(E)} cartes", flush=True)

    # Plus proche voisin de chaque carte dans l'index : le plafond structurel,
    # indépendant de toute requête.
    nn = np.zeros(len(E), np.float32)
    for s in range(0, len(E), 2048):
        sims = E[s:s + 2048] @ E.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -2.0
        nn[s:s + 2048] = sims.max(axis=1)

    # Meilleur résultat obtenu par les vues de chaque carte.
    best: dict[int, tuple[bool, float]] = defaultdict(lambda: (False, -1.0))
    for s in range(0, len(queries), 512):
        sims = queries[s:s + 512] @ E.T
        top = np.argpartition(-sims, 2, axis=1)[:, :2]
        for r in range(sims.shape[0]):
            order = top[r][np.argsort(-sims[r, top[r]])]
            target = query_rows[s + r]
            hit = bool(order[0] == target)
            margin = float(sims[r, order[0]] - sims[r, order[1]])
            previous = best[target]
            if (hit, margin) > previous:
                best[target] = (hit, margin)
        if s % 8192 == 0:
            print(f"  {s}/{len(queries)}", flush=True)

    labels = {}
    stats = Counter()
    for row, card_id in enumerate(ids):
        hit, margin = best.get(row, (False, -1.0))
        needs = not (hit and margin >= args.margin)
        labels[str(card_id)] = {
            "needs_printed_number": needs,
            "nearest_neighbour": round(float(nn[row]), 4),
            "best_margin": round(margin, 4),
        }
        stats["needs" if needs else "ok"] += 1
        if nn[row] >= 0.99:
            stats["quasi_doublon"] += 1
            stats["quasi_doublon_needs" if needs else "quasi_doublon_ok"] += 1

    total = len(ids)
    print(f"\n{stats['needs']} cartes sur {total} ({stats['needs'] / total:.1%}) "
          f"ne peuvent pas être tranchées par l'image seule")
    print(f"  dont voisin >= 0,99 : {stats['quasi_doublon_needs']} "
          f"sur {stats['quasi_doublon']} quasi-doublons")

    # Ventilation par ère de sortie : c'est là que l'information devient
    # actionnable côté produit.
    by_series: dict[str, list[bool]] = defaultdict(list)
    for card_id, label in labels.items():
        by_series[index.cards_by_id[card_id]["set_series"]].append(
            label["needs_printed_number"])
    print(f"\n{'série':<28} {'cartes':>7} {'numéro requis':>15}")
    print("-" * 54)
    for series, flags in sorted(by_series.items(),
                                key=lambda kv: -sum(kv[1]) / len(kv[1])):
        if len(flags) >= 50:
            print(f"{series[:27]:<28} {len(flags):>7} {sum(flags) / len(flags):>14.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model,
        "margin": args.margin,
        "views": args.views,
        "note": "mesuré sur des dégradations synthétiques ; indicateur de "
                "difficulté, pas prédiction d'échec. Périmé si l'encodeur ou "
                "l'espace change.",
        "labels": labels,
    }))
    print(f"\nétiquettes écrites : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
