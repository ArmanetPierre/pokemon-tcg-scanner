"""Prépare l'index pour Embedding Atlas, et le sert ou l'exporte.

    python scripts/explore_index.py                 # ouvre la carte dans le navigateur
    python scripts/explore_index.py --export site/  # application web autonome

Rend visible ce que les chiffres du README disent sans le montrer : les
réimpressions d'une même illustration collées les unes aux autres — d'où les
deux niveaux de confiance —, et le fait que tout l'espace soit resserré, ce qui
explique que deux cartes sans rapport partent déjà de ~0,79.

La projection et le voisinage sont calculés en métrique **cosinus**, la même que
la recherche embarquée : une carte proche à l'écran est une carte que le scanner
risque de confondre. Une projection euclidienne montrerait un voisinage qui
n'est pas celui du modèle.

La version exportée n'embarque **pas** les vecteurs — 42 Mo qu'un visiteur n'a
aucune raison de télécharger. Elle porte à la place la projection et les dix
plus proches voisins de chaque carte, tous deux calculés ici sur les vecteurs
réels : le visiteur voit donc le vrai voisinage du modèle, pas celui de la
projection 2D, qui n'est pas le même.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data" / "embeddings"
EXPORT = ROOT / "data" / "export"
ATLAS_DIR = ROOT / "data" / "atlas"

# Voisins retenus par carte dans la version exportée.
NEIGHBOURS = 10

# Lignes traitées d'un coup au calcul du voisinage. Le produit complet ferait
# 20 455² flottants, soit 1,6 Go.
CHUNK = 2048

# Regroupement par ère, à partir du préfixe d'identifiant de set. Sert de
# couleur par défaut : c'est la variable qui rend la structure lisible d'un
# coup d'œil, bien plus que la rareté.
ERAS = [
    ("Base / WotC", r"^(base|gym|neo|si|ecard|np|bp)"),
    ("EX", r"^(ex|pop|tk1|tk2)"),
    ("Diamond & Pearl / Platinum", r"^(dp|pl)"),
    ("HGSS / Call of Legends", r"^(hgss|hs|col|ru1)"),
    ("Black & White", r"^bw"),
    ("XY", r"^(xy|g1|dc1|det1|dv1)"),
    ("Sun & Moon", r"^sm"),
    ("Sword & Shield", r"^(swsh|cel25|pgo|fut20)"),
    ("Scarlet & Violet", r"^(sv|rsv|zsv)"),
    ("Mega Evolution", r"^me"),
    ("McDonald's", r"^mcd"),
]


def era_of(set_id: str) -> str:
    for label, pattern in ERAS:
        if re.match(pattern, set_id):
            return label
    return "autre"


def load(model: str) -> tuple[pd.DataFrame, np.ndarray]:
    directory = EMBEDDINGS / model
    vectors = np.load(directory / "embeddings.npy").astype(np.float32)
    ids = json.loads((directory / "card_ids.json").read_text())
    cards = {c["id"]: c for c in json.loads((EXPORT / "cards.json").read_text())}

    if len(ids) != len(vectors):
        sys.exit(f"{len(ids)} identifiants pour {len(vectors)} vecteurs — index désaligné")

    frame = pd.DataFrame([{
        "id": card_id,
        "nom": (card := cards.get(card_id, {})).get("name") or card_id,
        "extension": card.get("set_name") or "?",
        "ere": era_of(card.get("set_id") or ""),
        "rarete": card.get("rarity") or "—",
        "numero": f"{card.get('number')}/{card.get('set_printed_total')}",
        "image": card.get("image_small"),
    } for card_id in ids])

    return frame, vectors


def project(vectors: np.ndarray) -> np.ndarray:
    """UMAP en cosinus, graine fixe pour que deux exécutions donnent la même
    carte — sans quoi une capture d'écran ne peut être comparée à la suivante."""
    import umap

    print(f"UMAP sur {vectors.shape}… (quelques minutes)")
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    return reducer.fit_transform(vectors)


def neighbours(vectors: np.ndarray, k: int = NEIGHBOURS) -> list[dict]:
    """Les k plus proches voisins de chaque carte, sur les vecteurs réels.

    Renvoyés comme dictionnaires et non comme JSON : Atlas indexe la colonne
    par clé, et parquet sait stocker une structure imbriquée telle quelle.

    ⚠️ **Chaque carte figure en tête de sa propre liste, à distance 0.** C'est
    la convention kNN d'UMAP, et Atlas la suppose : son calcul de PageRank passe
    les distances à une construction d'ensemble simplicial flou qui prend le
    premier voisin pour la carte elle-même. La retirer — ce qui semble pourtant
    plus propre — fait dériver les rayons locaux, produit un tenseur creux
    invalide, et le processus meurt sur un SIGSEGV sans message.

    Les vecteurs étant L2-normalisés, le produit scalaire est la similarité
    cosinus.
    """
    print(f"voisinage : {k} plus proches sur {len(vectors)} cartes…")
    out: list[dict] = []

    for start in range(0, len(vectors), CHUNK):
        block = vectors[start:start + CHUNK] @ vectors.T

        for row in range(len(block)):
            index = start + row
            candidates = np.argpartition(-block[row], k)[:k + 1]
            order = candidates[np.argsort(-block[row, candidates])]

            # Explicitement en tête plutôt qu'en se fiant au tri : 114 paires de
            # cartes sont à distance nulle l'une de l'autre — même illustration
            # réimprimée — et une égalité parfaite peut placer la jumelle avant
            # la carte elle-même.
            keep = [index] + [int(i) for i in order if i != index][:k]
            out.append({
                "ids": keep,
                # Une distance, pas une similarité : Atlas attend que petit
                # veuille dire proche. Bornée à zéro, l'arrondi flottant
                # produisant sinon des -0.0.
                "distances": [max(0.0, round(1.0 - float(block[row, i]), 5)) for i in keep],
            })

    return out


def pagerank(neighbour_lists: list[dict], damping: float = 0.85, rounds: int = 50) -> np.ndarray:
    """Centralité de chaque carte dans le graphe des plus proches voisins.

    Calculé ici plutôt que laissé à Embedding Atlas, dont l'implémentation
    passe les distances à une construction d'ensemble simplicial flou et meurt
    sur un SIGSEGV avec un voisinage fourni de l'extérieur. Fournir la colonne
    toute faite court-circuite ce chemin.

    La grandeur a un sens propre ici : un score élevé désigne une carte que
    beaucoup d'autres ont pour voisine — un **attracteur**, du genre à remonter
    dans le top-k de photos qui n'ont rien à voir avec elle. C'est exactement ce
    qu'on observait sur appareil, où les mêmes quelques cartes revenaient en
    dauphines de scans sans rapport.
    """
    from scipy.sparse import csr_matrix

    size = len(neighbour_lists)
    rows, cols = [], []
    for source, entry in enumerate(neighbour_lists):
        for target in entry["ids"]:
            if int(target) != source:      # les boucles ne portent pas d'information
                rows.append(source)
                cols.append(int(target))

    weights = np.ones(len(rows), dtype=np.float64)
    graph = csr_matrix((weights, (rows, cols)), shape=(size, size))

    outgoing = np.asarray(graph.sum(axis=1)).ravel()
    outgoing[outgoing == 0] = 1
    transition = csr_matrix((1 / outgoing[rows], (rows, cols)), shape=(size, size)).T

    scores = np.full(size, 1 / size)
    for _ in range(rounds):
        scores = (1 - damping) / size + damping * (transition @ scores)

    return scores


def prepare(model: str, light: bool) -> Path:
    """Écrit le parquet et retourne son chemin. `light` omet les vecteurs."""
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    cache = ATLAS_DIR / f"{model}-projection.parquet"

    frame, vectors = load(model)

    if cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(frame) and (cached["id"] == frame["id"]).all():
            print(f"projection reprise de {cache.name}")
            frame["x"], frame["y"] = cached["x"], cached["y"]
            frame["voisins"] = cached["voisins"]
            frame["centralite"] = cached["centralite"]
        else:
            print(f"{cache.name} ne correspond plus à l'index — recalcul")
            cache.unlink()

    if "x" not in frame:
        coords = project(vectors)
        frame["x"], frame["y"] = coords[:, 0], coords[:, 1]
        frame["voisins"] = neighbours(vectors)
        frame["centralite"] = pagerank(list(frame["voisins"]))
        frame[["id", "x", "y", "voisins", "centralite"]].to_parquet(cache, index=False)
        print(f"projection mise en cache -> {cache}")

    if not light:
        frame["vecteur"] = list(vectors)

    path = ATLAS_DIR / f"{model}{'-web' if light else ''}.parquet"
    frame.to_parquet(path, index=False)
    print(f"{len(frame)} cartes -> {path} ({path.stat().st_size / 1e6:.1f} Mo)\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobileclip2-s2")
    parser.add_argument("--export", metavar="CHEMIN",
                        help="exporter une application web autonome au lieu de servir")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()

    dataset = prepare(args.model, light=bool(args.export))

    command = [
        str(ROOT / ".venv" / "bin" / "embedding-atlas"), str(dataset),
        "--x", "x", "--y", "y",
        "--neighbors", "voisins",
        "--pagerank", "centralite",
        "--image", "image",
    ]
    # Servi en local, les vecteurs restent disponibles : la recherche par
    # similarité d'Atlas s'en sert, et 42 Mo ne coûtent rien sur la machine qui
    # les a produits.
    command += ["--export-application", args.export] if args.export \
        else ["--vector", "vecteur", "--port", str(args.port)]

    print("$ " + " ".join(command[1:]) + "\n")
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
