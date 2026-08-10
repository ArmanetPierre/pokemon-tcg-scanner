"""Compacte l'index en un jeu de données embarquable dans une page web.

    python scripts/export_explainer_data.py            # -> data/atlas/explainer.json

Puis, pour obtenir la page complète :

    python - <<'EOF'
    import pathlib
    tpl = pathlib.Path("scripts/explainer.template.html").read_text()
    data = pathlib.Path("data/atlas/explainer.json").read_text()
    pathlib.Path("data/atlas/explainer.html").write_text(tpl.replace("__DATA__", data))
    EOF

Sort de quoi dessiner les 20 455 cartes et leurs voisinages dans un canvas, sans
moteur SQL ni WebAssembly. L'export d'Embedding Atlas pèse 112 Mo pour 2 Mo de
données utiles ; ici tout tient dans quelques centaines de kilo-octets, parce
qu'une page d'explication n'a pas besoin d'exécuter des requêtes — seulement de
montrer.

Trois compressions font l'essentiel :

- les coordonnées passent en entiers 16 bits, ce qui laisse 65 536 positions par
  axe là où l'écran en distingue au mieux 2 000 ;
- noms, extensions et numéros deviennent des dictionnaires indexés, 20 455
  cartes ne portant que 4 450 noms et 171 extensions ;
- les voisinages, eux, ne sont pas approximés : ce sont les vrais plus proches
  voisins calculés sur les vecteurs 512D, et c'est ce qui permet de montrer ce
  que le scanner confond réellement.

L'identifiant de carte n'est pas transporté : « Pitch Black · 51/84 » se lit
mieux que « me5-51 » et se reconstitue depuis les deux colonnes déjà là.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_index import ERAS, era_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "data" / "atlas"
EXPORT = ROOT / "data" / "export"

# Voisins conservés par carte. Dix seraient plus fidèles au moteur embarqué,
# mais huit suffisent à montrer un groupe de réimpressions et pèsent 80 Ko de
# moins une fois encodés.
NEIGHBOURS = 8


def quantize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ramène des flottants sur 16 bits signés, avec l'échelle pour revenir."""
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    scaled = np.round((values - low) / span * 65534 - 32767).astype(np.int16)
    return scaled, low, span


def encode(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def main() -> int:
    frame = pd.read_parquet(ATLAS / "mobileclip2-s2-web.parquet")
    cards = {c["id"]: c for c in json.loads((EXPORT / "cards.json").read_text())}

    names = sorted({str(n) for n in frame["nom"]})
    sets = sorted({str(s) for s in frame["extension"]})
    numbers = sorted({str(n) for n in frame["numero"]})
    eras = [label for label, _ in ERAS] + ["autre"]

    name_at = {v: i for i, v in enumerate(names)}
    set_at = {v: i for i, v in enumerate(sets)}
    number_at = {v: i for i, v in enumerate(numbers)}
    era_at = {v: i for i, v in enumerate(eras)}

    x, x_low, x_span = quantize(frame["x"].to_numpy(dtype=np.float64))
    y, y_low, y_span = quantize(frame["y"].to_numpy(dtype=np.float64))

    # L'ère se déduit de l'extension, donc une table de 171 entrées remplace une
    # colonne de 20 455.
    era_by_set_name: dict[str, str] = {}
    for card in cards.values():
        label = card.get("set_name")
        if label and label not in era_by_set_name:
            era_by_set_name[label] = era_of(card.get("set_id") or "")

    set_era = np.array(
        [era_at[era_by_set_name.get(name, "autre")] for name in sets], dtype=np.uint8
    )

    # 16 bits suffisent : l'index compte moins de 65 536 cartes, et passer de
    # 32 à 16 économise 110 Ko une fois encodé en base64.
    neighbours = np.stack([
        np.asarray(row["ids"][1:NEIGHBOURS + 1], dtype=np.uint16)
        for row in frame["voisins"]
    ]).astype(np.uint16)

    payload = {
        "count": len(frame),
        "eras": eras,
        "names": names,
        "sets": sets,
        "setEra": encode(set_era),
        "numbers": numbers,
        "scale": {"x": [x_low, x_span], "y": [y_low, y_span]},
        "x": encode(x),
        "y": encode(y),
        "name": encode(frame["nom"].map(name_at).to_numpy(dtype=np.uint16)),
        "set": encode(frame["extension"].map(set_at).to_numpy(dtype=np.uint16)),
        "number": encode(frame["numero"].map(number_at).to_numpy(dtype=np.uint16)),
        "neighbours": encode(neighbours),
        "neighbourCount": NEIGHBOURS,
    }

    out = ATLAS / "explainer.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(frame)} cartes -> {out} ({out.stat().st_size / 1e6:.2f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
