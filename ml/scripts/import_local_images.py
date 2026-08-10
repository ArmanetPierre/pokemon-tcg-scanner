"""Fait entrer dans le dataset des scans récupérés à la main.

    python scripts/import_local_images.py --check   # diagnostic seul
    python scripts/import_local_images.py           # convertit et installe

Certaines cartes sont dans les métadonnées mais n'ont d'image **nulle part** :
`images.pokemontcg.io` répond 404, scrydex ne sert que son placeholder, TCGdex
et Bulbapedia n'ont que le tirage d'origine. `add_missing_cards.py` les laisse
donc de côté, ce qui est le bon comportement automatique — mais elles existent,
et un humain peut les trouver.

Ce script est la porte d'entrée pour ces cas-là : on dépose les fichiers dans
`data/missingcard/<dossier>/<numéro>.<ext>`, il les convertit au format du
dataset et les range là où `build_embeddings.py` ira les chercher.

⚠️ **Le tirage compte autant que la carte.** Une promo McDonald's n'est pas sa
réimpression d'origine : mesuré à 0,928 de similarité, contre 0,79 pour deux
cartes sans aucun rapport. Ranger un scan de Kalos Starter Set sous `mcd14-1`
mettrait donc l'index à 0,07 de la vérité — plus du double de la marge qui
décide d'une identification. Le repère visuel est le numéro imprimé : une
McDonald's porte `X/12`, jamais `X/149`.

Vérifier après import, avec l'encodeur, que chaque image remonte bien une carte
du nom attendu. C'est ce contrôle qui a rattrapé deux lots erronés.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "missingcard"
CARDS_JSON = ROOT / "data" / "processed" / "cards.json"
IMAGES = ROOT / "data" / "raw" / "images_large"

# Qualité et côté maximum de `download_images.py` : les images importées doivent
# être indiscernables de celles du pipeline normal une fois dans le dataset.
MAX_SIDE = 1040
QUALITY = 95

# Dossier déposé -> set du dataset. Un dossier par set, un fichier par numéro.
FOLDERS = {
    "2014": "mcd14",
    "2015": "mcd15",
    "2017": "mcd17",
    "2018": "mcd18",
}

# Fichiers isolés à la racine, pour les cartes qui n'ont pas de set entier à
# combler. La clé est le nom du fichier sans extension.
SINGLES = {
    "HGSS18": "hsp-HGSS18",
}


def target_of(card_id: str) -> Path:
    return IMAGES / card_id.rsplit("-", 1)[0] / f"{card_id}.jpg"


def convert(source: Path, destination: Path) -> tuple[int, int]:
    """Recompresse en JPEG, sans agrandir : une source de 322 px le reste."""
    image = Image.open(source).convert("RGB")
    if max(image.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(image.size)
        image = image.resize(
            (round(image.width * ratio), round(image.height * ratio)), Image.LANCZOS
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=QUALITY)
    return image.size


def collect(cards: dict[str, dict]) -> list[tuple[Path, str]]:
    """Les couples (fichier, identifiant de carte), après contrôle de cohérence."""
    pairs: list[tuple[Path, str]] = []
    problems: list[str] = []

    for folder, set_id in FOLDERS.items():
        directory = SOURCE / folder
        if not directory.is_dir():
            problems.append(f"{folder}/ : dossier absent")
            continue

        expected = {c["number"]: c for c in cards.values() if c["set_id"] == set_id}
        # Les fichiers cachés que macOS sème un peu partout ne sont pas des
        # cartes manquantes, et les signaler comme telles noie les vrais trous.
        files = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]

        for path in sorted(files, key=lambda p: p.stem):
            number = path.stem.strip()
            card = expected.get(number)
            if card is None:
                problems.append(f"{folder}/{path.name} : aucun {set_id} numéro {number}")
                continue
            pairs.append((path, card["id"]))

        seen = {p.stem.strip() for p in files}
        for number in sorted(expected, key=lambda n: int(n) if n.isdigit() else 0):
            if number not in seen:
                problems.append(f"{set_id}-{number} ({expected[number]['name']}) : fichier manquant")

    for stem, card_id in SINGLES.items():
        matches = list(SOURCE.glob(f"{stem}.*"))
        if not matches:
            problems.append(f"{stem}.* : fichier absent")
        elif card_id not in cards:
            problems.append(f"{card_id} : inconnu des métadonnées")
        else:
            pairs.append((matches[0], card_id))

    for line in problems:
        print(f"  ⚠ {line}")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="diagnostic seul")
    args = parser.parse_args()

    if not CARDS_JSON.exists():
        sys.exit(f"{CARDS_JSON} absent — lancer d'abord scripts/build_metadata.py")

    cards = {c["id"]: c for c in json.loads(CARDS_JSON.read_text())}
    pairs = collect(cards)

    print(f"\n{len(pairs)} image(s) à importer")
    if args.check:
        for path, card_id in pairs:
            print(f"  {path.relative_to(SOURCE)!s:<20} -> {card_id:<14} {cards[card_id]['name']}")
        print("\n--check : rien n'a été écrit.")
        return 0

    for path, card_id in pairs:
        size = convert(path, target_of(card_id))
        print(f"  {card_id:<14} {cards[card_id]['name']:<18} {size[0]}x{size[1]}")

    print(f"\n{len(pairs)} image(s) installées dans {IMAGES}")
    print("Suite : build_embeddings.py, export_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
