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
    "energie": "mee",
}

# Sets que `pokemon-tcg-data` ignore entièrement — pas seulement leurs images,
# leurs métadonnées aussi. Elles sont alors reprises de TCGdex, qui les
# référence sans visuel. La valeur est l'identifiant du set chez eux.
FROM_TCGDEX = {
    "mee": "mee",
}

TCGDEX = "https://api.tcgdex.net/v2/en"

# Fichiers isolés à la racine, pour les cartes qui n'ont pas de set entier à
# combler. La clé est le nom du fichier sans extension.
SINGLES = {
    "HGSS18": "hsp-HGSS18",
}


def key(number: str) -> str:
    """Numéro réduit à sa valeur, pour comparer « 001 », « 1 » et « 1 »."""
    number = number.strip()
    return str(int(number)) if number.isdigit() else number.upper()


def target_of(card_id: str) -> Path:
    return IMAGES / card_id.rsplit("-", 1)[0] / f"{card_id}.jpg"


def metadata_from_tcgdex(set_id: str, tcgdex_id: str) -> list[dict]:
    """Fabrique les entrées `cards.json` d'un set que la source principale ignore.

    `image_small` et `image_large` restent nuls : il n'existe aucune URL, et
    inventer un lien mort serait pire que l'absence — l'app sait n'afficher
    aucune vignette, elle ne sait pas deviner qu'un lien ne répondra jamais.
    """
    import requests

    detail = requests.get(f"{TCGDEX}/sets/{tcgdex_id}", timeout=30).json()
    printed = detail.get("cardCount", {}) or {}
    total = printed.get("official") or printed.get("total")

    entries = []
    for card in detail.get("cards", []):
        local = str(card.get("localId", ""))
        entries.append({
            "id": f"{set_id}-{int(local) if local.isdigit() else local}",
            "name": card.get("name"),
            "number": local,
            "rarity": card.get("rarity"),
            "artist": card.get("illustrator"),
            "supertype": card.get("category"),
            "set_id": set_id,
            "set_name": detail.get("name"),
            "set_series": (detail.get("serie") or {}).get("name"),
            "set_printed_total": total,
            "release_date": detail.get("releaseDate"),
            "image_small": None,
            "image_large": None,
        })
    return entries


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

        # Rapprochement sur la valeur du numéro, pas sur son écriture : les
        # sources ne s'accordent pas sur le remplissage — TCGdex écrit « 001 »
        # là où pokemon-tcg-data écrit « 1 », et personne ne nomme ses fichiers
        # avec des zéros de tête.
        expected = {
            key(c["number"]): c for c in cards.values() if c["set_id"] == set_id
        }
        # Les fichiers cachés que macOS sème un peu partout ne sont pas des
        # cartes manquantes, et les signaler comme telles noie les vrais trous.
        files = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]

        for path in sorted(files, key=lambda p: p.stem):
            card = expected.get(key(path.stem.strip()))
            if card is None:
                problems.append(f"{folder}/{path.name} : aucun {set_id} numéro {path.stem.strip()}")
                continue
            pairs.append((path, card["id"]))

        seen = {key(p.stem.strip()) for p in files}
        for number, card in sorted(expected.items()):
            if number not in seen:
                problems.append(f"{card['id']} ({card['name']}) : fichier manquant")

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

    records = json.loads(CARDS_JSON.read_text())
    cards = {c["id"]: c for c in records}

    # Certains sets n'ont même pas de métadonnées : les créer avant d'associer
    # les fichiers, sinon chaque image serait signalée comme orpheline.
    created = []
    for set_id, tcgdex_id in FROM_TCGDEX.items():
        if any(c["set_id"] == set_id for c in cards.values()):
            continue
        if not (SOURCE / next(f for f, s in FOLDERS.items() if s == set_id)).is_dir():
            continue
        created = metadata_from_tcgdex(set_id, tcgdex_id)
        print(f"  + {set_id} : {len(created)} carte(s) reprises de TCGdex "
              f"({created[0]['set_name']})")
        for entry in created:
            cards[entry["id"]] = entry

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

    if created:
        merged = records + created
        merged.sort(key=lambda c: (c.get("release_date") or "", c["id"]))
        CARDS_JSON.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
        print(f"\n{len(merged)} cartes -> {CARDS_JSON}  (+{len(created)})")

    print(f"\n{len(pairs)} image(s) installées dans {IMAGES}")
    print("Suite : build_embeddings.py, export_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
