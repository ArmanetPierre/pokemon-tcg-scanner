"""Télécharge et normalise les métadonnées des cartes Pokémon TCG.

Source : https://github.com/PokemonTCG/pokemon-tcg-data (JSON brut, pas de rate-limit).

Produit data/processed/cards.json : une liste plate de cartes portant uniquement
les champs utiles au scanner (identification + affichage).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

RAW_BASE = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# Champs conservés par carte. Le reste (attaques, texte, légalité...) est inutile
# pour la reconnaissance visuelle et alourdirait le bundle iOS.
KEEP = ("id", "name", "number", "rarity", "artist", "supertype")


def fetch_json(url: str) -> object:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Téléchargement de la liste des sets...")
    sets = fetch_json(f"{RAW_BASE}/sets/en.json")
    (RAW_DIR / "sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=2))
    print(f"  {len(sets)} sets")

    set_by_id = {s["id"]: s for s in sets}

    print("Téléchargement des cartes par set...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda s: (s["id"], fetch_json(f"{RAW_BASE}/cards/en/{s['id']}.json")),
                sets,
            )
        )

    cards: list[dict] = []
    skipped_no_image = 0

    for set_id, raw_cards in results:
        set_meta = set_by_id[set_id]
        for card in raw_cards:
            images = card.get("images") or {}
            small, large = images.get("small"), images.get("large")
            if not small:
                skipped_no_image += 1
                continue

            entry = {k: card.get(k) for k in KEEP}
            entry.update(
                set_id=set_id,
                set_name=set_meta.get("name"),
                set_series=set_meta.get("series"),
                set_printed_total=set_meta.get("printedTotal"),
                release_date=set_meta.get("releaseDate"),
                image_small=small,
                image_large=large,
            )
            cards.append(entry)

    # Les IDs sont uniques dans la source, mais on vérifie : un doublon fausserait
    # silencieusement l'index vectoriel.
    seen: dict[str, dict] = {}
    duplicates = 0
    for card in cards:
        if card["id"] in seen:
            duplicates += 1
            continue
        seen[card["id"]] = card
    cards = sorted(seen.values(), key=lambda c: (c["release_date"] or "", c["id"]))

    out = OUT_DIR / "cards.json"
    out.write_text(json.dumps(cards, ensure_ascii=False, indent=1))

    print(f"\n{len(cards)} cartes -> {out}")
    if skipped_no_image:
        print(f"  {skipped_no_image} cartes ignorées (pas d'image)")
    if duplicates:
        print(f"  {duplicates} doublons d'ID supprimés")

    latest = max(cards, key=lambda c: c["release_date"] or "")
    print(f"  set le plus récent : {latest['set_name']} ({latest['release_date']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
