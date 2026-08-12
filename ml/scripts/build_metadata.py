"""Télécharge et normalise les métadonnées des cartes Pokémon TCG.

    python scripts/build_metadata.py            # à la révision épinglée
    python scripts/build_metadata.py --check    # la source a-t-elle bougé ?
    python scripts/build_metadata.py --ref HEAD # dernière révision en date

Source : https://github.com/PokemonTCG/pokemon-tcg-data (JSON brut, pas de rate-limit).

Produit data/processed/cards.json : une liste plate de cartes portant uniquement
les champs utiles au scanner (identification + affichage), et
data/processed/source.json qui note la révision utilisée.

**Pourquoi une révision épinglée.** Cette source est un dépôt git, et lire
`master` en fait une cible mouvante : deux reconstructions à un mois d'écart
donnent deux index différents, sans que rien ne le signale. Or l'index est
l'artefact central du projet — il doit être reconstructible à l'identique, ne
serait-ce que pour qu'une mesure d'il y a un mois reste comparable à celle
d'aujourd'hui. Le déplacement de la révision devient donc un geste délibéré,
qui laisse une trace dans l'historique.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = "PokemonTCG/pokemon-tcg-data"

# Révision épinglée. Celle-ci ajoute Pitch Black (me5), le set le plus récent de
# l'index — c'est exactement l'état qui a produit les 20 512 cartes mesurées.
# Pour la déplacer : `--check` pour voir ce qui a bougé, `--ref <sha>` pour
# reconstruire, puis reporter la valeur ici dans le même commit que le nouvel
# index.
DATA_REVISION = "8b4e387930ead7be6595b4d4c59b7ba7a3a79f08"
DATA_REVISION_DATE = "2026-07-17"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def raw_base(ref: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{ref}"


def latest_revision() -> tuple[str, str, str]:
    """(sha, date, titre) du dernier commit de la source."""
    resp = requests.get(f"https://api.github.com/repos/{REPO}/commits/master", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return (data["sha"], data["commit"]["committer"]["date"][:10],
            data["commit"]["message"].splitlines()[0])

# Champs conservés par carte. Le reste (attaques, texte, légalité...) est inutile
# pour la reconnaissance visuelle et alourdirait le bundle iOS.
KEEP = ("id", "name", "number", "rarity", "artist", "supertype")


def fetch_json(url: str) -> object:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DATA_REVISION,
                        help="révision de la source (défaut : celle épinglée)")
    parser.add_argument("--check", action="store_true",
                        help="comparer la révision épinglée au dernier commit, sans rien écrire")
    args = parser.parse_args()

    if args.check:
        sha, date, title = latest_revision()
        print(f"épinglée : {DATA_REVISION[:12]}  ({DATA_REVISION_DATE})")
        print(f"amont    : {sha[:12]}  ({date})  {title}")
        if sha == DATA_REVISION:
            print("\nà jour.")
        else:
            print(f"\nLa source a bougé. Pour l'adopter :"
                  f"\n    python scripts/build_metadata.py --ref {sha}"
                  f"\npuis reporter DATA_REVISION dans ce fichier, et reconstruire "
                  f"images, embeddings et index dans le même commit.")
        return 0

    ref = args.ref
    if ref == "HEAD":
        ref, date, title = latest_revision()
        print(f"dernière révision : {ref[:12]} ({date}) {title}")
    if ref != DATA_REVISION:
        print(f"ATTENTION : révision {ref[:12]}, différente de celle épinglée "
              f"({DATA_REVISION[:12]}). L'index produit ne sera pas comparable "
              f"aux mesures publiées tant que DATA_REVISION n'est pas mis à jour.")

    RAW_BASE = raw_base(ref)
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

    # Provenance, à côté des données qu'elle décrit. `export_index.py` la
    # recopie dans index.json : le kit livré dit alors de quel encodeur ET de
    # quelle révision de métadonnées il vient.
    (OUT_DIR / "source.json").write_text(json.dumps({
        "repo": REPO,
        "revision": ref,
        "pinned_revision": DATA_REVISION,
        "matches_pin": ref == DATA_REVISION,
        "sets": len(sets),
        "cards": len(cards),
    }, indent=2))

    print(f"\n{len(cards)} cartes -> {out}")
    print(f"  révision de la source : {ref[:12]}"
          f"{'' if ref == DATA_REVISION else '  (HORS ÉPINGLE)'}")
    if skipped_no_image:
        print(f"  {skipped_no_image} cartes ignorées (pas d'image)")
    if duplicates:
        print(f"  {duplicates} doublons d'ID supprimés")

    latest = max(cards, key=lambda c: c["release_date"] or "")
    print(f"  set le plus récent : {latest['set_name']} ({latest['release_date']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
