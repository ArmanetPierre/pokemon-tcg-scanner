"""Vérifie la fraîcheur de l'index et le reconstruit en une commande.

    python scripts/refresh_index.py --check     # diagnostic seul, rien n'est écrit
    python scripts/refresh_index.py             # diagnostic puis reconstruction

Deux sources sont croisées :

- **pokemon-tcg-data** (GitHub) fournit métadonnées *et* URLs d'images. C'est la
  source de l'index.
- **TCGdex** sert de sentinelle : il référence des sets que la première ignore
  encore (constaté : les promos « MEP » et les énergies « MEE » de 2026). Ses
  images manquent souvent pour ces sets récents, donc il ne peut pas les
  remplacer — mais il dit quand l'index est en retard, et sur quoi.

Une carte sans image reste hors index : le scanner la déclarera « inconnue »
plutôt que de lui attribuer une carte voisine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CARDS_JSON = ROOT / "data" / "processed" / "cards.json"
TCGDEX = "https://api.tcgdex.net/v2/en"

# Séries hors périmètre : jeu mobile, pas des cartes physiques à scanner.
IGNORED_SERIES = {"tcgp"}


def load_local_sets() -> dict[str, dict]:
    if not CARDS_JSON.exists():
        return {}
    sets: dict[str, dict] = {}
    for card in json.loads(CARDS_JSON.read_text()):
        entry = sets.setdefault(card["set_id"], {"name": card["set_name"], "count": 0})
        entry["count"] += 1
    return sets


def get_json(url: str):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def check_freshness() -> list[dict]:
    """Compare la série en cours à l'index local.

    Volontairement limité à la dernière série physique : comparer tout le
    catalogue produit surtout du bruit (« Base Set » contre notre « Base »,
    Trainer Kits, jeu mobile). Le seul signal qui compte en pratique est
    « un set de la génération actuelle vient de sortir et je ne l'ai pas ».
    """
    local = load_local_sets()
    local_names = {normalize(v["name"]) for v in local.values()}
    print(f"index local : {len(local)} sets, {sum(v['count'] for v in local.values())} cartes")

    try:
        series = [s for s in get_json(f"{TCGDEX}/series") if s["id"] not in IGNORED_SERIES]
        current = get_json(f"{TCGDEX}/series/{series[-1]['id']}")
    except Exception as exc:  # noqa: BLE001
        print(f"TCGdex injoignable ({exc}) — vérification de fraîcheur ignorée")
        return []

    print(f"série en cours d'après TCGdex : {current.get('name')}")
    missing = [
        s for s in current.get("sets", [])
        if normalize(s["name"]) not in local_names
        and (s.get("cardCount") or {}).get("total", 0) > 0
    ]
    if not missing:
        print("index à jour sur la série en cours")
        return []

    print(f"\n{len(missing)} sets de cette série absents de l'index :")
    for s in missing:
        total = (s.get("cardCount") or {}).get("total")
        print(f"  {s['id']:<10} {s['name'][:38]:<40} {total} cartes")
    print(
        "\nCes sets n'entreront dans l'index que lorsque pokemon-tcg-data les publiera\n"
        "AVEC des images — TCGdex les référence sans visuel, donc sans embedding\n"
        "possible. En attendant, ces cartes sont signalées « inconnues » plutôt que\n"
        "rattachées à une carte voisine."
    )
    return missing


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"\n$ {' '.join(command[1:])}")
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="diagnostic seul")
    parser.add_argument("--model", default="mobileclip2-s2")
    args = parser.parse_args()

    before = load_local_sets()
    check_freshness()
    if args.check:
        return 0

    run("build_metadata.py")
    run("download_images.py", "--size", "large")
    run("build_embeddings.py", "--model", args.model)

    after = load_local_sets()
    new_sets = sorted(set(after) - set(before))
    grown = [s for s in set(after) & set(before) if after[s]["count"] != before[s]["count"]]

    print("\n--- bilan ---")
    print(f"cartes : {sum(v['count'] for v in before.values())} -> "
          f"{sum(v['count'] for v in after.values())}")
    if new_sets:
        labels = ", ".join("{} ({})".format(s, after[s]["name"]) for s in new_sets)
        print(f"nouveaux sets : {labels}")
    if grown:
        print(f"sets complétés : {', '.join(grown)}")
    if not new_sets and not grown:
        print("index déjà à jour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
