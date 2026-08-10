"""Complète `cards.json` avec les cartes que pokemon-tcg-data ne couvre pas.

    python scripts/add_missing_cards.py --check   # diagnostic seul, rien n'est écrit
    python scripts/add_missing_cards.py           # complète data/processed/cards.json

Deux trous, tous deux relevés par `refresh_index.py --check` :

- **des sets entiers absents.** TCGdex les référence — c'est ainsi que les promos
  MEP de 2026 ont été repérées — mais sans visuel. Les images viennent donc de
  scrydex, les métadonnées de TCGdex.
- **des cartes dont l'image répond 404** sur `images.pokemontcg.io` alors que la
  carte est bien dans les métadonnées.

Une carte n'est ajoutée que si une **vraie** image existe. scrydex ne renvoie
jamais 404 : il sert un placeholder en HTTP 200, et l'ajouter au dataset
créerait des dos de carte identiques, attracteurs universels dans l'espace des
embeddings. Voir `src/fallback_images.py`.

Après ce script, la suite habituelle : `download_images.py`, `build_embeddings.py`,
`export_index.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fallback_images import has_real_image, scrydex_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARDS_JSON = ROOT / "data" / "processed" / "cards.json"
TCGDEX = "https://api.tcgdex.net/v2/en"

# Hors périmètre : jeu mobile, pas des cartes physiques à scanner.
IGNORED_SERIES = {"tcgp"}

# Assez pour saturer le CDN sans le fâcher.
WORKERS = 12


def get_json(url: str):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_cards() -> list[dict]:
    if not CARDS_JSON.exists():
        sys.exit(f"{CARDS_JSON} absent — lancer d'abord scripts/build_metadata.py")
    return json.loads(CARDS_JSON.read_text())


def normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def missing_sets(known_names: set[str]) -> list[dict]:
    """Sets de la série physique en cours que l'index local ignore.

    Comparaison par **nom** normalisé et non par identifiant : les deux sources
    ne s'accordent pas dessus — TCGdex écrit `me01` et `me02.5` là où
    pokemon-tcg-data écrit `me1` et `me2pt5`. Comparer les identifiants
    déclarerait manquants six sets déjà indexés.

    Même restriction que `refresh_index.py` : la dernière série physique
    seulement. Tout le catalogue produirait surtout du bruit.
    """
    series = [s for s in get_json(f"{TCGDEX}/series") if s["id"] not in IGNORED_SERIES]
    current = get_json(f"{TCGDEX}/series/{series[-1]['id']}")

    return [
        s for s in current.get("sets", [])
        if normalize(s["name"]) not in known_names
        and (s.get("cardCount") or {}).get("total", 0) > 0
    ]


def cards_of_set(set_id: str) -> tuple[dict, list[dict]]:
    detail = get_json(f"{TCGDEX}/sets/{set_id}")
    return detail, detail.get("cards", [])


def canonical_id(card: dict, set_id: str) -> str:
    """Identifiant à la convention de pokemon-tcg-data, qui est celle de l'index.

    TCGdex numérote sur trois chiffres (`mep-001`) là où le reste du dataset ne
    remplit pas (`me5-51`, `svp-102`). L'écart n'est pas cosmétique : scrydex
    sert son placeholder pour `mep-001` et la vraie carte pour `mep-1`, si bien
    qu'une carte sur trois chiffres est déclarée sans image et silencieusement
    écartée.

    Les numéros non numériques — `SWSH274` sur les vieilles promos — passent
    tels quels.
    """
    local = str(card.get("localId", ""))
    return f"{set_id}-{int(local)}" if local.isdigit() else f"{set_id}-{local}"


def entry_from_tcgdex(card: dict, set_detail: dict, series_name: str) -> dict:
    """Une carte TCGdex au format de `cards.json`, images pointées sur scrydex.

    `number` reste le numéro tel qu'imprimé, zéros de tête compris : c'est lui
    que l'OCR du bandeau lira, et la comparaison aval normalise déjà.
    """
    card_id = canonical_id(card, set_detail["id"])
    printed = set_detail.get("cardCount", {}) or {}

    # Un set de promos n'imprime pas de total sur ses cartes — le Meganium MEP
    # porte « 001 » seul — et TCGdex l'exprime par `official: 0`. Le dataset
    # existant y met malgré tout la taille du set (svp: 102, swshp: 307), et s'en
    # écarter mettrait un 0 là où tout le reste a un entier utile. Le champ ne
    # sert de toute façon pas à départager une promo : la lecture du bandeau
    # cherche un motif « numéro/total » qui n'y figure pas.
    total = printed.get("official") or printed.get("total")

    return {
        "id": card_id,
        "name": card.get("name"),
        "number": str(card.get("localId", "")),
        "rarity": card.get("rarity"),
        "artist": card.get("illustrator"),
        "supertype": card.get("category"),
        "set_id": set_detail["id"],
        "set_name": set_detail.get("name"),
        "set_series": series_name,
        "set_printed_total": total,
        "release_date": set_detail.get("releaseDate"),
        "image_small": scrydex_url(card_id, "small"),
        "image_large": scrydex_url(card_id, "large"),
    }


def find_broken_images(cards: list[dict], session: requests.Session) -> list[dict]:
    """Cartes déjà présentes dont l'image d'origine a disparu, mais que scrydex a.

    Repose sur le journal d'échecs de `download_images.py` plutôt que de
    re-tester 20 000 URLs : les échecs sont déjà connus, il ne reste qu'à voir
    lesquels sont rattrapables.
    """
    log = ROOT / "data" / "raw" / "download_failures_large.json"
    if not log.exists():
        return []

    failed_ids = set(json.loads(log.read_text()))
    by_id = {c["id"]: c for c in cards}
    candidates = [by_id[i] for i in failed_ids if i in by_id]
    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        usable = list(pool.map(lambda c: has_real_image(c["id"], session), candidates))

    return [card for card, ok in zip(candidates, usable) if ok]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="diagnostic seul")
    args = parser.parse_args()

    cards = load_cards()
    known_ids = {c["set_id"] for c in cards}
    known_names = {normalize(c["set_name"]) for c in cards if c.get("set_name")}
    print(f"index local : {len(known_ids)} sets, {len(cards)} cartes\n")

    session = requests.Session()
    additions: list[dict] = []

    absent = missing_sets(known_names)
    for meta in absent:
        detail, listed = cards_of_set(meta["id"])
        series_name = (detail.get("serie") or {}).get("name")

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            usable = list(pool.map(
                lambda c: has_real_image(canonical_id(c, detail["id"]), session), listed
            ))

        keep = [c for c, ok in zip(listed, usable) if ok]
        print(f"{meta['id']:<8} {meta.get('name', ''):<34} "
              f"{len(keep):>3}/{len(listed):<3} avec une vraie image")

        additions += [entry_from_tcgdex(c, detail, series_name) for c in keep]

    repairable = find_broken_images(cards, session)
    if repairable:
        print(f"\n{len(repairable)} carte(s) déjà indexée(s) dont l'image est "
              f"récupérable sur scrydex :")
        for card in repairable:
            print(f"  {card['id']:<14} {card['name']}")

    print(f"\n{len(additions)} carte(s) à ajouter, {len(repairable)} à réparer")

    if args.check:
        print("\n--check : rien n'a été écrit.")
        return 0

    if not additions and not repairable:
        print("Rien à faire.")
        return 0

    # Réparer avant d'ajouter : les deux touchent des entrées distinctes, mais
    # l'ordre rend le décompte final lisible.
    for card in repairable:
        card["image_small"] = scrydex_url(card["id"], "small")
        card["image_large"] = scrydex_url(card["id"], "large")

    merged = cards + additions
    merged.sort(key=lambda c: (c.get("release_date") or "", c["id"]))
    CARDS_JSON.write_text(json.dumps(merged, ensure_ascii=False, indent=1))

    print(f"\n{len(merged)} cartes -> {CARDS_JSON}")
    print("Suite : download_images.py, build_embeddings.py, export_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
