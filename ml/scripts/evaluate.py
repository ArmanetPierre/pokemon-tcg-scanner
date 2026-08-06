"""Mesure la précision d'identification sur les photos réelles.

    python scripts/evaluate.py
    python scripts/evaluate.py --model dinov2-b --top-k 5

Attend l'arborescence décrite dans data/eval/README.md : un dossier par carte
physique, nommé `<nom>_<numero>-<total>`, contenant une photo par condition.

Le résultat est ventilé par condition : une moyenne globale ne dit pas si
l'échec vient des reflets ou de la perspective.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pillow_heif
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.search import CardIndex  # noqa: E402

pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "data" / "eval" / "photos"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}

CONDITIONS = (
    "plat", "incline", "pivote", "reflet", "sombre",
    "sleeve", "loin", "masque", "fond", "flou",
)


def condition_of(path: Path) -> str:
    stem = path.stem.lower()
    for cond in CONDITIONS:
        if cond in stem:
            return cond
    return "autre"


def normalize(name: str) -> str:
    """'Nidoran ♂' -> 'nidoran', 'Kangaskhan ex' -> 'kangaskhanex'.

    Les noms de cartes contiennent des symboles de genre, des accents et des
    suffixes que personne ne tapera dans un nom de dossier.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in decomposed if ch.isalnum() and ch.isascii()).lower()


MIN_PREFIX = 4


def name_matches(card_name: str, query: str) -> bool:
    a, b = normalize(card_name), normalize(query)
    if not a or not b:
        return False
    if a == b:
        return True
    # Sans longueur minimale, la carte dresseur « N » serait préfixe de tout
    # nom commençant par un n. On n'accepte un préfixe que s'il est distinctif.
    shorter = min(len(a), len(b))
    return shorter >= MIN_PREFIX and (a.startswith(b) or b.startswith(a))


def resolve_card(folder: str, cards: list[dict]) -> tuple[str | None, str]:
    """Dossier `pikachu_58-102` -> card_id. Retourne (id, message)."""
    match = re.search(r"(\d+)\s*[-/]\s*(\d+)\s*$", folder)
    name_part = (folder[: match.start()] if match else folder).strip("_- ")

    if not match:
        by_name = [c for c in cards if name_matches(c["name"], name_part)]
        if len(by_name) == 1:
            return by_name[0]["id"], "résolu par nom seul"
        return None, f"numéro manquant, et {len(by_name)} cartes portent ce nom"

    number, total = match.group(1), int(match.group(2))
    candidates = [
        c for c in cards
        if c["number"] == number and c.get("set_printed_total") == total
    ]
    by_name = [c for c in candidates if name_matches(c["name"], name_part)]

    if len(by_name) == 1:
        return by_name[0]["id"], "ok"
    if len(by_name) > 1:
        return None, f"ambigu entre {', '.join(c['id'] for c in by_name)}"

    if not candidates:
        return None, f"aucune carte {number}/{total} dans l'index"

    # Aucun candidat ne porte ce nom. Si le nom existe ailleurs dans l'index,
    # c'est probablement le numéro qui est faux — accepter produirait une
    # vérité terrain silencieusement fausse, donc on refuse.
    elsewhere = [c for c in cards if name_matches(c["name"], name_part)]
    if elsewhere:
        example = ", ".join(f"{c['id']} ({c['number']}/{c['set_printed_total']})" for c in elsewhere[:3])
        return None, f"'{name_part}' existe mais pas en {number}/{total} — numéro erroné ? voir {example}"

    # Nom inconnu de l'index : cas attendu pour une carte française. Le couple
    # numéro/total tranche seul, à condition qu'il soit unique.
    if len(candidates) == 1:
        card = candidates[0]
        return card["id"], f"nom '{name_part}' absent de l'index (carte FR ?), retenu {card['name']}"
    listing = ", ".join(f"{c['id']} ({c['name']})" for c in candidates)
    return None, f"{number}/{total} correspond à {len(candidates)} cartes : {listing}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cards = json.loads((ROOT / "data" / "processed" / "cards.json").read_text())

    folders = sorted(p for p in PHOTO_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")) \
        if PHOTO_DIR.exists() else []
    if not folders:
        print(f"Aucune photo dans {PHOTO_DIR}. Voir data/eval/README.md.")
        return 1

    # Résoudre toutes les vérités terrain avant d'encoder quoi que ce soit :
    # inutile de charger le modèle si le nommage est à corriger.
    truth: dict[Path, str] = {}
    problems: list[str] = []
    for folder in folders:
        card_id, msg = resolve_card(folder.name, cards)
        if card_id is None:
            problems.append(f"  {folder.name} : {msg}")
        else:
            truth[folder] = card_id
            if msg != "ok":
                print(f"  {folder.name} -> {card_id} ({msg})")

    if problems:
        print("Dossiers non résolus :")
        print("\n".join(problems))
        if not truth:
            return 1
        print()

    photos = [
        (p, truth[folder])
        for folder in truth
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in EXTENSIONS
    ]
    print(f"{len(photos)} photos sur {len(truth)} cartes\n")

    encoder = load_encoder(args.model)
    index = CardIndex(args.model)

    images = [Image.open(p).convert("RGB") for p, _ in photos]
    vectors = encoder.encode(images)
    results = index.search(vectors, k=max(args.top_k, 2))

    by_condition: dict[str, list[tuple[bool, bool, float, str, str]]] = defaultdict(list)
    for (path, expected), hits in zip(photos, results):
        ids = [h.card_id for h in hits]
        top1 = ids[0] == expected
        topk = expected in ids[: args.top_k]
        margin = hits[0].score - hits[1].score
        by_condition[condition_of(path)].append((top1, topk, margin, ids[0], str(path)))

    print(f"{'condition':<12} {'n':>4} {'top-1':>7} {'top-' + str(args.top_k):>7} {'marge moy':>10}")
    print("-" * 44)
    all_rows = []
    for cond in list(CONDITIONS) + ["autre"]:
        rows = by_condition.get(cond)
        if not rows:
            continue
        all_rows += rows
        n = len(rows)
        t1 = sum(r[0] for r in rows) / n
        tk = sum(r[1] for r in rows) / n
        mg = sum(r[2] for r in rows) / n
        print(f"{cond:<12} {n:>4} {t1:>6.0%} {tk:>7.0%} {mg:>10.3f}")

    n = len(all_rows)
    print("-" * 44)
    print(
        f"{'TOTAL':<12} {n:>4} {sum(r[0] for r in all_rows) / n:>6.0%} "
        f"{sum(r[1] for r in all_rows) / n:>7.0%} {sum(r[2] for r in all_rows) / n:>10.3f}"
    )

    failures = [r for r in all_rows if not r[0]]
    if failures:
        print(f"\n{len(failures)} échecs top-1 :")
        for _, topk, margin, got, path in failures[:15]:
            name = index.cards_by_id[got]["name"]
            flag = "top-5 ok" if topk else "hors top-5"
            print(f"  {Path(path).parent.name}/{Path(path).name} -> {got} ({name}), {flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
