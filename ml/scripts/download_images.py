"""Télécharge les images de référence des cartes.

Par défaut `image_large` (745x1040). La haute résolution sert moins à
l'embedding lui-même (224-256 px en entrée) qu'à l'étape de départage des
cartes visuellement identiques : le bandeau bas portant le numéro et le symbole
du set est illisible en 245x342. Elle laisse aussi la porte ouverte aux modèles
à entrée 336/448 px.

Les PNG sont recompressés en JPEG à la volée, ce qui divise le dataset par ~4
sans perte visible.

Le script est reprenable : relancer après une interruption ne retélécharge que
ce qui manque.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CARDS_JSON = ROOT / "data" / "processed" / "cards.json"
RAW_DIR = ROOT / "data" / "raw"

# Par taille : URL du champ metadata, côté max conservé, qualité JPEG.
# Le côté max n'entraîne un redimensionnement que s'il est dépassé ; les valeurs
# ci-dessous sont les résolutions natives, donc on ne fait que recompresser.
SIZES = {
    "small": {"field": "image_small", "max_side": 384, "quality": 92},
    "large": {"field": "image_large", "max_side": 1040, "quality": 95},
}
RETRIES = 4

_print_lock = threading.Lock()


def image_dir(size: str) -> Path:
    return RAW_DIR / f"images_{size}"


def target_path(card: dict, size: str) -> Path:
    return image_dir(size) / card["set_id"] / f"{card['id']}.jpg"


def download_one(card: dict, session: requests.Session, size: str) -> str:
    """Retourne 'ok', 'skip' ou 'fail:<raison>'."""
    spec = SIZES[size]
    dest = target_path(card, size)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"

    url = card.get(spec["field"]) or card["image_small"]

    for attempt in range(RETRIES):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 404 and url != card["image_small"]:
                # Certains sets promo (McDonald's, etc.) n'ont pas de version
                # _hires. Mieux vaut la basse résolution qu'un trou dans l'index.
                url = card["image_small"]
                continue
            resp.raise_for_status()

            img = Image.open(io.BytesIO(resp.content))
            # Les PNG sources sont en RGBA ; JPEG exige du RGB.
            img = img.convert("RGB")
            if max(img.size) > spec["max_side"]:
                img.thumbnail((spec["max_side"], spec["max_side"]), Image.LANCZOS)

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".jpg.part")
            img.save(tmp, "JPEG", quality=spec["quality"], optimize=True)
            # Rename atomique : une interruption ne laisse jamais un .jpg tronqué
            # que la reprise considérerait comme valide.
            tmp.replace(dest)
            return "ok"
        except Exception as exc:  # noqa: BLE001 - on veut la raison, pas le type
            if attempt == RETRIES - 1:
                return f"fail:{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return "fail:unreachable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=sorted(SIZES), default="large")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="0 = toutes les cartes")
    args = parser.parse_args()

    cards = json.loads(CARDS_JSON.read_text())
    if args.limit:
        cards = cards[: args.limit]

    out_dir = image_dir(args.size)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "pokemon-card-scanner/0.1"

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: list[tuple[str, str]] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, c, session, args.size): c for c in cards}
        for i, future in enumerate(as_completed(futures), 1):
            card = futures[future]
            result = future.result()
            if result.startswith("fail"):
                counts["fail"] += 1
                failures.append((card["id"], result[5:]))
            else:
                counts[result] += 1

            if i % 250 == 0 or i == len(cards):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                eta = (len(cards) - i) / rate if rate else 0
                with _print_lock:
                    print(
                        f"{i}/{len(cards)}  ok={counts['ok']} skip={counts['skip']} "
                        f"fail={counts['fail']}  {rate:.0f}/s  ETA {eta / 60:.1f} min",
                        flush=True,
                    )

    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*.jpg")) / 1e6
    print(f"\nTerminé en {(time.time() - start) / 60:.1f} min — {size_mb:.0f} Mo sur disque")

    if failures:
        report = RAW_DIR / f"download_failures_{args.size}.json"
        report.write_text(json.dumps(dict(failures), ensure_ascii=False, indent=2))
        print(f"{len(failures)} échecs -> {report} (relancer le script pour réessayer)")
        for card_id, reason in failures[:10]:
            print(f"  {card_id}: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
