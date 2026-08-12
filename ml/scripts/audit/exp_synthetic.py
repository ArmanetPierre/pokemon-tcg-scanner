"""Banc synthetique : evaluer la recherche sur TOUT l'index, pas sur 31 photos.

Principe : partir du scan de reference, lui appliquer les degradations d'une
photo de telephone (perspective, flou, bruit, derive colorimetrique, reflet,
JPEG), puis chercher. La verite terrain est gratuite et exacte.

Ce n'est PAS un substitut aux vraies photos : les augmentations ne reproduisent
ni l'optique du capteur ni la vraie geometrie de detection. C'est une mesure
RELATIVE — elle compare des eres, des sets et des variantes de modele entre
elles sur des milliers de cartes, la ou 31 photos ne mesurent rien.
"""
import json, sys, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path("/Users/pierre/Projets/Pokemon/ml")
sys.path.insert(0, str(ROOT))
from src.encoder import load_encoder
from src.search import CardIndex

rng = random.Random(0)
np.random.seed(0)

index = CardIndex("mobileclip2-s2")
E = index.embeddings.astype(np.float32)
ids = np.array(index.card_ids)
cards = index.cards_by_id
IMG = ROOT / "data/raw/images_large"


def era(c):
    y = int(c["release_date"][:4])
    s = c["set_series"]
    if s in ("Base", "Gym", "Neo", "Legendary Collection"): return f"1 WotC {y//10*10}s"
    if s == "EX": return "2 EX"
    if s in ("Diamond & Pearl", "Platinum"): return "3 DP/Pt"
    if s in ("HeartGold & SoulSilver",): return "4 HGSS"
    if s == "Black & White": return "5 BW"
    if s == "XY": return "6 XY"
    if s == "Sun & Moon": return "7 SM"
    if s == "Sword & Shield": return "8 SWSH"
    if s == "Scarlet & Violet": return "9 SV"
    return "A autre/promo"


def degrade(img: Image.Image, r: random.Random) -> Image.Image:
    """Simule grossierement une photo de telephone d'une carte deja redressee."""
    a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    # perspective residuelle (le redressement n'est jamais parfait)
    j = 0.02
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[r.uniform(-j, j) * w, r.uniform(-j, j) * h] for _ in range(4)])
    a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h),
                            borderMode=cv2.BORDER_REPLICATE)
    # resolution reelle du crop, bien plus basse que le scan
    k = r.uniform(0.25, 0.45)
    a = cv2.resize(a, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    # flou de mise au point / bouge
    if r.random() < 0.6:
        a = cv2.GaussianBlur(a, (0, 0), r.uniform(0.6, 1.6))
    # exposition + balance des blancs
    a = a.astype(np.float32)
    a *= r.uniform(0.75, 1.25)
    a *= np.float32([r.uniform(.9, 1.1) for _ in range(3)])
    # reflet holo : tache claire diagonale
    if r.random() < 0.5:
        hh, ww = a.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww]
        g = np.exp(-(((xx - r.uniform(0, ww)) + (yy - r.uniform(0, hh))) ** 2)
                   / (2 * (0.45 * ww) ** 2)).astype(np.float32)
        a += (g * r.uniform(25, 70))[..., None]
    a += np.random.normal(0, r.uniform(1, 5), a.shape).astype(np.float32)
    a = np.clip(a, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, r.randint(55, 88)])
    a = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


PER_ERA = int(sys.argv[1]) if len(sys.argv) > 1 else 120
by_era = defaultdict(list)
for cid in index.card_ids:
    by_era[era(cards[cid])].append(cid)

sample = []
for e in sorted(by_era):
    pool = by_era[e]
    rng.shuffle(pool)
    for cid in pool[:PER_ERA]:
        p = IMG / cards[cid]["set_id"] / f"{cid}.jpg"
        if p.exists():
            sample.append((e, cid, p))
print(f"{len(sample)} requetes synthetiques sur {len(by_era)} eres", flush=True)

enc = load_encoder("mobileclip2-s2")
res = defaultdict(lambda: [0, 0, 0, []])  # n, top1, top5, marges
B = 64
for s0 in range(0, len(sample), B):
    chunk = sample[s0:s0 + B]
    imgs = [degrade(Image.open(p), random.Random(hash(cid) & 0xffff)) for _, cid, p in chunk]
    Q = enc.encode(imgs)
    S = Q @ E.T
    top = np.argpartition(-S, 5, axis=1)[:, :5]
    for i, (e, cid, _) in enumerate(chunk):
        o = top[i][np.argsort(-S[i, top[i]])]
        pred = ids[o]
        row = res[e]
        row[0] += 1
        row[1] += pred[0] == cid
        row[2] += cid in pred
        row[3].append(float(S[i, o[0]] - S[i, o[1]]))
    if s0 % (B * 8) == 0:
        print(f"  {s0+len(chunk)}/{len(sample)}", flush=True)

print(f"\n{'ere':<16} {'n':>5} {'top-1':>7} {'top-5':>7} {'marge med':>10}")
print("-" * 50)
tn = t1 = t5 = 0
for e in sorted(res):
    n, a, b, m = res[e]
    tn += n; t1 += a; t5 += b
    print(f"{e:<16} {n:>5} {a/n:>6.1%} {b/n:>7.1%} {np.median(m):>10.4f}")
print("-" * 50)
print(f"{'TOTAL':<16} {tn:>5} {t1/tn:>6.1%} {t5/tn:>7.1%}")
