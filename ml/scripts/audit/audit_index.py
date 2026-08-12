"""Audit de la structure de l'index d'embeddings (lecture seule)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/pierre/Projets/Pokemon/ml")
E = np.load(ROOT / "data/embeddings/mobileclip2-s2/embeddings.npy").astype(np.float32)
ids = json.loads((ROOT / "data/embeddings/mobileclip2-s2/card_ids.json").read_text())
cards = {c["id"]: c for c in json.loads((ROOT / "data/processed/cards.json").read_text())}
N, D = E.shape
print(f"index {N} x {D}")

# --- 1. anisotropie / hubness : la moyenne des vecteurs ---
mu = E.mean(0)
print(f"\n[anisotropie] ||mean vector|| = {np.linalg.norm(mu):.4f}  (0 = isotrope)")
sim_to_mean = E @ (mu / np.linalg.norm(mu))
print(f"  cos(carte, direction moyenne) : med {np.median(sim_to_mean):.3f}  "
      f"p5 {np.percentile(sim_to_mean,5):.3f}  p95 {np.percentile(sim_to_mean,95):.3f}")

# --- 2. voisins les plus proches, en blocs ---
CH = 1024
nn1 = np.zeros(N, np.float32); nn1_idx = np.zeros(N, np.int32)
nn2 = np.zeros(N, np.float32)
hub10 = np.zeros(N, np.int64)
rand_pairs = []
for s in range(0, N, CH):
    blk = E[s:s+CH]
    S = blk @ E.T
    for r in range(blk.shape[0]):
        S[r, s+r] = -2.0
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    for r in range(blk.shape[0]):
        o = top[r][np.argsort(-S[r, top[r]])]
        nn1[s+r], nn1_idx[s+r], nn2[s+r] = S[r, o[0]], o[0], S[r, o[1]]
        hub10[o] += 1
    if s == 0:
        rand_pairs = S[:200, ::97].ravel()

print(f"\n[baseline] similarite entre cartes quelconques : "
      f"med {np.median(rand_pairs):.3f}  p99 {np.percentile(rand_pairs,99):.3f}")

print("\n[voisin le plus proche dans l'index]")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q}: {np.percentile(nn1, q):.4f}")
for t in (0.99, 0.98, 0.97, 0.95, 0.92):
    n = int((nn1 >= t).sum())
    print(f"  cartes ayant un voisin >= {t}: {n:5d}  ({100*n/N:.1f} %)")

# separabilite intrinseque : nn1 - nn2 (marge disponible si la requete tombe pile)
gap = nn1 - nn2
print(f"\n[marge intrinseque 1er/2e voisin] med {np.median(gap):.4f}  "
      f"p10 {np.percentile(gap,10):.4f}  "
      f"< 0.03 : {100*(gap<0.03).mean():.1f} %  < 0.01 : {100*(gap<0.01).mean():.1f} %")

# --- 3. le voisin le plus proche est-il la meme carte (nom) ? ---
same_name = sum(cards[ids[i]]["name"] == cards[ids[nn1_idx[i]]]["name"] for i in range(N))
print(f"[voisin de meme nom] {100*same_name/N:.1f} %  "
      f"-> la confusion est {'majoritairement intra-nom' if same_name/N>0.5 else 'majoritairement inter-nom'}")

# --- 4. hubs : cartes qui apparaissent dans le top-10 de tout le monde ---
print("\n[hubness] apparitions dans le top-10 des autres (attendu ~10)")
print(f"  med {np.median(hub10):.0f}  p99 {np.percentile(hub10,99):.0f}  max {hub10.max()}")
order = np.argsort(-hub10)[:12]
for i in order:
    c = cards[ids[i]]
    print(f"  {hub10[i]:6d}  {ids[i]:<14} {c['name'][:24]:<25} {c['set_name'][:28]}")
n_hub = int((hub10 > 100).sum())
print(f"  cartes vues > 100 fois : {n_hub} ({100*n_hub/N:.2f} %) — "
      f"elles captent {100*hub10[hub10>100].sum()/hub10.sum():.1f} % de tous les top-10")

# --- 5. effet du centrage (correction d'anisotropie) ---
Ec = E - mu
Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
nn1c = np.zeros(N, np.float32); nn2c = np.zeros(N, np.float32)
hub10c = np.zeros(N, np.int64)
for s in range(0, N, CH):
    blk = Ec[s:s+CH]
    S = blk @ Ec.T
    for r in range(blk.shape[0]):
        S[r, s+r] = -2.0
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    for r in range(blk.shape[0]):
        o = top[r][np.argsort(-S[r, top[r]])]
        nn1c[s+r], nn2c[s+r] = S[r, o[0]], S[r, o[1]]
        hub10c[o] += 1
    if s == 0:
        rp_c = S[:200, ::97].ravel()
gapc = nn1c - nn2c
print(f"\n[apres centrage] baseline cartes quelconques med {np.median(rp_c):.3f} "
      f"(etait {np.median(rand_pairs):.3f})")
print(f"  marge intrinseque med {np.median(gapc):.4f} (etait {np.median(gap):.4f})  "
      f"-> x{np.median(gapc)/np.median(gap):.2f}")
print(f"  hubness max {hub10c.max()} (etait {hub10.max()}) ; "
      f"cartes vues >100 fois : {(hub10c>100).sum()} (etait {n_hub})")

# --- 6. difficulte par ere ---
print("\n[par set : sets les plus auto-confusants] (voisin >= 0.97 dans le set)")
by_set = defaultdict(list)
for i in range(N):
    by_set[cards[ids[i]]["set_name"]].append(nn1[i])
rows = [(np.mean(np.array(v) >= 0.97), len(v), k) for k, v in by_set.items() if len(v) >= 30]
for frac, n, k in sorted(rows, reverse=True)[:10]:
    print(f"  {100*frac:5.1f} %  n={n:4d}  {k}")
