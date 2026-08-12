"""Comparaison honnete : courbe risque/couverture, invariante a l'echelle.

Un seuil de marge n'est pas comparable entre deux espaces (le centrage
multiplie les marges par ~3,5). Ce qui se compare, c'est le pouvoir de tri :
a couverture egale, combien d'erreurs affichees comme fermes ?
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/pierre/Projets/Pokemon/ml")
sys.path.insert(0, str(ROOT))
from src.encoder import load_encoder
from src.pipeline import build_variants, full_photo_variant, load_bgr
from src.search import CardIndex

PROJECT = ROOT.parent
truth = json.loads((ROOT / "data/eval/truth.json").read_text())["photos"]
index = CardIndex("mobileclip2-s2")
E = index.embeddings.astype(np.float32)
mu = E.mean(0)
Ec = E - mu; Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
enc = load_encoder("mobileclip2-s2")
ids = np.array(index.card_ids)
names = np.array([index.cards_by_id[c]["name"] for c in index.card_ids])

def probe(q, M, k=20):
    s = q @ M.T
    top = np.argpartition(-s, k)[:k]; top = top[np.argsort(-s[top])]
    return top, s[top]

def csls(q, M, r_index, k=20):
    """CSLS : penalise les hubs par leur similarite moyenne a leurs voisins."""
    s = q @ M.T
    kq = np.partition(s, -10)[-10:].mean()
    s = 2 * s - r_index - kq
    top = np.argpartition(-s, k)[:k]; top = top[np.argsort(-s[top])]
    return top, s[top]

# r_index : similarite moyenne de chaque carte a ses 10 plus proches voisins
print("pre-calcul CSLS...", flush=True)
r = np.zeros(len(E), np.float32)
for s0 in range(0, len(E), 2048):
    S = E[s0:s0+2048] @ E.T
    for i in range(S.shape[0]): S[i, s0+i] = -2
    r[s0:s0+2048] = np.partition(S, -10, axis=1)[:, -10:].mean(1)

recs = []
for rel, info in truth.items():
    bgr = load_bgr(str(PROJECT / rel))
    if bgr is None: continue
    vs = build_variants(str(PROJECT / rel), bgr) or [full_photo_variant(bgr)]
    Q = enc.encode([v.image for v in vs])
    Qc = Q - mu; Qc /= np.linalg.norm(Qc, axis=1, keepdims=True)
    rec = {"exp": info["card_id"], "name": Path(rel).name}
    for space, Qs, fn in (("brut", Q, lambda q: probe(q, E)),
                          ("centre", Qc, lambda q: probe(q, Ec)),
                          ("csls", Q, lambda q: csls(q, E, r))):
        best, bv = None, -9
        for q in Qs:
            top, sc = fn(q)
            v = sc[0] + (sc[0] - sc[1])
            if v > bv: bv, best = v, (top, sc)
        top, sc = best
        n0 = names[top[0]]
        m_nm = next((sc[0]-s for t, s in zip(top[1:], sc[1:]) if names[t] != n0), sc[0]-sc[1])
        rec[space] = (str(ids[top[0]]), float(sc[0]-sc[1]), float(m_nm), list(map(str, ids[top[:5]])))
    recs.append(rec)
    print(".", end="", flush=True)
print()

ans = [r for r in recs if r["exp"] is not None]
print(f"\n{len(ans)} photos avec une reponse juste\n")
print(f"{'espace':<8} {'top-1':>7} {'top-5':>7} | precision a couverture fixee (tri par marge 1er/2e)")
print(f"{'':<8} {'':>7} {'':>7} | {'100%':>8} {'75%':>8} {'50%':>8} {'25%':>8} | AURC")
for space in ("brut", "centre", "csls"):
    ok = np.array([r[space][0] == r["exp"] for r in ans])
    t5 = np.mean([r["exp"] in r[space][3] for r in ans])
    m = np.array([r[space][1] for r in ans])
    order = np.argsort(-m)
    okr = ok[order]
    cells = []
    for cov in (1.0, .75, .50, .25):
        n = max(1, int(round(cov*len(ans))))
        cells.append(f"{okr[:n].mean():7.1%}")
    # AURC : erreur moyenne sur tous les prefixes de couverture (bas = bon)
    aurc = np.mean([1 - okr[:i+1].mean() for i in range(len(okr))])
    print(f"{space:<8} {ok.sum():>3}/{len(ans):<3} {t5:>6.0%} | " +
          " ".join(cells) + f" | {aurc:.4f}")

print("\nseuil qui garantit 100 % de precision, et ce qu'il retient :")
for space in ("brut", "centre", "csls"):
    ok = np.array([r[space][0] == r["exp"] for r in ans])
    m = np.array([r[space][1] for r in ans])
    order = np.argsort(-m); okr, mr = ok[order], m[order]
    n = 0
    while n < len(okr) and okr[:n+1].all(): n += 1
    print(f"  {space:<8} marge >= {mr[n-1]:.4f} -> {n}/{len(ans)} photos "
          f"({n/len(ans):.0%} de couverture) sans erreur")
