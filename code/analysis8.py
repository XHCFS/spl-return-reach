"""Are the two alignment gradients in the data significant? Elsayed alignment (prep->exec, exec->return)
with units subsampled to the same N in every area (N = 150), repeated draws; pairwise area differences.
"""
import numpy as np, json
from collections import defaultdict
from load_spl import load
from psth import unit_tensor, window_slices

rng = np.random.default_rng(8)
sl, T = window_slices()
units = load()
data = defaultdict(list)
for u in units:
    if min(len(c) for c in u['conds']) < 5:
        continue
    m, _ = unit_tensor(u)
    if m.mean() < 1.0:
        continue
    data[u['area']].append(m)


def softnorm(M):
    r = M.reshape(len(M), -1).max(1) - M.reshape(len(M), -1).min(1)
    return M / (r[:, None, None] + 5.0)


def epoch(M, ev, t0, t1):
    s, t = sl[ev]
    return M[:, :, s][:, :, (t >= t0) & (t < t1)]


def align(A, B, k=6):
    Xa = A.reshape(len(A), -1); Xa = Xa - Xa.mean(1, keepdims=True)
    Xb = B.reshape(len(B), -1); Xb = Xb - Xb.mean(1, keepdims=True)
    wa, Va = np.linalg.eigh(Xa @ Xa.T); Da = Va[:, np.argsort(wa)[::-1][:k]]
    Cb = Xb @ Xb.T; wb = np.sort(np.linalg.eigvalsh(Cb))[::-1]
    return np.trace(Da.T @ Cb @ Da) / wb[:k].sum()


N, REP = 150, 300
out = {}
vals = {}
for a in ['V6A', 'PEc', 'PE']:
    M = softnorm(np.array(data[a]))
    P, E, I = epoch(M, 'fix', 600, 1600), epoch(M, 'mout', -50, 350), epoch(M, 'min', -50, 350)
    pe, ei = [], []
    for _ in range(REP):
        k = rng.choice(len(M), N, replace=False)
        pe.append(align(P[k], E[k])); ei.append(align(E[k], I[k]))
    vals[a] = (np.array(pe), np.array(ei))
    out[a] = dict(prep_exec=[round(float(np.mean(pe)), 3), round(float(np.percentile(pe, 2.5)), 3), round(float(np.percentile(pe, 97.5)), 3)],
                  exec_return=[round(float(np.mean(ei)), 3), round(float(np.percentile(ei, 2.5)), 3), round(float(np.percentile(ei, 97.5)), 3)])
    print(a, out[a])
# pairwise: fraction of draws where the ordering is violated (subsampling-based)
for i, (a, b) in enumerate([('V6A', 'PEc'), ('PEc', 'PE'), ('V6A', 'PE')]):
    d1 = vals[a][0][:, None] - vals[b][0][None]; d2 = vals[b][1][:, None] - vals[a][1][None]
    out[f'{a}>{b} prep_exec P(violation)'] = round(float((d1 <= 0).mean()), 4)
    out[f'{b}>{a} exec_return P(violation)'] = round(float((d2 <= 0).mean()), 4)
for k, v in out.items():
    if 'violation' in k:
        print(k, v)
json.dump(out, open('/home/saif/fattori-lab/results/alignment_gradients.json', 'w'), indent=1)
