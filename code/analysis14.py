"""Is the return reach prepared in caudal SPL? Condition patterns in: HOLD [-300,0) before Red off,
RT = [Red off+50, Move-in-on) (return reaction time, hand still on target), RET = [0,300) after Move-in-on.
Noise-corrected trial-split correlations HOLD~RT, RT~RET, HOLD~RET; bootstrap over units."""
import numpy as np
from collections import defaultdict
from load_spl import load
rng = np.random.default_rng(14)
def cnt(sp, t0, t1): return ((sp >= t0) & (sp < t1)).sum() * 1000.0 / max(t1 - t0, 1)
S = defaultdict(list)
for u in load():
    if min(len(c) for c in u['conds']) < 6:
        continue
    X = np.zeros((2, 3, 9))
    for c, trials in enumerate(u['conds']):
        p = rng.permutation(len(trials)); h = len(trials) // 2
        for hi, idx in enumerate([p[:h], p[h:]]):
            for i in idx:
                sp, mk = trials[i]
                X[hi, 0, c] += cnt(sp, mk[6] - 300, mk[6]) / len(idx)
                X[hi, 1, c] += cnt(sp, mk[6] + 50, mk[7]) / len(idx)
                X[hi, 2, c] += cnt(sp, mk[7], mk[7] + 300) / len(idx)
    m = X.mean(0)
    if m.mean() < 1.0:
        continue
    X = X - X.mean(2, keepdims=True); X /= (np.sqrt((m - m.mean(1, keepdims=True)).var()) + 1.0)
    S[u['area']].append(X)
pc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
def cr(X, i, j):
    r = 0.5 * (pc(X[:, 0, i], X[:, 1, j]) + pc(X[:, 1, i], X[:, 0, j]))
    return r / np.sqrt(max(pc(X[:, 0, i], X[:, 1, i]), 0.05) * max(pc(X[:, 0, j], X[:, 1, j]), 0.05))
for a in ['V6A', 'PEc', 'PE']:
    X = np.array(S[a]); n = len(X)
    out = {}
    for nm, (i, j) in {'HOLD~RT': (0, 1), 'RT~RET': (1, 2), 'HOLD~RET': (0, 2)}.items():
        b = [cr(X[rng.integers(0, n, n)], i, j) for _ in range(300)]
        out[nm] = '%+.2f [%+.2f,%+.2f]' % (cr(X, i, j), np.percentile(b, 2.5), np.percentile(b, 97.5))
    rel = [round(pc(X[:, 0, k], X[:, 1, k]), 2) for k in range(3)]
    print(a, n, out, 'reliability HOLD/RT/RET', rel)
