"""Efference test: when does the inverted (velocity-like) target code appear around return onset?
r(t): trial-split correlation between the condition-centred outward pattern (0-300 ms after Move-out-on)
and the return pattern in [t, t+W) relative to Move-in-on; normalised by split-half reliabilities
(reliability of the short window is floored, so we also report raw r_cv). Bootstrap over units.
"""
import numpy as np, json
from collections import defaultdict
from load_spl import load

rng = np.random.default_rng(13)
W = 100
TS = np.arange(-300, 301, 50)


def cnt(sp, a, t0, t1):
    return ((sp >= a + t0) & (sp < a + t1)).sum() * 1000.0 / (t1 - t0)


S = defaultdict(list)
for u in load():
    if min(len(c) for c in u['conds']) < 6:
        continue
    X = np.zeros((2, 1 + len(TS), 9))
    for c, trials in enumerate(u['conds']):
        p = rng.permutation(len(trials)); h = len(trials) // 2
        for hi, idx in enumerate([p[:h], p[h:]]):
            X[hi, 0, c] = np.mean([cnt(trials[i][0], trials[i][1][4], 0, 300) for i in idx])
            for j, t in enumerate(TS):
                X[hi, 1 + j, c] = np.mean([cnt(trials[i][0], trials[i][1][7], t, t + W) for i in idx])
    m = X.mean(0)
    if m.mean() < 1.0:
        continue
    X = X - X.mean(2, keepdims=True)
    X /= (np.sqrt((m - m.mean(1, keepdims=True)).var()) + 1.0)
    S[u['area']].append(X)

pc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]


def curve(X):
    rel_o = pc(X[:, 0, 0], X[:, 1, 0])
    raw, cor = [], []
    for j in range(len(TS)):
        r = 0.5 * (pc(X[:, 0, 0], X[:, 1, 1 + j]) + pc(X[:, 1, 0], X[:, 0, 1 + j]))
        rel = pc(X[:, 0, 1 + j], X[:, 1, 1 + j])
        raw.append(r); cor.append(r / np.sqrt(max(rel_o, 0.05) * max(rel, 0.05)))
    return np.array(raw), np.array(cor)


res = {}
for a in ['V6A', 'PEc', 'PE']:
    X = np.array(S[a])
    raw, cor = curve(X)
    B = np.array([curve(X[rng.integers(0, len(X), len(X))])[0] for _ in range(500)])
    lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
    res[a] = dict(t=TS.tolist(), r_raw=np.round(raw, 3).tolist(), lo=np.round(lo, 3).tolist(), hi=np.round(hi, 3).tolist(), r_corr=np.round(cor, 2).tolist())
    print('=====', a, 'n', len(X))
    for j, t in enumerate(TS):
        flag = '  <-- CI<0' if hi[j] < 0 else ''
        print(f'  [{t:+4d},{t+W:+4d}) ms  r_raw {raw[j]:+.3f} [{lo[j]:+.3f},{hi[j]:+.3f}]  r_corr {cor[j]:+.2f}{flag}')
json.dump(res, open('/home/saif/fattori-lab/results/efference_timing.json', 'w'), indent=1)
