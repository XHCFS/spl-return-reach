"""Per-monkey replication of the PE-leads-PEc inversion timing (analysis13)."""
import numpy as np
from collections import defaultdict
import analysis13 as A  # builds S per area on import (pooled); rebuild per monkey here
from load_spl import load
rng = np.random.default_rng(131)
S = defaultdict(list)
for u in load():
    if min(len(c) for c in u['conds']) < 6:
        continue
    X = np.zeros((2, 1 + len(A.TS), 9))
    for c, trials in enumerate(u['conds']):
        p = rng.permutation(len(trials)); h = len(trials) // 2
        for hi, idx in enumerate([p[:h], p[h:]]):
            X[hi, 0, c] = np.mean([A.cnt(trials[i][0], trials[i][1][4], 0, 300) for i in idx])
            for j, t in enumerate(A.TS):
                X[hi, 1 + j, c] = np.mean([A.cnt(trials[i][0], trials[i][1][7], t, t + A.W) for i in idx])
    m = X.mean(0)
    if m.mean() < 1.0:
        continue
    X = X - X.mean(2, keepdims=True); X /= (np.sqrt((m - m.mean(1, keepdims=True)).var()) + 1.0)
    S[(u['area'], u['monkey'])].append(X)
for key in sorted(S):
    if key[0] == 'V6A':
        continue
    X = np.array(S[key]); raw, _ = A.curve(X)
    B = np.array([A.curve(X[rng.integers(0, len(X), len(X))])[0] for _ in range(300)])
    hi = np.percentile(B, 97.5, 0)
    sig = [int(t) for t, h in zip(A.TS, hi) if h < 0]
    print(key, 'n', len(X), 'min r_raw %.3f at t=%d' % (raw.min(), A.TS[np.argmin(raw)]), ' windows with CI<0 (start ms):', sig)
