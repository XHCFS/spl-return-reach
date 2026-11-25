"""Shared-gaze hypothesis for the prep/exec alignment gradient.
Epoch patterns (condition-centred, trial-split, noise-corrected):
  D = late delay (gaze on target, hand at home)   O = outward movement (gaze on target, hand moving)
  H = hold (gaze + hand on target)                 F = FREE epoch before target onset (control: no target)
D~H isolates what delay and hold share = gaze/target location without hand position.
Then: does removing the gaze-shared pattern (D~H axis per unit) erase the V6A>PEc>PE alignment gradient?
"""
import numpy as np, json
from collections import defaultdict
from load_spl import load

rng = np.random.default_rng(7)
units = load()
EP = {'F': (1, -600, -100), 'D': (2, 1100, 1600), 'O': (4, 0, 300), 'H': (6, -300, 0), 'I': (7, 0, 300)}


def cnt(sp, a, t0, t1):
    return ((sp >= a + t0) & (sp < a + t1)).sum() * 1000.0 / (t1 - t0)


S = defaultdict(list)
for u in units:
    if min(len(c) for c in u['conds']) < 6:
        continue
    X = np.zeros((2, len(EP), 9))
    for c, trials in enumerate(u['conds']):
        p = rng.permutation(len(trials)); h = len(trials) // 2
        for hi, idx in enumerate([p[:h], p[h:]]):
            for e, (ev, t0, t1) in enumerate(EP.values()):
                X[hi, e, c] = np.mean([cnt(trials[i][0], trials[i][1][ev], t0, t1) for i in idx])
    m = X.mean(0)
    if m.mean() < 1.0:
        continue
    X = X - X.mean(2, keepdims=True)
    X /= (np.sqrt((m - m.mean(1, keepdims=True)).var()) + 1.0)
    S[u['area']].append(X)

pc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
names = list(EP)


def cmat(X):
    n = len(names); M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = 0.5 * (pc(X[:, 0, i], X[:, 1, j]) + pc(X[:, 1, i], X[:, 0, j]))
            M[i, j] = r / np.sqrt(max(pc(X[:, 0, i], X[:, 1, i]), 0.02) * max(pc(X[:, 0, j], X[:, 1, j]), 0.02))
    return M


res = {}
for a in ['V6A', 'PEc', 'PE']:
    X = np.array(S[a])
    M = cmat(X)
    boots = np.array([cmat(X[rng.integers(0, len(X), len(X))]) for _ in range(200)])
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    rel = [round(pc(X[:, 0, i], X[:, 1, i]), 3) for i in range(len(names))]
    res[a] = dict(M=np.round(M, 3).tolist(), lo=np.round(lo, 3).tolist(), hi=np.round(hi, 3).tolist(), rel=rel, n=len(X))
    print('=====', a, 'n', len(X), 'split-half reliability', dict(zip(names, rel)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f'  {names[i]}~{names[j]}: {M[i,j]:+.2f} [{lo[i,j]:+.2f},{hi[i,j]:+.2f}]')
json.dump(res, open('/home/saif/fattori-lab/results/epoch_sharing.json', 'w'), indent=1)
