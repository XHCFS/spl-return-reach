"""Confound checks for the V6Av > V6Ad prep/exec alignment difference:
(1) monkey composition; (2) per-monkey alignment; (3) gaze-shared pattern D~H per subarea (analysis7 stats)."""
import numpy as np, glob, os, h5py
from collections import defaultdict, Counter
import analysis9 as A9
from load_spl import load
from psth import unit_tensor

cito = {}
for fn in glob.glob('/home/saif/fattori-lab/data/spl/data/*.h5'):
    mk, ar = os.path.basename(fn).split('_')[:2]
    with h5py.File(fn) as f:
        for un in f['DATA']:
            cito[(mk, ar, un)] = str(f['DATA'][un].attrs['Cito'])
D = defaultdict(list); EPI = defaultdict(list)
rng = np.random.default_rng(11)
def cnt(sp, a, t0, t1): return ((sp >= a + t0) & (sp < a + t1)).sum() * 1000.0 / (t1 - t0)
EP = {'D': (2, 1100, 1600), 'O': (4, 0, 300), 'H': (6, -300, 0)}
for u in load():
    if min(len(c) for c in u['conds']) < 6:
        continue
    m, tr = unit_tensor(u)
    if m.mean() < 1.0:
        continue
    key = (cito[(u['monkey'], u['area'], u['name'])], u['monkey'])
    a, b = A9.halves(tr); s = (m.max() - m.min()) + 5.0
    D[key].append((a / s, b / s))
    X = np.zeros((2, 3, 9))
    for c, trials in enumerate(u['conds']):
        p = rng.permutation(len(trials)); h = len(trials) // 2
        for hi, idx in enumerate([p[:h], p[h:]]):
            for e, (ev, t0, t1) in enumerate(EP.values()):
                X[hi, e, c] = np.mean([cnt(trials[i][0], trials[i][1][ev], t0, t1) for i in idx])
    mm = X.mean(0); X = X - X.mean(2, keepdims=True); X /= (np.sqrt((mm - mm.mean(1, keepdims=True)).var()) + 1.0)
    EPI[key].append(X)
print('composition', {k: len(v) for k, v in sorted(D.items())})
for key in sorted(D):
    H1 = np.array([x[0] for x in D[key]]); H2 = np.array([x[1] for x in D[key]])
    if len(H1) < 30:
        continue
    bs = [A9.ai(H1[i], H2[i], 'prep', 'exec', 3) for i in [A9.rng.integers(0, len(H1), 40) for _ in range(200)]]
    print(key, 'n', len(H1), 'prep_exec(k=3, N=40 boot) median %.3f [%.3f, %.3f]' % (np.median(bs), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
pc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
def corr_c(X, i, j):
    r = 0.5 * (pc(X[:, 0, i], X[:, 1, j]) + pc(X[:, 1, i], X[:, 0, j]))
    return r / np.sqrt(max(pc(X[:, 0, i], X[:, 1, i]), 0.02) * max(pc(X[:, 0, j], X[:, 1, j]), 0.02))
for sub in ['V6Av', 'V6Ad', 'PEc', 'PE']:
    X = np.concatenate([np.array(EPI[k]) for k in EPI if k[0] == sub])
    print(sub, 'n', len(X), 'D~O %.2f  D~H(gaze-shared) %.2f  O~H %.2f' % (corr_c(X, 0, 1), corr_c(X, 0, 2), corr_c(X, 1, 2)))
