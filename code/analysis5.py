"""Robustness of the outward/return orthogonality.
For each area: trial-split, noise-corrected correlation of condition-centred patterns
  (a) across windows, (b) per monkey, (c) per subarea (V6Ad/V6Av), (d) after partialling out the
  hold-epoch pattern (static hand + gaze at target), (e) Elsayed alignment of cond-specific subspaces.
"""
import numpy as np, json
from collections import defaultdict
from load_spl import load
import h5py, glob, os

rng = np.random.default_rng(5)
units = load()
# subarea labels ('Cito') from the h5 attributes
cito = {}
for fn in glob.glob('/home/saif/fattori-lab/data/spl/data/*.h5'):
    with h5py.File(fn) as f:
        for un in f['DATA']:
            cito[(os.path.basename(fn).split('_')[0], os.path.basename(fn).split('_')[1], un)] = f['DATA'][un].attrs['Cito']


def cnt(sp, a, t0, t1):
    return ((sp >= a + t0) & (sp < a + t1)).sum() * 1000.0 / (t1 - t0)


def build(win_out, win_in, win_hold=(-300, 0)):
    S = defaultdict(list)
    for u in units:
        if min(len(c) for c in u['conds']) < 6:
            continue
        X = np.zeros((2, 3, 9))  # half x epoch(out,in,hold) x cond
        for c, trials in enumerate(u['conds']):
            p = rng.permutation(len(trials)); h = len(trials) // 2
            for hi, idx in enumerate([p[:h], p[h:]]):
                X[hi, 0, c] = np.mean([cnt(trials[i][0], trials[i][1][4], *win_out) for i in idx])
                X[hi, 1, c] = np.mean([cnt(trials[i][0], trials[i][1][7], *win_in) for i in idx])
                X[hi, 2, c] = np.mean([cnt(trials[i][0], trials[i][1][6], *win_hold) for i in idx])
        m = X.mean(0)
        if m.mean() < 1.0:
            continue
        X = X - X.mean(2, keepdims=True)
        X /= (np.sqrt((m - m.mean(1, keepdims=True)).var()) + 1.0)
        S[u['area']].append(dict(X=X, monkey=u['monkey'], cito=cito[(u['monkey'], u['area'], u['name'])]))
    return S


def pc(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


def corr_cv(XA, XB, YA, YB):
    r = 0.5 * (pc(XA, YB) + pc(XB, YA))
    return r / np.sqrt(max(pc(XA, XB), 0.02) * max(pc(YA, YB), 0.02)), r


def resid(P, H):
    """remove, per unit, the component of pattern P (units x 9) explained by hold pattern H (units x 9)."""
    beta = (P * H).sum(1, keepdims=True) / ((H * H).sum(1, keepdims=True) + 1e-9)
    return P - beta * H


def stat(group, nboot=300, partial=False):
    X = np.array([g['X'] for g in group])  # U x 2 x 3 x 9
    def f(Xs):
        O, I, H = Xs[:, :, 0], Xs[:, :, 1], Xs[:, :, 2]
        if partial:  # use the *other* half's hold pattern to avoid fitting noise
            O = np.stack([resid(O[:, 0], H[:, 1]), resid(O[:, 1], H[:, 0])], 1)
            I = np.stack([resid(I[:, 0], H[:, 1]), resid(I[:, 1], H[:, 0])], 1)
        return corr_cv(O[:, 0], O[:, 1], I[:, 0], I[:, 1])[0]
    v = f(X)
    b = [f(X[rng.integers(0, len(X), len(X))]) for _ in range(nboot)]
    return dict(r=round(float(v), 3), ci=[round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)], n=len(X))


if __name__ == '__main__':
    out = {}
    for wname, (wo, wi) in {'0-300': ((0, 300), (0, 300)), '0-200': ((0, 200), (0, 200)), '100-400': ((100, 400), (100, 400)),
                            '-100-200': ((-100, 200), (-100, 200))}.items():
        S = build(wo, wi)
        for a in ['V6A', 'PEc', 'PE']:
            out[(a, wname, 'all')] = stat(S[a])
            if wname == '0-300':
                out[(a, wname, 'partial_hold')] = stat(S[a], partial=True)
                for mk in ['MonkeyF', 'MonkeyS']:
                    out[(a, wname, mk)] = stat([g for g in S[a] if g['monkey'] == mk])
                if a == 'V6A':
                    for ci in sorted(set(g['cito'] for g in S[a])):
                        grp = [g for g in S[a] if g['cito'] == ci]
                        if len(grp) > 30:
                            out[(a, wname, ci)] = stat(grp)
    for k, v in out.items():
        print(k, v)
    json.dump({'|'.join(k): v for k, v in out.items()}, open('/home/saif/fattori-lab/results/robust.json', 'w'), indent=1)
