"""Confirmatory statistics for the six headline claims (R1-R4), replacing bootstrap-over-units CIs.

Changes vs the exploratory scripts:
  * every statistic is averaged over K random trial splits (removes split-to-split noise);
  * area differences: unit-label permutation between the two areas (exact null for "no area difference");
  * PE inversion: condition-label permutation of the return pattern (null "return pattern unrelated to outward");
  * one-sided p-values, Holm correction across the 6-test family; per-monkey estimates as secondary.
The six tests and their windows are the ones already used in the exploratory analyses (no new windows).
"""
import numpy as np, json, glob, os, h5py, sys
from multiprocessing import Pool
from scipy.stats import spearmanr
from load_spl import load
from psth import unit_tensor, window_slices

K, NPERM, KDIM = 8, int(os.environ.get('NPERM', 1000)), 3
rng = np.random.default_rng(15)
sl, T = window_slices()
EPOCHS = {'prep': ('fix', 600, 1600), 'exec': ('mout', -50, 350), 'ret': ('min', -50, 350)}
CW = {'out': (4, 0, 300), 'ret': (7, 0, 300), 'early': (7, -50, 100), 'late': (7, 100, 300)}  # marker idx, t0, t1


def cnt(sp, a, t0, t1):
    return ((sp >= a + t0) & (sp < a + t1)).sum() * 1000.0 / (t1 - t0)


cito = {}
for fn in glob.glob('/home/saif/fattori-lab/data/spl/data/*.h5'):
    mk, ar = os.path.basename(fn).split('_')[:2]
    with h5py.File(fn) as f:
        for un in f['DATA']:
            cito[(mk, ar, un)] = str(f['DATA'][un].attrs['Cito'])

meta, P1, P2, C1, C2 = [], [], [], [], []
for u in load():
    if min(len(c) for c in u['conds']) < 6:
        continue
    m, tr = unit_tensor(u)
    if m.mean() < 1.0:
        continue
    s = (m.max() - m.min()) + 5.0
    cn = [np.array([[cnt(sp, mk[e], t0, t1) for e, t0, t1 in CW.values()] for sp, mk in trials]) for trials in u['conds']]
    p1, p2, c1, c2 = [], [], [], []
    for _ in range(K):
        a, b, ca, cb = [], [], [], []
        for c in range(9):
            p = rng.permutation(len(tr[c])); h = len(tr[c]) // 2
            a.append(tr[c][p[:h]].mean(0)); b.append(tr[c][p[h:]].mean(0))
            ca.append(cn[c][p[:h]].mean(0)); cb.append(cn[c][p[h:]].mean(0))
        p1.append(np.array(a) / s); p2.append(np.array(b) / s); c1.append(np.array(ca).T); c2.append(np.array(cb).T)  # W x 9
    cm = 0.5 * (np.array(c1) + np.array(c2)).mean(0)  # W x 9 mean pattern for normalisation (as analysis5)
    sd = np.sqrt((cm - cm.mean(1, keepdims=True)).var()) + 1.0
    nz = lambda X: (X - X.mean(-1, keepdims=True)) / sd
    P1.append(np.array(p1, np.float32)); P2.append(np.array(p2, np.float32)); C1.append(nz(np.array(c1))); C2.append(nz(np.array(c2)))
    meta.append(dict(area=u['area'], monkey=u['monkey'], cito=cito[(u['monkey'], u['area'], u['name'])]))
P1, P2, C1, C2 = map(np.array, (P1, P2, C1, C2))  # U x K x 9 x T ; U x K x W x 9
AREA = np.array([m['area'] for m in meta]); MONK = np.array([m['monkey'] for m in meta]); CITO = np.array([m['cito'] for m in meta])
WI = {k: i for i, k in enumerate(CW)}


def ep(X, name):
    ev, t0, t1 = EPOCHS[name]
    s, t = sl[ev]
    Y = X[..., s][..., (t >= t0) & (t < t1)].reshape(len(X), -1)
    return Y - Y.mean(1, keepdims=True)


def top(C, k):
    w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return V[:, o[:k]], w[o[:k]]


def ai(idx, src, dst):
    v = []
    for k in range(K):
        H1, H2 = P1[idx, k], P2[idx, k]
        S1, S2, D1, D2 = ep(H1, src), ep(H2, src), ep(H1, dst), ep(H2, dst)
        Cs = S1 @ S2.T; Cd = D1 @ D2.T
        Ds, _ = top(0.5 * (Cs + Cs.T), KDIM); Cd = 0.5 * (Cd + Cd.T); _, wd = top(Cd, KDIM)
        v.append(np.trace(Ds.T @ Cd @ Ds) / wd.sum())
    return float(np.mean(v))


def part_rho(idx):  # Spearman of per-unit cross-half variance, exec vs ret
    v = []
    for k in range(K):
        ve = (ep(P1[idx, k], 'exec') * ep(P2[idx, k], 'exec')).mean(1); vr = (ep(P1[idx, k], 'ret') * ep(P2[idx, k], 'ret')).mean(1)
        v.append(spearmanr(ve, vr)[0])
    return float(np.mean(v))


pc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]


def pat(idx, w1, w2, perm=None):  # cross-half pattern correlation (raw), optional condition permutation of w2
    q = slice(None) if perm is None else perm
    v = [0.5 * (pc(C1[idx, k, WI[w1]], C2[idx, k, WI[w2]][:, q]) + pc(C2[idx, k, WI[w1]], C1[idx, k, WI[w2]][:, q])) for k in range(K)]
    return float(np.mean(v))


def rel(idx, w):
    return float(np.mean([pc(C1[idx, k, WI[w]], C2[idx, k, WI[w]]) for k in range(K)]))


# ---- the six tests: (name, group A mask, group B mask, statistic(idx), sign) ; p = P(null stat*sign >= obs*sign)
def tests(mask):
    g = lambda **kw: np.where(mask & np.all([{'area': AREA, 'cito': CITO}[k] == v for k, v in kw.items()], 0))[0]
    return [
        ('T1 prep->exec alignment V6A > PE', g(area='V6A'), g(area='PE'), ('ai', 'prep', 'exec'), +1),
        ('T2 exec->ret alignment PE > V6A', g(area='PE'), g(area='V6A'), ('ai', 'exec', 'ret'), +1),
        ('T3 exec/ret unit reuse (Spearman) PE > V6A', g(area='PE'), g(area='V6A'), ('part_rho',), +1),
        ('T4 PE out~ret pattern < 0', g(area='PE'), None, None, -1),
        ('T5 early-return inversion PE < PEc', g(area='PE'), g(area='PEc'), ('pat', 'out', 'early'), -1),
        ('T6 prep->exec alignment V6Av > V6Ad', g(cito='V6Av'), g(cito='V6Ad'), ('ai', 'prep', 'exec'), +1),
    ]


def F(f, i):
    return {'ai': ai, 'part_rho': part_rho, 'pat': pat}[f[0]](i, *f[1:])


def perm_unit(args):
    a, b, f, seed = args
    r = np.random.default_rng(seed); u = r.permutation(np.concatenate([a, b]))
    return F(f, u[:len(a)]) - F(f, u[len(a):])


def run(mask, label, pool):
    out = {}
    for name, a, b, f, sgn in tests(mask):
        if b is None:  # T4: condition-label permutation
            obs = pat(a, 'out', 'ret')
            null = np.array([pat(a, 'out', 'ret', rng.permutation(9)) for _ in range(NPERM)])
            corr = obs / np.sqrt(max(rel(a, 'out'), .05) * max(rel(a, 'ret'), .05))
            row = dict(n=len(a), stat=round(obs, 3), corrected=round(corr, 2))
        else:
            if min(len(a), len(b)) < 20:
                out[name] = dict(skipped=f'n={len(a)},{len(b)}'); continue
            fa, fb = F(f, a), F(f, b); obs = fa - fb
            null = np.array(pool.map(perm_unit, [(a, b, f, 1000 + i) for i in range(NPERM)]))
            row = dict(nA=len(a), nB=len(b), A=round(fa, 3), B=round(fb, 3), diff=round(obs, 3))
        row['p'] = float((1 + np.sum(null * sgn >= obs * sgn)) / (1 + NPERM))
        out[name] = row
        print(label, name, row, flush=True)
    return out


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); run_max = 0
    for r, i in enumerate(o):
        run_max = max(run_max, (m - r) * ps[i]); adj[i] = min(1, run_max)
    return adj


if __name__ == '__main__':
    assert abs(holm(np.array([0.01, 0.04, 0.03]))[0] - 0.03) < 1e-9  # self-check: smallest p x m
    res = {}
    with Pool(11) as pool:
        res['pooled'] = run(np.ones(len(AREA), bool), 'pooled', pool)
        names = [k for k in res['pooled'] if 'p' in res['pooled'][k]]
        for k, a in zip(names, holm(np.array([res['pooled'][k]['p'] for k in names]))):
            res['pooled'][k]['p_holm'] = round(float(a), 4)
        for mk in sorted(set(MONK)):
            res[mk] = run(MONK == mk, mk, pool)
    print(json.dumps(res['pooled'], indent=1))
    json.dump(res, open('/home/saif/fattori-lab/results/confirmatory.json', 'w'), indent=1)
