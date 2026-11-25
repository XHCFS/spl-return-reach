"""Outward vs inward (return) movement: is condition structure inverted (velocity-like),
preserved (position-like), or collapsed (goal-like)? Per area, per monkey, bootstrap over units.
"""
import numpy as np, json
from collections import defaultdict
from load_spl import load
from psth import unit_tensor, window_slices

rng = np.random.default_rng(1)
sl, T = window_slices()
units = load()
AREAS = ['V6A', 'PEc', 'PE']
data = defaultdict(list)
for u in units:
    if min(len(c) for c in u['conds']) < 5:
        continue
    m, tr = unit_tensor(u)
    if m.mean() < 1.0:
        continue
    data[u['area']].append(dict(m=m, tr=tr, monkey=u['monkey']))


def ep(m, ev, t0, t1):
    s, t = sl[ev]
    return m[:, s][:, (t >= t0) & (t < t1)].mean(1)  # 9 conds


EPS = {'late delay': ('fix', 1100, 1600), 'out': ('mout', 0, 300), 'hold': ('redoff', -300, 0), 'in': ('min', 0, 300)}


def patterns(ds, split=None):
    """units x 9 condition-centred, z-scored-by-unit patterns per epoch. split: None | 'a' | 'b' (trial halves)."""
    P = {k: [] for k in EPS}
    for d in ds:
        if split is None:
            m = d['m']
        else:
            m = []
            for trc in d['tr']:
                h = len(trc) // 2
                m.append(trc[:h].mean(0) if split == 'a' else trc[h:].mean(0))
            m = np.array(m)
        sd = m.std() + 1.0
        for k, (ev, t0, t1) in EPS.items():
            x = ep(m, ev, t0, t1)
            P[k].append((x - x.mean()) / sd)
    return {k: np.array(v) for k, v in P.items()}


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def summary(ds, nboot=500):
    P = patterns(ds)
    Pa, Pb = patterns(ds, 'a'), patterns(ds, 'b')
    out = {}
    pairs = [('out', 'in'), ('out', 'hold'), ('hold', 'in'), ('late delay', 'out'), ('late delay', 'in')]
    n = len(ds)
    for a, b in pairs:
        r = corr(P[a], P[b])
        # cross-validated (trial-split) correlation, noise-corrected by within-epoch split-half reliability
        rab = 0.5 * (corr(Pa[a], Pb[b]) + corr(Pb[a], Pa[b]))
        rel_a, rel_b = corr(Pa[a], Pb[a]), corr(Pa[b], Pb[b])
        rc = rab / np.sqrt(max(rel_a, 1e-3) * max(rel_b, 1e-3))
        boots = []
        for _ in range(nboot):
            i = rng.integers(0, n, n)
            boots.append(corr(P[a][i], P[b][i]))
        out[f'{a}~{b}'] = dict(r=round(r, 3), ci=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                              r_cv=round(float(rab), 3), r_cv_corrected=round(float(rc), 3))
    # condition-variance in each epoch (how much target structure survives)
    out['cond_var'] = {k: round(float((P[k] ** 2).mean()), 4) for k in EPS}
    out['reliability'] = {k: round(corr(Pa[k], Pb[k]), 3) for k in EPS}
    return out


res = {}
for a in AREAS:
    res[a] = {'both': summary(data[a])}
    for mk in ['MonkeyF', 'MonkeyS']:
        res[a][mk] = summary([d for d in data[a] if d['monkey'] == mk], nboot=200)
    print('=====', a)
    for k, v in res[a].items():
        print(' ', k, json.dumps(v))
json.dump(res, open('/home/saif/fattori-lab/results/out_in.json', 'w'), indent=1)
