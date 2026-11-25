"""How is epoch orthogonality achieved: different units (recruitment) or the same units (rotation)?
Per unit, cross-half (noise-unbiased) variance in each epoch (prep, exec, return).
  overlap(a,b) = sum_i min(p_a(i), p_b(i)), with p_e the normalised participation (1 = same units, 0 = disjoint)
  spearman(a,b) of per-unit variance across units
Data vs trained networks (point-mass checkpoints), same definitions.
"""
import numpy as np, json, sys, glob, os
from scipy.stats import spearmanr

EPS_PAIRS = [('prep', 'exec'), ('exec', 'ret'), ('prep', 'ret')]


def overlap(va, vb):
    va = np.clip(va, 0, None); vb = np.clip(vb, 0, None)
    pa, pb = va / va.sum(), vb / vb.sum()
    return float(np.minimum(pa, pb).sum())


def null_overlap(va, vb, rng, n=200):  # same marginal distributions, random unit assignment
    return float(np.mean([overlap(va, rng.permutation(vb)) for _ in range(n)]))


def summarise(V, rng):  # V: dict epoch -> per-unit variance
    out = {}
    for a, b in EPS_PAIRS:
        out[f'{a}-{b}'] = dict(overlap=round(overlap(V[a], V[b]), 3), null=round(null_overlap(V[a], V[b], rng), 3),
                               spearman=round(float(spearmanr(V[a], V[b])[0]), 3))
    return out


def data_part():
    import analysis9 as A9
    D = A9.get_data()
    rng = np.random.default_rng(12)
    res = {}
    for a in ['V6A', 'PEc', 'PE']:
        H1 = np.array([x[0] for x in D[a]]); H2 = np.array([x[1] for x in D[a]])
        V = {e: (A9.ep(H1, e) * A9.ep(H2, e)).mean(1) for e in ['prep', 'exec', 'ret']}
        res[a] = summarise(V, rng)
        print('DATA', a, res[a])
    return res


def model_part(paths):
    import torch
    import analyze_model as AM
    import model as M
    rng = np.random.default_rng(13)
    res = {}
    for p in paths:
        models, it = AM.load_models(p)
        for name, mod in models:
            _, tr, hand, A, R, arm = AM.simulate(name, mod)
            rates = R if M.NONLIN == 'relu' else (R + 1) / 2
            mo, mi = AM.onsets(hand, tr)
            conds = tr['c']
            n = rates.shape[-1] // 3
            row = {}
            for mi_, sl in enumerate([slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)]):
                X = rates[..., sl]
                Tp = AM.epoch_traj(X, tr['t_go'], -50, 0, conds); Te = AM.epoch_traj(X, mo, -5, 35, conds); Ti = AM.epoch_traj(X, mi, -5, 35, conds)
                V = {e: T_.reshape(-1, T_.shape[-1]).var(0) for e, T_ in [('prep', Tp), ('exec', Te), ('ret', Ti)]}
                row[f'M{mi_+1}'] = summarise(V, rng)
            res[f'{os.path.basename(p)[:-3]}|{name}'] = row
            print('MODEL', os.path.basename(p)[:-3], name, {k: {kk: vv['overlap'] for kk, vv in v.items()} for k, v in row.items()})
    return res


if __name__ == '__main__':
    if sys.argv[1] == 'data':
        json.dump(data_part(), open('/home/saif/fattori-lab/results/participation_data.json', 'w'), indent=1)
    else:
        r = model_part(sys.argv[2:])
        json.dump(r, open(f'/home/saif/fattori-lab/results/participation_{sys.argv[1]}.json', 'w'), indent=1)
