"""Direction (left/right) vs depth (near/far) components of outward vs return target coding.
Conditions: 1-3 near (L,C,R), 4-6 intermediate, 7-9 far. Per unit contrasts from trial halves.
"""
import numpy as np, json
from analysis5 import build, rng

DIR = np.array([-1, 0, 1] * 3, float)          # right - left
DEP = np.repeat([-1, 0, 1], 3).astype(float)  # far - near


def contrast(P, w):  # P: units x 9 (cond-centred) -> units
    return P @ (w / np.abs(w).sum())


def corr_cv(a0, a1, b0, b1):
    pc = lambda x, y: np.corrcoef(x, y)[0, 1]
    r = 0.5 * (pc(a0, b1) + pc(a1, b0))
    return r / np.sqrt(max(pc(a0, a1), 0.02) * max(pc(b0, b1), 0.02)), pc(a0, a1), pc(b0, b1)


res = {}
for wname, wo in {'0-300': (0, 300), '-100-200': (-100, 200)}.items():
    S = build(wo, wo)
    for a in ['V6A', 'PEc', 'PE']:
        X = np.array([g['X'] for g in S[a]])  # U x half x (out,in,hold) x 9
        for cname, w in [('direction', DIR), ('depth', DEP)]:
            C = np.einsum('uhec,c->uhe', X, w / np.abs(w).sum())
            def f(C):
                r_oi = corr_cv(C[:, 0, 0], C[:, 1, 0], C[:, 0, 1], C[:, 1, 1])
                r_oh = corr_cv(C[:, 0, 0], C[:, 1, 0], C[:, 0, 2], C[:, 1, 2])
                r_hi = corr_cv(C[:, 0, 2], C[:, 1, 2], C[:, 0, 1], C[:, 1, 1])
                return r_oi, r_oh, r_hi
            (roi, relo, reli), (roh, _, relh), (rhi, _, _) = f(C)
            boots = []
            for _ in range(500):
                k = rng.integers(0, len(C), len(C)); boots.append(f(C[k])[0][0])
            res[f'{a}|{wname}|{cname}'] = dict(out_in=round(roi, 3), ci=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                                               out_hold=round(roh, 3), hold_in=round(rhi, 3), rel_out=round(relo, 3), rel_in=round(reli, 3), rel_hold=round(relh, 3))
            print(a, wname, cname, res[f'{a}|{wname}|{cname}'])
json.dump(res, open('/home/saif/fattori-lab/results/dir_depth.json', 'w'), indent=1)
