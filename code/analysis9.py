"""Noise-unbiased alignment index. For epochs A (source subspace) and B (target variance):
  C_B = sym(X_B^half1 X_B^half2^T) -- cross-half covariance, noise cancels in expectation
  D_A = top-k eigvecs of sym(X_A^1 X_A^2^T)
  AI = Tr(D_A^T C_B D_A) / sum top-k eigvals(C_B)
Bootstrap over units with replacement (N = 150 per area) -> CI; area differences tested on bootstrap draws.
"""
import numpy as np, json, sys
from collections import defaultdict
from load_spl import load
from psth import unit_tensor, window_slices

rng = np.random.default_rng(9)
sl, T = window_slices()
EPOCHS = {'prep': ('fix', 600, 1600), 'exec': ('mout', -50, 350), 'ret': ('min', -50, 350)}


def halves(trs):
    A, B = [], []
    for tr in trs:  # per condition: n_tr x T
        p = rng.permutation(len(tr)); h = len(tr) // 2
        A.append(tr[p[:h]].mean(0)); B.append(tr[p[h:]].mean(0))
    return np.array(A), np.array(B)


def get_data():
    units = load()
    D = defaultdict(list)
    for u in units:
        if min(len(c) for c in u['conds']) < 6:
            continue
        m, tr = unit_tensor(u)
        if m.mean() < 1.0:
            continue
        a, b = halves(tr)
        s = (m.max() - m.min()) + 5.0  # soft normalisation from the full mean
        D[u['area']].append((a / s, b / s))
    return D


def ep(X, name):
    ev, t0, t1 = EPOCHS[name]
    s, t = sl[ev]
    Y = X[..., s][..., (t >= t0) & (t < t1)]
    Y = Y.reshape(len(Y), -1)
    return Y - Y.mean(1, keepdims=True)


def xcov(X1, X2):
    C = X1 @ X2.T
    return 0.5 * (C + C.T)


def top(C, k):
    w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return V[:, o[:k]], w[o[:k]]


def ai(H1, H2, src, dst, k=6):
    Ds, _ = top(xcov(ep(H1, src), ep(H2, src)), k)
    Cd = xcov(ep(H1, dst), ep(H2, dst))
    _, wd = top(Cd, k)
    return float(np.trace(Ds.T @ Cd @ Ds) / wd.sum())


def run(D, N=150, B=300, k=6):
    res, draws = {}, {}
    for a in ['V6A', 'PEc', 'PE']:
        H1 = np.array([x[0] for x in D[a]]); H2 = np.array([x[1] for x in D[a]])
        full = dict(prep_exec=ai(H1, H2, 'prep', 'exec', k), exec_ret=ai(H1, H2, 'exec', 'ret', k), prep_ret=ai(H1, H2, 'prep', 'ret', k))
        bs = defaultdict(list)
        for _ in range(B):
            i = rng.integers(0, len(H1), N)
            for key, (s, d) in {'prep_exec': ('prep', 'exec'), 'exec_ret': ('exec', 'ret'), 'prep_ret': ('prep', 'ret')}.items():
                bs[key].append(ai(H1[i], H2[i], s, d, k))
        draws[a] = {kk: np.array(v) for kk, v in bs.items()}
        res[a] = {kk: [round(full[kk], 3), round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for kk, v in bs.items()}
        print(a, len(H1), res[a])
    for x, y in [('V6A', 'PEc'), ('PEc', 'PE'), ('V6A', 'PE')]:
        for kk in ['prep_exec', 'exec_ret', 'prep_ret']:
            d = draws[x][kk] - draws[y][kk]
            res[f'{x}-{y} {kk}'] = [round(float(d.mean()), 3), round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
            print(f'  {x}-{y} {kk}: diff {res[f"{x}-{y} {kk}"]}')
    return res


if __name__ == '__main__':
    D = get_data()
    out = {k_: run(D, k=k_) for k_ in [3, 6]}
    json.dump(out, open('/home/saif/fattori-lab/results/alignment_unbiased.json', 'w'), indent=1)
