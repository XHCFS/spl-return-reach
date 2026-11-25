"""Phase 0: reproduce T1 (planning/execution subspace alignment) and T4 (target decoding order),
plus a first look at outward vs inward (return) movement geometry.
"""
import sys, json
import numpy as np
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from load_spl import load
from psth import unit_tensor, window_slices

rng = np.random.default_rng(0)
units = load()
sl, T = window_slices()
AREAS = ['V6A', 'PEc', 'PE']

# ---------- build per-unit data ----------
MIN_TR = 5
data = defaultdict(list)  # area -> list of (means 9xT, trials list)
for u in units:
    nt = [len(c) for c in u['conds']]
    if len(nt) != 9 or min(nt) < MIN_TR:
        continue
    m, tr = unit_tensor(u)
    if m.mean() < 1.0:
        continue
    data[u['area']].append((m, tr, u['monkey']))
for a in AREAS:
    print(a, 'units kept', len(data[a]), {mk: sum(d[2] == mk for d in data[a]) for mk in 'MonkeyF MonkeyS'.split()})


def softnorm(M):  # M: units x cond x time
    rngs = M.reshape(len(M), -1).max(1) - M.reshape(len(M), -1).min(1)
    return M / (rngs[:, None, None] + 5.0)


def epoch(M, ev, t0, t1):
    s, t = sl[ev]
    idx = np.where((t >= t0) & (t < t1))[0]
    return M[:, :, s][:, :, idx]


def pcs(X, k):  # X: units x samples (centered)
    C = X @ X.T / X.shape[1]
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return V[:, o[:k]], w[o], C


def alignment(Mp, Me, k=6, nrand=1000):
    """Elsayed et al. 2016 alignment index: variance of exec data captured by prep PCs / max possible."""
    Xp = Mp.reshape(len(Mp), -1); Xp = Xp - Xp.mean(1, keepdims=True)
    Xe = Me.reshape(len(Me), -1); Xe = Xe - Xe.mean(1, keepdims=True)
    Dp, _, _ = pcs(Xp, k)
    De, we, Ce = pcs(Xe, k)
    ai = np.trace(Dp.T @ Ce @ Dp) / we[:k].sum()
    ai_rev = None
    # random baseline: subspaces drawn from the combined covariance (Elsayed's method)
    Ct = np.cov(np.concatenate([Xp, Xe], 1))
    wt, Vt = np.linalg.eigh(Ct); wt = np.clip(wt, 0, None)
    A = Vt * np.sqrt(wt)
    rnd = []
    for _ in range(nrand):
        v = A @ rng.standard_normal((len(A), k))
        q, _ = np.linalg.qr(v)
        rnd.append(np.trace(q.T @ Ce @ q) / we[:k].sum())
    return ai, np.array(rnd)


res = {}
for a in AREAS:
    M = softnorm(np.array([d[0] for d in data[a]]))
    Mp = epoch(M, 'fix', 600, 1600)      # late delay (planning)
    Me = epoch(M, 'mout', -50, 350)     # outward movement (execution)
    Mi = epoch(M, 'min', -50, 350)      # inward (return) movement
    Mh = epoch(M, 'redoff', -300, 0)    # hold
    out = {}
    for name, (A_, B_) in {'prep->exec': (Mp, Me), 'exec->prep': (Me, Mp), 'exec->inward': (Me, Mi), 'prep->inward': (Mp, Mi)}.items():
        ai, rnd = alignment(A_, B_)
        out[name] = (round(float(ai), 3), round(float(np.mean(rnd)), 3), round(float(np.mean(rnd >= ai)), 4))
    # condition-specific version: subtract cross-condition mean at each time first
    cs = lambda X: X - X.mean(1, keepdims=True)
    ai, rnd = alignment(cs(Mp), cs(Me))
    out['prep->exec (cond-specific)'] = (round(float(ai), 3), round(float(np.mean(rnd)), 3), round(float(np.mean(rnd >= ai)), 4))
    res[a] = out
    print(a, json.dumps(out))

# ---------- T4: pseudo-population decoding ----------
def pseudo_trials(trs, s_idx, n_pseudo, rng):
    """trs: list over units of list over cond of (n_tr, T). returns X (cond*n_pseudo, units) for train/test halves."""
    Xtr, Xte, ytr, yte = [], [], [], []
    for c in range(9):
        cols_tr, cols_te = [], []
        for tr in trs:
            x = tr[c][:, s_idx].mean(1)
            p = rng.permutation(len(x)); h = len(x) // 2
            a_, b_ = x[p[:h]], x[p[h:]]
            cols_tr.append(rng.choice(a_, n_pseudo)); cols_te.append(rng.choice(b_, n_pseudo))
        Xtr.append(np.array(cols_tr).T); Xte.append(np.array(cols_te).T)
        ytr += [c] * n_pseudo; yte += [c] * n_pseudo
    return np.vstack(Xtr), np.vstack(Xte), np.array(ytr), np.array(yte)


EPOCHS = {'early delay': ('fix', 100, 600), 'late delay': ('fix', 1100, 1600), 'move out': ('mout', 0, 300),
          'hold': ('redoff', -300, 0), 'move in': ('min', 0, 300)}
N_UNITS, N_REP = 100, 30
dec = {}
for a in AREAS:
    dec[a] = {}
    for en, (ev, t0, t1) in EPOCHS.items():
        s, t = sl[ev]; idx = np.arange(s.start, s.stop)[(t >= t0) & (t < t1)]
        accs = []
        for r in range(N_REP):
            pick = rng.choice(len(data[a]), N_UNITS, replace=False)
            trs = [data[a][i][1] for i in pick]
            Xtr, Xte, ytr, yte = pseudo_trials(trs, idx, 20, rng)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            clf = LDA(solver='lsqr', shrinkage='auto').fit((Xtr - mu) / sd, ytr)
            accs.append((clf.predict((Xte - mu) / sd) == yte).mean())
        dec[a][en] = round(float(np.mean(accs)), 3)
    print('decode', a, dec[a])

json.dump(dict(alignment=res, decoding=dec), open('/home/saif/fattori-lab/results/phase0.json', 'w'), indent=1)
