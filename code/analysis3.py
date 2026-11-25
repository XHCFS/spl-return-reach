"""Does a fixed kinematic encoding model (hand position + velocity, reconstructed as minimum-jerk
paths from the event times and target coordinates) fitted on the OUTWARD reach predict the
target-specific activity of the INWARD (return) reach? Plus time-resolved out/in pattern correlations.
"""
import numpy as np, json
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from load_spl import load

rng = np.random.default_rng(2)
units = load()
from kin import TGT, BIN, OUT_W, IN_W, LAG, minjerk, trial_kin


def trial_rate(sp, mk, ev, w):
    edges = mk[ev] + np.arange(w[0], w[1] + BIN, BIN)
    return np.histogram(sp, edges)[0] * (1000.0 / BIN)


def unit_epoch(u, ev, w):
    R, K = [], []  # cond-mean rates (9,T), kin (9,T,4)
    Rh = [[], []]
    for c, trials in enumerate(u['conds']):
        rr = np.array([gaussian_filter1d(trial_rate(sp, mk, ev, w).astype(float), 3) for sp, mk in trials])
        kk = np.array([np.concatenate(trial_kin(mk, c, ev, w), 1) for sp, mk in trials])
        R.append(rr.mean(0)); K.append(kk.mean(0))
        h = len(rr) // 2
        Rh[0].append(rr[:h].mean(0)); Rh[1].append(rr[h:].mean(0))
    return np.array(R), np.array(K), np.array(Rh)


def cc(X):  # condition-centre (remove cross-condition mean at each time)
    return X - X.mean(0, keepdims=True)


def fit_predict(Rtr, Ktr, Rte, Kte, lam=1.0):
    """Ridge on condition-centred data: rate ~ [pos, vel]."""
    Xtr = cc(Ktr).reshape(-1, Ktr.shape[-1]); ytr = cc(Rtr).ravel()
    sd = Xtr.std(0) + 1e-9
    Xtr = Xtr / sd
    beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    Xte = cc(Kte).reshape(-1, Kte.shape[-1]) / sd
    return (Xte @ beta).reshape(Rte.shape), (Xtr @ beta).reshape(Rtr.shape)


AREAS = ['V6A', 'PEc', 'PE']
store = defaultdict(list)
for u in units:
    if min(len(c) for c in u['conds']) < 6:
        continue
    Ro, Ko, Rho = unit_epoch(u, 4, OUT_W)
    Ri, Ki, Rhi = unit_epoch(u, 7, IN_W)
    if (Ro.mean() + Ri.mean()) / 2 < 1.0:
        continue
    store[u['area']].append((Ro, Ko, Rho, Ri, Ki, Rhi, u['monkey']))


def r2(y, yhat):
    y, yhat = y.ravel(), yhat.ravel()
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


res = {}
for a in AREAS:
    S = store[a]
    rows = []
    for Ro, Ko, Rho, Ri, Ki, Rhi, mk in S:
        # Fit on half A of outward, evaluate: (i) held-out half B outward (within-context), (ii) inward half B (cross-context)
        pred_in, fit_out = fit_predict(Rho[0], Ko, Rhi[1], Ki)
        pred_out_B, _ = fit_predict(Rho[0], Ko, Rho[1], Ko)
        # and the reverse direction: fit inward, predict outward
        pred_out_from_in, _ = fit_predict(Rhi[0], Ki, Rho[1], Ko)
        pred_in_B, _ = fit_predict(Rhi[0], Ki, Rhi[1], Ki)
        rows.append(dict(
            out_within=r2(cc(Rho[1]), pred_out_B), out_to_in=r2(cc(Rhi[1]), pred_in),
            in_within=r2(cc(Rhi[1]), pred_in_B), in_to_out=r2(cc(Rho[1]), pred_out_from_in),
            # ceiling: split-half predictability of cond-centred data
            ceil_out=r2(cc(Rho[1]), cc(Rho[0])), ceil_in=r2(cc(Rhi[1]), cc(Rhi[0])),
            cond_var_out=float(cc(Ro).var()), cond_var_in=float(cc(Ri).var())))
    # population-level: pool predictions across units (variance-weighted R2)
    agg = {}
    for k in rows[0]:
        agg[k + '_median'] = round(float(np.median([r[k] for r in rows])), 3)
    # population R2 = 1 - sum SSE / sum SST across units, recomputed
    def pop(key_true, which):
        num = den = 0.0
        for (Ro, Ko, Rho, Ri, Ki, Rhi, mk) in S:
            if which == 'out_within':
                y = cc(Rho[1]); yh, _ = fit_predict(Rho[0], Ko, Rho[1], Ko)
            elif which == 'out_to_in':
                y = cc(Rhi[1]); yh, _ = fit_predict(Rho[0], Ko, Rhi[1], Ki)
            elif which == 'in_within':
                y = cc(Rhi[1]); yh, _ = fit_predict(Rhi[0], Ki, Rhi[1], Ki)
            elif which == 'in_to_out':
                y = cc(Rho[1]); yh, _ = fit_predict(Rhi[0], Ki, Rho[1], Ko)
            elif which == 'ceil_out':
                y = cc(Rho[1]); yh = cc(Rho[0])
            elif which == 'ceil_in':
                y = cc(Rhi[1]); yh = cc(Rhi[0])
            num += ((y - yh) ** 2).sum(); den += (y ** 2).sum()
        return round(1 - num / den, 3)
    for w in ['out_within', 'out_to_in', 'in_within', 'in_to_out', 'ceil_out', 'ceil_in']:
        agg['POP_' + w] = pop(None, w)
    res[a] = agg
    print(a, len(S), json.dumps(agg))

# ---- time-resolved cross-correlation of condition patterns (out window x in window) ----
tc = {}
for a in AREAS:
    S = store[a]
    PoA = np.concatenate([cc(s[2][0])[None] for s in S])  # units x 9 x T (half A out)
    PiB = np.concatenate([cc(s[5][1])[None] for s in S])
    PoB = np.concatenate([cc(s[2][1])[None] for s in S])
    PiA = np.concatenate([cc(s[5][0])[None] for s in S])
    sd = np.array([np.sqrt(0.5 * (cc(s[0]).var() + cc(s[3]).var())) + 0.5 for s in S])[:, None, None]
    PoA, PiB, PoB, PiA = PoA / sd, PiB / sd, PoB / sd, PiA / sd
    Tn = PoA.shape[2]
    M = np.zeros((Tn, Tn))
    for i in range(0, Tn):
        for j in range(0, Tn):
            M[i, j] = 0.5 * (np.corrcoef(PoA[:, :, i].ravel(), PiB[:, :, j].ravel())[0, 1] + np.corrcoef(PoB[:, :, i].ravel(), PiA[:, :, j].ravel())[0, 1])
    tc[a] = M
    t = np.arange(OUT_W[0], OUT_W[1], BIN) + 5
    pick = [np.argmin(abs(t - x)) for x in (-100, 0, 100, 200, 300, 400)]
    print(a, 'rows: out t=', [int(t[p]) for p in pick], ' cols: in t= same')
    print(np.round(M[np.ix_(pick, pick)], 2))
np.save('/home/saif/fattori-lab/results/out_in_timecorr.npy', tc, allow_pickle=True)
json.dump(res, open('/home/saif/fattori-lab/results/kin_crosspred.json', 'w'), indent=1)
