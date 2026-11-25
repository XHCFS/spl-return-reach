"""Coarse (100 ms) time-resolved, cross-validated, noise-corrected comparison of target-specific
population patterns between outward and inward reaches, plus kinematic-model transfer.
Statistic: r_cv(X_A, Y_B) / sqrt(rel_X * rel_Y), trial halves A/B, with bootstrap over units.
"""
import numpy as np, json
from collections import defaultdict
from load_spl import load
from kin import TGT, trial_kin, minjerk

rng = np.random.default_rng(3)
units = load()
BIN = 100
W = (-200, 600)
NB = (W[1] - W[0]) // BIN
AREAS = ['V6A', 'PEc', 'PE']


def rates(sp, mk, ev):
    e = mk[ev] + np.arange(W[0], W[1] + BIN, BIN)
    return np.histogram(sp, e)[0] * (1000.0 / BIN)


def kin(mk, c, ev):
    # mean pos/vel within each 100 ms bin (sampled at 10 ms then averaged), shifted by 100 ms lag
    import kin as K
    old = K.BIN
    p, v = trial_kin(mk, c, ev, W)
    return np.concatenate([p, v], 1).reshape(NB, BIN // old, 4).mean(1)


store = defaultdict(list)
for u in units:
    if min(len(c) for c in u['conds']) < 6:
        continue
    O = [[], []]; I = [[], []]; KO = []; KI = []
    for c, trials in enumerate(u['conds']):
        ro = np.array([rates(sp, mk, 4) for sp, mk in trials]); ri = np.array([rates(sp, mk, 7) for sp, mk in trials])
        p = rng.permutation(len(trials)); h = len(trials) // 2
        O[0].append(ro[p[:h]].mean(0)); O[1].append(ro[p[h:]].mean(0))
        I[0].append(ri[p[:h]].mean(0)); I[1].append(ri[p[h:]].mean(0))
        KO.append(np.mean([kin(mk, c, 4) for sp, mk in trials], 0)); KI.append(np.mean([kin(mk, c, 7) for sp, mk in trials], 0))
    O, I = np.array(O), np.array(I)  # 2 x 9 x NB
    if (O.mean() + I.mean()) / 2 < 1.0:
        continue
    sd = np.sqrt(0.5 * (O.mean(0).var() + I.mean(0).var())) + 1.0
    store[u['area']].append(dict(O=O / sd, I=I / sd, KO=np.array(KO), KI=np.array(KI), monkey=u['monkey']))


def cc(X):
    return X - X.mean(-2, keepdims=True)  # centre across conditions (axis of 9)


def pc(a, b):
    a, b = a.ravel(), b.ravel()
    return np.corrcoef(a, b)[0, 1]


def corrected(XA, XB, YA, YB):
    r = 0.5 * (pc(XA, YB) + pc(XB, YA))
    rx, ry = pc(XA, XB), pc(YA, YB)
    return r, r / np.sqrt(max(rx, 0.02) * max(ry, 0.02)), rx, ry


res = {}
for a in AREAS:
    S = store[a]
    O = np.array([s['O'] for s in S]); I = np.array([s['I'] for s in S])  # U x 2 x 9 x NB
    O, I = cc(O), cc(I)
    # time-resolved matrix (out bin i vs in bin j)
    M = np.zeros((NB, NB)); Mraw = np.zeros((NB, NB))
    for i in range(NB):
        for j in range(NB):
            r, rc, _, _ = corrected(O[:, 0, :, i], O[:, 1, :, i], I[:, 0, :, j], I[:, 1, :, j])
            M[i, j] = rc; Mraw[i, j] = r
    # same-epoch reliability per bin (sanity)
    relO = [pc(O[:, 0, :, i], O[:, 1, :, i]) for i in range(NB)]
    relI = [pc(I[:, 0, :, j], I[:, 1, :, j]) for j in range(NB)]
    # movement-epoch summary: out bins 0..300 ms vs in bins 0..300 ms (bins 2,3,4)
    mv = slice(2, 5)
    r, rc, rx, ry = corrected(O[:, 0, :, mv], O[:, 1, :, mv], I[:, 0, :, mv], I[:, 1, :, mv])
    # hand-at-target bins: out late (400-600 => bins 6,7) vs in pre-movement (-200..0 => bins 0,1)
    r2_, rc2, _, _ = corrected(O[:, 0, :, 6:8], O[:, 1, :, 6:8], I[:, 0, :, 0:2], I[:, 1, :, 0:2])
    boots = []
    for _ in range(300):
        k = rng.integers(0, len(S), len(S))
        boots.append(corrected(O[k, 0, :, mv], O[k, 1, :, mv], I[k, 0, :, mv], I[k, 1, :, mv])[1])
    res[a] = dict(n=len(S), move_out_vs_move_in=dict(r_cv=round(r, 3), r_corrected=round(rc, 3), ci=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)], rel_out=round(rx, 3), rel_in=round(ry, 3)),
                  static_hand_at_target=dict(r_cv=round(r2_, 3), r_corrected=round(rc2, 3)),
                  relO=np.round(relO, 2).tolist(), relI=np.round(relI, 2).tolist(), M=np.round(M, 2).tolist())
    print('=====', a, 'n', len(S))
    print(' movement out vs in:', res[a]['move_out_vs_move_in'])
    print(' late-out (hand at target) vs pre-in (hand at target):', res[a]['static_hand_at_target'])
    print(' reliability out bins', res[a]['relO'], '\n reliability in bins ', res[a]['relI'])
    print(' corrected corr matrix rows=out bin start ms', list(range(W[0], W[1], BIN)), 'cols=in bins')
    print(np.round(M, 2))

# ---- kinematic transfer at population level ----
print('\n==== kinematic (pos+vel, lag 100ms) model transfer, cond-centred, cross-validated correlations')
kin_res = {}
for a in AREAS:
    S = store[a]
    pred = {'out->out': [], 'out->in': [], 'in->in': [], 'in->out': []}
    true = {k: [] for k in pred}
    for s in S:
        KO = cc(s['KO']).reshape(-1, 4); KI = cc(s['KI']).reshape(-1, 4)
        sdk = np.vstack([KO, KI]).std(0) + 1e-9
        KO, KI = KO / sdk, KI / sdk
        OA, OB = cc(s['O'][0]).ravel(), cc(s['O'][1]).ravel()
        IA, IB = cc(s['I'][0]).ravel(), cc(s['I'][1]).ravel()
        def fit(X, y, lam=5.0):
            return np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ y)
        bo, bi = fit(KO, OA), fit(KI, IA)
        pred['out->out'].append(KO @ bo); true['out->out'].append(OB)
        pred['out->in'].append(KI @ bo); true['out->in'].append(IB)
        pred['in->in'].append(KI @ bi); true['in->in'].append(IB)
        pred['in->out'].append(KO @ bi); true['in->out'].append(OB)
    kin_res[a] = {k: round(pc(np.concatenate(pred[k]), np.concatenate(true[k])), 3) for k in pred}
    print(a, kin_res[a])
json.dump(dict(geometry=res, kin=kin_res), open('/home/saif/fattori-lab/results/out_in_v2.json', 'w'), indent=1)
