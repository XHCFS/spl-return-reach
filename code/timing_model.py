"""Timing of the out/return inversion relative to return-movement onset, per module.
r(t) = corr(condition-centred outward pattern [0,300) ms after move-out onset, return pattern at t), t in 20 ms steps.
Reports t of the minimum and the first t where r < 0.5*min (onset of inversion)."""
import sys, json
import numpy as np
import analyze_model as AM
import model as M

ts = np.arange(-300, 320, 20)
res = {}
for p in sys.argv[1:]:
    models, it = AM.load_models(p)
    for name, mod in models:
        _, tr, hand, A, R, arm = AM.simulate(name, mod)
        rates = R if M.NONLIN == 'relu' else (R + 1) / 2
        mo, mi = AM.onsets(hand, tr); conds = tr['c']
        n = rates.shape[-1] // 3
        vel = np.concatenate([np.zeros((1,) + hand.shape[1:]), np.diff(hand, axis=0) / M.DT])
        row = {}
        for lab, X in [('M1', rates[..., :n]), ('M2', rates[..., n:2 * n]), ('M3', rates[..., 2 * n:]), ('hand_vel', vel)]:
            Po = AM.cc(AM.epoch_means(X, mo, 0, 30, conds))
            rr = []
            for t in ts:
                k = t // 10
                Pi = AM.cc(AM.epoch_means(X, mi, k, k + 2, conds))
                rr.append(np.corrcoef(Po.ravel(), Pi.ravel())[0, 1])
            rr = np.array(rr); imin = int(np.argmin(rr))
            on = next((int(ts[i]) for i in range(len(ts)) if rr[i] < 0.5 * rr[imin]), None) if rr[imin] < -0.2 else None
            row[lab] = dict(t_min=int(ts[imin]), r_min=round(float(rr[imin]), 2), t_onset=on)
        res[f'{p.split("/")[-1][:-3]}|{name}'] = row
        print(p.split('/')[-1][:-3], name, row)
json.dump(res, open(sys.argv[1].replace('.pt', '_timing.json'), 'w'), indent=1)
