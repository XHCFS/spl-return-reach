"""External reference: does a known proprioceptive area (S1 area 2, NLB Area2_Bump, Chowdhury et al. 2020)
show the PE signature -- an inverted (velocity-like) outward/return target code that appears at return onset --
and is that code shared with *passive* movement in the return direction?

Same statistics as the SPL analyses (analysis5/13): condition-centred patterns, cross-half (trial-split)
correlation, K random splits. Conditions = 8 target directions.
  out  : [0,300) ms after outward movement onset (rewarded trials)
  ret  : [0,300) ms after return onset (hand leaves the outer target back to centre, detected from speed)
  bump : [0,150) ms after a passive centre-hold bump; bump direction phi is matched to return direction
         theta+180 (the return from target theta moves the hand toward theta+180)
Time course: r(out, window [t,t+100) around return onset) for neurons and for hand velocity.
"""
import numpy as np, h5py, json
rng = np.random.default_rng(16)
K = 20
f = h5py.File('/home/saif/fattori-lab/data/nlb/area2_bump.nwb', 'r')
T = {k: f['intervals/trials'][k][:] for k in ['result', 'target_dir', 'move_onset_time', 'stop_time', 'start_time', 'ctr_hold_bump', 'bump_time', 'bump_dir']}
hp = f['processing/behavior/hand_pos/data'][:]; hv = f['processing/behavior/hand_vel/data'][:]  # 1 kHz, from t=0
st = f['units/spike_times'][:]; ix = f['units/spike_times_index'][:]
spikes = np.split(st, ix[:-1])
U = len(spikes)
speed = np.linalg.norm(hv, axis=1)
DIRS = np.arange(0, 360, 45)

# ---- events
out_ev, ret_ev, bump_ev = [], [], []
for i in range(len(T['result'])):
    if T['result'][i] == b'R' and not np.isnan(T['move_onset_time'][i]) and not np.isnan(T['target_dir'][i]):
        c = int(T['target_dir'][i] // 45)
        out_ev.append((T['move_onset_time'][i], c))
        a = int(T['stop_time'][i] * 1000)
        if a + 2000 > len(hp) or np.linalg.norm(hp[a]) < 7:  # hand must be out at the target at reward
            continue
        dist = np.linalg.norm(hp[a:a + 2000], axis=1)
        pk = int(np.argmax(dist[:800]))  # turnaround: farthest point after reward (task has no hold at target)
        vr = np.gradient(dist) * 1000  # radial velocity, cm/s
        seg = vr[pk:pk + 1000]
        if seg.min() > -10 or dist[pk:pk + 1000].min() > 4:  # must actually come back toward the centre
            continue
        on = a + pk + int(np.argmax(seg < 0.2 * seg.min()))  # inward radial speed > 20% of its peak
        ret_ev.append((on / 1000.0, c))
    if T['ctr_hold_bump'][i] and not np.isnan(T['bump_time'][i]):
        bump_ev.append((T['bump_time'][i], int(T['bump_dir'][i] // 45)))
print('events: out', len(out_ev), 'ret', len(ret_ev), 'bump', len(bump_ev))


def rates(evs, t0, t1):  # trials x units
    e = np.array([x for x, _ in evs])
    return np.stack([np.searchsorted(s, e + t1 / 1000) - np.searchsorted(s, e + t0 / 1000) for s in spikes], 1) * 1000.0 / (t1 - t0)


def kin(evs, t0, t1):  # trials x 2 mean hand velocity
    return np.array([np.nanmean(hv[int(e * 1000) + t0:int(e * 1000) + t1], 0) for e, _ in evs])


def kinpv(evs, t0, t1):  # trials x 5: mean hand position, velocity, 1
    return np.array([np.r_[np.nanmean(hp[int(e * 1000) + t0:int(e * 1000) + t1], 0), np.nanmean(hv[int(e * 1000) + t0:int(e * 1000) + t1], 0), 1] for e, _ in evs])


def halves(X, conds, ncond=8):  # -> K x 2 x U x ncond (condition-centred)
    out = np.zeros((K, 2, X.shape[1], ncond))
    for k in range(K):
        for c in range(ncond):
            idx = rng.permutation(np.where(conds == c)[0]); h = len(idx) // 2
            out[k, 0, :, c] = X[idx[:h]].mean(0); out[k, 1, :, c] = X[idx[h:]].mean(0)
    return out - out.mean(-1, keepdims=True)


pc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]


def xr(A, B, shift=0):  # cross-half pattern correlation, B's conditions rolled by `shift` (in 45-deg steps)
    B = np.roll(B, shift, axis=-1)
    raw = np.mean([0.5 * (pc(A[k, 0], B[k, 1]) + pc(A[k, 1], B[k, 0])) for k in range(K)])
    rel = lambda X: np.mean([pc(X[k, 0], X[k, 1]) for k in range(K)])
    return float(raw), float(raw / np.sqrt(max(rel(A), .05) * max(rel(B), .05)))


def norm(X):  # soft per-unit normalisation, as in the SPL analyses
    return X / (X.std(0) + 1.0)


co = np.array([c for _, c in out_ev]); cr = np.array([c for _, c in ret_ev]); cb = np.array([c for _, c in bump_ev])
Xo = rates(out_ev, 0, 300); Xr = rates(ret_ev, 0, 300); Xb = rates(bump_ev, 0, 150)
sd = np.concatenate([Xo, Xr, Xb]).std(0) + 1.0
O, R, Bp = halves(Xo / sd, co), halves(Xr / sd, cr), halves(Xb / sd, cb)
# bump in direction phi ~ return from target theta when phi = theta + 180 -> roll bump patterns by 4 steps
res = dict(n_units=U, n=dict(out=len(out_ev), ret=len(ret_ev), bump=len(bump_ev)),
           out_ret=xr(O, R), out_bump_same_dir=xr(O, Bp), ret_bump_matched_dir=xr(R, Bp, shift=-4), ret_bump_same_target_dir=xr(R, Bp))
Ko, Kr = halves(kin(out_ev, 0, 300), co), halves(kin(ret_ev, 0, 300), cr)
res['hand_vel_out_ret'] = xr(Ko, Kr)
for k, v in res.items():
    print(k, v)

# ---- null for out_ret: condition-label permutation of the return pattern
null = np.array([0.5 * np.mean([pc(O[k, 0], R[k, 1][:, p]) + pc(O[k, 1], R[k, 0][:, p]) for k in range(4)]) for p in [rng.permutation(8) for _ in range(2000)]])
res['out_ret_p_perm'] = float((1 + np.sum(null <= res['out_ret'][0])) / 2001)
print('out_ret permutation p', res['out_ret_p_perm'])

# ---- time course around return onset (100 ms windows, 50 ms steps), neurons and hand velocity
# kinematic encoding model (position + velocity, linear) fitted on outward-phase 100 ms windows only
fitw = range(-100, 500, 50)
Xf = np.concatenate([kinpv(out_ev, t, t + 100) for t in fitw]); Yf = np.concatenate([rates(out_ev, t, t + 100) / sd for t in fitw])
ok = ~np.isnan(Xf).any(1); Wk = np.linalg.lstsq(Xf[ok], Yf[ok], rcond=None)[0]
pred = lambda evs, t0, t1: np.nan_to_num(kinpv(evs, t0, t1)) @ Wk
Op = halves(pred(out_ev, 0, 300), co)
TS = np.arange(-300, 301, 50); tc = dict(t=TS.tolist(), neural=[], neural_corr=[], kin_model=[], hand_vel=[], neural_bump=[])
for t in TS:
    Rt = halves(rates(ret_ev, t, t + 100) / sd, cr); Kt = halves(kin(ret_ev, t, t + 100), cr)
    r_, rc_ = xr(O, Rt); tc['neural'].append(round(r_, 3)); tc['neural_corr'].append(round(rc_, 2)); tc['hand_vel'].append(round(xr(Ko, Kt)[0], 3))
    tc['kin_model'].append(round(xr(Op, halves(pred(ret_ev, t, t + 100), cr))[0], 3))
for t in np.arange(-100, 201, 25):  # passive: outward pattern vs bump response, relative to bump onset
    Bt = halves(rates(bump_ev, t, t + 50) / sd, cb); tc['neural_bump'].append((int(t), round(xr(O, Bt)[0], 3)))
res['timecourse'] = tc
for i, t in enumerate(TS):
    print(f'  ret [{t:+4d},{t + 100:+4d})  neural r {tc["neural"][i]:+.3f} (corr {tc["neural_corr"][i]:+.2f})  kin-model r {tc["kin_model"][i]:+.3f}  hand-vel r {tc["hand_vel"][i]:+.3f}')
print('  bump-aligned (out~bump, 50 ms windows):', tc['neural_bump'])
json.dump(res, open('/home/saif/fattori-lab/results/area2_return.json', 'w'), indent=1)
