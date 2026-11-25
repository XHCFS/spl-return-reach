"""Apply the data metrics to trained models, per module.
Metrics: out~in condition-pattern correlation (movement 0-300 ms after each onset), late-delay~out,
hold~in, Elsayed alignment (prep -> exec), and the same for muscles / hand velocity as references.
"""
import sys, glob, json, os
import numpy as np
import torch
import model as M

DEV = 'cpu'


def load_models(path):
    ck = torch.load(path, map_location=DEV)
    st = M.Stack(ck['specs']); st.load_state_dict(ck['state'])
    return [(f"{v}_s{s}", st.unstack(k)) for k, (v, s) in enumerate(ck['specs'])], ck['it']


def simulate(name, mod, n_rep=10, seed=123, noise_scale=1.0):
    arm = M.make_arm(DEV)
    mod.noise *= noise_scale
    rng = np.random.default_rng(seed)
    conds = np.repeat(np.arange(9), n_rep)
    tr = M.sample_trials(len(conds), rng, conds=conds, delay=80, hold=60)
    with torch.no_grad():
        hand, A, R = M.run(mod, arm, tr, rng, DEV, record=True)
    return name, tr, hand.numpy(), A.numpy(), R.numpy(), arm


def onsets(hand, tr):
    sp = np.linalg.norm(np.diff(hand, axis=0), axis=-1) / M.DT  # T-1 x B
    B = hand.shape[1]
    mo, mi = np.zeros(B, int), np.zeros(B, int)
    for b in range(B):
        g, r = tr['t_go'][b], tr['t_red'][b]
        s1 = sp[g:r, b]; s2 = sp[r:, b]
        mo[b] = g + np.argmax(s1 > 0.1 * s1.max())
        mi[b] = r + np.argmax(s2 > 0.1 * s2.max())
    return mo, mi


def epoch_means(X, align, t0, t1, conds):
    """X: T x B x N ; returns 9 x N mean over [align+t0, align+t1) bins (10 ms), averaged over trials of a cond."""
    out = []
    for c in range(9):
        idx = np.where(conds == c)[0]
        out.append(np.mean([X[align[b] + t0: align[b] + t1, b].mean(0) for b in idx], 0))
    return np.array(out)


def epoch_traj(X, align, t0, t1, conds):
    out = []
    for c in range(9):
        idx = np.where(conds == c)[0]
        out.append(np.mean([X[align[b] + t0: align[b] + t1, b] for b in idx], 0))
    return np.array(out)  # 9 x T x N


def cc(P):
    return P - P.mean(0, keepdims=True)


def pcorr(a, b):
    return float(np.corrcoef(cc(a).ravel(), cc(b).ravel())[0, 1])


def alignment(Xp, Xe, k=6):
    Xp = Xp.reshape(-1, Xp.shape[-1]).T; Xe = Xe.reshape(-1, Xe.shape[-1]).T
    Xp = Xp - Xp.mean(1, keepdims=True); Xe = Xe - Xe.mean(1, keepdims=True)
    wp, Vp = np.linalg.eigh(Xp @ Xp.T); Dp = Vp[:, np.argsort(wp)[::-1][:k]]
    Ce = Xe @ Xe.T; we = np.sort(np.linalg.eigvalsh(Ce))[::-1]
    return float(np.trace(Dp.T @ Ce @ Dp) / we[:k].sum())


def analyze(name, mod):
    name, tr, hand, A, R, arm = simulate(name, mod)
    conds = tr['c']
    mo, mi = onsets(hand, tr)
    tgt = M.targets_xy(arm, conds, DEV).numpy()
    end_err = np.linalg.norm(hand[tr['t_red'] - 5, np.arange(len(conds))] - tgt, axis=1).mean() * 100
    home = M.home_xy(arm).numpy()[0]
    ret_err = np.linalg.norm(hand[-5] - home, axis=1).mean() * 100
    rates = R if M.NONLIN == 'relu' else (R + 1) / 2  # tanh -> [0,1] "rate"
    vel = np.concatenate([np.zeros((1,) + hand.shape[1:]), np.diff(hand, axis=0) / M.DT])
    res = dict(name=name, endpoint_err_cm=round(float(end_err), 2), return_err_cm=round(float(ret_err), 2),
               rt_out_ms=float(np.mean(mo - tr['t_go']) * 10), rt_in_ms=float(np.mean(mi - tr['t_red']) * 10))
    n = rates.shape[-1] // 3
    mods = {'M1(V6A-like)': rates[..., :n], 'M2(PEc-like)': rates[..., n:2 * n], 'M3(PE-like)': rates[..., 2 * n:]}
    refs = {'muscles': A, 'hand_vel': vel, 'hand_pos': hand}
    for k, X in list(mods.items()) + list(refs.items()):
        P_out = epoch_means(X, mo, 0, 30, conds); P_in = epoch_means(X, mi, 0, 30, conds)
        P_del = epoch_means(X, tr['t_go'], -50, 0, conds); P_hold = epoch_means(X, tr['t_red'], -30, 0, conds)
        d = dict(out_in=round(pcorr(P_out, P_in), 3), delay_out=round(pcorr(P_del, P_out), 3),
                 hold_in=round(pcorr(P_hold, P_in), 3), delay_in=round(pcorr(P_del, P_in), 3),
                 out_hold=round(pcorr(P_out, P_hold), 3),
                 condvar_out=round(float(cc(P_out).var()), 5), condvar_in=round(float(cc(P_in).var()), 5))
        if k in mods:
            Tp = epoch_traj(X, tr['t_go'], -50, 0, conds); Te = epoch_traj(X, mo, -5, 35, conds)
            d['align_prep_exec'] = round(alignment(Tp, Te), 3)
            Ti = epoch_traj(X, mi, -5, 35, conds)
            d['align_exec_in'] = round(alignment(Te, Ti), 3)
            d['align_prep_in'] = round(alignment(Tp, Ti), 3)
            d['mean_sq_rate'] = round(float((R[..., :] ** 2).mean()), 4)
        res[k] = d
    return res


if __name__ == '__main__':
    torch.set_num_threads(2)
    models, it = load_models(sys.argv[1])
    print('checkpoint iteration', it)
    allres = []
    for name, mod in models:
        r = analyze(name, mod); allres.append(r)
        print('\n=====', r['name'], {k: r[k] for k in ['endpoint_err_cm', 'return_err_cm', 'rt_out_ms', 'rt_in_ms']})
        for k, v in r.items():
            if isinstance(v, dict):
                print(f'  {k:14s}', v)
    json.dump(allres, open(sys.argv[1].replace('.pt', f'_metrics_it{it}.json'), 'w'), indent=1)
