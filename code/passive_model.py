"""Passive-movement experiment on trained point-mass networks.
The hand is moved along minimum-jerk out and return paths (same timing as the active run); the network's
muscle output is ignored. Per module: condition-specific variance during passive vs active movement
(passive/active ratio) and the out~return pattern correlation under passive movement.
"""
import sys, json
import numpy as np
import torch
import analyze_model as AM
import model as M


def minjerk(s):
    s = np.clip(s, 0, 1); return 10 * s**3 - 15 * s**4 + 6 * s**5


def run_passive(mod, arm, tr, rng, mo, mi, dur_out, dur_in):
    B, T = len(tr['c']), tr['T']
    U = M.build_inputs(tr, rng, 'cpu')
    arm.reset(options={'batch_size': B, 'joint_state': M.HOME_JOINT.repeat(B, 1)})
    tgt = M.targets_xy(arm, tr['c'], 'cpu').numpy()
    t = np.arange(T)[:, None]
    frac = np.where(t < mi[None], minjerk((t - mo[None]) / dur_out[None]), 1 - minjerk((t - mi[None]) / dur_in[None]))
    pos = torch.tensor(frac[..., None] * tgt[None], dtype=torch.float32)  # T x B x 2
    vel = torch.cat([torch.zeros(1, B, 2), torch.diff(pos, dim=0) / M.DT])
    x = torch.zeros(B, mod.N); buf, R = [], []
    with torch.no_grad():
        for k in range(T):
            arm.pos, arm.vel = pos[k], vel[k]; arm._update()
            ms = arm.states['muscle']; p = torch.cat([ms[:, 1, :], ms[:, 2, :]], 1); buf.append(p)
            up = buf[max(0, k - 5)] + 0.01 * torch.randn_like(p)
            x, r, out = mod.step(x, U[k] + 0.02 * torch.randn_like(U[k]), up)
            arm.act = out  # motor output ignored for movement (only stored)
            R.append(r)
    return torch.stack(R).numpy()


res = {}
for p in sys.argv[1:]:
    models, it = AM.load_models(p)
    for name, mod in models:
        _, tr, hand, A, R, arm = AM.simulate(name, mod)
        mo, mi = AM.onsets(hand, tr)
        sp = np.linalg.norm(np.diff(hand, axis=0), axis=-1)
        dur_out = np.array([max(10, np.argmax(sp[mo[b]:, b] < 0.1 * sp[mo[b]:mo[b] + 60, b].max()) if True else 30) for b in range(len(mo))])
        dur_out = np.clip(dur_out, 15, 50); dur_in = dur_out.copy()
        Rp = run_passive(mod, arm, tr, np.random.default_rng(5), mo, mi, dur_out, dur_in)
        f = (lambda X: X) if M.NONLIN == 'relu' else (lambda X: (X + 1) / 2)
        Ra, Rp_ = f(R), f(Rp)
        n = Ra.shape[-1] // 3; conds = tr['c']; row = {}
        for lab, sl in [('M1', slice(0, n)), ('M2', slice(n, 2 * n)), ('M3', slice(2 * n, 3 * n))]:
            a_o = AM.cc(AM.epoch_means(Ra[..., sl], mo, 0, 30, conds)); a_i = AM.cc(AM.epoch_means(Ra[..., sl], mi, 0, 30, conds))
            p_o = AM.cc(AM.epoch_means(Rp_[..., sl], mo, 0, 30, conds)); p_i = AM.cc(AM.epoch_means(Rp_[..., sl], mi, 0, 30, conds))
            row[lab] = dict(active_out_in=round(AM.pcorr(a_o, a_i), 2), passive_out_in=round(AM.pcorr(p_o, p_i), 2),
                            passive_over_active_var_in=round(float(p_i.var() / (a_i.var() + 1e-12)), 2),
                            passive_vs_active_in_r=round(AM.pcorr(p_i, a_i), 2))
        res[f'{p.split("/")[-1][:-3]}|{name}'] = row
        print(p.split('/')[-1][:-3], name, row)
json.dump(res, open('/home/saif/fattori-lab/results/passive_model.json', 'w'), indent=1)
