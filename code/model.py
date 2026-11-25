"""Task-trained modular RNN for the Bologna 9-target out-and-back reaching task (Diomedi et al. 2024).

Module 1 (V6A-like) gets target/gaze (version, vergence) + task cues; module 3 (PE-like) gets delayed
proprioception and drives the 6 muscles of MotorNet's planar arm26; module 2 (PEc-like) sits between.
Variants change where inputs enter (see VARIANTS).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import motornet as mn

DT = 0.01
TGT_CM = np.array([[-2.5, 9.3], [0, 10], [2.5, 9.3], [-3.56, 13.2], [0, 15], [3.56, 13.2], [-6.2, 23], [0, 25], [6.2, 23]])
EYE_Y, EYE_Z, IOD = -8.0, 12.5, 3.0  # cm: eyes behind the home button, at target height; interocular distance
# targets are at eye height (z = 12.5 above HB), so only horizontal version/vergence vary


def eye_coords(xy_cm):
    """version (deg) and vergence (deg) of fixating a point at height EYE_Z in the horizontal plane."""
    x, y = xy_cm[..., 0], xy_cm[..., 1] - EYE_Y
    le = np.arctan2(x + IOD / 2, y); re = np.arctan2(x - IOD / 2, y)
    return np.degrees((le + re) / 2), np.degrees(le - re)


VV = np.stack(eye_coords(TGT_CM), -1)  # 9 x 2
VV_MU, VV_SD = VV.mean(0), VV.std(0)

NONLIN = os.environ.get('NONLIN', 'tanh')
PHASE_ALL = os.environ.get('PHASE_ALL', '0') == '1'  # task-phase cue channels (green, red, free) to every module
f_rate = torch.relu if NONLIN == 'relu' else torch.tanh
PROP_MODE = os.environ.get('PROP_MODE', 'both')  # 'both' (length + velocity) or 'vel'

VARIANTS = {
    # name: (vision->modules, proprio->modules, output module, inter-module density)
    'hier': ([0], [2], 2, 0.1),
    'all_inputs': ([0, 1, 2], [0, 1, 2], 2, 0.1),
    'swap': ([2], [0], 2, 0.1),
    'single': ([0], [0], 0, 1.0),  # one module; modules are just thirds of a dense net
    'out_first': ([0], [2], 0, 0.1),  # vision->M1, proprio->M3, but muscles read from M1 (reversed output)
    'vis_prop_m1': ([0], [0], 2, 0.1),  # both inputs at M1, output at M3 (pure distance-from-output test)
    'vp12': ([0, 1], [0, 1], 2, 0.1),  # sensory inputs to M1 and M2 (V6A, bimodal PEc); M3 (PE) input-free, drives output
}


class ModRNN(nn.Module):
    def __init__(self, n_vis, n_prop, n_out, variant, n_per=100, alpha=0.2, noise=0.05, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        vis_m, prop_m, out_m, dens = VARIANTS[variant]
        self.N, self.n_per, self.alpha, self.noise = 3 * n_per, n_per, alpha, noise
        N = self.N
        mod = np.repeat(np.arange(3), n_per)
        mask = np.zeros((N, N), np.float32)
        for i in range(3):
            for j in range(3):
                if i == j:
                    mask[np.ix_(mod == i, mod == j)] = 1
                elif abs(i - j) == 1:
                    mask[np.ix_(mod == i, mod == j)] = (torch.rand(n_per, n_per, generator=g).numpy() < dens)
        if variant == 'single':
            mask[:] = 1
        self.register_buffer('mask', torch.tensor(mask))
        if NONLIN == 'relu':  # fan-in scaled, sub-critical gain so non-saturating units don't explode
            fan = torch.tensor(mask.sum(1, keepdims=True)).clamp(min=1)
            self.W = nn.Parameter(torch.randn(N, N, generator=g) * 0.8 / fan.sqrt())
        else:
            self.W = nn.Parameter(torch.randn(N, N, generator=g) * 1.2 / math.sqrt(n_per))
        vmask = np.zeros((N, n_vis), np.float32); pmask = np.zeros((N, n_prop), np.float32)
        for m in vis_m: vmask[mod == m] = 1
        if PHASE_ALL:
            vmask[:, 2:5] = 1
        for m in prop_m: pmask[mod == m] = 1
        self.register_buffer('vmask', torch.tensor(vmask)); self.register_buffer('pmask', torch.tensor(pmask))
        self.Wv = nn.Parameter(torch.randn(N, n_vis, generator=g) / math.sqrt(n_vis))
        self.Wp = nn.Parameter(torch.randn(N, n_prop, generator=g) / math.sqrt(n_prop))
        self.b = nn.Parameter(torch.zeros(N))
        omask = np.zeros((n_out, N), np.float32); omask[:, mod == out_m] = 1
        self.register_buffer('omask', torch.tensor(omask))
        self.Wo = nn.Parameter(torch.randn(n_out, N, generator=g) * 0.1 / math.sqrt(n_per))
        self.bo = nn.Parameter(torch.full((n_out,), -2.0))

    def step(self, x, uv, up):
        r = f_rate(x)
        inp = r @ (self.W * self.mask).T + uv @ (self.Wv * self.vmask).T + up @ (self.Wp * self.pmask).T + self.b
        x = (1 - self.alpha) * x + self.alpha * (inp + self.noise * torch.randn_like(x))
        r = f_rate(x)
        out = torch.sigmoid(r @ (self.Wo * self.omask).T + self.bo)
        return x, r, out


class Stack(nn.Module):
    """K independent ModRNNs evaluated in one batched pass (weights stacked on a leading K axis)."""
    def __init__(self, specs, n_vis=5, n_prop=12, n_out=6):
        super().__init__()
        self.specs = specs
        n_per, noise = int(os.environ.get('N_PER', 100)), float(os.environ.get('NOISE', 0.05))
        ms = [ModRNN(n_vis, n_prop, n_out, v, seed=s, n_per=n_per, noise=noise) for v, s in specs]
        self.N, self.alpha, self.noise = ms[0].N, ms[0].alpha, ms[0].noise
        for k in ['W', 'Wv', 'Wp', 'b', 'Wo', 'bo']:
            setattr(self, k, nn.Parameter(torch.stack([getattr(m, k).data for m in ms])))
        for k in ['mask', 'vmask', 'pmask', 'omask']:
            self.register_buffer(k, torch.stack([getattr(m, k) for m in ms]))

    def step(self, x, uv, up):  # x: K,B,N ; uv: B,5 (shared) ; up: K,B,12
        r = f_rate(x)
        inp = (torch.einsum('kbn,kmn->kbm', r, self.W * self.mask) + torch.einsum('bi,kni->kbn', uv, self.Wv * self.vmask)
               + torch.einsum('kbi,kni->kbn', up, self.Wp * self.pmask) + self.b[:, None])
        x = (1 - self.alpha) * x + self.alpha * (inp + self.noise * torch.randn_like(x))
        r = f_rate(x)
        out = torch.sigmoid(torch.einsum('kbn,kon->kbo', r, self.Wo * self.omask) + self.bo[:, None])
        return x, r, out

    def unstack(self, k):
        v, s = self.specs[k]
        m = ModRNN(self.Wv.shape[2], self.Wp.shape[2], self.Wo.shape[1], v, seed=s, n_per=self.N // 3, noise=self.noise)
        for n in ['W', 'Wv', 'Wp', 'b', 'Wo', 'bo']:
            getattr(m, n).data.copy_(getattr(self, n).data[k].cpu())
        for n in ['mask', 'vmask', 'pmask', 'omask']:
            getattr(m, n).copy_(getattr(self, n)[k].cpu())
        return m


def run_stack(model, arm, tr, rng, device, prop_delay=5):
    K, B, T = len(model.specs), len(tr['c']), tr['T']
    U = build_inputs(tr, rng, device)
    arm.reset(options={'batch_size': K * B, 'joint_state': HOME_JOINT.repeat(K * B, 1).to(device)})
    x = torch.zeros(K, B, model.N, device=device)
    buf, hand, A = [], [], []
    act = 0.0
    for t in range(T):
        ms = arm.states['muscle']
        p = torch.cat([ms[:, 1, :], ms[:, 2, :]], 1).view(K, B, -1)
        buf.append(p)
        up = buf[max(0, t - prop_delay)] + 0.01 * torch.randn_like(p)
        if PROP_MODE == 'vel':  # spindle-like: velocity channels only (lengths zeroed)
            up = torch.cat([torch.zeros_like(up[..., :6]), up[..., 6:]], -1)
        uv = U[t] + 0.02 * torch.randn_like(U[t])
        x, r, out = model.step(x, uv, up)
        act = act + (r ** 2).mean(dim=(1, 2)) / T
        arm.step(out.reshape(K * B, -1))
        hand.append(arm.states['cartesian'][:, :2].view(K, B, 2))
        A.append(out)
    run_stack.last_act = act  # K: mean squared rate (for activity regularisation)
    return torch.stack(hand, 1), torch.stack(A, 1)  # K,T,B,2 ; K,T,B,6


def train_stack(specs, iters, device, out_dir, B=256, tag='stack', lr=1e-3):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    arm = make_arm(device)
    model = Stack(specs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr / 20)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); log = []
    for it in range(iters):
        tr = sample_trials(B, rng)
        hand, A = run_stack(model, arm, tr, rng, device)
        pos_k = torch.stack([loss_fn(hand[k], A[k], None, tr, arm, device)[1] for k in range(len(specs))])
        loss = pos_k.sum() + 1e-3 * (A ** 2).mean() * len(specs) + float(os.environ.get('ACT_REG', 0)) * run_stack.last_act.sum()
        opt.zero_grad(); loss.backward()
        for p in model.parameters():  # per-model gradient clipping
            if p.grad is not None:
                n = p.grad.flatten(1).norm(dim=1).clamp(min=1.0)
                p.grad /= n.view(-1, *([1] * (p.grad.dim() - 1)))
        opt.step(); sched.step()
        if it % 25 == 0 or it == iters - 1:
            errs = [round(float(e.detach()) * 100, 2) for e in pos_k]
            log.append((it, errs))
            print(f'it {it} pos-err cm {errs} ({time.time()-t0:.0f}s)', flush=True)
        if it % 100 == 99 or it == iters - 1:
            torch.save(dict(specs=specs, state=model.state_dict(), it=it), f'{out_dir}/{tag}.pt')
            json.dump(log, open(f'{out_dir}/{tag}_log.json', 'w'))
    return model


class PointMass:
    """Lean effector: 2-D point mass pulled by 6 non-negative actuators (directions every 60 deg) with
    first-order activation dynamics. Mimics the bits of the motornet effector interface that run() uses."""
    n_muscles = 6

    def __init__(self, device, m=1.0, damping=10.0, fmax=20.0, tau=0.05):
        ang = torch.arange(6, dtype=torch.float32) * math.pi / 3
        self.D = torch.stack([torch.cos(ang), torch.sin(ang)], 1).to(device)  # 6 x 2
        self.device, self.m, self.b, self.fmax, self.tau = torch.device(device), m, damping, fmax, tau
        self.states = {}

    def joint2cartesian(self, joint):
        return torch.zeros(joint.shape[0], 4, device=self.device)

    def reset(self, options):
        B = options['batch_size']
        self.pos = torch.zeros(B, 2, device=self.device); self.vel = torch.zeros(B, 2, device=self.device)
        self.act = torch.zeros(B, 6, device=self.device)
        self._update()

    def _update(self):
        L = -(self.pos @ self.D.T) / 0.1; V = -(self.vel @ self.D.T) / 0.5  # actuator "length"/"velocity" proxies
        ms = torch.stack([self.act, L, V] + [self.act] * 4, 1)  # B x 7 x 6 ; rows 1,2 used as proprioception
        self.states = {'muscle': ms, 'cartesian': torch.cat([self.pos, self.vel], 1)}

    def step(self, u):
        self.act = self.act + DT * (u - self.act) / self.tau
        F = (self.fmax * self.act) @ self.D - self.b * self.vel
        self.vel = self.vel + DT * F / self.m
        self.pos = self.pos + DT * self.vel
        self._update()


def make_arm(device):
    if os.environ.get('EFFECTOR', 'motornet') == 'pointmass':
        return PointMass(device)
    arm = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle(), timestep=DT)
    arm = arm.to(device); arm.muscle.to(device); arm.skeleton.to(device)  # motornet's .to() doesn't propagate
    return arm


HOME_JOINT = torch.tensor([[1.0, 2.2, 0.0, 0.0]])  # shoulder, elbow (rad), velocities; far targets at ~81% of max reach


def home_xy(arm):
    return arm.joint2cartesian(HOME_JOINT.to(arm.device))[:, :2]


def sample_trials(B, rng, conds=None, delay=None, hold=None):
    c = rng.integers(0, 9, B) if conds is None else np.asarray(conds)
    t_fix = 30
    d = rng.integers(40, 81, B) if delay is None else np.full(B, delay)
    h = rng.integers(30, 61, B) if hold is None else np.full(B, hold)
    t_go = t_fix + d
    t_red = t_go + 60 + h          # 600 ms to reach, then hold
    T = int((t_red + 70).max())
    return dict(c=c, t_fix=t_fix, t_go=t_go, t_red=t_red, T=T)


def build_inputs(tr, rng, device):
    """vis input per time: [version, vergence, target_on(green), red, free_gaze_flag]; gaze = target in fixation."""
    B, T = len(tr['c']), tr['T']
    U = np.zeros((T, B, 5), np.float32)
    tvv = (VV[tr['c']] - VV_MU) / VV_SD
    for b in range(B):
        f, g, r = tr['t_fix'], tr['t_go'][b], tr['t_red'][b]
        # free gaze: random walk of fixations before fixation onset and after red off (dark, no target)
        free = (rng.standard_normal((T, 2)) * 0.0)
        jumps = np.cumsum(rng.random(T) < 0.03)
        gz = rng.standard_normal((jumps.max() + 1, 2))[jumps] * 1.0
        U[:, b, 0:2] = gz
        U[f:r, b, 0:2] = tvv[b]
        U[f:g, b, 2] = 1.0          # green
        U[g:r, b, 3] = 1.0          # red (go)
        U[:f, b, 4] = 1.0; U[r:, b, 4] = 1.0
    return torch.tensor(U, device=device)


def run(model, arm, tr, rng, device, record=False, prop_delay=5):
    B, T = len(tr['c']), tr['T']
    U = build_inputs(tr, rng, device)
    arm.reset(options={'batch_size': B, 'joint_state': HOME_JOINT.repeat(B, 1).to(device)})
    x = torch.zeros(B, model.N, device=device)
    prop_buf = []
    hand, R, A = [], [], []
    for t in range(T):
        ms = arm.states['muscle']  # B x 7 x 6 ; rows 1,2 = length, velocity (normalised in motornet)
        p = torch.cat([ms[:, 1, :], ms[:, 2, :]], 1)
        prop_buf.append(p)
        up = prop_buf[max(0, t - prop_delay)] + 0.01 * torch.randn_like(p)
        if PROP_MODE == 'vel':
            up = torch.cat([torch.zeros_like(up[..., :6]), up[..., 6:]], -1)
        uv = U[t] + 0.02 * torch.randn_like(U[t])
        x, r, out = model.step(x, uv, up)
        arm.step(out)
        hand.append(arm.states['cartesian'][:, :2])
        A.append(out)
        if record:
            R.append(r.detach())
    hand = torch.stack(hand)  # T x B x 2
    A = torch.stack(A)
    return hand, A, (torch.stack(R) if record else None)


def targets_xy(arm, c, device):
    h = home_xy(arm)
    off = torch.tensor(TGT_CM[c] / 100.0, dtype=torch.float32, device=device)
    return h + off  # x lateral, y forward (arm26 frame: +y forward of shoulder)


def loss_fn(hand, A, R_act, tr, arm, device):
    T, B = hand.shape[:2]
    home = home_xy(arm)[0]
    tgt = targets_xy(arm, tr['c'], device)
    t = torch.arange(T, device=device)[:, None]
    tg = torch.tensor(tr['t_go'], device=device)[None]; trd = torch.tensor(tr['t_red'], device=device)[None]
    w_home_pre = (t < tg + 15).float()
    w_tgt = ((t >= tg + 45) & (t < trd + 15)).float()
    w_home_post = (t >= trd + 50).float()
    d_home = (hand - home).abs().sum(-1)
    d_tgt = (hand - tgt[None]).abs().sum(-1)
    pos = (w_home_pre * d_home + w_tgt * d_tgt + w_home_post * d_home).sum() / (w_home_pre + w_tgt + w_home_post).sum()
    effort = (A ** 2).mean()
    return pos + 1e-3 * effort, pos


def train(variant, seed, iters, device, out_dir, B=256):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    arm = make_arm(device)
    model = ModRNN(5, 12, arm.n_muscles, variant, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = []
    t0 = time.time()
    for it in range(iters):
        tr = sample_trials(B, rng)
        hand, A, _ = run(model, arm, tr, rng, device)
        loss, pos = loss_fn(hand, A, None, tr, arm, device)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            log.append((it, float(pos)))
            print(f'{variant} s{seed} it {it} pos-err {float(pos)*100:.2f} cm  ({time.time()-t0:.0f}s)', flush=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{out_dir}/{variant}_s{seed}.pt')
    json.dump(log, open(f'{out_dir}/{variant}_s{seed}_log.json', 'w'))
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', default='hier:0,hier:1,hier:2,all_inputs:0,all_inputs:1,swap:0,swap:1,single:0')
    ap.add_argument('--iters', type=int, default=800); ap.add_argument('--B', type=int, default=256)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='../results/models'); ap.add_argument('--tag', default='stack'); ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(1)
    specs = [(x.split(':')[0], int(x.split(':')[1])) for x in a.specs.split(',')]
    train_stack(specs, a.iters, a.device, a.out, B=a.B, tag=a.tag, lr=a.lr)
