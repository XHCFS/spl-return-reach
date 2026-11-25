"""Figures for the report (static PNG, light surface). Area colours are fixed across all figures:
V6A blue, PEc orange, PE aqua. Monkey identity is marker shape. Every line is direct-labelled."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/home/saif/fattori-lab/results'
OUT = '/home/saif/fattori-lab/figures'
os.makedirs(OUT, exist_ok=True)
AREAS = ['V6A', 'PEc', 'PE']
COL = {'V6A': '#2a78d6', 'PEc': '#eb6834', 'PE': '#1baf7a'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MK = {'pooled': ('o', 9), 'MonkeyF': ('s', 6), 'MonkeyS': ('^', 6)}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2, 'legend.frameon': False})


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=160, bbox_inches='tight'); plt.close(fig); print('wrote', name)


def per_area_stats():
    import analysis15 as A
    out = {}
    for lab, m in [('pooled', np.ones(len(A.AREA), bool)), ('MonkeyF', A.MONK == 'MonkeyF'), ('MonkeyS', A.MONK == 'MonkeyS')]:
        for a in AREAS:
            i = np.where(m & (A.AREA == a))[0]
            out[f'{lab}|{a}'] = dict(n=len(i), prep_exec=A.ai(i, 'prep', 'exec'), exec_ret=A.ai(i, 'exec', 'ret'), reuse=A.part_rho(i),
                                     out_ret=A.pat(i, 'out', 'ret'), early=A.pat(i, 'out', 'early'), late=A.pat(i, 'out', 'late'))
    json.dump(out, open(f'{R}/per_area_stats.json', 'w'), indent=1)
    return out


def fig_gradients(S):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, title in zip(axs, ['prep_exec', 'exec_ret', 'reuse'],
                              ['planning to movement\nsubspace alignment', 'outward to return\nsubspace alignment', 'outward to return\nunit reuse (Spearman)']):
        x = np.arange(3)
        ax.plot(x, [S[f'pooled|{a}'][key] for a in AREAS], color=INK2, lw=1, zorder=1)
        for lab, (m, ms) in MK.items():
            for j, a in enumerate(AREAS):
                ax.plot(j + {'pooled': 0, 'MonkeyF': -0.12, 'MonkeyS': 0.12}[lab], S[f'{lab}|{a}'][key], m, ms=ms, color=COL[a],
                        mec=SURF if lab == 'pooled' else COL[a], mfc=COL[a] if lab == 'pooled' else SURF, mew=2 if lab == 'pooled' else 1.5, zorder=3)
        ax.set_xticks(x, AREAS); ax.set_title(title, fontsize=10, color=INK); ax.axhline(0, color=INK2, lw=0.8)
    h = [plt.Line2D([], [], marker=MK[k][0], ms=MK[k][1], color=INK2, mfc=INK2 if k == 'pooled' else SURF, ls='') for k in MK]
    axs[2].legend(h, ['both monkeys', 'monkey F', 'monkey S'], loc='upper left', fontsize=9)
    axs[0].set_ylabel('alignment index or Spearman rho')
    save(fig, 'fig1_gradients')


def fig_timing():
    d = json.load(open(f'{R}/efference_timing.json'))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for a in AREAS:
        t = np.array(d[a]['t']) + 50
        ax.fill_between(t, d[a]['lo'], d[a]['hi'], color=COL[a], alpha=0.15, lw=0)
        ax.plot(t, d[a]['r_raw'], color=COL[a]); ax.text(t[-1] + 12, d[a]['r_raw'][-1] + {'V6A': 0, 'PEc': -0.012, 'PE': 0.012}[a], a, color=INK, va='center', fontsize=9)
        sig = np.array(d[a]['hi']) < 0
        ax.plot(t[sig], np.array(d[a]['r_raw'])[sig], 'o', ms=8, color=COL[a], mec=SURF, mew=2)
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8, ls=':')
    ax.set_xlabel('centre of 100 ms window, ms from return onset (Move in on)'); ax.set_ylabel('outward vs return pattern r (raw)')
    ax.set_title('return target code relative to the outward one\nfilled dots: 95% bootstrap interval below zero', fontsize=10, color=INK)
    ax.set_xlim(-280, 420)
    save(fig, 'fig2_timing')


def fig_area2():
    d = json.load(open(f'{R}/area2_return.json'))['timecourse']
    t = np.array(d['t']) + 50
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, lab, c in [('neural_corr', 'area 2 neurons', '#6250d6'), ('kin_model', 'kinematic model', '#eda100'), ('hand_vel', 'hand velocity', INK2)]:
        y = np.array(d[key], float); ok = ~np.isnan(y)
        ax.plot(t[ok], y[ok], color=c, ls='--' if key == 'hand_vel' else '-'); ax.text(t[ok][-1] + 12, y[ok][-1], lab, color=INK, va='center', fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8, ls=':')
    pe = json.load(open(f'{R}/confirmatory.json'))['pooled']['T4 PE out~ret pattern < 0']['corrected']
    ax.axhline(pe, color=COL['PE'], lw=1.2, ls=':'); ax.text(-270, pe + 0.05, f'PE, 0-300 ms (corrected {pe:+.2f})', color=INK, fontsize=8)
    ax.set_xlabel('centre of 100 ms window, ms from return onset'); ax.set_ylabel('outward vs return pattern r')
    ax.set_title('S1 area 2 reference (NLB Area2_Bump, 65 units)', fontsize=10, color=INK); ax.set_xlim(-280, 520)
    save(fig, 'fig3_area2')


def fig_model_timing():
    pts = []
    for f in glob.glob(f'{R}/timing_model.json') + glob.glob(f'{R}/kaggle_*/*timing*.json'):
        for k, v in json.load(open(f)).items():
            arch = k.split('|')[1].rsplit('_s', 1)[0]
            inverting_arch = arch in ('vis_prop_m1', 'vp12') or k.startswith('vel_tanh|hier')  # the three architectures that invert (REPORT M3/M4)
            if inverting_arch and v['M2']['t_onset'] is not None and v['M3']['t_onset'] is not None:
                pts.append((v['M2']['t_onset'], v['M3']['t_onset'], k))
    pts = sorted(set(pts))
    first = dict(pec=sum(b > a for a, b, _ in pts), tie=sum(abs(b - a) <= 20 and not b > a for a, b, _ in pts), pe=sum(b < a - 20 for a, b, _ in pts))
    json.dump(dict(points=pts, counts=first), open(f'{R}/model_timing_order.json', 'w'), indent=1); print('order counts', first)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    jit = np.random.default_rng(0).uniform(-6, 6, (len(pts), 2))
    ax.plot([p[0] for p in pts] + jit[:, 0], [p[1] for p in pts] + jit[:, 1], 'o', ms=8, color='#6250d6', mec=SURF, mew=2, label=f'trained networks (n={len(pts)})')
    ax.plot([100], [-50], '*', ms=16, color=COL['PE'], mec=INK, mew=0.8, label='cortex (PEc +100, PE -50)')
    lim = [-260, 320]; ax.plot(lim, lim, color=INK2, lw=0.8, ls=':'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(150, 230, 'PE-like first', color=INK2, fontsize=9); ax.text(-240, 260, 'above diagonal:\nPEc-like first', color=INK2, fontsize=9)
    ax.set_xlabel('PEc-like module inversion onset (ms)'); ax.set_ylabel('PE-like module inversion onset (ms)')
    ax.legend(loc='lower right', fontsize=8); ax.set_title('order of the velocity-like return code', fontsize=10, color=INK)
    save(fig, 'fig4_model_timing')
    return pts


def fig_power():
    d = json.load(open(f'{R}/power.json'))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, test, title in zip(axs, ['T4', 'T5'], ['PE inversion (T4)', 'PE before PEc (T5)']):
        for lab, c, ls in [('pooled', INK, '-'), ('F', '#6250d6', '-'), ('S', '#008300', '--')]:
            p = d[lab][test]['power_at_N']; n = [int(k) for k in p]
            ax.plot(n, list(p.values()), color=c, ls=ls, label={'pooled': 'effect size of both monkeys', 'F': 'effect size of monkey F', 'S': 'effect size of monkey S'}[lab])
        ax.axhline(0.8, color=INK2, lw=0.8, ls=':'); ax.set_xlabel('neurons per area'); ax.set_title(title, fontsize=10, color=INK); ax.set_xlim(30, 520)
    axs[0].set_ylabel('power (one-sided alpha 0.05/6)'); axs[1].legend(loc='lower right', fontsize=8)
    save(fig, 'fig5_power')


if __name__ == '__main__':
    S = per_area_stats()
    fig_gradients(S); fig_timing(); fig_area2(); pts = fig_model_timing(); fig_power()
    print('model timing points', sorted(pts))
