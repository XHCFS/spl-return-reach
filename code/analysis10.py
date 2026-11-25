"""Within-area test of the hierarchy hypothesis: V6Av (visual) vs V6Ad (somatomotor) vs PEc vs PE,
noise-unbiased alignment (analysis9), matched N, bootstrap."""
import numpy as np, json, glob, os, h5py
from collections import defaultdict
import analysis9 as A9
from load_spl import load
from psth import unit_tensor

cito = {}
for fn in glob.glob('/home/saif/fattori-lab/data/spl/data/*.h5'):
    mk, ar = os.path.basename(fn).split('_')[:2]
    with h5py.File(fn) as f:
        for un in f['DATA']:
            cito[(mk, ar, un)] = str(f['DATA'][un].attrs['Cito'])
print('subarea labels', {k: sum(1 for v in cito.values() if v == k) for k in set(cito.values())})
D = defaultdict(list)
for u in load():
    if min(len(c) for c in u['conds']) < 6:
        continue
    m, tr = unit_tensor(u)
    if m.mean() < 1.0:
        continue
    a, b = A9.halves(tr); s = (m.max() - m.min()) + 5.0
    D[cito[(u['monkey'], u['area'], u['name'])]].append((a / s, b / s))
groups = [g for g in ['V6Av', 'V6Ad', 'PEc', 'PE'] if len(D[g]) >= 60]
N = min(80, min(len(D[g]) for g in groups))
res, draws = {}, {}
for g in groups:
    H1 = np.array([x[0] for x in D[g]]); H2 = np.array([x[1] for x in D[g]])
    bs = defaultdict(list)
    for _ in range(400):
        i = A9.rng.integers(0, len(H1), N)  # bootstrap with replacement
        bs['prep_exec'].append(A9.ai(H1[i], H2[i], 'prep', 'exec', 3)); bs['exec_ret'].append(A9.ai(H1[i], H2[i], 'exec', 'ret', 3))
    draws[g] = {k: np.array(v) for k, v in bs.items()}
    res[g] = {k: [round(float(np.median(v)), 3), round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in bs.items()}
    print(g, len(H1), 'N', N, res[g])
for x, y in [('V6Av', 'V6Ad'), ('V6Ad', 'PEc'), ('V6Av', 'PE')]:
    if x in draws and y in draws:
        for k in ['prep_exec', 'exec_ret']:
            d = draws[x][k][:, None] - draws[y][k][None]
            print(f'  {x}-{y} {k}: median diff {np.median(d):+.3f}, P(diff<=0) {np.mean(d<=0):.3f}')
json.dump(res, open('/home/saif/fattori-lab/results/subarea_alignment_boot.json', 'w'), indent=1)
