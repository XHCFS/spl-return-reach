"""Per-network fit to the data's area gradients. For each metric, the 3-vector (M1, M2, M3) is compared
with data (V6A, PEc, PE): shape score = Pearson r across the 3 stages; gradient-sign agreement = sign(M3-M1)
== sign(PE-V6A). Reported per (config, variant) as mean and the fraction of networks matching each sign."""
import json, sys, os
import numpy as np
from collections import defaultdict
from summarize_models import DATA, MODS

KEYS = ['out_in', 'delay_out', 'align_prep_exec', 'align_exec_in']
rows = defaultdict(list)
for p in sys.argv[1:]:
    cfg = os.path.basename(p).split('_metrics')[0]
    for r in json.load(open(p)):
        v = r['name'].rsplit('_s', 1)[0]
        sc, sg = [], []
        for k in KEYS:
            m = np.array([r[mm][k] for mm in MODS]); d = np.array(DATA[k])
            sc.append(np.corrcoef(m, d)[0, 1]); sg.append(np.sign(m[2] - m[0]) == np.sign(d[2] - d[0]))
        rows[(cfg, v)].append((sc, sg))
print(f"{'config/variant':26s} n  " + ' '.join(f'{k:>18s}' for k in KEYS) + '   all-4-signs')
for key, L in sorted(rows.items()):
    sc = np.array([x[0] for x in L]); sg = np.array([x[1] for x in L])
    cells = [f'r={sc[:, j].mean():+.2f} sgn={sg[:, j].mean():.2f}' for j in range(len(KEYS))]
    print(f'{key[0] + "/" + key[1]:26s} {len(L)}  ' + ' '.join(f'{c:>18s}' for c in cells) + f'   {sg.all(1).mean():.2f}')
