"""Summarise model metric JSONs by (config, variant) against the data fingerprint."""
import json, sys, glob, os
import numpy as np
from collections import defaultdict

KEYS = ['out_in', 'delay_out', 'hold_in', 'align_prep_exec', 'align_exec_in', 'align_prep_in']
MODS = ['M1(V6A-like)', 'M2(PEc-like)', 'M3(PE-like)']
DATA = {'align_prep_exec': [0.24, 0.16, 0.11], 'align_exec_in': [0.02, 0.04, 0.12], 'align_prep_in': [0.02, 0.02, 0.03],
        'out_in': [-0.13, -0.14, -0.74], 'delay_out': [0.43, 0.43, 0.09], 'hold_in': [0.07, 0.20, 0.19]}


def load(paths):
    agg = defaultdict(lambda: defaultdict(list))
    for p in paths:
        cfg = os.path.basename(p).split('_metrics')[0]
        for r in json.load(open(p)):
            v = r['name'].rsplit('_s', 1)[0]
            agg[(cfg, v)]['err'].append(r['endpoint_err_cm'])
            for m in MODS:
                for k in KEYS:
                    if k in r[m]:
                        agg[(cfg, v)][(m, k)].append(r[m][k])
    return agg


if __name__ == '__main__':
    paths = sys.argv[1:]
    agg = load(paths)
    print(f"{'config/variant':28s} {'err':>5s} " + ' '.join(f'{k:>22s}' for k in KEYS))
    print(f"{'DATA V6A/PEc/PE':28s} {'':5s} " + ' '.join(f"{'/'.join(f'{x:.2f}' for x in DATA[k]):>22s}" for k in KEYS))
    for (cfg, v), d in sorted(agg.items()):
        cells = []
        for k in KEYS:
            vals = [np.mean(d[(m, k)]) if d[(m, k)] else np.nan for m in MODS]
            cells.append('/'.join(f'{x:.2f}' for x in vals))
        print(f'{cfg + "/" + v:28s} {np.mean(d["err"]):5.2f} ' + ' '.join(f'{c:>22s}' for c in cells))
