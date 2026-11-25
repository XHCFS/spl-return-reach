"""Load the Diomedi et al. 2024 SPL dataset (GIN 10.12751/g-node.7q2dbp) into plain Python structures.

units: list of dict(area, monkey, name, conds=[ [ (spikes_ms, markers_ms), ... trials ], ... 9 conds ])
"""
import glob, os, pickle
import h5py
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'spl', 'data')
CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'spl_units.pkl')


def load(cache=True):
    if cache and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    units = []
    for fn in sorted(glob.glob(os.path.join(DATA, '*_reach9pos.h5'))):
        monkey, area = os.path.basename(fn).split('_')[:2]
        with h5py.File(fn, 'r') as f:
            for uname in sorted(f['DATA'].keys()):
                ug = f['DATA'][uname]
                conds = []
                for cname in sorted(ug.keys()):
                    trials = []
                    for tname in sorted(ug[cname].keys()):
                        tg = ug[cname][tname]
                        sp = np.asarray(tg['spike_trains'][()], float).ravel()
                        mk = np.asarray(tg['event_markers'][()], float).ravel()
                        trials.append((sp, mk))
                    conds.append(trials)
                units.append(dict(area=area, monkey=monkey, name=uname, conds=conds))
    if cache:
        with open(CACHE, 'wb') as f:
            pickle.dump(units, f)
    return units


if __name__ == '__main__':
    u = load(cache=False)
    from collections import Counter
    print(Counter((x['monkey'], x['area']) for x in u))
    print('conds per unit', Counter(len(x['conds']) for x in u))
    nt = [len(t) for x in u for t in x['conds']]
    print('trials per cond: min %d median %d max %d' % (min(nt), np.median(nt), max(nt)))
    print('markers per trial', Counter(len(t[1]) for x in u for c in x['conds'] for t in c))
    sp, mk = u[0]['conds'][0][0]
    print('example markers', mk, 'n spikes', len(sp), 'spike range', sp.min() if len(sp) else None, sp.max() if len(sp) else None)
