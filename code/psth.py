"""Condition-averaged, event-aligned firing rates for the SPL dataset.

Marker order (Diomedi et al. 2024, Fig. 3): 0 Start, 1 Green on, 2 Fix on, 3 Green->red (go),
4 Move out on, 5 Move out off (touch), 6 Red off, 7 Move in on, 8 Move in off, 9 End.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

EV = dict(start=0, green=1, fix=2, go=3, mout=4, tout=5, redoff=6, min=7, tin=8, end=9)

# (event, t0, t1) windows in ms; concatenated in this order
WINDOWS = [('fix', -200, 1600), ('go', -300, 200), ('mout', -200, 500), ('redoff', -300, 200), ('min', -200, 500)]
BIN = 10
SIGMA = 3  # bins (30 ms)


def trial_rates(sp, mk, ev, t0, t1, bin_ms=BIN):
    a = mk[EV[ev]]
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    h, _ = np.histogram(sp - a, edges)
    return h * (1000.0 / bin_ms)


def unit_tensor(unit, windows=WINDOWS, smooth=True):
    """Return (n_cond, n_time) mean rate and list of per-trial arrays (n_trials, n_time) per cond."""
    means, trials_all = [], []
    for trials in unit['conds']:
        tr = []
        for sp, mk in trials:
            tr.append(np.concatenate([trial_rates(sp, mk, *w) for w in windows]))
        tr = np.array(tr)
        if smooth:
            tr = gaussian_filter1d(tr, SIGMA, axis=1, mode='nearest')
        trials_all.append(tr)
        means.append(tr.mean(0))
    return np.array(means), trials_all


def window_slices(windows=WINDOWS, bin_ms=BIN):
    out, i = {}, 0
    for ev, t0, t1 in windows:
        n = (t1 - t0) // bin_ms
        out[ev] = (slice(i, i + n), np.arange(t0, t1, bin_ms) + bin_ms / 2)
        i += n
    return out, i
