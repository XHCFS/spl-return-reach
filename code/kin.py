"""Reconstructed hand kinematics (minimum-jerk) from event markers and target coordinates."""
import numpy as np

TGT = np.array([[-2.5, 9.3], [0, 10], [2.5, 9.3], [-3.56, 13.2], [0, 15], [3.56, 13.2], [-6.2, 23], [0, 25], [6.2, 23]])  # x,y cm; z constant
BIN = 10
OUT_W = (-200, 600)   # relative to Move out on
IN_W = (-200, 600)    # relative to Move in on
LAG = 100             # neural activity leads kinematics by LAG ms


def minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def trial_kin(mk, c, ev, w):
    """Hand x,y and velocity on bins of window w around event ev (4=out on, 7=in on), shifted by LAG."""
    t = mk[ev] + np.arange(w[0], w[1], BIN) + BIN / 2 + LAG
    tgt = TGT[c]
    pos = np.zeros((len(t), 2))
    # outward: mk4 -> mk5 ; hold at target until mk7 ; inward mk7 -> mk8
    so = (t - mk[4]) / max(mk[5] - mk[4], 1)
    si = (t - mk[7]) / max(mk[8] - mk[7], 1)
    frac = np.where(t < mk[7], minjerk(so), 1 - minjerk(si))
    pos = frac[:, None] * tgt[None]
    vel = np.gradient(pos, BIN / 1000.0, axis=0)
    return pos, vel


