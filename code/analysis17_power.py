"""Power for the follow-up experiment: how many neurons per area are needed to detect
  T4  PE outward~return inversion (< 0)                     [condition-label permutation null]
  T5  early-return inversion stronger in PE than PEc        [unit-label permutation null]
Null SD is estimated at the recorded N from permutations and scaled by sqrt(N0/N) (the statistic is a
mean over units, so its null SD ~ 1/sqrt(N)); effect = observed value (pooled, and the weaker monkey).
One-sided alpha 0.05/6 (Holm worst case for a 6-test family)."""
import numpy as np, json
from scipy.stats import norm
import analysis15 as A

rng = np.random.default_rng(17)
NP, ALPHA = 1000, 0.05 / 6
res = {}
for lab, mask in [('pooled', np.ones(len(A.AREA), bool)), ('F', A.MONK == 'MonkeyF'), ('S', A.MONK == 'MonkeyS')]:
    if not mask.any():
        mk = sorted(set(A.MONK)); mask = A.MONK == (mk[0] if lab == 'F' else mk[1])
    pe = np.where(mask & (A.AREA == 'PE'))[0]; pec = np.where(mask & (A.AREA == 'PEc'))[0]
    obs4 = A.pat(pe, 'out', 'ret'); n4 = np.array([A.pat(pe, 'out', 'ret', rng.permutation(9)) for _ in range(NP)])
    obs5 = A.pat(pe, 'out', 'early') - A.pat(pec, 'out', 'early')
    both = np.concatenate([pe, pec]); n5 = []
    for _ in range(NP):
        u = rng.permutation(both); n5.append(A.pat(u[:len(pe)], 'out', 'early') - A.pat(u[len(pe):], 'out', 'early'))
    n5 = np.array(n5)
    row = {}
    for name, obs, null, N0 in [('T4', obs4, n4, len(pe)), ('T5', obs5, n5, min(len(pe), len(pec)))]:
        sd = null.std()
        curve = {N: round(float(norm.cdf(abs(obs) / (sd * np.sqrt(N0 / N)) - norm.ppf(1 - ALPHA))), 2) for N in [40, 60, 100, 150, 200, 300, 400]}
        need = next((N for N in range(20, 2001, 10) if norm.cdf(abs(obs) / (sd * np.sqrt(N0 / N)) - norm.ppf(1 - ALPHA)) >= 0.8), None)
        row[name] = dict(obs=round(obs, 3), null_sd=round(float(sd), 3), N_recorded=N0, power_at_N=curve, N_for_80pct=need)
    res[lab] = row
    print(lab, json.dumps(row))
json.dump(res, open('/home/saif/fattori-lab/results/power.json', 'w'), indent=1)
