"""Monkey-stratified version of the confirmatory family (analysis15): the areas have different monkey
mixes (e.g. PE is 63% monkey F, V6A 45%), so a pooled area difference could be a monkey difference.
Statistic = mean over the two monkeys of the within-monkey statistic (equal weight).
Null = area labels permuted *within* each monkey (T1-T3, T5, T6); condition labels permuted
independently per monkey (T4). Holm across the 6 tests."""
import numpy as np, json, os
from multiprocessing import Pool
import analysis15 as A

NPERM = int(os.environ.get('NPERM', 2000))
MKS = sorted(set(A.MONK))


def groups(name):
    t = {n: (a, b, f, s) for n, a, b, f, s in A.tests(np.ones(len(A.AREA), bool))}[name]
    a, b = t[0], t[1]
    return [(a[A.MONK[a] == m], None if b is None else b[A.MONK[b] == m]) for m in MKS], t[2], t[3]


def stat(pairs, f):
    return float(np.mean([A.F(f, a) - A.F(f, b) for a, b in pairs]))


def perm(args):
    pairs, f, seed = args
    r = np.random.default_rng(seed); out = []
    for a, b in pairs:
        u = r.permutation(np.concatenate([a, b])); out.append((u[:len(a)], u[len(a):]))
    return stat(out, f)


if __name__ == '__main__':
    names = [t[0] for t in A.tests(np.ones(len(A.AREA), bool))]
    res = {}
    with Pool(11) as pool:
        for name in names:
            pairs, f, sgn = groups(name)
            if pairs[0][1] is None:  # T4
                obs = float(np.mean([A.pat(a, 'out', 'ret') for a, _ in pairs]))
                null = np.array([np.mean([A.pat(a, 'out', 'ret', A.rng.permutation(9)) for a, _ in pairs]) for _ in range(NPERM)])
            else:
                obs = stat(pairs, f)
                null = np.array(pool.map(perm, [(pairs, f, 5000 + i) for i in range(NPERM)]))
            p = float((1 + np.sum(null * sgn >= obs * sgn)) / (1 + NPERM))
            res[name] = dict(stat_mean_over_monkeys=round(obs, 3), p=p, n=[[len(a), 0 if b is None else len(b)] for a, b in pairs])
            print(name, res[name], flush=True)
    for k, a in zip(names, A.holm(np.array([res[k]['p'] for k in names]))):
        res[k]['p_holm'] = round(float(a), 4)
    print(json.dumps(res, indent=1))
    json.dump(res, open('/home/saif/fattori-lab/results/confirmatory_stratified.json', 'w'), indent=1)
