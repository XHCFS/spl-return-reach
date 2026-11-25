# Outward vs return reaching in the macaque superior parietal lobule: new results from the lab's public data, and what a trained-network model says about them

24–25 Nov 2026. Data: Diomedi et al. 2024, *Sci Data* 11:624, GIN doi 10.12751/g-node.7q2dbp (CC-BY 4.0).
- 682 single units: V6A 285, PEc 231, PE 166.
- Two *M. fascicularis*, 9 targets (3 directions × 3 depths), instructed delay, foveated reach in the dark.
- After the filters (≥ 6 trials per target, mean rate ≥ 1 Hz), 165–284 units remain per area.

Code is in `~/fattori-lab/code`, results in `~/fattori-lab/results`, and the Kaggle kernels are `spl-rnn-*`. Every analysis uses pseudo-populations, the same as the lab's own papers.

## 1. What was reproduced
| Published finding | Here |
|---|---|
| Target decoding V6A > PEc > PE (Vaccari 2024; Filippini 2022) | Pseudo-population LDA with 100 units, chance 0.11. Outward movement: 0.94 / 0.90 / 0.48. |
| Planning and execution subspaces partly shared in V6A/PEc, nearly orthogonal in PE (Diomedi 2025) | Elsayed alignment index, noise-unbiased: 0.24 / 0.16 / 0.11. |

## 2. New empirical results
These use only the public data. No model is involved.

**R1. Two opposite gradients along V6A → PEc → PE.**
- Planning→execution alignment *falls*: 0.24 → 0.16 → 0.11. V6A − PE = +0.13 [95% bootstrap +0.02, +0.22].
- Execution→return alignment *rises*: 0.02 → 0.04 → 0.12. V6A − PE = −0.10 [−0.18, −0.03].
- Method: alignment computed from cross-covariances between independent trial halves, so noise cancels. Bootstrap over units at matched N = 150.
- The return reach is carried in a subspace almost orthogonal to the outward reach everywhere (alignment ≤ 0.12, against ≈ 0.35–0.40 for random subspaces). PE reuses the most.

**R2. The return is carried by a re-drawn set of neurons in V6A and PEc, and by the same neurons in PE.**
- Overlap of which units carry the variance in each epoch (1 = same units, chance ≈ 0.45):
  - V6A and PEc, outward→return: 0.42–0.45, at chance, Spearman ≈ −0.05.
  - PE: 0.54, Spearman 0.31.

**R3. The return's target code is inverted, as for a velocity code, only in PE. It appears first in PE and later in PEc.** (Update 25 Nov: confirmatory tests in `confirmatory*.json`. R3 holds pooled and in monkey F, but not in monkey S. It is carried by a minority of large-amplitude units, and the median PE unit does not invert. See the root README, E2 and E3.)
- Statistic: correlation between the outward target pattern and the return pattern. A pure velocity code predicts −1.
- PE −0.74; PEc −0.14; V6A −0.13. The PE value is noisy.
- Time-resolved around return onset (the hand holds at the target until then):
  - PE's inversion is significant from the [−50, +50) ms window and peaks at [0, +100) ms.
  - PEc's inversion is significant only from +100 ms and stays weak.
  - V6A never inverts.
- PEc's late inversion replicates in both monkeys. PE's early one is significant in monkey F and has the same direction, but is not significant, in monkey S, which has only 61 PE units.
- A kinematic encoding model (minimum-jerk position + velocity) fitted to the outward reach predicts the return at chance in every area.

**R4. The planning→execution gradient holds inside V6A.**
- V6Av (more visual) 0.46 > V6Ad 0.23 ≈ PEc 0.20 > PE 0.15. V6Av − V6Ad bootstrap P = 0.002.
- Per monkey: S has V6Av 0.48 > V6Ad 0.20; F has V6Ad 0.37 > PEc 0.26 > PE 0.18.
- The sharing is not a static-gaze artefact. Delay and hold share gaze position but only 0.12–0.23 of their target pattern, while delay and movement share 0.36–0.54.

## 3. What the trained networks show
**Setup:**
- Modular RNNs (3 × 100 units) were trained only to do the Bologna out-and-back task.
- The effector was a point mass with 6 pull-only actuators, or MotorNet's arm26.
- Over 100 networks were trained across 13 Kaggle runs, varying input placement, output placement, nonlinearity, regularization, noise and size.
- Every network was scored with the same metrics as the data.

**M1. Near-orthogonal outward/return target codes are generic.** They appear in every network, even though hand velocity is exactly inverted. So the weak out/return relation in V6A/PEc is what a goal-driven controller does. It isn't evidence about kinematic coding.

**M2. Rectified (ReLU) units are needed for cortex-like separation between epochs.**
- Execution→return alignment: tanh networks 0.4–0.8 with any regularization, noise or size; ReLU networks 0.08–0.19; cortex 0.02–0.12.
- ReLU networks achieve this by giving epochs *different* units (overlap below chance), while cortex is at chance. That is a remaining mismatch.

**M3. Both gradients in R1 appear when sensory input enters at the caudal end, far from the motor output.**
- Scored network by network on four gradient signs:
  - All inputs at the V6A-like module: 8/8 networks right.
  - All inputs everywhere: 4/4 right.
  - Textbook placement (vision → V6A-like, proprioception → PE-like): 0–25% right. It misses PE's inversion.
  - Swapped inputs, reversed output and a single module: fail.
- **Confirmation run** (fresh seeds): on the point mass, `vis_prop_m1` and `vp12` (sensory input to V6A- and PEc-like modules) each got all 4 signs right in 3/4 networks.
  - `vp12` confines the inversion more to the PE-like module: −0.23 / −0.26 / −0.44.
- **Not effector-robust.**
  - On MotorNet's arm (networks undertrained at 1–6 cm error), the planning→execution gradient reversed for these placements.
  - With ReLU units the PE-like inversion vanishes for every placement.
- **Treat M3 as tentative:** a point-mass result about input placement, not a general one.

**M4. In these networks, passive movement reproduces active activity (r ≈ 1).**
- They are feedback controllers, so they cannot separate command (efference) from sensory feedback (reafference).
- The PE-like inversion in the "all inputs at V6A" network is relayed proprioceptive velocity. Its timing order (PEc-like before PE-like) is the *opposite* of the data's (PE before PEc).
- **Velocity-dominated (spindle-like) proprioception entering the PE-like module** (textbook input placement, `PROP_MODE=vel`, 6 networks):
  - It reproduces PE's inversion (−0.76 vs −0.74), the delay~outward drop in PE (0.20 vs 0.09) and both alignment gradients. All 4 signs are right in 4/6 networks.
  - Its PEc-like module is also inverted (−0.80 vs −0.14), and ReLU removes the inversion.

**M5. The trained networks get the timing order wrong, and the data's order is the novel part.**
- In the 3 architectures that produce the inversion (17 networks with a measurable onset in both modules), the velocity-like return code appears first in the intermediate (PEc-like) module in 15 networks. One is within a 20 ms step, and one (`vis_prop_m1` seed 13) has the PE-like module 160 ms first (sign test 15/16, p = 0.0005). An earlier version of this section said "14 networks, none PE-first", which missed the confirmation batch. Onset ranges:
  - `vis_prop_m1`: onset −200…0 ms vs −40…+100 ms.
  - velocity-proprioception `hier`: −220…+20 ms vs −40…+100 ms, even with proprioception entering the PE-like module directly.
  - `vp12`: 0…+60 ms vs +20…+120 ms.
- Only 1 of 17 networks reproduces cortex's PE-first order.
- Trained controllers generate the return's kinematic code predictively, upstream. Cortex does the reverse (R3): PE at onset, PEc about 100–150 ms later, V6A never.
- Supporting this (analysis14): during the return reaction time (light off → hand lifts), all areas stay hold-like.
  - HOLD~RT: V6A 0.82, PEc 0.75, PE ≈ 1.
  - PEc shifts modestly toward the upcoming return pattern (RT~RET 0.37 [0.24, 0.52]). V6A's shift is marginal (0.23 [−0.01, 0.40]).
- **Hypothesis this raises:** the visually guided outward reach is planned through caudal SPL (shared delay→movement subspaces in V6A/PEc, R1/R4). The return reach in the dark is not: its kinematic code originates at the somatomotor end (PE, or upstream motor areas) and reaches PEc as feedback.

## 4. Predictions that need testing with the lab's data
1. **F2 in the same task** (De Vitis 2025 recordings). Apply this exact pipeline.
   - If the gradients track distance from the motor output, F2 should show planning→execution ≤ PE (≈ 0.11) and execution→return ≥ PE (≈ 0.12), with a strong outward/return inversion.
   - If F2 looks like V6A instead, the distance-from-output reading fails.
2. **Passive versus active out-and-back movements in PE.**
   - If PE's inversion is proprioceptive, it should be present during passive arm displacement along the same paths.
   - If it is a command or efference copy, it should be absent or much weaker.
   - The models can't decide this (M4), so the experiment is informative.
3. **PE timing with kinematics or EMG.** Does PE's inverted code lead or lag hand velocity on single trials? R3's 100 ms windows can't tell, and the lab's newer datasets with kinematics can.
4. **V6Av under gaze/target dissociation** (De Vitis 2022, Buonfiglio 2025 tasks). R4 argues the planning→execution sharing is not gaze. That predicts it survives when gaze and reach target are separated.
5. **Visually guided vs dark return.** If the return is cued to a visible target (for example a lit home target), the hypothesis in M5 predicts that V6A/PEc acquire a predictive, pre-onset return code and that the PE-before-PEc order weakens or reverses.
6. **The return reach in other lab tasks.** Most published SPL analyses stop at the outward reach. R1–R3 say the return is a near-orthogonal, separately recruited state in V6A/PEc. That should hold in any dataset where the return is recorded.

## 5. Caveats
- **Pseudo-populations.** These are condition-averaged geometries from neurons recorded on different days. Trial-level structure is unknown.
- **No eye traces after the light goes off.** Gaze during the return is uncontrolled and could shape V6A's return code.
- **Low PE reliability.** Split-half reliability is ≈ 0.15, so PE estimates have wide intervals.
- **Model results** are point-mass robust but not yet effector-robust. They are qualitative (gradient signs), not quantitative fits.
- **Novelty** was checked by searching published work and preprints. Unpublished lab analyses of the return epoch would not show up.
