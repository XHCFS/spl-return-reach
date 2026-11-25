# Running findings log (Topic A exploration)

The dataset is Diomedi et al. 2024 (GIN 10.12751/g-node.7q2dbp).
- 682 single units: V6A 285, PEc 231, PE 166.
- Two monkeys, 9 targets, 7–10 trials per target.
- After the filters (≥5–6 trials per condition, mean rate ≥ 1 Hz), 165–284 units remain per area.

## Reproductions (Phase 0)
- **Target decoding** with pseudo-populations of 100 units, LDA, chance 0.11. V6A > PEc > PE in every epoch.
  - Early delay: 0.96 / 0.84 / 0.46
  - Move out: 0.94 / 0.90 / 0.48
  - This matches Vaccari 2024 and Filippini 2022.
- **Planning-to-execution alignment** (Elsayed index, k = 6): V6A 0.25, PEc 0.16, PE 0.14. Random baseline ≈ 0.33–0.40.
  - All three areas are more orthogonal than chance, with the gradient V6A > PEc > PE.
  - This is consistent with Diomedi 2025: PE nearly orthogonal, V6A/PEc partly shared.

## New observations from the public data
1. **Outward vs return reach target code.**
   - Statistic: condition-centred, trial-split, noise-corrected pattern correlation over 0–300 ms after each movement onset.
   - V6A −0.13 [−0.28, −0.03]; PEc −0.14 [−0.26, −0.04]; PE −0.74 [−8.3, −0.10] (PE is very noisy, reliability ≈ 0.15).
   - The value depends on the window. V6A ranges from −0.02 to −0.17.
   - A pure hand-velocity code predicts −1.
   - A kinematic encoding model (minimum-jerk position + velocity) fitted on the outward reach predicts the return at chance in every area (r ≈ 0.01, against 0.08–0.12 within the outward reach).
   - **Reading:** the return movement is mostly encoded in a different (near-orthogonal) target subspace in V6A/PEc, with a larger inverted component in PE.
2. **Elsayed alignment of the exec subspace to the return subspace** (not condition-centred): V6A 0.04, PEc 0.07, PE 0.15.
3. **Epoch-sharing matrix** (noise-corrected):
   - D~O: V6A 0.43, PEc 0.43, PE 0.09.
   - D~H: 0.16 / 0.19 / −0.04.
   - Delay and hold share gaze position on the target but not hand position. Their weak sharing means the large delay/movement sharing in V6A/PEc is **not** static gaze or target position.
   - Control: the FREE epoch has no target structure (reliability ≈ −0.07), as expected.
4. **Direction vs depth split** of the out/return relation: no clean dissociation.

## Model runs
- MotorNet arm26 modular RNN, 8 networks per stack. Locally this took ~5 s per step and was too slow.
  - Kaggle kernel `spl-rnn-train`: 16 models, 2500 steps, learning rate 3e-3.
- Point-mass lean effector: Kaggle kernel `spl-rnn-pointmass`, 16 models, 2000 steps, plus analysis.
- An early checkpoint (step 99, undertrained) had module out~in of +0.3 to +0.8, the opposite sign to the data. It needs a converged check.

## Update: point-mass sweep (Kaggle `spl-rnn-pointmass`, 16 nets, 2000 steps, endpoint error < 1 mm)
- **Generic near-orthogonality.** The outward and return target codes are close to orthogonal in every variant and every module (r from −0.2 to +0.4), even though hand velocity is exactly inverted (−1.00).
  - So the weak out/return relation in SPL is what a goal-driven controller does. It isn't evidence for a fixed kinematic code.
- **Two gradients set by position in the hierarchy.**
  - Elsayed index (full data), module 1 → 2 → 3 for hier: planning→execution 0.62 → 0.44 → 0.35; execution→return 0.35 → 0.60 → 0.72.
  - The same orderings appear with swapped inputs and with all inputs to every module. They are flat in the single-module network.
- **Input placement shows up instead in hold~return and delay~outward of module 1.**
  - Hold~return: hier −0.02, swap 0.91. The data (V6A) give 0.07, which favours hier.
  - Delay~outward with inputs swapped: 0.11, against 0.43 in the data.

## Update: noise-unbiased alignment in the data
Method: cross-half covariance, bootstrap over units, N = 150.
- **Planning→execution:** V6A 0.24 > PEc 0.16 > PE 0.11. V6A − PE = +0.13 [+0.02, +0.22].
- **Execution→return:** V6A 0.02 < PEc 0.04 < PE 0.12. V6A − PE = −0.10 [−0.18, −0.03].
- **Planning→return:** about 0.03 everywhere.
- Both gradients exist in the data, in opposite directions, and the hierarchical model predicts both orderings.
- **Mismatch:** the absolute values are much lower in cortex. The models are far less orthogonal between epochs. This is being tested with an activity-regularization, noise and size sweep (Kaggle `spl-rnn-reg`).
- **Decisive control (queued):** read the muscles out of module 1 (`out_first`), and put all inputs at module 1 (`vis_prop_m1`). If distance from the output is the cause, both gradients should flip with `out_first`.

## Update: regularization/noise/size sweep (Kaggle `spl-rnn-reg`, 32 nets)
- **Robust in hier under every config:** planning→execution alignment falls M1 > M2 > M3, and execution→return rises M1 < M2 < M3.
- **Correction to the earlier claim:** with inputs swapped, the planning→execution gradient is inconsistent and even reverses in one config (reg1: 0.21/0.35/0.71). It depends on where the goal/visual input enters relative to the output. It is not set by distance from the output alone.
- **The main cortex–model discrepancy remains.**
  - Cortex: execution→return 0.02–0.12, planning→return 0.02–0.03.
  - Networks: 0.25–0.85 and 0.1–0.5.
  - Activity L2 (0.01, 0.1), 4× noise and 2× size don't close the gap.

## Update: V6A subareas (Cito labels), noise-unbiased alignment, bootstrap N = 80
- **Planning→execution:** V6Av 0.46 > V6Ad 0.23 ≈ PEc 0.20 > PE 0.15. V6Av − V6Ad = +0.23 (P = 0.002).
- **Execution→return:** 0.09 / 0.07 / 0.08 / 0.16. Only PE stands out.
- **Per monkey:**
  - Monkey S: V6Av 0.48 > V6Ad 0.20.
  - Monkey F: V6Ad 0.37 > PEc 0.26 > PE 0.18 (V6Av has only 23 units).
- **Gaze-shared component** (delay~hold): V6Av 0.23 / V6Ad 0.12. Delay~outward is far larger (0.54 / 0.36), so the sharing is not a gaze artefact.

## Update: participation (which neurons carry each epoch)
Noise-unbiased per-unit variance. Overlap: 1 = same units, 0 = disjoint.
- **V6A:** planning→execution 0.56 (null 0.48), execution→return 0.42 (null 0.45), planning→return 0.42 (null 0.43).
- **PEc:** 0.47 (0.44), 0.45 (0.48), 0.40 (0.41).
- **PE:** 0.46 (0.40), 0.54 (0.46), 0.46 (0.43).
- In V6A/PEc the return is carried by neurons unrelated to those of the outward reach or the delay (chance overlap, Spearman ≈ −0.05). PE reuses them (Spearman 0.31).
- **Tanh networks:** output-module execution→return overlap is 0.74–0.80, i.e. heavy reuse.

## Update: controls (Kaggle `spl-rnn-ctrl`) and per-network scoring (`score_models.py`)
The score asks whether each network gets the sign of the M1→M3 gradient right on out_in, delay_out, align_prep_exec and align_exec_in, against data V6A→PE.
- **All 4 signs right in every network:** `vis_prop_m1` (8/8; all inputs at M1, output at M3) and `all_inputs` (4/4, small magnitudes).
- **Fails:** `hier` (0–25% of networks; it misses PE's inversion), `swap`, `out_first` and `single`.
- **`vis_prop_m1` M3 reproduces PE's magnitudes:** out_in −0.73 to −0.90 (data −0.74), delay_out 0.10 (data 0.09).
  - But its M2 is also strongly inverted (−0.66 to −0.85), while the data's PEc is −0.14. This led to the `vp12` variant.
- **MotorNet arm** (Kaggle `spl-rnn-train`, 2500 steps): `hier` networks are undertrained (1–6 cm endpoint error).
  - Well-trained `all_inputs` networks reproduce the planning→execution gradient but not the execution→return one.
  - So the model conclusions are not yet effector-robust.

## Update: efference timing in the data (analysis13)
- **Method:** correlation between the outward pattern (0–300 ms) and 100 ms return windows around Move-in-on; bootstrap over units.
- **PE:** inversion significant from [−50, +50) ms, peaking at [0, +100) (r_corr −0.68).
- **PEc:** inversion only from [+100, +200) to [+250, +350) ms (r_corr −0.15).
- **V6A:** none.
- **Before onset:** all areas carry a positive, hold-like pattern.
- **Reading:** the velocity-like code appears in PE at movement onset, about 100–150 ms before PEc. That fits PE as a command/efference stage with PEc receiving it later.
  - It contradicts the `vis_prop_m1` timing (M2 before M3) and is what `vp12` predicts.

## Update: ReLU / phase sweep (Kaggle `spl-rnn-relu`)
- **ReLU closes the epoch-orthogonality gap.**
  - Execution→return in hier with ReLU (activity reg 0.01): 0.08 / 0.13 / 0.19. Planning→return: 0.08 / 0.12 / 0.06.
  - Data: 0.02 / 0.04 / 0.12 and 0.02–0.03.
  - tanh with any setting: 0.4–0.8.
- **But ReLU overshoots unit specialization.** Execution/return participation overlap is 0.10–0.38, below the null (Spearman −0.2 to −0.36). Cortex is at the null (Spearman ≈ −0.05).
- ReLU hier still has no PE-like inversion (M3 out_in ≈ 0.1).
- Phase cues to every module (tanh): 3/4 hier networks get all 4 signs right. It doesn't fix the magnitudes.

## Update: per-monkey efference timing
- PEc's late inversion (from +100 ms) replicates in both monkeys.
- PE's onset inversion is significant in monkey F (n = 103). In monkey S (n = 61) the minimum is also at t = 0 but not significant.

## Update: passive-movement simulation (`passive_model.py`)
- Imposing the same out/return paths with the network output ignored reproduces the active patterns in every module (r ≈ 0.96–1.0) for both `vis_prop_m1` and `hier`. The networks act as feedback controllers.
- So the `vis_prop_m1` M3 inversion is relayed proprioceptive velocity, not an internal command. **The "PE = efference stage" reading is withdrawn.**
- **Next model:** velocity-dominated (spindle-like) proprioception entering the PE-like module (`PROP_MODE=vel`, Kaggle `spl-rnn-vel`, queued). It should predict PE inverting first and PEc later, as in the data.

## Update: velocity proprioception (Kaggle `spl-rnn-vel`) and confirmation (`spl-rnn-confirm`)
- **vel_tanh hier:**
  - PE-like inversion −0.76; delay_out 0.64 / 0.46 / 0.20; both alignment gradients present.
  - All 4 signs right in 4/6 networks.
  - Its PEc-like module is also inverted (−0.80).
  - With ReLU the inversion is lost.
- **Timing:** in every inverting architecture the PEc-like module inverts before or with the PE-like one.
  - vis_prop_m1, vel hier and vp12 (14 networks): onsets M2 −220…+60 vs M3 −40…+120.
  - The data's order (PE at onset, PEc +100–150 ms) is never reproduced.
- **Confirmation:**
  - Point-mass vis_prop_m1 and vp12: 3/4 networks each get all signs right.
  - MotorNet (undertrained, 1–6 cm): the planning→execution gradient reverses.
  - ReLU: the inversion vanishes for every placement.
  - So input-placement conclusions are point-mass only.
- **Return reaction time (analysis14):** HOLD~RT is 0.75–1.07 in all areas. PEc's RT~RET is 0.37 [0.24, 0.52] (modest return preparation); V6A's is 0.23 [−0.01, 0.40].
