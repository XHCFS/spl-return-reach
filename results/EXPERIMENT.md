# Proposed experiment: is the return reach in the dark planned in caudal SPL, or relayed from the somatomotor end?

25 Nov 2026. This follows from `REPORT.md` (results R1–R3, M5) and is sized with `power.json`.

## The question
In the public SPL data, the reach *out* to a visible target and the reach *back* to the home button in the dark use different population codes. The difference changes along the V6A → PEc → PE axis:
- **V6A and PEc:** the return is carried by a different set of neurons, in a subspace nearly orthogonal to the outward one. Its target code is not inverted.
- **PE:** the return reuses the outward neurons, with an inverted, velocity-like code. That code appears at return onset in PE and about 100–150 ms later in PEc. It never appears in V6A.

Trained network controllers predict the opposite order: the upstream module goes first. Two readings fit the data:
- **(H-efference)** PE carries a command or efference copy for the return. PEc receives it later.
- **(H-afference)** PE carries proprioceptive feedback of the moving arm, and so does PEc, with a longer delay.

Either way, caudal SPL (V6A) would not be planning the dark return. The existing data can't separate H-efference from H-afference, for three reasons:
1. There are no kinematics, only button events.
2. Gaze is uncontrolled after the fixation light goes off.
3. Neurons were recorded one session at a time, so latencies are pseudo-population averages.

## What the public area 2 data already tell us (reference, `area2_return.json`)
Area 2 of S1 is a proprioceptive area. NLB Area2_Bump (Chowdhury et al. 2020) has 65 simultaneously recorded units during centre-out reaches and passive bumps.
- **The out~return inversion is weak and late.**
  - Raw r is −0.11 over 0–300 ms after return onset (corrected −0.15, permutation p = 0.11).
  - It reaches about −0.34 (corrected) only 200–300 ms into the return.
  - Hand velocity is already inverted (r ≈ −0.86) 50–150 ms into the return.
- **A kinematic model predicts almost no inversion.** A linear position + velocity encoding model fitted to the outward reach predicts r ≈ −0.04 on the return, because position and velocity changes cancel.
- **Passive bumps resemble the active outward reach in the same direction** (r = +0.12 from 0–100 ms). They barely resemble the active return (+0.04).

**Implication.** A purely proprioceptive area shows a weaker and slower inversion than PE does (PE: −0.58 corrected when averaged over 8 trial splits, significant from [−50, +50) ms, though mostly from monkey F). That points toward H-efference, but only weakly: the tasks differ (area 2 has no hold at the target) and so do the animals. The area 2 result also shows that a passive-only test is ambiguous, because active and passive responses differ even in S1. The decisive measure is therefore **timing relative to the arm's own movement on single trials**, not passive–active similarity alone.

## Design
The existing Reach-to-Press setup is used, with three additions. All conditions are interleaved within sessions.

| Factor | Levels | Tests |
|---|---|---|
| Return guidance | (a) dark return, as now; (b) the home button is lit when the fixation LED goes off | Visual planning of the return |
| Gaze during the return | (a) free, as now; (b) fixate the target LED, which stays on dim until the home button is pressed | Gaze confound in V6A |
| Movement mode (separate blocks) | (a) active; (b) passive: the arm is moved along each monkey's own mean active trajectories by a robotic manipulandum/exoskeleton | Afferent latency baseline |

**Measurements**
- Continuous eye tracking.
- 3D hand and arm kinematics (markerless video or markers, ≥ 200 Hz).
- EMG from deltoid, biceps and triceps, if feasible. EMG onset gives a command-timing reference.
- **Simultaneous** recordings in PE and PEc, plus V6A when possible, with multi-contact linear probes, so latencies can be measured within trials.

**Minimum viable version (no manipulandum).** Factors 1 and 2 plus kinematics and eye tracking. This is a change to the task program in the current rig, and it already tests the main predictions (P1, P2, P4).

## Pre-registered predictions and analyses
Every analysis uses the pipeline in `~/fattori-lab/code` (analysis9/13/15) with fixed windows and the 6-test Holm family from `confirmatory.json`.

| # | Prediction | Supports | Against |
|---|---|---|---|
| P1 | Dark return replicates R1–R3: exec→ret alignment PE > V6A; PE out~ret r < 0; early inversion PE < PEc | Replication | Failure in new animals means R1–R3 were animal-specific (note that monkey S is already weak for the PE inversion) |
| P2 | Lit-home return gives V6A/PEc a return-specific code **before** movement onset (RT window, Red off → Move in), which the dark return lacks; PE-before-PEc weakens | Caudal SPL plans visually guided returns, but not dark ones | No pre-onset change: caudal SPL ignores the return target |
| P3 | Single-trial latency of PE's inverted direction code, measured against hand-velocity onset: **H-efference** says it leads velocity onset (≤ 0 ms, and leads EMG or is near it); **H-afference** says it lags (≥ +20–40 ms, matching the passive-block latency) | Decides H-efference vs H-afference | — |
| P4 | With gaze held at the target, V6A's return code and the V6Av planning-execution sharing (R4) survive | R1/R4 are not gaze artefacts | They vanish, meaning gaze effects |
| P5 | Passive block: PE responds with a latency L_pass; in active trials PE's code leads L_pass by > 30 ms | Central (efference) contribution in PE | Equal latencies point to afference |

## Sample size (from `power.json`)
- Pseudo-population tests at one-sided α = 0.05/6:
  - **PE inversion** (P1/T4) needs about 90 PE neurons for 80% power at the pooled effect size, and about 60 at monkey F's.
  - At monkey S's effect size it needs about 500.
  - **PE-before-PEc** (T5) needs about 210 PE and 210 PEc neurons at the pooled effect size, about 150 at monkey F's, and about 690 at monkey S's.
- **Plan for at least 200 well-isolated PE and 200 PEc neurons per animal, in 2 animals.** This is similar to a typical Fattori-lab recording campaign (Diomedi 2024 has 166 PE and 231 PEc neurons across 2 animals).
- Simultaneous probes reduce the requirement for P3/P5, because latencies are then estimated per session from single trials. A 32-contact probe giving about 10 units per session needs about 20 sessions per area.

## Cheaper step that should come first: pre-registered reanalysis of existing lab data
Before any new recording, the fixed pipeline should run on the lab's **unpublished** datasets that include the return movement. Candidates:
- other Reach-to-Press sessions in V6A/PEc/PE;
- the De Vitis 2025 F2 recordings;
- the gaze–reach dissociation tasks.

Register the six tests and windows on OSF first. This turns R1–R3 from exploratory into confirmed results at no recording cost, and the F2 data tests where premotor cortex sits on the two gradients.

## Public datasets checked for replication
- **NLB MC_Maze (PMd/M1)** has no usable return: the trials are cut at reward and the return to centre is missing (the hand position jumps back between trials).
- **NLB Area2_Bump** has returns and was used above.
- A DANDI search ("parietal", "area 5", "reach") and the GIN search found no other public macaque SPL reaching dataset, so the return analysis can't be replicated in SPL outside the lab.
