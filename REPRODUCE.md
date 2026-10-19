# CSCI 645 HW2 — Reproduction Guide

Task: MuJoCo Playground `PandaPickCube` (state observations, `impl="jax"`).
Teacher: Playground PPO. Student: ACT. Ablation: dataset size (25 / 50 / 100).

## 0. Environment setup

Tested on Ubuntu 22.04.5, AMD Ryzen 7 7800X3D, 30 GB RAM, RTX 4070 Ti SUPER (16 GB),
NVIDIA driver 580.178.04 (see `configs/hardware.txt`).

```bash
# Playground source at the pinned commit (needed for learning/train_jax_ppo.py)
git clone https://github.com/google-deepmind/mujoco_playground third_party/mujoco_playground
git -C third_party/mujoco_playground checkout d59156e099a8785dd58fbce2e25a6ee1f7b5800c

# Python 3.12 venv with exact versions from uv.lock
uv sync --python 3.12

# Smoke test (downloads mujoco_menagerie on first run); writes results/env_info.json
# and videos/smoke_{random,zero}_actions.mp4
.venv/bin/python code/s0_smoke_test.py
```

Key versions: Python 3.12.14, JAX 0.11.2 (cuda12 plugin), MuJoCo / MJX 3.16.0,
Brax 0.14.2, Playground 0.2.0 @ `d59156e`, PyTorch 2.9.0+cu126 (cuDNN 9.10).
Full list: `configs/env_freeze.txt`.

JAX and PyTorch share one venv and one GPU; `code/common/env_utils.py` sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MUJOCO_GL=egl` before importing JAX.

## 1. PPO teacher

```bash
# ~4 min each on an RTX 4070 Ti SUPER; one checkpoint every 1.64M env steps
.venv/bin/python code/s1_train_teacher.py --seed 0 --out checkpoints/teacher_seed0
.venv/bin/python code/s1_train_teacher.py --seed 1 --out checkpoints/teacher_seed1
# Upstream normalizer setting, for the diagnostic comparison only
.venv/bin/python code/s1_train_teacher.py --seed 0 --obs_std_eps 0 --out checkpoints/upstream_eps0_seed0

# Evaluate every checkpoint on the 100 held-out eval seeds, plot, render videos
.venv/bin/python code/s1_eval_teacher.py --run checkpoints/teacher_seed0
.venv/bin/python code/s1_eval_teacher.py --run checkpoints/teacher_seed1 --results results/teacher_seed1 --n_videos 0
.venv/bin/python code/s1_eval_teacher.py --run checkpoints/upstream_eps0_seed0 --results results/teacher_upstream_eps0 --n_videos 0
.venv/bin/python code/s1_plot_normalizer_fix.py
```

- Hyperparameters: Playground's tuned `manipulation_params.brax_ppo_config("PandaPickCube")`
  with two changes:
  1. `normalize_observations_std_eps` 0 → **0.01** (see deviations below).
  2. `num_evals` 4 → 16. Brax rounds each eval interval up to whole blocks of
     10 resets × 163,840 env steps, so both settings train for the same 24.58M env steps;
     16 evals simply gives one checkpoint per block (15 checkpoints).
- Policy: 4×32 SiLU MLP, tanh-Gaussian, observation normalization; evaluated
  deterministically (tanh of the mean). Value net 5×256.
- Seeds: `configs/seeds.yaml` (teacher seed 0; eval keys `split(PRNGKey(10000), 100)`).
- The final seed-0 checkpoint is committed at
  `checkpoints/teacher_seed0/checkpoints/000024576000` (1.1 MB). Every run's
  `progress.csv` / `train_config.json` is committed too.

| Run | Final success (100 held-out seeds) | 95% Wilson CI |
|---|---:|---:|
| `teacher_seed0` (**the teacher used downstream**) | 98% | [93.0, 99.4] |
| `teacher_seed1` | 99% | [94.6, 99.8] |
| `upstream_eps0_seed0` (std_eps = 0) | 0% | [0.0, 3.7] |

The seed-0 teacher solves episodes in 36 steps (0.72 s) on average. Its two failures are a
near-miss on a far target (5.9 cm, threshold 5 cm) and a box in the far corner that it
never lifts. Success per checkpoint is in `results/teacher/eval_by_checkpoint.csv`
(0% up to 11.5M steps, then 25%, 94%, … 98%).

## 2. Demonstrations

```bash
.venv/bin/python code/s2_collect_demos.py   # ~40 s
```

- Teacher: `checkpoints/teacher_seed0/checkpoints/000024576000`, deterministic.
- Collection reset keys: `split(PRNGKey(0), 200)`, which are disjoint from the eval keys (asserted).
- **Teacher success on the 200 collection seeds: 197/200 = 98.5%** (95% CI [95.7, 99.5]).
- **Filtering:** the baseline dataset is the first 100 successful episodes in seed-index
  order (collection indices 0–102, skipping 3 failures). Failures are kept in
  `data/demos_raw.npz` for analysis only.
- Every episode is the full 150 steps (no early termination). The teacher typically reaches
  the target by step 39 ± 13 and then holds the box there, which matters because success is
  judged at the final step.
- Stored per step: obs (66), executed action (8, teacher output in (−1, 1), never clipped),
  reward, done, qpos/qvel (t = 0..150), box/gripper position, and box–target distance. Stored
  per episode: target position, success, final distance, first-success step, reset key, and
  collection index.
- Alignment checks, both exact (0.0): re-querying the teacher on each stored obs reproduces
  the stored action, and open-loop replay of the stored actions reproduces the stored obs.
- Ablation subsets (`configs/datasets.yaml`): `subset_50` and `subset_25` are nested prefixes
  of `np.random.default_rng(0).permutation(100)`; `baseline_100` is all of them.
- `data/demos_baseline.npz` (6 MB) is committed. Collection is bitwise reproducible: the
  sha256 of (obs, actions) is recorded in `results/demos/collection_summary.json`.

## 3. Dataset inspection

```bash
.venv/bin/python code/s3_inspect_demos.py   # ~10 s
```

Outputs are in `results/demos/`: `dataset_stats.json`, `demo_episodes.csv`, `demo_frames.png`,
`trajectories.png`, `init_coverage.png`, `timing.png`, and `actions.png`. Videos are in `videos/`:
`expert_demo_ep000.mp4`, `expert_demos_grid.mp4`, and `expert_demo_slowest_ep088.mp4`.

| | collected | baseline dataset |
|---|---|---|
| trajectories | 200 | 100 (subsets: 50, 25) |
| length (steps) | 150 ± 0 | 150 ± 0 (15,000 transitions) |
| success fraction | 0.985 | 1.00 (filtered) |
| grasp / lift / first-success step | | 17.0 ± 5.5 / 22.2 ± 6.8 / 39.0 ± 13.0 |
| final box–target distance | | 0.9 ± 0.9 cm (max 3.5 cm) |

Episode length is fixed at 150 steps because the environment only terminates early on
failure. The phase timings are therefore the meaningful measure of how long a demonstration
takes. Grasp is the first step with the gripper within 2 cm of the box centre. Lift is the
first step with the box centre above 4.5 cm.

Observations, for the report discussion:

- **Consistency.** Every demonstration follows the same strategy: reach down with the gripper
  open, close it on the box at about step 17, lift about 5 steps later, then carry the
  box to the target. No demonstration re-grasps or drops the box, and none leaves the 5 cm
  success region once it has entered it. After success, the joint actions decay to about 0
  (mean |a| is 0.06 after success versus 0.42 before) and the arm holds still.
- **Coverage.** Box start positions and targets cover the whole randomization range
  (x 0.50–0.89, y −0.19–0.20, target z 0.23–0.43). The 25- and 50-demo subsets cover the same
  ranges. The three teacher failures are at the far corners (box x ≥ 0.85, |y| ≥ 0.15), and
  filtering on success removes them, so the dataset is slightly thinner exactly where the task
  is hardest.
- **Time to success.** Time to success depends mostly on how far the box is from the robot:
  the correlation between initial box x and first-success step is r = 0.73, compared with
  0.40 for the initial box–target distance. All demonstrations slower than 65 steps start with
  box x > 0.84.
- **Errors and inefficiencies.**
  - The carried box travels 1.18 ± 0.10 times the straight-line distance.
  - During the reach (steps 0–30) the actions for joints 1, 3, 5 and 7 jitter from step to
    step.
  - The teacher stops correcting once the box is inside the success radius: 14 of 100
    demonstrations end 2–3.5 cm from the target instead of near 0.
  - The gripper target sits at its limit 99% of the time, so after the first few steps only
    the sign of the gripper action matters. The teacher keeps outputting about −0.35, which
    the clip turns into a no-op.

## 4. ACT student

```bash
.venv/bin/python code/s4_train_act.py --dataset baseline_100 --seed 0   # ~14 min
.venv/bin/python code/s4_plot_training.py                               # results/act/training_baseline.png
# sanity check: overfit a single demonstration (~1.5 min)
.venv/bin/python code/s4_train_act.py --episodes 0 --steps 2000 --rollout_every 1000 --val_every 500 --wandb disabled
```

Each run writes `checkpoints/act_<dataset>_seed<seed>/`, which contains `policy.pt` (42 MB, not
committed), `progress.csv` and `train_config.json`. Curves are also logged to Weights &
Biases (project `csci645-hw2-act`; the run URL is in `train_config.json`). Everything needed for
the report is also in the local CSV and figures.

**Implementation.** `code/common/act.py` is our own PyTorch implementation of ACT (Zhao et
al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023;
reference code at github.com/tonyzhaozh/act). It keeps the reference design:

- a CVAE whose encoder reads [CLS, obs, a_t..a_{t+k−1}] and outputs the latent z;
- a transformer encoder–decoder policy whose decoder has k learned query tokens;
- the loss L1 + β·KL;
- z = 0 at test time, with temporal ensembling.

Changes for a state-only task:

- There is no image backbone. The 66-D observation is split into its 9 semantic groups (qpos,
  qvel, gripper pos, gripper rotation, box rotation, box−gripper, target−box, target rotation
  difference, ctrl−qpos). Each group is linearly projected to its own token, so the encoder
  attends over 10 tokens: [z, 9 observation tokens].
- Positional embeddings are learned and added once at the input. The reference adds them
  inside every attention layer.
- The layers are the stock `torch.nn.Transformer{Encoder,Decoder}` layers (post-norm, ReLU),
  and the model is smaller than the reference (see below).

| | |
|---|---|
| Observation | 66-D state (layout under *Environment facts*), standardized with training-set mean/std; std floored at 0.01 (affects 8 dims: finger qpos, two box-quaternion components, ctrl−qpos of joints 5–7 and gripper) |
| Action | 8-D delta joint/gripper targets in [−1, 1], standardized; predictions de-standardized and clipped to [−1, 1] |
| Chunk length k | 20 steps (0.4 s) |
| Architecture | hidden 256, 8 heads, FF 1024, dropout 0.1; 4 CVAE-encoder, 4 encoder, 4 decoder layers; latent 32; 10.6M parameters |
| Objective | masked L1 on standardized actions + 10 · KL(q(z∣o, a) ‖ N(0, I)); chunks running past step 150 are zero-padded and masked out of the loss and the CVAE encoder |
| Optimizer | AdamW, lr 1e-4 (constant), weight decay 1e-4, batch 256, no gradient clipping |
| Duration | fixed 30,000 gradient steps (512 epochs of the 15,000 baseline samples); the final weights are used, with no checkpoint selection |
| Sampling | (episode, t) pairs drawn uniformly with replacement |
| Inference | query every step, z = 0, temporal ensembling with weights exp(−0.01·i) (i = 0 is the oldest prediction) |
| Seed | `--seed 0` (initialization, batches, CVAE noise, dropout); seeds 1 and 2 are reserved for the ablation |

Monitoring during training, never used to select a model:

- Held-out L1: computed on the 97 successful collected episodes that are not in the baseline.
- Closed-loop success on 50 validation seeds (`val_base_seed` in `configs/seeds.yaml`). These
  seeds are disjoint from the collection and evaluation seeds, and this is asserted.
- Closed-loop success from the reset keys of 50 training episodes.

**Baseline run (`act_baseline_100_seed0`), figure `results/act/training_baseline.png`:**

- Train L1 keeps falling (0.11 at 1k steps, 0.030 at 30k). Held-out L1 flattens at about 0.079
  from roughly 8k steps on, so the model is fitting the 100 demonstrations more closely than it
  generalizes.
- KL collapses to about 1e-6, so the latent z carries essentially no information. This is
  expected: the teacher is a deterministic policy, so the demonstrations contain no
  multimodality for z to explain.
- Closed-loop success at 5k / 10k / … / 30k steps is 64 / 56 / 60 / 62 / 74 / 72% on the
  validation seeds and 84 / 88 / 94 / 94 / 94 / 94% from the training initial conditions.
- Of the 14 validation failures of the final model:
  - 8 are near-misses: the box is grasped, lifted and carried, but ends 5.1–9.2 cm from the
    target.
  - 2 never grasp the box.
  - 4 grasp the box but lose it or carry it elsewhere.

  Phase 5 runs the formal evaluation on the 100 held-out seeds.

**Checks.**

- *Overfitting one demonstration:* training on demonstration 0 only for 2,000 steps
  (`checkpoints/act_debug_ep0_seed0`) reproduces it in closed loop from its own reset key
  (success, final distance 0.3 cm) and gets 0% on the validation seeds, as expected.
- *Reproducibility:* two identical 500-step runs give bitwise-identical losses and weights.
  - Matmuls use TF32. Attention uses PyTorch's deterministic "math" kernel; the default
    memory-efficient kernel is nondeterministic.
  - The run also sets `torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG`.
- Reloading `policy.pt` and re-running the 50 validation seeds reproduces the 72% logged during
  training.

## 5. Evaluation

```bash
.venv/bin/python code/s5_evaluate.py --policies teacher act_baseline_100_seed0   # ~30 s
.venv/bin/python code/s5_plot_eval.py                                            # results/eval/baseline_vs_teacher.png
```

- Every policy is rolled out from the same 100 held-out reset keys, `split(PRNGKey(10000), 100)`.
  These are disjoint from the demonstration and validation keys. No training trajectory is
  replayed.
- The student runs with z = 0 and temporal ensembling (m = 0.01). The teacher runs
  deterministically.
- Success means the box centre is within 5 cm of the target at step 150. CIs are Wilson 95%.
- Outputs:
  - per-episode results in `results/eval/<name>/episodes.csv`;
  - summaries in `results/eval/<name>/summary.json` and `results/eval/summary.csv`;
  - full rollouts in `data/eval_rollouts/<name>.npz` (regenerable, not committed).
- Evaluation is deterministic: re-running it gives identical results.

| Condition | Number of demonstrations | Teacher success rate | Student success rate | Evaluation episodes |
|---|---:|---:|---:|---:|
| Baseline (ACT, seed 0) | 100 | 98% [93.0, 99.4] | **81%** [72.2, 87.5] | 100 |

Failures are categorized automatically by `code/common/analysis.py`. Grasp means the gripper
comes within 2 cm of the box centre; lift means the box centre rises above 4.5 cm.

| Category | Teacher | Student |
|---|---:|---:|
| no grasp | 1 | 5 |
| grasp, no lift | 0 | 1 |
| dropped (lifted, back on the table at the end) | 0 | 4 |
| near miss (holding, 5–10 cm from the target) | 1 | 5 |
| wrong place (holding, ≥ 10 cm from the target) | 0 | 4 |

Observations:

- The comparison is paired: on 81 seeds both policies succeed, on 17 only the teacher
  succeeds, and on 2 both fail. The student never succeeds where the teacher fails.
- When the student succeeds it is as fast as the teacher (first success at step 36.6 vs. 36.3),
  and its actions are slightly smoother than the teacher's (mean |Δa| per step 0.023 vs. 0.027),
  as expected from temporal ensembling.
- The student's failures are spread across all stages of the task, not concentrated in one.
  They cluster at **initial box positions close to the robot** (x < 0.6 m, mostly |y| > 0.05):
  success by initial box x is 59% (0.5–0.6 m), 88% (0.6–0.7), 97% (0.7–0.8) and 78% (0.8–0.9).
  The teacher's hard cases (Section 3) are instead at the far corners. Phase 7 analyses this.
- Videos are in `videos/`: `act_baseline_100_seed0_success_ep{000,002}.mp4` and one video per
  failure category, `act_baseline_100_seed0_failure_<category>_epXXX.mp4`. The teacher's videos
  on the same seeds are `teacher_{success,failure}_ep*.mp4`.

## 6. Data ablation (Option A: dataset quantity)

```bash
bash code/s6_run_ablation.sh   # 8 new training runs (~14 min each), evaluation, summary
```

**What varies.**
- *Independent variable:* the number of demonstrations, N = 25, 50 or 100.
- *Nested datasets:* `subset_25` ⊂ `subset_50` ⊂ `baseline_100`, all prefixes of one fixed
  permutation of the baseline (`configs/datasets.yaml`). Every condition therefore uses the
  same teacher, the same filtering, and identical trajectories.
- *Training seeds:* each N is trained with seeds 0, 1 and 2.

**What stays fixed.**
- Everything else in Section 4: architecture, normalization (statistics fit on each run's own
  training set), optimizer, and the **same 30,000 gradient steps**.
- A fixed number of steps means the smaller datasets get more passes over the data: 2048, 1024
  and 512 epochs for N = 25, 50 and 100.
- Evaluation is identical: the same 100 held-out seeds, temporal ensembling, and no checkpoint
  selection.

Outputs:
- `results/ablation/ablation_table.{md,csv}`, `ablation_runs.csv` and `ablation_stats.json`;
- the figure `results/ablation/ablation.png`;
- per-run evaluations under `results/eval/`;
- videos for training seed 0 of each N in `videos/act_<dataset>_seed0_*.mp4`.

| Condition | Number of demonstrations | Teacher success rate | Student success rate (mean ± std over 3 training seeds) | Per seed | Pooled [95% CI] | Evaluation episodes |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 100 | 98% [93.0, 99.4] | 82.3% ± 2.3 | 81% / 81% / 85% | 82.3% [77.6, 86.2] | 100 per seed (300 total) |
| Ablation (subset_50) | 50 | 98% [93.0, 99.4] | 56.0% ± 7.2 | 64% / 50% / 54% | 56.0% [50.3, 61.5] | 100 per seed (300 total) |
| Ablation (subset_25) | 25 | 98% [93.0, 99.4] | 16.7% ± 1.5 | 17% / 15% / 18% | 16.7% [12.9, 21.3] | 100 per seed (300 total) |

The teacher success rate is the same in every row because all datasets come from the same
teacher. It succeeded on 197 of 200 collection episodes (98.5%), and only successes were kept.

Findings:

- **More demonstrations help a lot, and the effect is far larger than training-seed noise.**
  - Success rises from 17% to 56% to 82% as N goes from 25 to 50 to 100.
  - Pooled pairwise Fisher exact tests, 300 episodes per condition: 25 vs 50, p ≈ 4e-24;
    50 vs 100, p ≈ 3e-12.
  - The std across training seeds is only 1.5–7 points.
  - The curve has not saturated at N = 100 (82% vs. the teacher's 98%), so more data would
    likely help further.
- **The models memorize equally well but generalize very differently.**
  - From the reset keys of their own training episodes, all students succeed 96–97% of the time
    regardless of N.
  - Lower N gives a *lower* final train L1 (0.018 / 0.023 / 0.031 for N = 25 / 50 / 100)
    but a *higher* L1 on the 97 unseen teacher demonstrations (0.152 / 0.112 / 0.080).
  - So the gap is entirely in generalizing to new initial conditions: 25 demonstrations cover
    the 5-D space of box position and target position too sparsely.
- **The type of failure changes with N.** Failures are pooled over the 3 seeds:
  - N = 25: 131 of 250 failures are *no grasp*. The arm does not even reach an unseen box
    position correctly.
  - N = 50: grasping mostly works (31 no-grasp failures), and *near misses* (box carried to
    within 5–10 cm of the target) become the largest category, 65 of 132.
  - N = 100: the remaining 53 failures are spread over near miss (20), no grasp (12), dropped
    (10) and wrong place (10).

  More data first fixes reaching and grasping, then placement precision.
- **Overfitting with more epochs.** For N = 25 and 50, held-out L1 reaches its minimum around
  5–10k steps and then rises slightly (by about 0.005). For N = 100 it stays flat. Early
  stopping would therefore help small datasets a little, but the effect is far too small to
  explain the success gap.
- The 50 validation seeds used during training rank the conditions identically:
  - 16 / 14 / 18% for N = 25;
  - 42 / 36 / 44% for N = 50;
  - 72 / 72 / 56% for N = 100.

  These are lower than on the 100 evaluation seeds, which reflects sampling noise between two
  random seed sets.

Limitations:

- Only one draw of the nested subsets was used, so the variance from *which* demonstrations are
  selected is not measured. Only training-seed variance is.
- Hyperparameters were set once, for N = 100, and not re-tuned per N.
- Holding the gradient steps fixed confounds dataset size with the number of epochs. The
  held-out L1 curves suggest this matters little.
- All conclusions are for one deterministic PPO teacher and one task.

## Environment facts (from `results/env_info.json`)

- Observation: 66-D flat vector — qpos(16: 9 robot + 7 box freejoint), qvel(15),
  gripper pos(3), gripper rotmat[3:](6), box rotmat[3:](6), box−gripper(3),
  target−box(3), target−box rotmat diff(6), ctrl−robot qpos(8).
- Action: 8-D in [-1, 1]; delta position control `ctrl ← clip(ctrl + 0.04·a)`
  for 7 arm joints + gripper (gripper ctrl range [0, 0.04]).
- Control at 50 Hz (ctrl_dt 0.02 s, 4 × 0.005 s physics substeps), 150-step episodes.
  `done` fires only if the box leaves bounds or the sim produces NaNs.
- Reset: box xy uniform ±0.2 m around (0.7, 0); target uniform ±0.2 m in xy and
  0.2–0.4 m above the box's initial height.
- Success (ours, `env_utils.is_success`): box centre within 0.05 m of the target at
  the final step. The box is 4 × 4 × 6 cm.

## Notes / deviations from upstream

- **PPO observation-normalizer std floor (`normalize_observations_std_eps=0.01`).** With
  Playground's settings on this software stack, PPO collapses for both seeds we tried
  (0 and 1). Eval return peaks at ~5M steps, the value loss then jumps from ~1 to ~1e14, and
  no checkpoint ever succeeds. We traced this to observation dim 11 (box z): its running std
  is ~0.003 while the box rests on the table, so once the policy starts lifting the box
  (z ≈ 0.3–0.4 m), that input reaches ~100σ. Rollouts from the last pre-collapse checkpoint
  contain no NaNs, reward outliers, or exploding velocities, so the simulation is not the
  cause. Flooring the std at 0.1 fixes it: return rises smoothly to ~1400 and success
  reaches 98–99% (figure: `results/teacher/normalizer_fix.png`).

- Brax is installed from GitHub commit `6d03937` instead of PyPI 0.14.2: the release still
  calls `jax.device_put_replicated`, which was removed in JAX 0.10.
- `code/common/teacher.py` re-implements Brax's `ppo.checkpoint.load_config`: at this commit
  `save()` skips `None` kernel-init entries but `load_config()` looks them up
  (`KeyError: None`), and `observation_size` is serialized as a dict.

- Playground's default backend for this task is `warp`; we use `impl="jax"` as in the
  homework README.
- Upstream `reset()` does not call `mj_forward`, so at t = 0 `xpos`/`site_xpos` and the
  observation fields derived from them are all zero; they are correct from step 1 on.
  We keep this unchanged because the PPO teacher is trained on the same observations.
- **GPU determinism:** by default MJX rollouts on the GPU are not reproducible. Scatter-add
  summation order varies, and the resulting ~1e-6 differences grow chaotically through
  contacts, so closed-loop reruns of the teacher diverged by up to ~10 in obs.
  `env_utils.py` therefore sets `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. With it,
  batched rollouts are bitwise identical across runs and across batch sizes (100 vs 200);
  a batch of 1 still compiles differently. All evaluation and collection numbers use this
  flag. PPO training ran before the flag was added, so retraining yields a slightly
  different teacher; the exact teacher checkpoint is committed instead.
- GPU float32 matmuls (TF32) are batch-shape dependent at the ~1e-3 level, so policy
  re-queries in checks use the same batch shape as the original rollout.
