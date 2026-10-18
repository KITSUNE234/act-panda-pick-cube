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
near-miss on a far target (7.7 cm, threshold 5 cm) and a box in the far corner that it
never lifts. Success per checkpoint is in `results/teacher/eval_by_checkpoint.csv`
(0% up to 11.5M steps, then 25%, 94%, … 98%).

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
- Simulation is bitwise deterministic on our GPU: replaying the same actions from the
  same reset key reproduces the trajectory exactly (max |Δobs| = 0 after 200 steps).
