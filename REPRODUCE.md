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

- Playground's default backend for this task is `warp`; we use `impl="jax"` as in the
  homework README.
- Upstream `reset()` does not call `mj_forward`, so at t = 0 `xpos`/`site_xpos` and the
  observation fields derived from them are all zero; they are correct from step 1 on.
  We keep this unchanged because the PPO teacher is trained on the same observations.
- Simulation is bitwise deterministic on our GPU: replaying the same actions from the
  same reset key reproduces the trajectory exactly (max |Δobs| = 0 after 200 steps).
