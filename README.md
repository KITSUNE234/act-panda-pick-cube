# CSCI 645 HW2: Imitation Learning on PandaPickCube

Yixin Liu. The write-up is `report.pdf`.

- Environment: MuJoCo Playground `PandaPickCube`, state observations, `impl="jax"`.
- Teacher: PPO (Brax). Student: ACT (PyTorch).
- Ablation: Option A, 25 / 50 / 100 demonstrations.

## Layout

```text
code/          all scripts, run in the order s0 ... s7; shared modules in code/common/
configs/       seeds, dataset subsets, hardware and package versions
checkpoints/   teacher checkpoint (seed 0), training curves and configs of every run
data/          demos_baseline.npz (100 successful teacher demonstrations)
results/       CSV/JSON summaries and figures, one folder per stage
videos/        expert demonstrations and student rollouts (success and failure)
```

ACT student weights (42 MB each) are not included. Run steps 4 and 6 below to recreate them; training is bitwise deterministic.

## Setup

Tested on Ubuntu 22.04, an RTX 4070 Ti SUPER (driver 580) and CUDA 12.

```bash
uv sync --python 3.12            # exact versions from uv.lock
git clone https://github.com/google-deepmind/mujoco_playground third_party/mujoco_playground
git -C third_party/mujoco_playground checkout d59156e099a8785dd58fbce2e25a6ee1f7b5800c
```

Main versions:

- Python 3.12.14, JAX 0.11.2, MuJoCo/MJX 3.16.0, PyTorch 2.9.0+cu126.
- Playground at commit `d59156e`.
- Brax at commit `6d03937`. PyPI brax 0.14.2 calls `jax.device_put_replicated`, which was removed in JAX 0.10.

The full package list is in `configs/env_freeze.txt`.

ACT training logs to Weights & Biases by default. Without an account, run `export WANDB_MODE=disabled` first.

## Reproduce

```bash
.venv/bin/python code/s0_smoke_test.py                       # environment check

# 1. PPO teacher (~4 min). The committed checkpoint can be used instead.
.venv/bin/python code/s1_train_teacher.py --seed 0 --out checkpoints/teacher_seed0
.venv/bin/python code/s1_eval_teacher.py --run checkpoints/teacher_seed0

# 2-3. Collect demonstrations (~40 s), then plot dataset statistics and render videos
.venv/bin/python code/s2_collect_demos.py
.venv/bin/python code/s3_inspect_demos.py

# 4-5. Train the ACT baseline (~14 min) and evaluate it on the 100 held-out seeds
.venv/bin/python code/s4_train_act.py --dataset baseline_100 --seed 0
.venv/bin/python code/s4_plot_training.py
.venv/bin/python code/s5_evaluate.py --policies teacher act_baseline_100_seed0

# 6. Ablation: 3 dataset sizes x 3 training seeds, evaluation and summary (~2 h)
bash code/s6_run_ablation.sh

# 7. Failure analysis: teacher takeover and distance to training data, plus figures
.venv/bin/python code/s7_failure_analysis.py
.venv/bin/python code/s7_plot_failure.py
```

Seeds are listed in `configs/seeds.yaml`:

- Teacher: training seed 0.
- Demonstration collection: `split(PRNGKey(0), 200)`.
- Validation: `split(PRNGKey(20000), 50)`.
- Evaluation: `split(PRNGKey(10000), 100)`.
- ACT: training seeds 0, 1, 2.

## Deviations from upstream

- **PPO normalizer.** With Playground's default PPO config, training diverged on this setup. The std of the box height is about 0.003 while the box rests on the table, so lifting the box gives inputs of about 100σ. `s1_train_teacher.py` sets `normalize_observations_std_eps=0.01`; see `results/teacher/normalizer_fix.png`.
- **GPU determinism.** `code/common/env_utils.py` sets `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. Without it, MJX rollouts on the GPU are not reproducible. With it, demonstration collection and evaluation are bitwise reproducible.
- **Checkpoint loading.** `code/common/teacher.py` re-implements Brax's checkpoint `load_config`, because the version at this commit fails with `KeyError: None`.
