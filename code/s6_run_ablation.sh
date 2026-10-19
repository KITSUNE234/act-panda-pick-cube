#!/usr/bin/env bash
# Phase 6 (Option A, dataset quantity): train ACT on the nested 25/50/100-demo
# datasets with 3 training seeds each, identical hyperparameters and 30k gradient
# steps, then evaluate every run on the 100 held-out seeds. Runs that already
# have a policy.pt are skipped. ~14 min per run on an RTX 4070 Ti SUPER.
#
#   bash code/s6_run_ablation.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../logs
runs=()
for seed in 0 1 2; do
  for dataset in subset_25 subset_50 baseline_100; do
    name="act_${dataset}_seed${seed}"
    runs+=("$name")
    if [[ -f "../checkpoints/$name/policy.pt" ]]; then
      echo "skip $name (exists)"
      continue
    fi
    echo "train $name"
    ../.venv/bin/python s4_train_act.py --dataset "$dataset" --seed "$seed" > "../logs/$name.log" 2>&1
  done
done
# Videos only for training seed 0 of each dataset size (seeds 1-2 are numbers only).
../.venv/bin/python s5_evaluate.py --policies $(printf '%s\n' "${runs[@]}" | grep 'seed0$')
../.venv/bin/python s5_evaluate.py --no_videos --policies $(printf '%s\n' "${runs[@]}" | grep -v 'seed0$')
../.venv/bin/python s6_plot_ablation.py
