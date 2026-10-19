"""Phase 4 figure: ACT training curves (loss terms, held-out L1, closed-loop
validation success) for one or more runs. Writes results/act/<out>.png.

  python code/s4_plot_training.py --runs act_baseline_100_seed0 --out training_baseline
"""

import argparse

from common import plot_style

import matplotlib.pyplot as plt
import pandas as pd

from common.env_utils import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["act_baseline_100_seed0"])
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default="training_baseline")
    args = p.parse_args()
    labels = args.labels or args.runs

    plot_style.apply()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for run, label, color in zip(args.runs, labels, plot_style.SERIES):
        df = pd.read_csv(ROOT / "checkpoints" / run / "progress.csv")
        k = df["step"] / 1e3
        axes[0].plot(k, df["l1"], color=color, label=label)
        axes[1].plot(k, df["kl"], color=color, label=label)
        val = df.dropna(subset=["val_l1"])
        axes[2].plot(val["step"] / 1e3, val["val_l1"], color=color, marker="o", markersize=3, label=label)
        roll = df.dropna(subset=["val_success"])
        axes[3].plot(roll["step"] / 1e3, roll["val_success"] * 100, color=color, marker="o", label=label)
    axes[0].set(title="Train L1 (normalized actions)", xlabel="gradient steps (k)", ylabel="L1", yscale="log")
    axes[1].set(title="Train KL (weight 10)", xlabel="gradient steps (k)", ylabel="KL", yscale="log")
    axes[2].set(title="Held-out demo L1 (z = 0)", xlabel="gradient steps (k)", ylabel="L1")
    axes[3].set(title="Closed-loop success, 50 val seeds", xlabel="gradient steps (k)",
                ylabel="success (%)", ylim=(-2, 102))
    if len(args.runs) > 1:
        axes[3].legend(loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "act" / f"{args.out}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
