"""Phase 1 diagnostic figure: PPO with upstream normalizer settings (std_eps=0)
vs. our fix (std_eps=0.01), same seed. Writes results/teacher/normalizer_fix.png.

  python code/s1_plot_normalizer_fix.py
"""

from common import plot_style

import matplotlib.pyplot as plt
import pandas as pd

from common.env_utils import ROOT

RUNS = [  # (label, training run, held-out eval results)
    ("upstream (std_eps = 0)", "checkpoints/upstream_eps0_seed0", "results/teacher_upstream_eps0"),
    ("ours (std_eps = 0.01)", "checkpoints/teacher_seed0", "results/teacher"),
]


def main():
    plot_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for (label, run, res), color in zip(RUNS, plot_style.SERIES):
        prog = pd.read_csv(ROOT / run / "progress.csv")
        evals = pd.read_csv(ROOT / res / "eval_by_checkpoint.csv")
        axes[0].plot(prog["step"] / 1e6, prog["eval/episode_reward"], color=color, marker="o", label=label)
        train = prog.dropna(subset=["training/v_loss"])
        axes[1].plot(train["step"] / 1e6, train["training/v_loss"], color=color, marker="o", label=label)
        axes[2].plot(evals["step"] / 1e6, evals["success_rate"] * 100, color=color, marker="o", label=label)
    axes[0].set(title="Eval return", xlabel="env steps (M)", ylabel="episode return")
    axes[1].set(title="Value loss (log scale)", xlabel="env steps (M)", ylabel="v_loss", yscale="log")
    axes[2].set(title="Success on 100 held-out seeds", xlabel="env steps (M)",
                ylabel="success rate (%)", ylim=(-2, 102))
    axes[0].legend(loc="upper left")
    fig.suptitle("PPO teacher, seed 0: observation-normalizer std floor", color=plot_style.TEXT, y=1.02)
    fig.tight_layout()
    out = ROOT / "results" / "teacher" / "normalizer_fix.png"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
