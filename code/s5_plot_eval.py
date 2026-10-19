"""Phase 5 figure: teacher vs. ACT student on the held-out evaluation seeds.
Writes results/eval/<out>.png.

  python code/s5_plot_eval.py --student act_baseline_100_seed0
"""

import argparse
import json

from common import plot_style
from common import analysis as an

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.env_utils import ROOT

CATEGORY_LABELS = {"no_grasp": "no grasp", "grasp_no_lift": "grasp, no lift", "dropped": "dropped",
                   "near_miss": "near miss (5-10 cm)", "wrong_place": "wrong place (>10 cm)",
                   "out_of_bounds": "out of bounds"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="act_baseline_100_seed0")
    p.add_argument("--out", default="baseline_vs_teacher")
    args = p.parse_args()

    res = ROOT / "results" / "eval"
    names = [("teacher", "PPO teacher"), (args.student, "ACT student (100 demos)")]
    summaries = [json.loads((res / n / "summary.json").read_text()) for n, _ in names]
    student = pd.read_csv(res / args.student / "episodes.csv")

    plot_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [0.7, 1.1, 1]})

    ax = axes[0]
    for i, ((_, label), s, color) in enumerate(zip(names, summaries, plot_style.SERIES)):
        rate = 100 * s["success_rate"]
        lo, hi = (100 * v for v in s["success_ci95"])
        ax.bar(i, rate, width=0.6, color=color)
        ax.errorbar(i, rate, yerr=[[rate - lo], [hi - rate]], color=plot_style.TEXT, capsize=5, lw=1.5)
        ax.text(i, 3, f"{rate:.0f}%", ha="center", color=plot_style.SURFACE, fontweight="bold")
    ax.set_xticks(range(len(names)), [label.replace(" (", "\n(") for _, label in names])
    ax.set(title="Success, 100 held-out seeds (95% CI)", ylabel="success rate (%)", ylim=(0, 105))
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    cats = an.CATEGORIES[1:]
    y = np.arange(len(cats))
    for k, ((_, label), s, color) in enumerate(zip(names, summaries, plot_style.SERIES)):
        counts = [s["categories"][c] for c in cats]
        ax.barh(y + (k - 0.5) * 0.38, counts, height=0.36, color=color, label=label)
    ax.set_yticks(y, [CATEGORY_LABELS[c] for c in cats])
    ax.invert_yaxis()
    ax.set(title="Failures by category", xlabel="# episodes (of 100)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    ax = axes[2]
    ok = student["success"]
    ax.scatter(student.loc[ok, "box_x0"], student.loc[ok, "box_y0"], s=22, color=plot_style.GOOD,
               edgecolors=plot_style.SURFACE, linewidths=0.8, label="student success")
    ax.scatter(student.loc[~ok, "box_x0"], student.loc[~ok, "box_y0"], s=55, marker="x",
               color=plot_style.BAD, linewidths=2, label="student failure")
    ax.set(title="Student outcome by initial box position", xlabel="box x (m)", ylabel="box y (m)",
           xlim=(0.48, 0.92), ylim=(-0.22, 0.22))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    out = res / f"{args.out}.png"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
