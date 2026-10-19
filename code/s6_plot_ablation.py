"""Phase 6 (Option A): summarize the dataset-quantity ablation.

Reads results/eval/summary.csv and results/eval/<run>/episodes.csv for the runs
act_<dataset>_seed<s> (default inference: temporal ensembling, m = 0.01) and
writes
  results/ablation/ablation_runs.csv    one row per (dataset size, training seed)
  results/ablation/ablation_table.csv   one row per dataset size (pooled over seeds)
  results/ablation/ablation_table.md    the report table
  results/ablation/ablation_stats.json  pooled pairwise Fisher exact tests
  results/ablation/ablation.png

  python code/s6_plot_ablation.py
"""

import itertools
import json

from common import plot_style
from common import analysis as an
from common import rollout as ro

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common.env_utils import ROOT

DATASETS = [("subset_25", 25), ("subset_50", 50), ("baseline_100", 100)]
SEEDS = [0, 1, 2]
RES = ROOT / "results" / "ablation"
CATEGORY_LABELS = {"no_grasp": "no grasp", "grasp_no_lift": "grasp, no lift", "dropped": "dropped",
                   "near_miss": "near miss", "wrong_place": "wrong place", "out_of_bounds": "out of bounds"}


def load_runs():
    summary = pd.read_csv(ROOT / "results" / "eval" / "summary.csv").set_index("name")
    rows = []
    for (dataset, n), seed in itertools.product(DATASETS, SEEDS):
        name = f"act_{dataset}_seed{seed}"
        if name not in summary.index:
            continue
        ep = pd.read_csv(ROOT / "results" / "eval" / name / "episodes.csv")
        prog = pd.read_csv(ROOT / "checkpoints" / name / "progress.csv")
        last = prog.dropna(subset=["val_success"]).iloc[-1]
        rows.append({"name": name, "dataset": dataset, "n_demos": n, "train_seed": seed,
                     "eval_success": ep["success"].mean(), "eval_n": len(ep),
                     "eval_k": int(ep["success"].sum()),
                     "final_dist_median": ep["final_dist"].median(),
                     "train_l1_final": prog["l1"].iloc[-1], "val_l1_final": last["val_l1"],
                     "val_success_final": last["val_success"],
                     "train_init_success_final": last["train_success"],
                     **{f"n_{c}": int((ep["category"] == c).sum()) for c in an.CATEGORIES}})
    return pd.DataFrame(rows)


def pooled_table(runs, teacher):
    out = []
    for dataset, n in DATASETS:
        r = runs[runs.dataset == dataset]
        if r.empty:
            continue
        k, total = int(r.eval_k.sum()), int(r.eval_n.sum())
        lo, hi = ro.wilson_ci(k, total)
        out.append({"dataset": dataset, "n_demos": n, "n_seeds": len(r),
                    "per_seed_success": " / ".join(f"{s:.0%}" for s in r.eval_success),
                    "mean_success": r.eval_success.mean(), "std_success": r.eval_success.std(ddof=1),
                    "pooled_k": k, "pooled_n": total, "pooled_ci95_lo": lo, "pooled_ci95_hi": hi,
                    "teacher_success": teacher["success_rate"],
                    "train_init_success_mean": r.train_init_success_final.mean(),
                    "val_l1_mean": r.val_l1_final.mean(), "train_l1_mean": r.train_l1_final.mean(),
                    **{f"n_{c}": int(r[f"n_{c}"].sum()) for c in an.CATEGORIES}})
    return pd.DataFrame(out)


def markdown(table, teacher):
    lines = ["| Condition | Number of demonstrations | Teacher success rate | Student success rate "
             "(mean ± std over 3 training seeds) | Per seed | Pooled [95% CI] | Evaluation episodes |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    lo_t, hi_t = teacher["success_ci95"]
    for _, r in table.iloc[::-1].iterrows():
        cond = "Baseline" if r.dataset == "baseline_100" else f"Ablation ({r.dataset})"
        lines.append(
            f"| {cond} | {r.n_demos} | {100 * r.teacher_success:.0f}% [{100 * lo_t:.1f}, {100 * hi_t:.1f}] | "
            f"{100 * r.mean_success:.1f}% ± {100 * r.std_success:.1f} | {r.per_seed_success} | "
            f"{100 * r.pooled_k / r.pooled_n:.1f}% [{100 * r.pooled_ci95_lo:.1f}, {100 * r.pooled_ci95_hi:.1f}] | "
            f"{r.pooled_n // r.n_seeds} per seed ({r.pooled_n} total) |")
    return "\n".join(lines) + "\n"


def pairwise_tests(table):
    tests = {}
    for (_, a), (_, b) in itertools.combinations(table.iterrows(), 2):
        ct = [[a.pooled_k, a.pooled_n - a.pooled_k], [b.pooled_k, b.pooled_n - b.pooled_k]]
        odds, p = stats.fisher_exact(ct)
        tests[f"{a.dataset} vs {b.dataset}"] = {
            "success": [a.pooled_k / a.pooled_n, b.pooled_k / b.pooled_n],
            "difference": b.pooled_k / b.pooled_n - a.pooled_k / a.pooled_n,
            "fisher_exact_p": p}
    return tests


def plot(runs, table, teacher):
    plot_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    xs = {n: i for i, (_, n) in enumerate(DATASETS)}

    ax = axes[0]
    ax.axhline(100 * teacher["success_rate"], color=plot_style.TEXT_2, lw=1.2, ls="--")
    ax.text(0.5, 100 * teacher["success_rate"] + 1, "teacher (held-out seeds)", color=plot_style.TEXT_2,
            ha="center", va="bottom", fontsize=8)
    for seed, color in zip(SEEDS, plot_style.SERIES):
        r = runs[runs.train_seed == seed]
        ax.scatter([xs[n] + 0.12 * (seed - 1) for n in r.n_demos], 100 * r.eval_success, s=26,
                   color=color, edgecolors=plot_style.SURFACE, linewidths=0.8, zorder=3,
                   label=f"training seed {seed}")
    ax.plot([xs[n] for n in table.n_demos], 100 * table.train_init_success_mean, color=plot_style.TEXT_2,
            lw=1.5, ls=":", marker="s", markersize=6, label="student, training initial conditions")
    pooled = 100 * table.pooled_k / table.pooled_n
    ax.errorbar([xs[n] for n in table.n_demos], pooled,
                yerr=[pooled - 100 * table.pooled_ci95_lo, 100 * table.pooled_ci95_hi - pooled],
                color=plot_style.TEXT, lw=2, capsize=6, marker="o", markersize=7, zorder=2,
                label="pooled over seeds (95% CI)")
    ax.set_xticks(range(len(DATASETS)), [str(n) for _, n in DATASETS])
    ax.set(title="Student success, 100 held-out seeds", xlabel="number of demonstrations (log scale)",
           ylabel="success rate (%)", ylim=(0, 106), xlim=(-0.5, len(DATASETS) - 0.5))
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    cats = an.CATEGORIES[1:]
    width = 0.6
    for i, (_, r) in enumerate(table.iterrows()):
        bottom = 0.0
        for c, color in zip(cats, plot_style.SERIES):
            v = r[f"n_{c}"] / r.n_seeds
            if v > 0:
                ax.bar(i, v, width, bottom=bottom, color=color, edgecolor=plot_style.SURFACE, linewidth=2)
            bottom += v
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in plot_style.SERIES[:len(cats)]]
    ax.legend(handles, [CATEGORY_LABELS[c] for c in cats], loc="upper right", fontsize=8)
    ax.set_xticks(range(len(table)), [str(n) for n in table.n_demos])
    ax.set(title="Failures per 100 episodes (mean over seeds)", xlabel="number of demonstrations",
           ylabel="# failed episodes")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax = axes[2]
    for (dataset, n), color in zip(DATASETS, plot_style.SERIES):
        curves = []
        for seed in SEEDS:
            path = ROOT / "checkpoints" / f"act_{dataset}_seed{seed}" / "progress.csv"
            if path.exists():
                curves.append(pd.read_csv(path).dropna(subset=["val_l1"]).set_index("step")["val_l1"])
        if not curves:
            continue
        df = pd.concat(curves, axis=1)
        ax.plot(df.index / 1e3, df.mean(axis=1), color=color, label=f"{n} demos")
        ax.fill_between(df.index / 1e3, df.min(axis=1), df.max(axis=1), color=color, alpha=0.2, lw=0)
    ax.set(title="Held-out demo L1 (97 unseen demos; band = seed range)", xlabel="gradient steps (k)",
           ylabel="L1 (normalized actions)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(RES / "ablation.png")
    plt.close(fig)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    teacher = json.loads((ROOT / "results" / "eval" / "teacher" / "summary.json").read_text())
    runs = load_runs()
    table = pooled_table(runs, teacher)
    runs.to_csv(RES / "ablation_runs.csv", index=False)
    table.to_csv(RES / "ablation_table.csv", index=False)
    (RES / "ablation_table.md").write_text(markdown(table, teacher))
    tests = pairwise_tests(table)
    (RES / "ablation_stats.json").write_text(json.dumps(tests, indent=2))
    plot(runs, table, teacher)
    print(runs[["name", "eval_success", "train_init_success_final", "val_success_final",
                "val_l1_final", "train_l1_final"]].round(3).to_string(index=False))
    print(markdown(table, teacher))
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
