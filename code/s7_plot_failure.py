"""Phase 7 figures (run after s7_failure_analysis.py). Writes results/failure/
  failure_modes.png       box tilt near the robot, distance to the training states, teacher takeover
  exec_modes.png          temporal ensembling vs. open-loop chunk execution
and videos/failure_tilt_student_vs_teacher_epXXX.mp4: student (left) and teacher
(right) from the same held-out seed, for the first near-robot dropped episode.

  python code/s7_plot_failure.py
"""

import json

from common import plot_style
from common import env_utils as eu
from common import rollout as ro

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd

from s7_failure_analysis import BASELINE_RUNS, NEAR_ROBOT_X, RES, ROLLOUTS, box_tilt_deg

ROOT = eu.ROOT
GRASP_PHASE = ["no_grasp", "grasp_no_lift", "dropped", "wrong_place", "out_of_bounds"]


def band(ax, x, y, color, label, ls="-"):
    """Median line with an interquartile band; y is [episodes, steps]."""
    q1, med, q3 = np.percentile(y, [25, 50, 75], axis=0)
    ax.plot(x, med, color=color, ls=ls, label=f"{label} (n={len(y)})")
    ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)


def failure_modes(env, ep, cs, tk, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    teacher = np.load(ROLLOUTS / "teacher.npz")
    tilt_t = box_tilt_deg(teacher["qpos"], env)
    steps = np.arange(61)

    ax = axes[0]
    tilts, labels = {"ok": [], "fail": []}, []
    for run in BASELINE_RUNS:
        tilt_s = box_tilt_deg(np.load(ROLLOUTS / f"{run}.npz")["qpos"], env)
        e = ep[(ep.run == run) & ep.near_robot]
        tilts["ok"].append(tilt_s[e.episode[e.success]][:, steps])
        tilts["fail"].append(tilt_s[e.episode[e.category.isin(GRASP_PHASE)]][:, steps])
    near_eps = ep[(ep.run == BASELINE_RUNS[0]) & ep.near_robot].episode
    band(ax, steps, tilt_t[near_eps][:, steps], plot_style.TEXT_2, "teacher", ls="--")
    band(ax, steps, np.concatenate(tilts["ok"]), plot_style.SERIES[0], "student, success")
    band(ax, steps, np.concatenate(tilts["fail"]), plot_style.SERIES[1], "student, grasp-phase failure")
    ax.axvspan(10, 17, color=plot_style.GRID, alpha=0.6, lw=0)
    ax.text(13.5, 92, "grasp + lift", ha="center", va="top", fontsize=8, color=plot_style.TEXT_2)
    ax.set(title=f"Box tilt, initial box x < {NEAR_ROBOT_X} m (3 seeds)", xlabel="control step",
           ylabel="tilt of box from upright (deg)", ylim=(0, 95), xlim=(0, 60))

    ax = axes[1]
    base = cs[cs.run.isin(BASELINE_RUNS)].groupby("step")[
        ["teacher_median", "student_success_median", "student_failure_median"]].mean()
    ax.plot(base.index, base.teacher_median, color=plot_style.TEXT_2, ls="--", label="teacher, same seeds")
    ax.plot(base.index, base.student_success_median, color=plot_style.SERIES[0], label="student, success")
    ax.plot(base.index, base.student_failure_median, color=plot_style.SERIES[1], label="student, failure")
    ax.axhline(summary["off_distribution_threshold"], color=plot_style.TEXT_2, lw=1, ls=":")
    ax.text(148, summary["off_distribution_threshold"] * 1.12, "teacher 99th pct.", ha="right",
            fontsize=8, color=plot_style.TEXT_2)
    ax.set_yscale("log")
    ax.set(title="Distance to nearest training state (median)", xlabel="control step",
           ylabel="RMS z-score to nearest demo state", xlim=(0, 150))
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    fails = ep[~ep.success & ep.run.isin(tk.run.unique())]
    switch = sorted(tk.switch_step.unique())
    groups = [("near miss", fails.category == "near_miss", plot_style.SERIES[2]),
              ("grasp-phase failure", fails.category.isin(GRASP_PHASE), plot_style.SERIES[1])]
    for label, m, color in groups:
        rescued = [100 * fails[m][f"takeover_{s}"].mean() for s in switch[:-1]]
        ax.plot(switch[:-1], rescued, color=color, marker="o", label=f"{label} (n={m.sum()})")
    ax.axvspan(10, 17, color=plot_style.GRID, alpha=0.6, lw=0)
    ax.text(13.5, 103, "grasp + lift", ha="center", va="top", fontsize=8, color=plot_style.TEXT_2)
    ax.set(title="Teacher takes over at step s: student failures rescued", xlabel="takeover step s",
           ylabel="% of student failures that succeed", ylim=(0, 105), xlim=(-3, 103))
    ax.legend(loc="center right", fontsize=8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "failure_modes.png")
    plt.close(fig)


def exec_modes(ex, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(ex))[::-1]
    colors = [plot_style.SERIES[0] if m.startswith("ensemble") else plot_style.SERIES[1] for m in ex["mode"]]
    ax = axes[0]
    pct = 100 * ex.success
    ax.errorbar(pct, y, xerr=[pct - 100 * ex.ci95_lo, 100 * ex.ci95_hi - pct], fmt="none",
                ecolor=plot_style.TEXT_2, capsize=4, lw=1.2)
    ax.scatter(pct, y, color=colors, s=60, zorder=3, edgecolors=plot_style.SURFACE, linewidths=1)
    for yi, (_, r) in zip(y, ex.iterrows()):
        ax.text(100 * r.ci95_hi + 1, yi, f"{100 * r.success:.1f}%  (p={r.fisher_p_vs_default:.2g})",
                va="center", fontsize=8, color=plot_style.TEXT_2)
    ax.set_yticks(y, ex["mode"])
    ax.set(title="Success, 3 seeds x 100 held-out episodes (95% CI)", xlabel="success rate (%)",
           xlim=(60, 100))
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.barh(y, ex.mean_abs_arm_action_change, color=colors, height=0.6)
    ax.axvline(summary["teacher_mean_abs_arm_action_change"], color=plot_style.TEXT_2, ls="--", lw=1.2)
    ax.text(summary["teacher_mean_abs_arm_action_change"], y.max() + 0.6, "teacher", ha="center",
            fontsize=8, color=plot_style.TEXT_2)
    for yi, (_, r) in zip(y, ex.iterrows()):
        ax.text(0.037, yi, f"{r.gripper_flips_per_episode:.1f} gripper flips/ep",
                va="center", fontsize=8, color=plot_style.TEXT_2)
    ax.set_yticks(y, [""] * len(y))
    ax.set(title="Action smoothness", xlabel="mean |a(t+1) - a(t)|, arm dims", xlim=(0, 0.052),
           ylim=(-0.6, len(ex) - 0.1))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(RES / "exec_modes.png")
    plt.close(fig)


def side_by_side_video(env, ep):
    e = ep[(ep.run == BASELINE_RUNS[0]) & ep.near_robot & (ep.category == "dropped")].iloc[0]
    i = int(e.episode)
    student = np.load(ROLLOUTS / f"{BASELINE_RUNS[0]}.npz")
    teacher = np.load(ROLLOUTS / "teacher.npz")
    frames = [ro.render_episode(env, d["qpos"][i], d["target_pos"][i], every=2) for d in (student, teacher)]
    path = ROOT / "videos" / f"failure_tilt_student_vs_teacher_ep{i:03d}.mp4"
    media.write_video(path, np.concatenate(frames, axis=2), fps=eu.video_fps(env, every=2))
    return path


def main():
    plot_style.apply()
    env = eu.load_env()
    ep = pd.read_csv(RES / "episodes.csv")
    cs = pd.read_csv(RES / "covariate_shift.csv")
    tk = pd.read_csv(RES / "takeover.csv")
    ex = pd.read_csv(RES / "exec_modes.csv")
    summary = json.loads((RES / "summary.json").read_text())
    failure_modes(env, ep, cs, tk, summary)
    exec_modes(ex, summary)
    video = side_by_side_video(env, ep)
    print(f"wrote {video.relative_to(ROOT)}")
    print(f"wrote {', '.join(p.name for p in sorted(RES.glob('*.png')))}")


if __name__ == "__main__":
    main()
