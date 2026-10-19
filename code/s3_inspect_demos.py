"""Phase 3: visualize and inspect the demonstration dataset.

Writes
  results/demos/dataset_stats.json      counts, lengths, success, phase timings, per-subset stats
  results/demos/demo_episodes.csv       per-episode diagnostics of the baseline dataset
  results/demos/demo_frames.png         frame strips of several demonstrations
  results/demos/trajectories.png        box / gripper paths (top-down and height over time)
  results/demos/init_coverage.png       initial box and target positions, subsets and failures
  results/demos/timing.png              phase timings; time to success vs. initial box x
  results/demos/actions.png             per-dimension action time series (median + band)
  videos/expert_demo_ep000.mp4          one full expert demonstration
  videos/expert_demos_grid.mp4          2x2 montage of four demonstrations
  videos/expert_demo_slowest_epXXX.mp4  the demonstration that takes longest to succeed

  python code/s3_inspect_demos.py
"""

import json

from common import plot_style
from common import env_utils as eu
from common import rollout as ro
from common.demos import dataset_indices, load_demos

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd

ROOT = eu.ROOT
RES = ROOT / "results" / "demos"
VID = ROOT / "videos"
LIFT_Z = 0.045      # box counts as lifted once its centre is 1.5 cm above the rest height
GRASP_DIST = 0.02   # gripper site within 2 cm of the box centre
FRAME_STEPS = [0, 12, 24, 36, 60, 149]
STRIP_EPISODES = [0, 1, 2, 3]
GRID_EPISODES = [0, 1, 2, 3]
ACTION_NAMES = [f"joint {i + 1}" for i in range(7)] + ["gripper"]


def first_true(mask):
    """Index of the first True along the last axis, or -1."""
    idx = np.argmax(mask, axis=-1)
    return np.where(mask.any(axis=-1), idx, -1)


def reconstruct_ctrl(env, actions):
    """Actuator targets after each step: ctrl_t = clip(ctrl_{t-1} + scale * a_t)."""
    lo, hi = np.asarray(env._lowers), np.asarray(env._uppers)
    ctrl = np.broadcast_to(np.asarray(env._init_ctrl), actions.shape[:1] + lo.shape).copy()
    out = np.empty(actions.shape)
    for t in range(actions.shape[1]):
        ctrl = np.clip(ctrl + env._config.action_scale * actions[:, t], lo, hi)
        out[:, t] = ctrl
    at_bound = np.isclose(out, lo, atol=1e-6) | np.isclose(out, hi, atol=1e-6)
    return out, at_bound


def episode_table(d, env):
    """Per-episode diagnostics. Index 0 of the [T+1] arrays is the reset state."""
    box, grip = d["box_pos"][:, 1:], d["gripper_pos"][:, 1:]
    adr = env._obj_qposadr
    box0 = d["qpos"][:, 0, adr:adr + 3]
    grasp = first_true(np.linalg.norm(box - grip, axis=-1) < GRASP_DIST) + 1
    lift = first_true(box[..., 2] > LIFT_Z) + 1
    path = np.linalg.norm(np.diff(d["box_pos"][:, 1:], axis=1), axis=-1).sum(1)
    straight = np.linalg.norm(d["target_pos"] - box0, axis=-1)
    fss = d["first_success_step"]
    after = np.array([np.abs(a[s:]).mean() for a, s in zip(d["actions"], fss)])
    before = np.array([np.abs(a[:s]).mean() for a, s in zip(d["actions"], fss)])
    return pd.DataFrame({
        "episode": np.arange(len(fss)),
        "collect_index": d["collect_index"],
        "box_x0": box0[:, 0], "box_y0": box0[:, 1],
        "target_x": d["target_pos"][:, 0], "target_y": d["target_pos"][:, 1],
        "target_z": d["target_pos"][:, 2],
        "init_box_target_dist": straight,
        "grasp_step": grasp, "lift_step": lift, "first_success_step": fss,
        "final_dist": d["final_dist"],
        "box_path_len": path, "path_ratio": path / straight,
        "return": d["rewards"].sum(1),
        "mean_abs_action_before_success": before,
        "mean_abs_action_after_success": after,
        "left_success_region": [bool((dd[s:] >= eu.SUCCESS_DIST).any())
                                for dd, s in zip(d["box_target_dist"], fss)],
    })


def mean_std(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x)),
            "min": float(np.min(x)), "max": float(np.max(x))}


def plot_frames(env, d):
    rows = []
    for ep in STRIP_EPISODES:
        frames = ro.render_episode(env, d["qpos"][ep], d["target_pos"][ep], height=240, width=320, every=1)
        rows.append(frames[FRAME_STEPS])
    fig, axes = plt.subplots(len(rows), len(FRAME_STEPS), figsize=(2.1 * len(FRAME_STEPS), 1.65 * len(rows)))
    for r, (ep, row) in enumerate(zip(STRIP_EPISODES, rows)):
        for c, (t, img) in enumerate(zip(FRAME_STEPS, row)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"t = {t}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"demo {ep}", fontsize=9)
    fig.suptitle("Expert demonstrations (green = box, red = target; t in control steps of 0.02 s)",
                 color=plot_style.TEXT, y=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(RES / "demo_frames.png")
    plt.close(fig)


def plot_trajectories(d, table):
    box, grip, tgt = d["box_pos"][:, 1:], d["gripper_pos"][:, 1:], d["target_pos"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    c_box, c_grip, c_tgt = plot_style.SERIES[0], plot_style.SERIES[1], plot_style.SERIES[2]
    ax = axes[0]
    for i in range(len(box)):
        ax.plot(grip[i, :, 0], grip[i, :, 1], color=c_grip, lw=0.6, alpha=0.25)
        ax.plot(box[i, :, 0], box[i, :, 1], color=c_box, lw=0.8, alpha=0.4)
    ax.scatter(table["box_x0"], table["box_y0"], s=10, color=c_box, label="box start", zorder=3)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=14, marker="x", color=c_tgt, label="target", zorder=3)
    ax.plot([], [], color=c_box, label="box path")
    ax.plot([], [], color=c_grip, label="gripper path")
    ax.set(title="Top-down paths (all 100 demos)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    t = np.arange(box.shape[1])
    ax = axes[1]
    for i in range(len(box)):
        ax.plot(t, box[i, :, 2], color=c_box, lw=0.6, alpha=0.3)
    ax.plot(t, np.median(box[..., 2], 0), color=plot_style.TEXT, lw=2, label="median")
    ax.set(title="Box height", xlabel="control step", ylabel="box z (m)")
    ax.legend(loc="lower right")

    ax = axes[2]
    dist = d["box_target_dist"][:, 1:]
    for i in range(len(dist)):
        ax.plot(t, dist[i], color=c_box, lw=0.6, alpha=0.3)
    ax.plot(t, np.median(dist, 0), color=plot_style.TEXT, lw=2, label="median")
    ax.axhline(eu.SUCCESS_DIST, color=plot_style.TEXT_2, lw=1, ls="--", label="success radius 5 cm")
    ax.set(title="Box-target distance", xlabel="control step", ylabel="distance (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "trajectories.png")
    plt.close(fig)


def plot_coverage(table, raw, env, datasets):
    adr = env._obj_qposadr
    fail = ~raw["success"]
    sub25 = np.isin(table["episode"], datasets["subset_25"])
    sub50 = np.isin(table["episode"], datasets["subset_50"]) & ~sub25
    rest = ~(sub25 | sub50)
    groups = [("in subset_25", sub25, plot_style.SERIES[0]),
              ("added in subset_50", sub50, plot_style.SERIES[1]),
              ("added in baseline_100", rest, plot_style.SERIES[2])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, (xcol, ycol, title) in zip(axes[:2], [("box_x0", "box_y0", "Initial box position"),
                                                   ("target_x", "target_y", "Target position (x, y)")]):
        for label, m, color in groups:
            ax.scatter(table.loc[m, xcol], table.loc[m, ycol], s=22, color=color, label=label,
                       edgecolors=plot_style.SURFACE, linewidths=0.8)
        if xcol == "box_x0":
            fx, fy = raw["qpos"][fail, 0, adr], raw["qpos"][fail, 0, adr + 1]
        else:
            fx, fy = raw["target_pos"][fail, 0], raw["target_pos"][fail, 1]
        ax.scatter(fx, fy, s=60, marker="x", color=plot_style.BAD, linewidths=2,
                   label="teacher failure (excluded)")
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)", xlim=(0.48, 0.92), ylim=(-0.22, 0.22))
        ax.set_aspect("equal", adjustable="box")
    ax = axes[2]
    ax.hist(table["target_z"], bins=np.linspace(0.22, 0.44, 12), color=plot_style.SERIES[0],
            edgecolor=plot_style.SURFACE, linewidth=2)
    ax.set(title="Target height", xlabel="target z (m)", ylabel="# demos")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RES / "init_coverage.png")
    plt.close(fig)


def plot_timing(table):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    phases = [("grasp_step", "grasp"), ("lift_step", "lift"), ("first_success_step", "first success")]
    rng = np.random.default_rng(0)
    for k, (col, label) in enumerate(phases):
        y = k + rng.uniform(-0.18, 0.18, len(table))
        ax.scatter(table[col], y, s=12, color=plot_style.SERIES[k], alpha=0.7, linewidths=0)
        q1, med, q3 = np.percentile(table[col], [25, 50, 75])
        ax.plot([q1, q3], [k, k], color=plot_style.TEXT, lw=2)
        ax.plot([med], [k], marker="|", markersize=16, mew=2.5, color=plot_style.TEXT)
    ax.set_yticks(range(len(phases)), [p[1] for p in phases])
    ax.set(title="When each phase is reached (bar = IQR, tick = median)", xlabel="control step",
           xlim=(0, 100), ylim=(-0.6, len(phases) - 0.4))
    ax = axes[1]
    ax.scatter(table["box_x0"], table["first_success_step"], s=22,
               color=plot_style.SERIES[2], edgecolors=plot_style.SURFACE, linewidths=0.8)
    r = np.corrcoef(table["box_x0"], table["first_success_step"])[0, 1]
    ax.set(title=f"Time to success vs. box distance from robot (r = {r:.2f})",
           xlabel="initial box x (m; robot base at x = 0)", ylabel="first success step")
    fig.tight_layout()
    fig.savefig(RES / "timing.png")
    plt.close(fig)


def plot_actions(d, ctrl, at_bound):
    a = d["actions"]
    t = np.arange(a.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True)
    for j, ax in enumerate(axes.flat):
        lo, med, hi = np.percentile(a[..., j], [10, 50, 90], axis=0)
        for i in range(3):
            ax.plot(t, a[i, :, j], color=plot_style.SERIES[1], lw=0.7, alpha=0.6)
        ax.fill_between(t, lo, hi, color=plot_style.SERIES[0], alpha=0.25, lw=0)
        ax.plot(t, med, color=plot_style.SERIES[0], lw=2)
        title = ACTION_NAMES[j]
        if at_bound[..., j].mean() > 0.5:
            title += f" (ctrl at limit {at_bound[..., j].mean():.0%} of steps)"
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-1.05, 1.05)
    axes[0, 0].plot([], [], color=plot_style.SERIES[0], label="median, 10-90% band")
    axes[0, 0].plot([], [], color=plot_style.SERIES[1], lw=0.7, label="demos 0-2")
    axes[0, 0].legend(loc="lower right", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("control step")
    for ax in axes[:, 0]:
        ax.set_ylabel("action")
    fig.suptitle("Teacher actions (delta joint targets, scaled to [-1, 1])", color=plot_style.TEXT, y=1.0)
    fig.tight_layout()
    fig.savefig(RES / "actions.png")
    plt.close(fig)


def write_videos(env, d):
    fps = eu.video_fps(env, every=1)
    ep = 0
    frames = ro.render_episode(env, d["qpos"][ep], d["target_pos"][ep], every=1)
    media.write_video(VID / f"expert_demo_ep{ep:03d}.mp4", frames, fps=fps)
    tiles = [ro.render_episode(env, d["qpos"][i], d["target_pos"][i], height=240, width=320, every=1)
             for i in GRID_EPISODES]
    top = np.concatenate(tiles[:2], axis=2)
    bottom = np.concatenate(tiles[2:], axis=2)
    media.write_video(VID / "expert_demos_grid.mp4", np.concatenate([top, bottom], axis=1), fps=fps)
    slow = int(np.argmax(d["first_success_step"]))
    frames = ro.render_episode(env, d["qpos"][slow], d["target_pos"][slow], every=1)
    media.write_video(VID / f"expert_demo_slowest_ep{slow:03d}.mp4", frames, fps=fps)


def main():
    plot_style.apply()
    RES.mkdir(parents=True, exist_ok=True)
    VID.mkdir(exist_ok=True)
    env = eu.load_env()
    d = load_demos("baseline_100")
    datasets = dataset_indices()
    with np.load(ROOT / "data" / "demos_raw.npz") as f:
        raw = {k: f[k] for k in f.files}

    table = episode_table(d, env)
    table.to_csv(RES / "demo_episodes.csv", index=False)
    ctrl, at_bound = reconstruct_ctrl(env, d["actions"])
    # Sanity: the reconstructed ctrl matches obs[ctrl - robot qpos] after the first step
    # (obs at t holds ctrl_{t-1}; qpos[:, t] is the state at that observation).
    lay = eu.obs_layout(env)
    sl = lay["ctrl_minus_robot_qpos"]
    recon = ctrl[:, :-1] - d["qpos"][:, 1:-1, :8]
    ctrl_err = float(np.abs(recon - d["obs"][:, 1:, sl]).max())
    assert ctrl_err < 1e-4, ctrl_err

    da = np.diff(d["actions"], axis=1)
    stats = {
        "collected": {"n_trajectories": int(len(raw["success"])),
                      "success_fraction": float(raw["success"].mean()),
                      "length_steps": mean_std(raw["length"])},
        "baseline_100": {
            "n_trajectories": int(len(table)),
            "success_fraction": float(d["success"].mean()),
            "length_steps": mean_std(d["length"]),
            "n_transitions": int(d["length"].sum()),
            "grasp_step": mean_std(table["grasp_step"]),
            "lift_step": mean_std(table["lift_step"]),
            "first_success_step": mean_std(table["first_success_step"]),
            "final_dist_m": mean_std(table["final_dist"]),
            "box_path_ratio": mean_std(table["path_ratio"]),
            "episodes_leaving_success_region": int(table["left_success_region"].sum()),
            "corr_init_dist_vs_success_step": float(np.corrcoef(
                table["init_box_target_dist"], table["first_success_step"])[0, 1]),
            "corr_box_x0_vs_success_step": float(np.corrcoef(
                table["box_x0"], table["first_success_step"])[0, 1]),
            "n_final_dist_above_2cm": int((table["final_dist"] > 0.02).sum()),
            "mean_abs_action_before_success": float(table["mean_abs_action_before_success"].mean()),
            "mean_abs_action_after_success": float(table["mean_abs_action_after_success"].mean()),
            "mean_abs_action_change_per_step": float(np.abs(da).mean()),
            "ctrl_at_limit_fraction": dict(zip(ACTION_NAMES, at_bound.mean((0, 1)).round(4).tolist())),
            "action_abs_max": float(np.abs(d["actions"]).max()),
            "ctrl_reconstruction_max_err": ctrl_err,
        },
        "subsets": {name: {"n": len(idx),
                           "first_success_step": mean_std(table.loc[idx, "first_success_step"]),
                           "box_x0_range": [float(table.loc[idx, "box_x0"].min()),
                                            float(table.loc[idx, "box_x0"].max())],
                           "box_y0_range": [float(table.loc[idx, "box_y0"].min()),
                                            float(table.loc[idx, "box_y0"].max())]}
                    for name, idx in datasets.items()},
    }
    (RES / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    plot_frames(env, d)
    plot_trajectories(d, table)
    plot_coverage(table, raw, env, datasets)
    plot_timing(table)
    plot_actions(d, ctrl, at_bound)
    write_videos(env, d)
    print("wrote figures to results/demos/ and videos to videos/")


if __name__ == "__main__":
    main()
