"""Phase 1: evaluate PPO teacher checkpoints on the held-out eval seeds.

Every checkpoint is rolled out deterministically on the same eval keys
(configs/seeds.yaml) with the shared success criterion. Writes
  results/teacher/eval_by_checkpoint.csv   success vs. training step
  results/teacher/final_episodes.csv       per-episode results, last checkpoint
  results/teacher/final_summary.json
  results/teacher/teacher_training.png     reward + success curves
  videos/teacher_{success,failure}_ep*.mp4

  python code/s1_eval_teacher.py --run checkpoints/teacher_seed0
"""

import argparse
import json
import pathlib

from common import env_utils as eu
from common import plot_style
from common import rollout as ro
from common import teacher as tch

import jax
import mediapy as media
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = eu.ROOT


def episode_table(out, keys_base, n):
    return pd.DataFrame({
        "episode": np.arange(n),
        "eval_base_seed": keys_base,
        "success": out["success"],
        "final_dist": out["final_dist"],
        "first_success_step": out["first_success_step"],
        "reached_box": out["reached_box"],
        "max_box_height": out["box_pos"][:, :, 2].max(axis=1),
        "length": out["length"],
        "return": out["rewards"].sum(axis=1),
    })


def plot_training(progress, by_ckpt, path):
    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    x = progress["step"] / 1e6
    y, s = progress["eval/episode_reward"], progress["eval/episode_reward_std"]
    ax.fill_between(x, y - s, y + s, color=plot_style.SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(x, y, color=plot_style.SERIES[0], marker="o")
    ax.set(title="PPO eval return (Brax evaluator, 128 envs)",
           xlabel="env steps (M)", ylabel="episode return")
    ax = axes[1]
    x = by_ckpt["step"] / 1e6
    lo, hi = by_ckpt["ci_lo"] * 100, by_ckpt["ci_hi"] * 100
    ax.fill_between(x, lo, hi, color=plot_style.SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(x, by_ckpt["success_rate"] * 100, color=plot_style.SERIES[0], marker="o")
    ax.set(title=f"Teacher success on {int(by_ckpt['n_episodes'].iloc[0])} held-out seeds",
           xlabel="env steps (M)", ylabel="success rate (%)", ylim=(-2, 102))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="checkpoints/teacher_seed0")
    p.add_argument("--n_videos", type=int, default=2, help="per outcome")
    p.add_argument("--results", default="results/teacher")
    p.add_argument("--video_prefix", default="teacher")
    args = p.parse_args()

    run = ROOT / args.run
    res_dir = ROOT / args.results
    res_dir.mkdir(parents=True, exist_ok=True)
    seeds = eu.load_seeds()
    keys = eu.eval_keys()
    n = len(keys)

    env = eu.load_env()
    benv = ro.BatchedEnv(env)
    rng = jax.random.PRNGKey(0)  # unused by the deterministic policy

    rows, out = [], None
    ckpts = tch.list_checkpoints(run)
    for ckpt in ckpts:
        act = tch.load_teacher(ckpt, deterministic=True)
        out = ro.rollout(benv, lambda obs, t: act(obs, rng), keys)
        summ = ro.summarize(out)
        rows.append({"step": int(ckpt.name), "checkpoint": str(ckpt.relative_to(ROOT)),
                     "n_episodes": n, "success_rate": summ["success_rate"],
                     "ci_lo": summ["success_ci95"][0], "ci_hi": summ["success_ci95"][1],
                     "final_dist_mean": summ["final_dist_mean"],
                     "reached_box_rate": summ["reached_box_rate"],
                     "mean_return": summ["mean_return"]})
        print(f"{ckpt.name}: success {summ['success_rate']:.2f} "
              f"[{summ['success_ci95'][0]:.2f}, {summ['success_ci95'][1]:.2f}], "
              f"final dist {summ['final_dist_mean']:.3f}, reached {summ['reached_box_rate']:.2f}",
              flush=True)
    by_ckpt = pd.DataFrame(rows)
    by_ckpt.to_csv(res_dir / "eval_by_checkpoint.csv", index=False)

    # `out` now holds the final checkpoint's rollouts.
    final = ckpts[-1]
    table = episode_table(out, seeds["eval_base_seed"], n)
    table.to_csv(res_dir / "final_episodes.csv", index=False)
    summary = {"checkpoint": str(final.relative_to(ROOT)), "step": int(final.name),
               "eval_base_seed": seeds["eval_base_seed"], "success_dist_m": eu.SUCCESS_DIST,
               **ro.summarize(out),
               "first_success_step_mean": float(table.loc[table.success, "first_success_step"].mean())}
    (res_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    progress = pd.read_csv(run / "progress.csv")
    plot_training(progress, by_ckpt, res_dir / "teacher_training.png")

    vid_dir = ROOT / "videos"
    for label, mask in [("success", out["success"]), ("failure", ~out["success"])]:
        for ep in np.flatnonzero(mask)[: args.n_videos]:
            frames = ro.render_episode(env, out["qpos"][ep], out["target_pos"][ep])
            path = vid_dir / f"{args.video_prefix}_{label}_ep{ep:03d}.mp4"
            media.write_video(path, frames, fps=eu.video_fps(env))
            print(f"wrote {path.relative_to(ROOT)} (final dist {out['final_dist'][ep]:.3f})")


if __name__ == "__main__":
    main()
