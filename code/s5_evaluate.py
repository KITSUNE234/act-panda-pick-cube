"""Phase 5: evaluate policies on the held-out evaluation seeds.

Every policy (teacher or ACT student) is rolled out from the same reset keys,
split(PRNGKey(eval_base_seed), n_eval) from configs/seeds.yaml, which are
disjoint from the demonstration and validation keys. Each episode is labelled
with the shared success criterion and a failure category (common/analysis.py).

Writes, for each policy <name>:
  results/eval/<name>/episodes.csv     per-episode outcome, phase timings, category
  results/eval/<name>/summary.json     success rate + Wilson 95% CI, category counts
  data/eval_rollouts/<name>.npz        full rollouts (regenerable; for failure analysis)
  videos/<name>_success_epXXX.mp4, videos/<name>_failure_<category>_epXXX.mp4
and updates results/eval/summary.csv (one row per policy).

  python code/s5_evaluate.py --policies teacher act_baseline_100_seed0
  python code/s5_evaluate.py --policies act_baseline_100_seed0 --mode open_loop --query_every 20
"""

import argparse
import json

from common import analysis as an
from common import env_utils as eu
from common import rollout as ro
from common import teacher as tch
from common.act import ChunkExecutor, load_policy, setup_torch

import jax
import mediapy as media
import numpy as np
import pandas as pd
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = eu.ROOT
TEACHER = "checkpoints/teacher_seed0/checkpoints/000024576000"
SAVE_KEYS = ["obs", "actions", "rewards", "dones", "qpos", "qvel", "box_pos", "gripper_pos",
             "target_pos", "box_target_dist", "success", "final_dist", "first_success_step"]


def make_policy(spec, args):
    """Returns (act_fn, info) for 'teacher' or the name of a checkpoints/ ACT run."""
    if spec == "teacher":
        act = tch.load_teacher(ROOT / TEACHER, deterministic=True)
        rng = jax.random.PRNGKey(0)  # unused by the deterministic policy
        return (lambda obs, t: act(obs, rng)), {"policy": "teacher", "checkpoint": TEACHER}
    run = ROOT / "checkpoints" / spec
    model, obs_norm, act_norm = load_policy(run / "policy.pt")
    cfg = json.loads((run / "train_config.json").read_text())
    executor = ChunkExecutor(model, obs_norm, act_norm, mode=args.mode, m=args.ensemble_m,
                             query_every=args.query_every)
    info = {"policy": "act", "checkpoint": f"checkpoints/{spec}/policy.pt",
            "dataset": cfg["dataset"], "n_demos": cfg["n_train_episodes"], "train_seed": cfg["seed"],
            "mode": args.mode, "ensemble_m": args.ensemble_m if args.mode == "ensemble" else None,
            "query_every": executor.query_every if args.mode == "open_loop" else 1}
    return executor, info


def result_name(spec, args):
    if spec == "teacher" or (args.mode == "ensemble" and args.ensemble_m == 0.01):
        return spec
    if args.mode == "ensemble":
        return f"{spec}__ens_m{args.ensemble_m:g}"
    return f"{spec}__open_loop{args.query_every or ''}"


def episode_table(out, env):
    grasp, lift = an.phase_steps(out)
    adr = env._obj_qposadr
    return pd.DataFrame({
        "episode": np.arange(len(out["success"])),
        "success": out["success"],
        "category": an.failure_category(out),
        "final_dist": out["final_dist"],
        "first_success_step": out["first_success_step"],
        "grasp_step": grasp, "lift_step": lift,
        "max_box_z": out["box_pos"][:, 1:, 2].max(1),
        "box_x0": out["qpos"][:, 0, adr], "box_y0": out["qpos"][:, 0, adr + 1],
        "target_x": out["target_pos"][:, 0], "target_y": out["target_pos"][:, 1],
        "target_z": out["target_pos"][:, 2],
        "return": out["rewards"].sum(1),
        "mean_abs_action_change": np.abs(np.diff(out["actions"], axis=1)).mean((1, 2)),
    })


def write_videos(env, out, table, name, n_success):
    """The first n_success successes and the first failure of each category."""
    picks = [(ep, "success") for ep in np.flatnonzero(out["success"])[:n_success]]
    for cat in an.CATEGORIES[1:]:
        eps = np.flatnonzero(table["category"] == cat)
        if len(eps):
            picks.append((eps[0], f"failure_{cat}"))
    for ep, label in picks:
        frames = ro.render_episode(env, out["qpos"][ep], out["target_pos"][ep], every=2)
        media.write_video(ROOT / "videos" / f"{name}_{label}_ep{ep:03d}.mp4", frames,
                          fps=eu.video_fps(env, every=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["teacher", "act_baseline_100_seed0"])
    p.add_argument("--mode", default="ensemble", choices=["ensemble", "open_loop"])
    p.add_argument("--ensemble_m", type=float, default=0.01)
    p.add_argument("--query_every", type=int, default=None, help="open_loop: default = chunk length")
    p.add_argument("--n_videos", type=int, default=2,
                   help="success videos; plus one per failure category (teacher videos come from s1)")
    p.add_argument("--no_videos", action="store_true")
    args = p.parse_args()

    setup_torch()
    seeds = eu.load_seeds()
    keys = eu.eval_keys()
    env = eu.load_env()
    benv = ro.BatchedEnv(env)
    res_root = ROOT / "results" / "eval"
    (ROOT / "data" / "eval_rollouts").mkdir(parents=True, exist_ok=True)
    (ROOT / "videos").mkdir(exist_ok=True)

    rows = []
    for spec in args.policies:
        name = result_name(spec, args)
        act_fn, info = make_policy(spec, args)
        with sdpa_kernel(SDPBackend.MATH):
            out = ro.rollout(benv, act_fn, keys)
        table = episode_table(out, env)
        summary = {"name": name, **info, "eval_base_seed": seeds["eval_base_seed"],
                   **ro.summarize(out),
                   "first_success_step_mean": float(table.loc[table.success, "first_success_step"].mean()),
                   "categories": {c: int((table.category == c).sum()) for c in an.CATEGORIES}}
        res_dir = res_root / name
        res_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(res_dir / "episodes.csv", index=False)
        (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        np.savez_compressed(ROOT / "data" / "eval_rollouts" / f"{name}.npz",
                            **{k: out[k] for k in SAVE_KEYS})
        if spec != "teacher" and not args.no_videos:
            write_videos(env, out, table, name, args.n_videos)
        lo, hi = summary["success_ci95"]
        print(f"{name}: success {summary['success_rate']:.0%} [{lo:.1%}, {hi:.1%}]  "
              f"{summary['categories']}", flush=True)
        rows.append({k: summary.get(k) for k in
                     ["name", "policy", "dataset", "n_demos", "train_seed", "mode", "ensemble_m",
                      "query_every", "n_episodes", "success_rate"]}
                    | {"ci95_lo": lo, "ci95_hi": hi, "final_dist_mean": summary["final_dist_mean"],
                       "final_dist_median": summary["final_dist_median"]}
                    | {f"n_{c}": summary["categories"][c] for c in an.CATEGORIES})

    path = res_root / "summary.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.name.isin(new.name)], new], ignore_index=True)
    new.sort_values("name").to_csv(path, index=False)
    print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
