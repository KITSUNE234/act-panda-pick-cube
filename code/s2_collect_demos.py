"""Phase 2: collect demonstrations from the PPO teacher.

Rolls out the deterministic teacher on the collection seeds (configs/seeds.yaml),
labels every episode with the shared success criterion, keeps the first
`n_keep` successful episodes (in seed-index order) as the baseline dataset, and
fixes the nested ablation subsets. Two alignment checks guard the data:
  1. re-querying the teacher on each stored obs reproduces the stored action;
  2. open-loop replay of the stored actions from the same reset key reproduces
     the stored observations.

Writes
  data/demos_raw.npz             all collected episodes (incl. failures)
  data/demos_baseline.npz        the kept successful episodes, re-indexed 0..n_keep-1
  configs/datasets.yaml          episode indices of each dataset (nested subsets)
  results/demos/collection_episodes.csv, results/demos/collection_summary.json

  python code/s2_collect_demos.py
"""

import argparse
import hashlib
import json

from common import env_utils as eu
from common import rollout as ro
from common import teacher as tch

import jax
import numpy as np
import pandas as pd
import yaml

ROOT = eu.ROOT
DEFAULT_TEACHER = "checkpoints/teacher_seed0/checkpoints/000024576000"
SUBSET_SIZES = (25, 50)
SUBSET_PERM_SEED = 0

# Per-episode arrays saved to the .npz files.
EPISODE_KEYS = ["obs", "actions", "rewards", "dones", "qpos", "qvel", "box_pos",
                "gripper_pos", "target_pos", "box_target_dist", "reached_box",
                "length", "success", "final_dist", "first_success_step"]


def sha256(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def replay_max_obs_diff(benv, keys, actions, obs):
    """Open-loop replay of stored actions; returns max |obs_replay - obs_stored|."""
    out = ro.rollout(benv, lambda o, t: actions[:, t], keys, n_steps=actions.shape[1])
    return float(np.abs(out["obs"] - obs).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    p.add_argument("--n_collect", type=int, default=200)
    p.add_argument("--n_keep", type=int, default=100)
    args = p.parse_args()

    seeds = eu.load_seeds()
    keys = eu.episode_keys(seeds["collect_base_seed"], args.n_collect)
    eval_keys = np.asarray(eu.eval_keys())
    overlap = (np.asarray(keys)[:, None, :] == eval_keys[None, :, :]).all(-1).any()
    assert not overlap, "collection and eval reset keys overlap"

    env = eu.load_env()
    benv = ro.BatchedEnv(env)
    act = tch.load_teacher(ROOT / args.teacher, deterministic=True)
    rng = jax.random.PRNGKey(0)  # unused by the deterministic policy

    out = ro.rollout(benv, lambda obs, t: act(obs, rng), keys)
    out["actions"] = out["actions"].astype(np.float32)
    n_success = int(out["success"].sum())
    print(f"teacher success on {args.n_collect} collection seeds: {n_success}/{args.n_collect}")

    succ_idx = np.flatnonzero(out["success"])
    assert len(succ_idx) >= args.n_keep, f"only {len(succ_idx)} successes, need {args.n_keep}"
    kept = succ_idx[: args.n_keep]

    # Check 1: stored action == teacher(stored obs). Re-query step by step with
    # the same batch shape as collection: GPU float32 matmuls (TF32) can pick
    # shape-dependent algorithms that differ at the 1e-3 level.
    T = out["obs"].shape[1]
    requery = np.stack([np.asarray(act(out["obs"][:, t], rng)) for t in range(T)], axis=1)
    action_diff = float(np.abs(requery[kept] - out["actions"][kept]).max())
    # Check 2: open-loop replay reproduces the observations.
    obs_diff = replay_max_obs_diff(benv, keys[kept], out["actions"][kept], out["obs"][kept])
    print(f"alignment: max |teacher(obs) - action| = {action_diff:.2e}, "
          f"replay max |obs diff| = {obs_diff:.2e}")
    assert action_diff < 1e-4 and obs_diff < 1e-3, "alignment check failed"

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    key_data = np.asarray(jax.random.key_data(keys)) if jax.dtypes.issubdtype(
        keys.dtype, jax.dtypes.prng_key) else np.asarray(keys)
    raw = {k: out[k] for k in EPISODE_KEYS}
    raw.update(collect_index=np.arange(args.n_collect), reset_key=key_data,
               kept=np.isin(np.arange(args.n_collect), kept))
    np.savez_compressed(data_dir / "demos_raw.npz", **raw)
    base = {k: out[k][kept] for k in EPISODE_KEYS}
    base.update(collect_index=kept, reset_key=key_data[kept])
    np.savez_compressed(data_dir / "demos_baseline.npz", **base)

    # Nested ablation subsets: prefixes of one fixed permutation of the baseline.
    perm = np.random.default_rng(SUBSET_PERM_SEED).permutation(args.n_keep)
    datasets = {f"baseline_{args.n_keep}": list(range(args.n_keep))}
    for n in sorted(SUBSET_SIZES, reverse=True):
        datasets[f"subset_{n}"] = sorted(int(i) for i in perm[:n])
    sizes = sorted(SUBSET_SIZES)
    for small, big in zip(sizes, sizes[1:]):
        assert set(datasets[f"subset_{small}"]) <= set(datasets[f"subset_{big}"])
    (ROOT / "configs" / "datasets.yaml").write_text(
        "# Episode indices into data/demos_baseline.npz. Subsets are nested prefixes of\n"
        f"# np.random.default_rng({SUBSET_PERM_SEED}).permutation({args.n_keep}).\n"
        + yaml.safe_dump(datasets, default_flow_style=None, width=100))

    res_dir = ROOT / "results" / "demos"
    res_dir.mkdir(parents=True, exist_ok=True)
    adr = env._obj_qposadr
    table = pd.DataFrame({
        "collect_index": np.arange(args.n_collect),
        "kept": raw["kept"],
        "success": out["success"],
        "final_dist": out["final_dist"],
        "first_success_step": out["first_success_step"],
        "return": out["rewards"].sum(axis=1),
        "length": out["length"],
        "box_x0": out["qpos"][:, 0, adr], "box_y0": out["qpos"][:, 0, adr + 1],
        "target_x": out["target_pos"][:, 0], "target_y": out["target_pos"][:, 1],
        "target_z": out["target_pos"][:, 2],
    })
    table.to_csv(res_dir / "collection_episodes.csv", index=False)

    lo, hi = ro.wilson_ci(n_success, args.n_collect)
    summary = {
        "teacher_checkpoint": args.teacher,
        "collect_base_seed": seeds["collect_base_seed"],
        "n_collected": args.n_collect,
        "teacher_success_rate": n_success / args.n_collect,
        "teacher_success_ci95": [lo, hi],
        "filtering": f"kept the first {args.n_keep} successful episodes in seed-index order",
        "n_kept": len(kept),
        "last_kept_collect_index": int(kept[-1]),
        "episode_length_steps": int(T),
        "obs_dim": int(out["obs"].shape[-1]),
        "action_dim": int(out["actions"].shape[-1]),
        "kept_first_success_step_mean": float(out["first_success_step"][kept].mean()),
        "kept_first_success_step_std": float(out["first_success_step"][kept].std()),
        "kept_final_dist_mean": float(out["final_dist"][kept].mean()),
        "alignment_max_action_diff": action_diff,
        "alignment_replay_max_obs_diff": obs_diff,
        "datasets": {k: len(v) for k, v in datasets.items()},
        "sha256_baseline_obs_actions": sha256(base["obs"], base["actions"]),
    }
    (res_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
