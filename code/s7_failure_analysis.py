"""Phase 7: failure analysis of the ACT students.

Uses the evaluation rollouts saved by s5_evaluate.py (data/eval_rollouts/) and
runs one new experiment, teacher takeover: on the same 100 held-out seeds the
student acts for steps t < s and the PPO teacher for t >= s. Because batched
rollouts are bitwise deterministic, the first s steps are exactly the student's
evaluation episode; s = 0 is the teacher alone and s = 150 the student alone.
If the teacher succeeds after taking over from a failing student, the student
reached states the teacher can still solve but the student cannot.

Writes results/failure/
  takeover.csv          success per (run, switch step) and among student failures
  episodes.csv          per-episode diagnostics for the baseline runs
  covariate_shift.csv   distance of visited states to the nearest training state, per step
  coverage_vs_n.csv     the same distance summarized per run of the 25/50/100-demo ablation
  exec_modes.csv        success and smoothness for each chunk execution mode
  summary.json          the numbers quoted in REPRODUCE.md

  python code/s7_failure_analysis.py
"""

import argparse
import json

from common import analysis as an
from common import env_utils as eu
from common import rollout as ro
from common import teacher as tch
from common.act import ChunkExecutor, load_policy, setup_torch
from common.demos import load_demos

import jax
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = eu.ROOT
RES = ROOT / "results" / "failure"
ROLLOUTS = ROOT / "data" / "eval_rollouts"
TEACHER = "checkpoints/teacher_seed0/checkpoints/000024576000"
BASELINE_RUNS = [f"act_baseline_100_seed{s}" for s in (0, 1, 2)]
ABLATION_RUNS = [f"act_{d}_seed{s}" for d in ("subset_25", "subset_50", "baseline_100") for s in (0, 1, 2)]
SWITCH_STEPS = [0, 10, 15, 20, 30, 50, 100, 150]
NEAR_ROBOT_X = 0.6  # initial box x (m) below which the student fails most often
# Execution modes of the baseline runs: (label, result-name suffix).
EXEC_MODES = [("ensemble m=0.01 (default)", ""), ("ensemble m=0", "__ens_m0"),
              ("ensemble m=0.1", "__ens_m0.1"), ("open loop, re-plan every 1", "__open_loop1"),
              ("open loop, re-plan every 5", "__open_loop5"), ("open loop, re-plan every 10", "__open_loop10"),
              ("open loop, re-plan every 20", "__open_loop20")]


def box_tilt_deg(qpos, env):
    """Angle between the box's z-axis and the world z-axis, folded to [0, 90]."""
    adr = env._obj_qposadr
    x, y = qpos[..., adr + 4], qpos[..., adr + 5]
    return np.degrees(np.arccos(np.clip(np.abs(1 - 2 * (x * x + y * y)), 0, 1)))


def nearest_demo_dist(obs, demo_obs, obs_norm):
    """Euclidean distance (in the policy's normalized input space, divided by
    sqrt(D) so it reads as an RMS per-dimension z-score) from every state in
    obs [B,T,D] to the nearest state of the run's training demonstrations."""
    B, T, D = obs.shape
    ref = torch.tensor(obs_norm.norm(demo_obs.reshape(-1, D)), device="cuda")
    x = torch.tensor(obs_norm.norm(obs.reshape(-1, D)), device="cuda")
    d = torch.cat([torch.cdist(c, ref).min(1).values for c in x.split(4096)])
    return d.cpu().numpy().reshape(B, T) / np.sqrt(D)


def gripper_flips(actions):
    """Number of open/close command changes (only the sign of the gripper action matters)."""
    return (np.diff(np.sign(actions[..., 7]), axis=1) != 0).sum(1)


def takeover(benv, keys, run, teacher_act, switch_steps):
    model, obs_norm, act_norm = load_policy(ROOT / "checkpoints" / run / "policy.pt")
    executor = ChunkExecutor(model, obs_norm, act_norm)
    rng = jax.random.PRNGKey(0)  # unused by the deterministic teacher
    out = {}
    for s in switch_steps:
        act_fn = lambda obs, t: executor(obs, t) if t < s else teacher_act(obs, rng)
        with sdpa_kernel(SDPBackend.MATH):
            out[s] = ro.rollout(benv, act_fn, keys)["success"]
        print(f"{run} takeover at {s:3d}: {out[s].mean():.0%}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=BASELINE_RUNS, help="runs for the takeover experiment")
    args = p.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    setup_torch()
    env = eu.load_env()
    benv = ro.BatchedEnv(env)
    keys = eu.eval_keys()
    teacher_act = tch.load_teacher(ROOT / TEACHER, deterministic=True)
    teacher = dict(np.load(ROLLOUTS / "teacher.npz"))
    summary = {}

    # 1. Teacher takeover.
    rows, tk = [], {}
    for run in args.runs:
        tk[run] = takeover(benv, keys, run, teacher_act, SWITCH_STEPS)
        student_ok = tk[run][150]
        saved = np.load(ROLLOUTS / f"{run}.npz")["success"]
        assert (student_ok == saved).all(), "takeover prefix does not reproduce the evaluation rollout"
        for s in SWITCH_STEPS:
            rows.append({"run": run, "switch_step": s, "success": tk[run][s].mean(),
                         "n_student_failures": int((~student_ok).sum()),
                         "rescued_student_failures": int((tk[run][s] & ~student_ok).sum())})
    tk_df = pd.DataFrame(rows)
    tk_df.to_csv(RES / "takeover.csv", index=False)

    # 2. Per-episode diagnostics and covariate shift.
    ep_rows, cs_rows, cov_rows = [], [], []
    nn_cache = {}
    for run in ABLATION_RUNS:
        cfg = json.loads((ROOT / "checkpoints" / run / "train_config.json").read_text())
        demo_obs = load_demos(cfg["dataset"])["obs"]
        _, obs_norm, _ = load_policy(ROOT / "checkpoints" / run / "policy.pt")
        st = dict(np.load(ROLLOUTS / f"{run}.npz"))
        nn_s = nearest_demo_dist(st["obs"], demo_obs, obs_norm)
        nn_t = nearest_demo_dist(teacher["obs"], demo_obs, obs_norm)
        nn_cache[run] = (nn_s, nn_t, st)
        ok = st["success"]
        run_thr = np.percentile(nn_t, 99)
        cov_rows.append({"run": run, "dataset": cfg["dataset"], "n_demos": cfg["n_train_episodes"],
                         "success": ok.mean(), "teacher_dist_step0_median": np.median(nn_t[:, 0]),
                         "teacher_dist_steps60_150_median": np.median(nn_t[:, 60:]),
                         "student_dist_steps60_150_median": np.median(nn_s[:, 60:]),
                         "student_episodes_off_distribution": (nn_s > run_thr).any(1).mean(),
                         "student_failures_off_distribution": (nn_s[~ok] > run_thr).any(1).mean()})
        for t in range(nn_s.shape[1]):
            cs_rows.append({"run": run, "dataset": cfg["dataset"], "step": t,
                            "student_success_median": np.median(nn_s[ok, t]) if ok.any() else np.nan,
                            "student_failure_median": np.median(nn_s[~ok, t]) if (~ok).any() else np.nan,
                            "teacher_median": np.median(nn_t[:, t])})
    cs_df = pd.DataFrame(cs_rows)
    cs_df.to_csv(RES / "covariate_shift.csv", index=False)
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(RES / "coverage_vs_n.csv", index=False)

    # Off-distribution threshold: 99th percentile of the teacher's own distances on
    # the evaluation seeds under the baseline normalizer (pooled over the 3 seeds).
    thr = float(np.percentile(np.concatenate([nn_cache[r][1].ravel() for r in BASELINE_RUNS]), 99))
    tilt_t = box_tilt_deg(teacher["qpos"], env)
    for run in BASELINE_RUNS:
        nn_s, nn_t, st = nn_cache[run]
        ep = pd.read_csv(ROOT / "results" / "eval" / run / "episodes.csv")
        tilt_s = box_tilt_deg(st["qpos"], env)
        off = an.first_true(nn_s > thr)
        # Last 50 steps: box speed (m/s), change in box-target distance, arm action size.
        speed = np.linalg.norm(np.diff(st["box_pos"][:, -51:], axis=1), axis=-1).mean(1) / env.dt
        dist_change = st["box_target_dist"][:, -1] - st["box_target_dist"][:, -51]
        arm_action = np.abs(st["actions"][:, -50:, :7]).mean((1, 2))
        for i in range(len(ep)):
            ep_rows.append({"run": run, "episode": i, "category": ep.category[i], "success": ep.success[i],
                            "box_x0": ep.box_x0[i], "box_y0": ep.box_y0[i],
                            "near_robot": ep.box_x0[i] < NEAR_ROBOT_X,
                            "teacher_success": bool(teacher["success"][i]),
                            "grasp_step": ep.grasp_step[i], "lift_step": ep.lift_step[i],
                            "max_tilt_student": tilt_s[i].max(), "max_tilt_teacher": tilt_t[i].max(),
                            "tilt_step20_student": tilt_s[i, 20], "tilt_step20_teacher": tilt_t[i, 20],
                            "nn_dist_step0": nn_s[i, 0], "nn_dist_max_student": nn_s[i].max(),
                            "nn_dist_max_teacher": nn_t[i].max(), "off_dist_step": off[i],
                            "gripper_flips": int(gripper_flips(st["actions"][i:i + 1])[0]),
                            "final_dist": ep.final_dist[i], "box_speed_last50": speed[i],
                            "dist_change_last50": dist_change[i], "arm_action_last50": arm_action[i],
                            **({f"takeover_{s}": bool(tk[run][s][i]) for s in SWITCH_STEPS} if run in tk else {})})
    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(RES / "episodes.csv", index=False)

    # 3. Chunk execution modes (evaluation results from s5_evaluate.py).
    ex_rows = []
    for label, suffix in EXEC_MODES:
        k = n = 0
        jerk, flips, per_seed = [], [], []
        for run in BASELINE_RUNS:
            out = np.load(ROLLOUTS / f"{run}{suffix}.npz")
            ok = out["success"]
            k, n = k + int(ok.sum()), n + len(ok)
            per_seed.append(f"{ok.mean():.0%}")
            jerk.append(np.abs(np.diff(out["actions"][..., :7], axis=1)).mean())
            flips.append(gripper_flips(out["actions"]).mean())
        lo, hi = ro.wilson_ci(k, n)
        ex_rows.append({"mode": label, "suffix": suffix, "success": k / n, "k": k, "n": n,
                        "ci95_lo": lo, "ci95_hi": hi, "per_seed": " / ".join(per_seed),
                        "mean_abs_arm_action_change": float(np.mean(jerk)),
                        "gripper_flips_per_episode": float(np.mean(flips))})
    jerk_t = float(np.abs(np.diff(teacher["actions"][..., :7], axis=1)).mean())
    ex_df = pd.DataFrame(ex_rows)
    ref = ex_df.iloc[0]
    ex_df["fisher_p_vs_default"] = [stats.fisher_exact([[ref.k, ref.n - ref.k], [r.k, r.n - r.k]])[1]
                                    for _, r in ex_df.iterrows()]
    ex_df.to_csv(RES / "exec_modes.csv", index=False)

    # 4. Numbers for the write-up.
    b = ep_df
    fail, near = ~b.success, b.near_robot
    summary["off_distribution_threshold"] = thr
    summary["student_success_near_robot"] = float(b[near].success.mean())
    summary["student_success_far"] = float(b[~near].success.mean())
    summary["teacher_success_near_robot"] = float(b[near].teacher_success.mean())
    summary["fraction_failures_near_robot"] = float(near[fail].mean())
    summary["median_max_tilt"] = {
        "student_success": float(b[~fail].max_tilt_student.median()),
        "student_failure": float(b[fail].max_tilt_student.median()),
        "teacher_on_student_failure_seeds": float(b[fail].max_tilt_teacher.median())}
    summary["fraction_failures_tilt_over_30deg"] = float((b[fail].max_tilt_student > 30).mean())
    summary["fraction_successes_tilt_over_30deg"] = float((b[~fail].max_tilt_student > 30).mean())
    summary["fraction_failures_off_distribution"] = float((b[fail].off_dist_step >= 0).mean())
    summary["fraction_successes_off_distribution"] = float((b[~fail].off_dist_step >= 0).mean())
    summary["median_off_dist_step_failures"] = float(b[fail & (b.off_dist_step >= 0)].off_dist_step.median())
    summary["nn_dist_step0_mean"] = {"student_success": float(b[~fail].nn_dist_step0.mean()),
                                     "student_failure": float(b[fail].nn_dist_step0.mean())}
    summary["nn_dist_step0_mannwhitney_p"] = float(stats.mannwhitneyu(b[fail].nn_dist_step0,
                                                                       b[~fail].nn_dist_step0).pvalue)
    summary["last50_steps_median_by_category"] = b.groupby("category")[
        ["box_speed_last50", "dist_change_last50", "arm_action_last50", "final_dist"]].median().round(4).to_dict(
        orient="index")
    summary["category_counts_baseline"] = b.category.value_counts().to_dict()
    tk_pool = tk_df.groupby("switch_step")[["success", "rescued_student_failures", "n_student_failures"]].agg(
        {"success": "mean", "rescued_student_failures": "sum", "n_student_failures": "sum"})
    summary["takeover"] = {int(s): {"success": float(r.success),
                                    "rescued_of_student_failures": f"{int(r.rescued_student_failures)}/{int(r.n_student_failures)}"}
                           for s, r in tk_pool.iterrows()}
    summary["teacher_mean_abs_arm_action_change"] = jerk_t
    summary["coverage_vs_n"] = cov_df.groupby("n_demos")[
        ["success", "teacher_dist_step0_median", "teacher_dist_steps60_150_median",
         "student_dist_steps60_150_median", "student_episodes_off_distribution",
         "student_failures_off_distribution"]].mean().round(3).to_dict(orient="index")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(ex_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
