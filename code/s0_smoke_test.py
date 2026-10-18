"""Phase 0 smoke test: load PandaPickCube, inspect obs/action spaces, check
determinism and batched throughput, render random/zero-action videos, and
confirm JAX + PyTorch can share the GPU. Writes results/env_info.json."""

import json
import pathlib
import platform
import subprocess
import time

from common import env_utils as eu  # sets MUJOCO_GL / XLA env vars first

import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_STEPS = 200  # same as the README example


def rollout(env, step_fn, state, action_fn):
    traj = [state]
    for i in range(N_STEPS):
        state = step_fn(state, action_fn(i))
        traj.append(state)
    return traj


def main():
    env = eu.load_env()
    layout = eu.obs_layout(env)
    lowers, uppers = env.mj_model.actuator_ctrlrange.T

    print(f"obs size {env.observation_size}, action size {env.action_size}")
    print(f"dt {env.dt}, sim_dt {env.sim_dt}, n_substeps {env.n_substeps}")
    print("obs layout:", {k: (v.start, v.stop) for k, v in layout.items()})
    print("actuators:", [env.mj_model.actuator(i).name for i in range(env.mj_model.nu)])
    print("ctrl range:", np.round(np.stack([lowers, uppers], 1), 3).tolist())

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Determinism: the same key must give the same initial box/target.
    s_a, s_b = reset_fn(jax.random.PRNGKey(0)), reset_fn(jax.random.PRNGKey(0))
    assert np.array_equal(s_a.obs, s_b.obs), "reset is not deterministic"
    s_c = reset_fn(jax.random.PRNGKey(1))
    assert not np.array_equal(s_a.obs, s_c.obs), "different keys gave same reset"

    # README random-action rollout (resample every 10 steps) and zero actions.
    rng = jax.random.PRNGKey(42)
    rand_actions = []
    for i in range(N_STEPS):
        if i % 10 == 0:
            rng, k = jax.random.split(rng)
            a = jax.random.uniform(k, (env.action_size,), minval=-1.0, maxval=1.0)
        rand_actions.append(a)
    traj_rand = rollout(env, step_fn, s_a, lambda i: rand_actions[i])
    traj_zero = rollout(env, step_fn, s_a, lambda i: jnp.zeros(env.action_size))

    # Open-loop replay of the same actions from the same key must reproduce
    # the trajectory (the property Phase 2's alignment check relies on).
    traj_rand2 = rollout(env, step_fn, s_b, lambda i: rand_actions[i])
    replay_err = float(np.max(np.abs(np.asarray(traj_rand[-1].obs) - np.asarray(traj_rand2[-1].obs))))
    print(f"replay max |obs diff| after {N_STEPS} steps: {replay_err:.2e}")

    for name, traj in [("random", traj_rand), ("zero", traj_zero)]:
        d = float(eu.box_target_dist(env, traj[-1]))
        print(f"{name}: final box-target dist {d:.3f} m, success {bool(eu.is_success(d))}, "
              f"done {float(traj[-1].done)}")
        frames = eu.render(env, traj)
        out = ROOT / "videos" / f"smoke_{name}_actions.mp4"
        media.write_video(out, frames, fps=eu.video_fps(env))
        print(f"  wrote {out.relative_to(ROOT)}")

    # Batched throughput: 100 envs x 150 steps, as needed for eval in Phase 5.
    n_envs, n_steps = 100, env._config.episode_length
    keys = eu.episode_keys(10_000, n_envs)
    v_reset, v_step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    states = v_reset(keys)
    zeros = jnp.zeros((n_envs, env.action_size))
    jax.block_until_ready(v_step(states, zeros))  # compile
    t0 = time.time()
    for _ in range(n_steps):
        states = v_step(states, zeros)
    jax.block_until_ready(states.obs)
    batch_sec = time.time() - t0
    print(f"batched {n_envs} envs x {n_steps} steps: {batch_sec:.2f}s "
          f"({n_envs * n_steps / batch_sec:.0f} env-steps/s, python loop)")

    # Upstream quirk: reset() builds data without mj_forward, so xpos/site_xpos
    # (and the obs fields derived from them) are all zero at t=0. They become
    # valid after the first step. We keep this as-is since the PPO teacher is
    # trained on the same observations; check geometry after one zero step.
    reset_states = v_reset(keys)
    assert np.allclose(np.asarray(eu.box_pos(env, reset_states)), 0.0)
    init_states = v_step(reset_states, zeros)
    init_box = np.asarray(eu.box_pos(env, init_states))
    init_target = np.asarray(init_states.info["target_pos"])
    qpos_box = np.asarray(reset_states.data.qpos[:, env._obj_qposadr:env._obj_qposadr + 3])
    assert np.allclose(init_box, qpos_box, atol=1e-3), "box moved during the first step"

    # Success metric must agree with the target_minus_box slice of the obs.
    dist = np.asarray(eu.box_target_dist(env, init_states))
    dist_from_obs = np.linalg.norm(np.asarray(init_states.obs)[:, layout["target_minus_box"]], axis=-1)
    assert np.allclose(dist, dist_from_obs, atol=1e-5)
    assert eu.is_success(0.0) and eu.is_success(eu.SUCCESS_DIST - 1e-3)
    assert not eu.is_success(eu.SUCCESS_DIST + 1e-3)
    assert not eu.is_success(dist).any(), "some reset already counts as success"
    print(f"initial box-target dist: min {dist.min():.3f}, mean {dist.mean():.3f} m")

    # JAX -> torch hand-off on the same GPU.
    obs_t = torch.from_numpy(np.array(states.obs)).cuda()
    assert obs_t.shape == (n_envs, env.observation_size)
    torch_ok = bool(torch.isfinite(obs_t).all())

    info = {
        "env_name": eu.ENV_NAME,
        "env_overrides": eu.ENV_OVERRIDES,
        "env_config": env._config.to_dict(),
        "observation_size": int(env.observation_size),
        "action_size": int(env.action_size),
        "ctrl_dt": float(env.dt),
        "sim_dt": float(env.sim_dt),
        "obs_layout": {k: [v.start, v.stop] for k, v in layout.items()},
        "actuators": [env.mj_model.actuator(i).name for i in range(env.mj_model.nu)],
        "ctrl_range": np.stack([lowers, uppers], 1).tolist(),
        "success_dist_m": eu.SUCCESS_DIST,
        "replay_max_obs_diff": replay_err,
        "note_reset_xpos_zero": "reset() skips mj_forward: xpos-derived obs fields are 0 at t=0",
        "init_box_xy_range": [init_box[:, :2].min(0).tolist(), init_box[:, :2].max(0).tolist()],
        "init_target_range": [init_target.min(0).tolist(), init_target.max(0).tolist()],
        "batched_100x150_seconds": batch_sec,
        "torch_handoff_ok": torch_ok,
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "cuda_torch": torch.version.cuda,
            "cudnn_torch": torch.backends.cudnn.version(),
            "playground_commit": subprocess.run(
                ["git", "-C", str(ROOT / "third_party" / "mujoco_playground"), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "jax_devices": [str(d) for d in jax.devices()],
            "platform": platform.platform(),
        },
    }
    out = ROOT / "results" / "env_info.json"
    out.write_text(json.dumps(info, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
