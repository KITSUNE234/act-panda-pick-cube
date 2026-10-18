"""Batched closed-loop rollouts on the raw (un-wrapped) env, shared by teacher
evaluation, demonstration collection, and student evaluation."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from common import env_utils as eu


class BatchedEnv:
    """jit(vmap(...)) reset/step for one env instance, compiled once."""

    def __init__(self, env):
        self.env = env
        self.reset = jax.jit(jax.vmap(env.reset))
        self.step = jax.jit(jax.vmap(env.step))
        self.episode_length = int(env._config.episode_length)


def rollout(benv: BatchedEnv, act_fn, keys, n_steps=None):
    """Runs len(keys) episodes in parallel.

    act_fn(obs[B, D], t) -> actions[B, A]; it may return a JAX or NumPy array.
    The action stored at index t is exactly the one executed from obs[:, t].
    Episodes are not auto-reset; after an early `done` (box out of bounds) the
    sim keeps stepping but those steps are masked out via `length`.

    Returns a dict of NumPy arrays:
      obs [B,T,D], actions [B,T,A], rewards [B,T], dones [B,T],
      qpos [B,T+1,nq], qvel [B,T+1,nv]  (t=0 is the reset state),
      box_pos / gripper_pos [B,T+1,3], target_pos [B,3], box_target_dist [B,T+1],
      length [B], success [B], final_dist [B], reached_box [B], first_success_step [B].
    """
    env = benv.env
    T = n_steps or benv.episode_length
    state = benv.reset(keys)

    obs, actions, rewards, dones = [], [], [], []
    qpos, qvel = [state.data.qpos], [state.data.qvel]
    box, grip = [eu.box_pos(env, state)], [eu.gripper_pos(env, state)]
    for t in range(T):
        a = jnp.asarray(act_fn(state.obs, t), dtype=jnp.float32)
        obs.append(state.obs)
        actions.append(a)
        state = benv.step(state, a)
        rewards.append(state.reward)
        dones.append(state.done)
        qpos.append(state.data.qpos)
        qvel.append(state.data.qvel)
        box.append(eu.box_pos(env, state))
        grip.append(eu.gripper_pos(env, state))

    stack = lambda xs: np.asarray(jnp.stack(xs, axis=1))
    out = {
        "obs": stack(obs),
        "actions": stack(actions),
        "rewards": stack(rewards),
        "dones": stack(dones),
        "qpos": stack(qpos),
        "qvel": stack(qvel),
        "box_pos": stack(box),
        "gripper_pos": stack(grip),
        "target_pos": np.asarray(state.info["target_pos"]),
        "reached_box": np.asarray(state.info["reached_box"]),
    }
    dist = np.linalg.norm(out["box_pos"] - out["target_pos"][:, None], axis=-1)
    # xpos is zero at reset (no mj_forward upstream); use the qpos box position.
    adr = env._obj_qposadr
    dist[:, 0] = np.linalg.norm(out["qpos"][:, 0, adr:adr + 3] - out["target_pos"], axis=-1)
    out["box_target_dist"] = dist

    done_any = out["dones"].any(axis=1)
    first_done = np.where(done_any, out["dones"].argmax(axis=1), T - 1)
    out["length"] = first_done + 1
    out["final_dist"] = dist[:, -1]
    out["success"] = eu.is_success(out["final_dist"]) & ~done_any
    within = eu.is_success(dist[:, 1:])
    out["first_success_step"] = np.where(within.any(axis=1), within.argmax(axis=1) + 1, -1)
    return out


def summarize(out):
    s = out["success"]
    n = len(s)
    lo, hi = wilson_ci(int(s.sum()), n)
    return {
        "n_episodes": n,
        "success_rate": float(s.mean()),
        "success_ci95": [lo, hi],
        "final_dist_mean": float(out["final_dist"].mean()),
        "final_dist_median": float(np.median(out["final_dist"])),
        "reached_box_rate": float(out["reached_box"].mean()),
        "early_done_rate": float(out["dones"].any(axis=1).mean()),
        "mean_return": float(out["rewards"].sum(axis=1).mean()),
    }


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [float(centre - half), float(centre + half)]


def render_episode(env, qpos, target_pos, height=480, width=640, every=2, qvel=None):
    """Renders one episode from recorded qpos [T+1, nq] and its target position."""
    m = env.mj_model
    renderer = mujoco.Renderer(m, height=height, width=width)
    d = mujoco.MjData(m)
    mocap_id = int(np.asarray(env._mocap_target).reshape(-1)[0])
    frames = []
    for t in range(0, len(qpos), every):
        d.qpos[:] = qpos[t]
        if qvel is not None:
            d.qvel[:] = qvel[t]
        d.mocap_pos[mocap_id] = target_pos
        d.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d)
        frames.append(renderer.render())
    renderer.close()
    return np.stack(frames)
