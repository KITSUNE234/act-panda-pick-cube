"""Shared environment helpers: the single source of truth for env loading,
observation layout, and the success criterion used by teacher, dataset, and
student evaluation alike."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
# Let JAX and PyTorch share the GPU instead of JAX grabbing 75% up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from mujoco_playground import registry

ENV_NAME = "PandaPickCube"
# The README example uses the JAX MJX backend; Playground's default is "warp".
ENV_OVERRIDES = {"impl": "jax"}

# Success: box centre within this distance (m) of the target at the final step.
# The box is 4 x 4 x 6 cm, so 5 cm means "box overlaps the target volume".
SUCCESS_DIST = 0.05


def load_env():
    return registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)


def obs_layout(env):
    """Named slices of the flat observation vector (mirrors PandaPickCube._get_obs)."""
    m = env.mj_model
    sizes = [
        ("qpos", m.nq),
        ("qvel", m.nv),
        ("gripper_pos", 3),
        ("gripper_mat[3:]", 6),
        ("box_mat[3:]", 6),
        ("box_minus_gripper", 3),
        ("target_minus_box", 3),
        ("target_mat_minus_box_mat[:6]", 6),
        ("ctrl_minus_robot_qpos", m.nu),
    ]
    layout, start = {}, 0
    for name, n in sizes:
        layout[name] = slice(start, start + n)
        start += n
    assert start == env.observation_size, (start, env.observation_size)
    return layout


def box_pos(env, state):
    return state.data.xpos[..., env._obj_body, :]


def gripper_pos(env, state):
    return state.data.site_xpos[..., env._gripper_site, :]


def box_target_dist(env, state):
    """Distance from box centre to target; works on single or batched states."""
    return jnp.linalg.norm(state.info["target_pos"] - box_pos(env, state), axis=-1)


def is_success(final_dist):
    return final_dist < SUCCESS_DIST


def episode_keys(base_seed: int, n: int):
    """Deterministic per-episode reset keys; episode i always gets the same key."""
    return jax.random.split(jax.random.PRNGKey(base_seed), n)


def render(env, states, height=480, width=640, every=2):
    """Render a list of *unbatched* states to frames."""
    return env.render(states[::every], height=height, width=width)


def video_fps(env, every=2):
    return round(1.0 / env.dt / every)


def to_numpy(tree):
    return jax.tree_util.tree_map(np.asarray, tree)
