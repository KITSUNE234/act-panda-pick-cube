"""Loading Brax PPO checkpoints as batched teacher policies."""

import json
import pathlib

import jax
from brax.training import checkpoint as brax_checkpoint
from brax.training import networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict


def list_checkpoints(run_dir):
    """Step-numbered checkpoint dirs under run_dir/checkpoints, sorted by step."""
    ckpt_root = pathlib.Path(run_dir).resolve() / "checkpoints"  # orbax needs absolute paths
    dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def _load_network_config(ckpt_path):
    """Like brax's ppo checkpoint.load_config, with two fixes for this brax commit:
    save() skips kernel-init fields that are None but load_config() looks them
    up anyway (KeyError: None), and observation_size is stored as
    {"shape": [66], "dtype": <unserializable>} instead of an int."""
    d = json.loads((pathlib.Path(ckpt_path) / "ppo_network_config.json").read_text())
    kw = d["network_factory_kwargs"]
    if "activation" in kw:
        kw["activation"] = networks.ACTIVATION[kw["activation"]]
    for name in brax_checkpoint._KERNEL_INIT_FN_KEYWORDS:
        if kw.get(name) is not None:
            kw[name] = networks.KERNEL_INITIALIZER[kw[name]]
    if isinstance(d["observation_size"], dict):
        d["observation_size"] = int(d["observation_size"]["shape"][0])
    return config_dict.create(**d)


def load_teacher(ckpt_path, deterministic=True):
    """Returns act(obs[B, D], key) -> actions[B, A] in [-1, 1] (tanh-squashed)."""
    config = _load_network_config(ckpt_path)
    params = ppo_checkpoint.load(ckpt_path)
    network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
    policy = ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)

    @jax.jit
    def act(obs, key):
        keys = jax.random.split(key, obs.shape[0])
        return jax.vmap(policy)(obs, keys)[0]

    return act
