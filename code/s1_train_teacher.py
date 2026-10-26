"""Phase 1: train the PPO teacher for PandaPickCube.

Uses Playground's tuned Brax PPO hyperparameters (manipulation_params) and the
same training wrapper as learning/train_jax_ppo.py, but saves a checkpoint and
a metrics row at every evaluation so teacher competence can be tracked over time.

  python code/s1_train_teacher.py --seed 0 --out checkpoints/teacher_seed0
"""

import argparse
import functools
import json
import pathlib
import time

from common import env_utils as eu

import jax
import numpy as np
import pandas as pd
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/teacher_seed0")
    p.add_argument("--num_timesteps", type=int, default=None,
                   help="default: Playground's tuned value (20M)")
    # Brax rounds each eval interval up to whole blocks of 10 resets x 163,840
    # env steps. Upstream (20M, 4 evals) therefore trains 3 x 5 blocks = 24.58M
    # steps; 16 evals gives the same 24.58M total with a checkpoint every block.
    p.add_argument("--num_evals", type=int, default=16,
                   help="evals incl. step 0; one checkpoint per eval after step 0")
    p.add_argument("--impl", default=eu.ENV_OVERRIDES["impl"], choices=["jax", "warp"],
                   help="MJX backend used for PPO training")
    # Deviation from upstream: floor the observation-normalizer std at
    # sqrt(0.01) = 0.1. With the default 0.0, the box-height dim has std ~0.003
    # while the box rests on the table; once the policy starts lifting, that
    # input jumps to ~100 sigma, the value loss explodes (~1e14), and PPO
    # collapses to 0% success (see README.md and the report). 0.0 reproduces upstream.
    p.add_argument("--obs_std_eps", type=float, default=0.01,
                   help="brax normalize_observations_std_eps (upstream: 0.0)")
    p.add_argument("--ppo_override", default="{}",
                   help='JSON dict of extra brax ppo.train kwargs, e.g. \'{"normalize_observations_std_eps": 0.01}\'')
    args = p.parse_args()

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ppo_params = manipulation_params.brax_ppo_config(eu.ENV_NAME, args.impl)
    if args.num_timesteps is not None:
        ppo_params.num_timesteps = args.num_timesteps
    ppo_params.num_evals = args.num_evals
    ppo_params.normalize_observations_std_eps = args.obs_std_eps
    for k, v in json.loads(args.ppo_override).items():
        ppo_params[k] = v

    env = eu.load_env(args.impl)
    eval_env = eu.load_env(args.impl)

    training_params = dict(ppo_params)
    del training_params["network_factory"]
    num_eval_envs = training_params.pop("num_eval_envs", 128)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory)

    config = {
        "seed": args.seed,
        "env_name": eu.ENV_NAME,
        "impl": args.impl,
        "env_config": env._config.to_dict(),
        "ppo_params": ppo_params.to_dict(),
        "num_eval_envs": num_eval_envs,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(json.dumps(config["ppo_params"], indent=2, default=str))

    # Metric keys differ between the step-0 eval and later ones, so log JSON lines.
    log_path = out / "progress.jsonl"
    log_f = open(log_path, "w")
    t_start = time.monotonic()

    def progress(num_steps, metrics):
        row = {"step": int(num_steps), "wall_sec": time.monotonic() - t_start}
        row.update({k: float(np.asarray(v)) for k, v in metrics.items()})
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(f"step {num_steps:>10,d} | eval reward {row.get('eval/episode_reward', float('nan')):8.2f}"
              f" ± {row.get('eval/episode_reward_std', float('nan')):6.2f} | {row['wall_sec']:7.1f}s",
              flush=True)

    _, _, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
        **training_params,
        network_factory=network_factory,
        seed=args.seed,
        save_checkpoint_path=out / "checkpoints",
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )
    log_f.close()
    pd.read_json(log_path, lines=True).to_csv(out / "progress.csv", index=False)
    print(f"done in {time.monotonic() - t_start:.1f}s; checkpoints in {out / 'checkpoints'}")


if __name__ == "__main__":
    main()
