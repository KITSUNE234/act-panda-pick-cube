"""Phase 4: train a state-only ACT student on a demonstration dataset.

Training samples are (obs_t, a_t..a_{t+k-1}) for every step t of every training
episode; chunks running past the episode end are zero-padded and masked out of
the loss (and of the CVAE encoder), as in the reference ACT implementation.
Observations and actions are standardized with statistics of the training
episodes (std floored at --std_floor). Training runs for a fixed number of
gradient steps; the final weights are used, with no checkpoint selection.

Monitoring (never used for model selection):
  * val L1: z = 0 prediction error on the 97 successful collected episodes that
    are not in the baseline dataset (demos_raw.npz, kept = False);
  * val success: closed-loop success on the validation reset keys
    (configs/seeds.yaml), disjoint from the collection and evaluation keys;
  * train success: closed-loop success from the reset keys of (up to 50)
    training episodes, i.e. on initial conditions seen in training.

Writes checkpoints/act_<dataset>_seed<seed>/{policy.pt, progress.csv, train_config.json}

  python code/s4_train_act.py --dataset baseline_100 --seed 0
"""

import argparse
import dataclasses
import json
import os
import time

from common import env_utils as eu
from common import rollout as ro
from common.act import ACT, ACTConfig, ChunkExecutor, Normalizer, act_loss, save_policy, setup_torch
from common.demos import load_demos

import jax
import numpy as np
import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = eu.ROOT


def chunk_targets(actions, k):
    """actions [N,T,A] -> zero-padded [N,T+k,A] and pad mask [T+k] (True past the end)."""
    N, T, A = actions.shape
    padded = np.concatenate([actions, np.zeros((N, k, A), actions.dtype)], axis=1)
    return padded, np.arange(T + k) >= T


class ChunkData:
    """All (episode, t) samples of a set of episodes, resident on the GPU."""

    def __init__(self, obs, actions, k, obs_norm, act_norm, device):
        self.N, self.T = obs.shape[:2]
        self.k = k
        padded, pad = chunk_targets(act_norm.norm(actions), k)
        padded[:, self.T:] = 0.0  # padding stays zero after normalization
        self.obs = torch.as_tensor(obs_norm.norm(obs).reshape(self.N * self.T, -1), device=device)
        self.act = torch.as_tensor(padded, device=device)
        self.pad = torch.as_tensor(pad, device=device)
        self.offsets = torch.arange(k, device=device)

    def __len__(self):
        return self.N * self.T

    def get(self, flat_idx):
        ep, t = flat_idx // self.T, flat_idx % self.T
        steps = t[:, None] + self.offsets
        return self.obs[flat_idx], self.act[ep[:, None], steps], self.pad[steps]


@torch.no_grad()
def val_l1(model, data, batch=4096):
    model.eval()
    total, count = 0.0, 0.0
    for s in range(0, len(data), batch):
        idx = torch.arange(s, min(s + batch, len(data)), device=data.obs.device)
        obs, act, pad = data.get(idx)
        pred = model(obs)
        valid = (~pad).unsqueeze(-1).float()
        total += float((torch.abs(pred - act) * valid).sum())
        count += float(valid.sum()) * act.shape[-1]
    model.train()
    return total / count


def rollout_success(model, obs_norm, act_norm, benv, keys, m):
    model.eval()
    out = ro.rollout(benv, ChunkExecutor(model, obs_norm, act_norm, mode="ensemble", m=m), keys)
    model.train()
    return ro.summarize(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="baseline_100", help="name in configs/datasets.yaml")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="debug: train on these baseline episode indices instead of --dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--kl_weight", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4, help="CVAE encoder, encoder and decoder layers")
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--std_floor", type=float, default=0.01)
    p.add_argument("--ensemble_m", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--rollout_every", type=int, default=5000)
    p.add_argument("--wandb", default=os.environ.get("WANDB_MODE", "online"),
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", default="csci645-hw2-act")
    args = p.parse_args()

    name = f"act_{args.dataset}_seed{args.seed}" if args.episodes is None else \
        f"act_debug_ep{'-'.join(map(str, args.episodes))}_seed{args.seed}"
    out_dir = ROOT / (args.out or f"checkpoints/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_torch()
    device = "cuda"

    # Data.
    if args.episodes is None:
        train = load_demos(args.dataset)
    else:
        full = load_demos("baseline_100")
        train = {k: v[args.episodes] for k, v in full.items()}
    with np.load(ROOT / "data" / "demos_raw.npz") as f:
        held = f["success"] & ~f["kept"]
        val = {"obs": f["obs"][held], "actions": f["actions"][held]}
    obs_norm = Normalizer.fit(train["obs"], args.std_floor)
    act_norm = Normalizer.fit(train["actions"], args.std_floor)
    data = ChunkData(train["obs"], train["actions"], args.chunk, obs_norm, act_norm, device)
    val_data = ChunkData(val["obs"], val["actions"], args.chunk, obs_norm, act_norm, device)

    # Model.
    env = eu.load_env()
    layout = eu.obs_layout(env)
    cfg = ACTConfig(obs_dim=train["obs"].shape[-1], action_dim=train["actions"].shape[-1],
                    chunk=args.chunk, hidden=args.hidden, heads=8, ff=4 * args.hidden,
                    enc_layers=args.layers, dec_layers=args.layers, cvae_layers=args.layers,
                    latent=args.latent, dropout=args.dropout,
                    obs_groups=tuple((s.start, s.stop) for s in layout.values()))
    model = ACT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    benv = ro.BatchedEnv(env)
    val_keys = eu.val_keys()
    train_keys = jax.numpy.asarray(train["reset_key"][:50])
    for other in (eu.eval_keys(), eu.episode_keys(eu.load_seeds()["collect_base_seed"], 200)):
        assert not (np.asarray(val_keys)[:, None] == np.asarray(other)[None]).all(-1).any()

    config = {**vars(args), "run_name": name, "n_train_episodes": data.N,
              "n_train_samples": len(data), "n_val_episodes": val_data.N,
              "n_params": n_params, "model": dataclasses.asdict(cfg),
              "train_episode_collect_index": train["collect_index"].tolist(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    run = None
    if args.wandb != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name=name, config=config,
                         dir=str(ROOT), mode=args.wandb)
        config["wandb_url"] = run.url
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    print(f"{name}: {data.N} episodes, {len(data)} samples, {n_params / 1e6:.2f}M params")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows, acc, t0 = [], [], time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(data), (args.batch,), device=device, generator=gen)
        obs, act, pad = data.get(idx)
        pred, mu, logvar = model(obs, act, pad)
        loss, l1, kl = act_loss(pred, act, pad, mu, logvar, args.kl_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        opt.step()
        acc.append(torch.stack([loss, l1, kl, gnorm]).detach())

        if step % args.log_every == 0:
            m = torch.stack(acc).mean(0).tolist()
            acc = []
            row = {"step": step, "loss": m[0], "l1": m[1], "kl": m[2], "grad_norm": m[3],
                   "epoch": step * args.batch / len(data), "time_s": time.time() - t0}
            if step % args.val_every == 0 or step == args.steps:
                row["val_l1"] = val_l1(model, val_data)
            if step % args.rollout_every == 0 or step == args.steps:
                s = rollout_success(model, obs_norm, act_norm, benv, val_keys, args.ensemble_m)
                row.update(val_success=s["success_rate"], val_final_dist=s["final_dist_mean"])
                s = rollout_success(model, obs_norm, act_norm, benv, train_keys, args.ensemble_m)
                row.update(train_success=s["success_rate"], train_final_dist=s["final_dist_mean"])
            rows.append(row)
            if run is not None:
                run.log(row, step=step)
            if "val_l1" in row:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in row.items()), flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "progress.csv", index=False)
    save_policy(out_dir / "policy.pt", model, obs_norm, act_norm,
                extra={"train_config": config, "steps": args.steps})
    if run is not None:
        run.finish()
    print(f"wrote {out_dir.relative_to(ROOT)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    with sdpa_kernel(SDPBackend.MATH):
        main()
