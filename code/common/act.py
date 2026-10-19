"""State-only ACT (Action Chunking with Transformers) in PyTorch.

Follows the architecture of Zhao et al., "Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware" (RSS 2023; github.com/tonyzhaozh/act),
with these changes for a 66-D state observation and no cameras:
  * no image backbone; the observation is split into its semantic groups
    (env_utils.obs_layout) and each group is linearly projected to one token,
    so the main encoder attends over [z, 9 observation tokens];
  * learned positional embeddings everywhere, added once at the input
    (the reference adds fixed sinusoidal / learned embeddings in every layer);
  * stock torch.nn.Transformer{Encoder,Decoder} layers (post-norm, ReLU).
The CVAE structure, L1 + beta * KL objective, z = 0 at test time, and
temporal ensembling with weights exp(-m * i) (i = 0 for the oldest prediction)
are as in the reference.
"""

import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class ACTConfig:
    obs_dim: int = 66
    action_dim: int = 8
    chunk: int = 20
    hidden: int = 256
    heads: int = 8
    ff: int = 1024
    enc_layers: int = 4
    dec_layers: int = 4
    cvae_layers: int = 4
    latent: int = 32
    dropout: float = 0.1
    # (start, stop) of each observation group, from env_utils.obs_layout.
    obs_groups: tuple = ()


def _encoder(cfg, n_layers):
    layer = nn.TransformerEncoderLayer(cfg.hidden, cfg.heads, cfg.ff, cfg.dropout, batch_first=True)
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        groups = cfg.obs_groups or ((0, cfg.obs_dim),)
        H = cfg.hidden
        # CVAE encoder: [CLS, obs, a_1..a_k] -> (mu, logvar) of z.
        self.cls = nn.Parameter(torch.zeros(1, 1, H))
        self.cvae_obs = nn.Linear(cfg.obs_dim, H)
        self.cvae_act = nn.Linear(cfg.action_dim, H)
        self.cvae_pos = nn.Parameter(torch.zeros(1, 2 + cfg.chunk, H))
        self.cvae_enc = _encoder(cfg, cfg.cvae_layers)
        self.to_latent = nn.Linear(H, 2 * cfg.latent)
        # Policy: encoder over [z, obs groups], decoder with k learned queries.
        self.groups = [tuple(g) for g in groups]
        self.obs_proj = nn.ModuleList(nn.Linear(b - a, H) for a, b in self.groups)
        self.latent_proj = nn.Linear(cfg.latent, H)
        self.enc_pos = nn.Parameter(torch.zeros(1, 1 + len(self.groups), H))
        self.encoder = _encoder(cfg, cfg.enc_layers)
        self.queries = nn.Parameter(torch.zeros(1, cfg.chunk, H))
        dec_layer = nn.TransformerDecoderLayer(H, cfg.heads, cfg.ff, cfg.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers, norm=nn.LayerNorm(H))
        self.head = nn.Linear(H, cfg.action_dim)
        for p in (self.cls, self.cvae_pos, self.enc_pos, self.queries):
            nn.init.normal_(p, std=0.02)

    def encode_latent(self, obs, actions, pad):
        """obs [B,D], actions [B,k,A], pad [B,k] (True = padded) -> mu, logvar [B,latent]."""
        B = obs.shape[0]
        tokens = torch.cat([self.cls.expand(B, -1, -1), self.cvae_obs(obs)[:, None],
                            self.cvae_act(actions)], dim=1) + self.cvae_pos
        mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=obs.device), pad], dim=1)
        h = self.cvae_enc(tokens, src_key_padding_mask=mask)[:, 0]
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs, z):
        """obs [B,D], z [B,latent] -> predicted action chunk [B,k,A]."""
        B = obs.shape[0]
        tokens = [self.latent_proj(z)] + [proj(obs[:, a:b]) for proj, (a, b) in zip(self.obs_proj, self.groups)]
        memory = self.encoder(torch.stack(tokens, dim=1) + self.enc_pos)
        h = self.decoder(self.queries.expand(B, -1, -1), memory)
        return self.head(h)

    def forward(self, obs, actions=None, pad=None):
        """Training (actions given): returns (pred, mu, logvar). Inference: pred with z = 0."""
        if actions is None:
            z = torch.zeros(obs.shape[0], self.cfg.latent, device=obs.device)
            return self.decode(obs, z)
        mu, logvar = self.encode_latent(obs, actions, pad)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(obs, z), mu, logvar


def act_loss(pred, target, pad, mu, logvar, kl_weight):
    """Masked L1 over valid chunk steps + kl_weight * KL(q(z|o,a) || N(0, I))."""
    valid = (~pad).unsqueeze(-1).float()
    l1 = (F.l1_loss(pred, target, reduction="none") * valid).sum() / (valid.sum() * pred.shape[-1])
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl


class Normalizer:
    """Per-dimension (x - mean) / max(std, std_floor), fit on the training data."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    @classmethod
    def fit(cls, x, std_floor):
        x = x.reshape(-1, x.shape[-1])
        return cls(x.mean(0), np.maximum(x.std(0), std_floor))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def save_policy(path, model, obs_norm, act_norm, extra=None):
    torch.save({"config": dataclasses.asdict(model.cfg), "model": model.state_dict(),
                "obs_norm": obs_norm.state_dict(), "act_norm": act_norm.state_dict(),
                **(extra or {})}, path)


def load_policy(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ACTConfig(**{**ckpt["config"], "obs_groups": tuple(map(tuple, ckpt["config"]["obs_groups"]))})
    model = ACT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Normalizer(**ckpt["obs_norm"]), Normalizer(**ckpt["act_norm"])


class ChunkExecutor:
    """Turns a chunk-predicting model into act_fn(obs[B,D], t) for rollout.rollout.

    mode="ensemble": query every step and average all predictions for step t with
      weights exp(-m * i), i = 0 for the oldest (ACT temporal ensembling).
    mode="open_loop": query every `query_every` steps and execute the chunk open loop.
    """

    def __init__(self, model, obs_norm, act_norm, mode="ensemble", m=0.01, query_every=None,
                 device="cuda"):
        assert mode in ("ensemble", "open_loop")
        self.model, self.obs_norm, self.act_norm = model, obs_norm, act_norm
        self.mode, self.m, self.device = mode, m, device
        self.k = model.cfg.chunk
        self.query_every = query_every or self.k
        assert 1 <= self.query_every <= self.k
        self.buf = None

    @torch.no_grad()
    def _predict(self, obs):
        x = torch.as_tensor(self.obs_norm.norm(np.asarray(obs, np.float32)), device=self.device)
        pred = self.model(x).cpu().numpy()
        return np.clip(self.act_norm.denorm(pred), -1.0, 1.0)

    def __call__(self, obs, t):
        if t == 0:
            self.buf = {}
        if self.mode == "open_loop":
            if t % self.query_every == 0:
                self.chunk, self.t0 = self._predict(obs), t
            return self.chunk[:, t - self.t0]
        self.buf[t] = self._predict(obs)
        self.buf.pop(t - self.k, None)
        # Predictions for step t made at steps s = t-k+1..t, oldest first.
        preds = np.stack([self.buf[s][:, t - s] for s in sorted(self.buf)], axis=1)
        w = np.exp(-self.m * np.arange(preds.shape[1]))
        return (preds * (w / w.sum())[None, :, None]).sum(1).astype(np.float32)
