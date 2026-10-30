"""Training: supervised (from scratch or from a pretrained encoder), linear probes, contrastive pretraining.

No early stopping and no model selection on any test domain: every run follows a fixed schedule from
config/experiment.yaml. The bench validation split is used for one thing only, temperature scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.encoder import Classifier, Encoder, Projector
from ..signals.dataset import Windows
from .augment import augment


def tensors(w: Windows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.from_numpy(w.x), torch.from_numpy(w.mask.astype(np.float32)), torch.from_numpy(w.y)


@dataclass
class Outputs:
    logits: np.ndarray
    embeddings: np.ndarray


@torch.no_grad()
def predict(model: Classifier, w: Windows, batch: int = 1024) -> Outputs:
    model.eval()
    x, m, _ = tensors(w)
    logits, emb = [], []
    for i in range(0, len(w), batch):
        z, lo = model.embed_and_logits(x[i : i + batch], m[i : i + batch])
        logits.append(lo.numpy())
        emb.append(z.numpy())
    return Outputs(
        np.concatenate(logits) if logits else np.zeros((0, 0)), np.concatenate(emb) if emb else np.zeros((0, 0))
    )


def _cosine(step: int, total: int) -> float:
    return 0.5 * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))


def train_classifier(
    train: Windows,
    n_classes: int,
    cfg: dict[str, Any],
    aug: dict[str, Any],
    epochs: int,
    min_steps: int,
    seed: int,
    encoder: Encoder | None = None,
) -> Classifier:
    """Supervised training; with `encoder`, fine-tunes it (lower learning rate for the encoder)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Classifier(n_classes)
    if encoder is not None:
        model.encoder.load_state_dict(encoder.state_dict())
        groups = [
            {"params": model.encoder.parameters(), "lr": cfg["finetune_encoder_lr"]},
            {"params": model.head.parameters(), "lr": cfg["lr"]},
        ]
    else:
        groups = [{"params": model.parameters(), "lr": cfg["lr"]}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    base = [gr["lr"] for gr in opt.param_groups]
    x, m, y = tensors(train)
    bs = min(cfg["batch"], len(train))
    steps = max(min_steps, epochs * math.ceil(len(train) / bs))
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(train), (bs,), generator=g)
        xb, mb = augment(x[idx], m[idx], aug, g)
        loss = F.cross_entropy(model(xb, mb), y[idx])
        for gr, b in zip(opt.param_groups, base, strict=True):
            gr["lr"] = b * _cosine(step, steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def linear_probe(encoder: Encoder, train: Windows, n_classes: int, steps: int, seed: int) -> Classifier:
    """Frozen encoder; only the linear head is trained, on precomputed embeddings."""
    torch.manual_seed(seed)
    model = Classifier(n_classes)
    model.encoder.load_state_dict(encoder.state_dict())
    model.eval()
    z = torch.from_numpy(predict(model, train).embeddings)
    y = torch.from_numpy(train.y)
    mu, sd = z.mean(0), z.std(0).clamp_min(1e-6)
    zn = (z - mu) / sd
    head = torch.nn.Linear(z.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(head(zn), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    # fold the standardisation into the head so the model is a plain Classifier
    with torch.no_grad():
        model.head.weight.copy_(head.weight / sd[None, :])
        model.head.bias.copy_(head.bias - (head.weight * (mu / sd)[None, :]).sum(1))
    return model


def nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(torch.cat([a, b]), dim=1)
    sim = z @ z.T / temperature
    n = a.shape[0]
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def pretrain_contrastive(
    pool: Windows, cfg: dict[str, Any], aug: dict[str, Any], epochs: int, seed: int
) -> tuple[Encoder, list[float]]:
    """SimCLR on unlabelled windows: two strongly augmented views of a window should embed together."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    enc, proj = Encoder(), Projector()
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["ssl_lr"], weight_decay=cfg["weight_decay"])
    x, m, _ = tensors(pool)
    bs = min(cfg["ssl_batch"], len(pool))
    steps = max(1, epochs * (len(pool) // bs))
    losses = []
    enc.train()
    proj.train()
    for step in range(steps):
        idx = torch.randint(0, len(pool), (bs,), generator=g)
        v1 = augment(x[idx], m[idx], aug, g)
        v2 = augment(x[idx], m[idx], aug, g)
        loss = nt_xent(proj(enc(*v1)), proj(enc(*v2)), cfg["ssl_temperature"])
        opt.param_groups[0]["lr"] = cfg["ssl_lr"] * _cosine(step, steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    enc.eval()
    return enc, losses
