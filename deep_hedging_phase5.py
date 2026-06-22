"""
Deep Hedging — Phase 5: robustness to model misspecification.

The premise
-----------
You never know the true dynamics. You calibrate a model, train your hedger on
it, sell at its price — and reality differs. A good hedge should degrade
gracefully. Here we take the Phase 2 deep hedger (no vol feature, trained on ONE
Heston calibration) and evaluate the FIXED learned policy on worlds it never saw:
  * Heston, base params (matched — the Phase 2 result, as a reference)
  * Heston, higher vol-of-vol (xi up)
  * Heston, stronger skew (rho more negative)
  * Heston, higher vol level (v0, theta up — a regime shift)
  * Merton jump-diffusion (structurally different: discontinuous crashes)
For each world we RE-PIN the premium to that world's fair price (so mean P&L is
centred and std / CVaR measure hedging quality, not mispricing) and compare the
deep hedger to the constant-vol BS delta (sigma = sqrt(theta_base), also
misspecified under shifts — exactly as a real desk's fixed assumption would be).

Reading it
----------
If the deep hedger keeps its edge over BS delta across parameter shifts, the
learned hedge is robust, not overfit to one calibration. Under jumps (a risk the
diffusive training never contained) both should degrade — the honest question is
by how much, and whether the edge survives.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Config:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.0
    mu: float = 0.0
    T: float = 60.0 / 365.0
    n_steps: int = 30
    # base Heston calibration (the training world = Phase 2)
    v0: float = 0.04
    theta: float = 0.04
    kappa: float = 1.5
    xi: float = 0.5
    rho: float = -0.7
    alpha: float = 0.99
    hidden: Tuple[int, ...] = (32, 32)
    batch: int = 8192
    epochs: int = 1200
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def sigma_bs(self) -> float:
        return math.sqrt(self.theta)        # the desk's fixed constant-vol assumption

    def base_params(self) -> Dict[str, float]:
        return dict(v0=self.v0, theta=self.theta, kappa=self.kappa, xi=self.xi, rho=self.rho)


# --------------------------------------------------------------------------- #
def simulate_heston(cfg, p, n, gen):
    dt = cfg.dt; sdt = math.sqrt(dt)
    Zv = torch.randn(n, cfg.n_steps, generator=gen, device=cfg.device)
    Zp = torch.randn(n, cfg.n_steps, generator=gen, device=cfg.device)
    Zs = p["rho"] * Zv + math.sqrt(1 - p["rho"] ** 2) * Zp
    logS = torch.empty(n, cfg.n_steps + 1, device=cfg.device); logS[:, 0] = math.log(cfg.S0)
    v = torch.empty(n, cfg.n_steps + 1, device=cfg.device); v[:, 0] = p["v0"]
    for k in range(cfg.n_steps):
        vp = torch.clamp(v[:, k], min=0.0); sv = torch.sqrt(vp)
        logS[:, k + 1] = logS[:, k] + (cfg.mu - 0.5 * vp) * dt + sv * sdt * Zs[:, k]
        v[:, k + 1] = v[:, k] + p["kappa"] * (p["theta"] - vp) * dt + p["xi"] * sv * sdt * Zv[:, k]
    return torch.exp(logS)


def simulate_merton(cfg, n, gen, sig=0.18, lam=1.0, mJ=-0.10, sJ=0.05):
    dt = cfg.dt; sdt = math.sqrt(dt)
    Z = torch.randn(n, cfg.n_steps, generator=gen, device=cfg.device)
    # compensator so the process is a martingale under mu=0
    comp = lam * (math.exp(mJ + 0.5 * sJ ** 2) - 1.0)
    counts = torch.poisson(torch.full((n, cfg.n_steps), lam * dt), generator=gen)
    Zj = torch.randn(n, cfg.n_steps, generator=gen, device=cfg.device)
    jump = counts * mJ + torch.sqrt(torch.clamp(counts, min=0.0)) * sJ * Zj
    incr = (cfg.mu - comp - 0.5 * sig ** 2) * dt + sig * sdt * Z + jump
    logS = math.log(cfg.S0) + torch.cat(
        [torch.zeros(n, 1, device=cfg.device), torch.cumsum(incr, 1)], 1)
    return torch.exp(logS)


def gen_paths(cfg, kind, p, n, seed):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    return simulate_merton(cfg, n, gen) if kind == "merton" else simulate_heston(cfg, p, n, gen)


def premium_of(cfg, kind, p, n=200_000):
    S = gen_paths(cfg, kind, p, n, seed=999)
    return torch.clamp(S[:, -1] - cfg.K, min=0.0).mean().item()


# --------------------------------------------------------------------------- #
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_delta(S, K, sigma, tau):
    tau = torch.clamp(tau, min=1e-12)
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * torch.sqrt(tau))
    return _norm_cdf(d1)


class Hedger(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims = (3,) + cfg.hidden
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor([10.0, 1.0, 1.0]))

    def forward(self, log_m, tau_n, prev):
        return self.net(torch.stack([log_m, tau_n, prev], dim=-1) * self.scale).squeeze(-1)


def rollout_nn(cfg, S, hedger, premium):
    n = S.shape[0]
    holding = torch.zeros(n, device=cfg.device)
    pnl = torch.zeros(n, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = cfg.T - k * cfg.dt
        log_m = torch.log(S[:, k] / cfg.K)
        tau_n = torch.full((n,), tau / cfg.T, device=cfg.device)
        d = hedger(log_m, tau_n, holding)
        pnl = pnl + d * (S[:, k + 1] - S[:, k]); holding = d
    return premium + pnl - torch.clamp(S[:, -1] - cfg.K, min=0.0)


def bs_pnl(cfg, S, premium):
    n = S.shape[0]; trading = torch.zeros(n, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = torch.full((n,), cfg.T - k * cfg.dt, device=cfg.device)
        trading += bs_delta(S[:, k], cfg.K, cfg.sigma_bs, tau) * (S[:, k + 1] - S[:, k])
    return premium + trading - torch.clamp(S[:, -1] - cfg.K, min=0.0)


def cvar_loss(pnl, alpha, w):
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(-pnl - w, min=0.0).mean()


def cvar_of(pnl, alpha=0.99):
    k = max(1, int(round((1.0 - alpha) * pnl.numel())))
    return -torch.topk(-pnl, k).values.mean().item()


def train(cfg, seed=0, verbose=False):
    """Train the deep hedger on the BASE Heston calibration only."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    p = cfg.base_params()
    premium = premium_of(cfg, "heston", p)
    hedger = Hedger(cfg).to(cfg.device)
    w = torch.zeros(1, device=cfg.device, requires_grad=True)
    opt = torch.optim.Adam(list(hedger.parameters()) + [w], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.85)], gamma=0.3)
    for epoch in range(cfg.epochs):
        S = simulate_heston(cfg, p, cfg.batch, gen)
        pnl = rollout_nn(cfg, S, hedger, premium)
        loss = cvar_loss(pnl, cfg.alpha, w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch % 300 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  loss {loss.item():+.4f}")
    return hedger


def evaluate_on(cfg, hedger, kind, p, n=100_000):
    """Fixed policy + BS delta, both on this world, with this world's premium."""
    prem = premium_of(cfg, kind, p)
    S = gen_paths(cfg, kind, p, n, seed=123_456)
    with torch.no_grad():
        d = rollout_nn(cfg, S, hedger, prem)
        b = bs_pnl(cfg, S, prem)
    return (d.std().item(), cvar_of(d)), (b.std().item(), cvar_of(b))


TESTS = [
    ("Heston base",        "heston", {}),
    ("Hi vol-of-vol",      "heston", {"xi": 0.9}),
    ("Hi skew",            "heston", {"rho": -0.9}),
    ("Hi vol level",       "heston", {"v0": 0.09, "theta": 0.09}),
    ("Merton jumps",       "merton", {}),
]


if __name__ == "__main__":
    cfg = Config()
    print("Training deep hedger on BASE Heston ...")
    hedger = train(cfg, cfg.seed, verbose=True)

    print(f"\n{'world':18s} {'deep std':>9s} {'deep CVaR':>10s} | {'BS std':>8s} {'BS CVaR':>9s}")
    rows = []
    for name, kind, override in TESTS:
        p = cfg.base_params(); p.update(override)
        (ds, dc), (bs, bc) = evaluate_on(cfg, hedger, kind, p)
        rows.append((name, ds, dc, bs, bc))
        print(f"{name:18s} {ds:9.4f} {dc:+10.4f} | {bs:8.4f} {bc:+9.4f}")

    names = [r[0] for r in rows]
    x = np.arange(len(names)); wbar = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.bar(x - wbar/2, [-r[2] for r in rows], wbar, label="Deep hedge")
    a1.bar(x + wbar/2, [-r[4] for r in rows], wbar, label="BS delta")
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    a1.set_ylabel("CVaR99 loss (lower = better)"); a1.set_title("Tail risk across worlds")
    a1.legend()
    a2.bar(x - wbar/2, [r[1] for r in rows], wbar, label="Deep hedge")
    a2.bar(x + wbar/2, [r[3] for r in rows], wbar, label="BS delta")
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    a2.set_ylabel("P&L std (lower = better)"); a2.set_title("Dispersion across worlds")
    a2.legend()
    fig.suptitle("Phase 5 — robustness: deep hedger (trained on base Heston) vs BS delta")
    fig.tight_layout(); fig.savefig("phase5_robustness.png", dpi=130)
