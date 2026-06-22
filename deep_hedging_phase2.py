"""
Deep Hedging — Phase 2: Heston stochastic volatility (incomplete market).

Why this is the conceptual jump
-------------------------------
Under GBM the market is complete: the option can be replicated by trading the
underlying, and BS delta is (near) optimal. Under Heston the volatility is
itself random and only partially correlated with the spot, so the option
carries volatility risk that CANNOT be removed by trading the underlying alone.
The market is incomplete, the constant-vol BS delta is the WRONG hedge ratio,
and a hedger that adapts to the volatility regime can do strictly better —
even with zero transaction costs. That is what we test here.

Honesty rule (information set)
------------------------------
The network never sees the latent instantaneous variance v_t (unobservable in
practice). It sees only observable quantities: log-moneyness, time-to-maturity,
current holding, and a causal EWMA realized-vol estimate built from past
returns. We also train a 3-feature variant WITHOUT the vol proxy, so the
ablation shows how much edge comes from (a) a better average hedge ratio vs
(b) genuine volatility-regime adaptation.

Benchmark
---------
BS delta at a constant sigma = sqrt(theta) (the long-run vol), receiving the
correct Heston premium. Both hedgers start from the same (Heston) premium, so
the only difference is the hedging strategy. Cost = 0 throughout this phase.

Finding (verified, seed 0)
--------------------------
The deep hedger beats constant-vol BS delta on CVaR99 (~-2.95 vs -3.38) at
essentially equal std -> the incomplete-market claim holds. But the realized-vol
proxy did NOT help (the 3-feature net edged out the 4-feature one). So the edge
is STRUCTURAL: the network learns the Heston-appropriate hedge ratio that the
constant-vol delta gets wrong, not volatility-regime timing (the EWMA proxy is
too noisy/lagged at this horizon). Both stds (~0.88) dwarf the complete-market
case (~0.36) because vol risk is un-hedgeable with the underlying alone — which
is exactly the motivation for adding a second instrument in Phase 4.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

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
    # Heston params (realistic equity-style, leverage via rho<0)
    v0: float = 0.04
    theta: float = 0.04
    kappa: float = 1.5
    xi: float = 0.5
    rho: float = -0.7
    ewma_lambda: float = 0.94
    # objective / net / training
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
        return math.sqrt(self.theta)     # constant vol the naive BS hedger uses


# --------------------------------------------------------------------------- #
def simulate_heston(cfg, n_paths, gen):
    """Full-truncation Euler. Returns (S, v), both (n_paths, n_steps+1)."""
    dt = cfg.dt
    Zv = torch.randn(n_paths, cfg.n_steps, generator=gen, device=cfg.device)
    Zp = torch.randn(n_paths, cfg.n_steps, generator=gen, device=cfg.device)
    Zs = cfg.rho * Zv + math.sqrt(1 - cfg.rho ** 2) * Zp
    logS = torch.empty(n_paths, cfg.n_steps + 1, device=cfg.device)
    v = torch.empty(n_paths, cfg.n_steps + 1, device=cfg.device)
    logS[:, 0] = math.log(cfg.S0); v[:, 0] = cfg.v0
    sdt = math.sqrt(dt)
    for k in range(cfg.n_steps):
        vp = torch.clamp(v[:, k], min=0.0)
        sv = torch.sqrt(vp)
        logS[:, k + 1] = logS[:, k] + (cfg.mu - 0.5 * vp) * dt + sv * sdt * Zs[:, k]
        v[:, k + 1] = v[:, k] + cfg.kappa * (cfg.theta - vp) * dt + cfg.xi * sv * sdt * Zv[:, k]
    return torch.exp(logS), v


def heston_premium(cfg, n=400_000):
    gen = torch.Generator(device=cfg.device).manual_seed(999)
    S, _ = simulate_heston(cfg, n, gen)
    return torch.clamp(S[:, -1] - cfg.K, min=0.0).mean().item()   # r=0, no discount


# --------------------------------------------------------------------------- #
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_delta(S, K, sigma, tau, r=0.0):
    tau = torch.clamp(tau, min=1e-12)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * torch.sqrt(tau))
    return _norm_cdf(d1)


# --------------------------------------------------------------------------- #
class Hedger(nn.Module):
    def __init__(self, n_in, hidden, scale):
        super().__init__()
        dims = (n_in,) + tuple(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, feats):
        return self.net(feats * self.scale).squeeze(-1)


def realized_vol_feature(cfg, S):
    """Causal EWMA realized-vol estimate, shape (n, n_steps+1). Observable only."""
    n = S.shape[0]
    logret = torch.log(S[:, 1:] / S[:, :-1])
    rv2 = torch.empty(n, cfg.n_steps + 1, device=cfg.device)
    rv2[:, 0] = cfg.theta
    lam = cfg.ewma_lambda
    for j in range(1, cfg.n_steps + 1):
        rv2[:, j] = lam * rv2[:, j - 1] + (1 - lam) * logret[:, j - 1] ** 2 / cfg.dt
    return torch.sqrt(torch.clamp(rv2, min=1e-8))


# --------------------------------------------------------------------------- #
def rollout(cfg, S, hedger, premium, use_vol):
    n = S.shape[0]
    rv = realized_vol_feature(cfg, S) if use_vol else None
    holding = torch.zeros(n, device=cfg.device)
    pnl_trade = torch.zeros(n, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = cfg.T - k * cfg.dt
        log_m = torch.log(S[:, k] / cfg.K)
        tau_norm = torch.full((n,), tau / cfg.T, device=cfg.device)
        if use_vol:
            feats = torch.stack([log_m, tau_norm, holding, rv[:, k]], dim=-1)
        else:
            feats = torch.stack([log_m, tau_norm, holding], dim=-1)
        delta = hedger(feats)
        pnl_trade = pnl_trade + delta * (S[:, k + 1] - S[:, k])
        holding = delta
    return premium + pnl_trade - torch.clamp(S[:, -1] - cfg.K, min=0.0)


def bs_pnl(cfg, S, premium):
    n = S.shape[0]
    trading = torch.zeros(n, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = torch.full((n,), cfg.T - k * cfg.dt, device=cfg.device)
        d = bs_delta(S[:, k], cfg.K, cfg.sigma_bs, tau, cfg.r)
        trading += d * (S[:, k + 1] - S[:, k])
    return premium + trading - torch.clamp(S[:, -1] - cfg.K, min=0.0)


# --------------------------------------------------------------------------- #
def cvar_loss(pnl, alpha, w):
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(-pnl - w, min=0.0).mean()


def cvar_of(pnl, alpha=0.99):
    k = max(1, int(round((1.0 - alpha) * pnl.numel())))
    return -torch.topk(-pnl, k).values.mean().item()


def train(cfg, seed, use_vol, premium, verbose=False):
    torch.manual_seed(seed)
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    n_in = 4 if use_vol else 3
    scale = [10., 1., 1., 5.] if use_vol else [10., 1., 1.]
    hedger = Hedger(n_in, cfg.hidden, scale).to(cfg.device)
    w = torch.zeros(1, device=cfg.device, requires_grad=True)
    opt = torch.optim.Adam(list(hedger.parameters()) + [w], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.85)], gamma=0.3)
    for epoch in range(cfg.epochs):
        S, _ = simulate_heston(cfg, cfg.batch, gen)
        pnl = rollout(cfg, S, hedger, premium, use_vol)
        loss = cvar_loss(pnl, cfg.alpha, w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch % 300 == 0 or epoch == cfg.epochs - 1):
            print(f"  [{'vol' if use_vol else 'novol'}] epoch {epoch:4d} loss {loss.item():+.4f}")
    return hedger


def evaluate(cfg, premium, hedger_vol, hedger_novol, n_test=100_000):
    gen = torch.Generator(device=cfg.device).manual_seed(123_456)
    S, _ = simulate_heston(cfg, n_test, gen)
    with torch.no_grad():
        p_bs = bs_pnl(cfg, S, premium)
        p_nv = rollout(cfg, S, hedger_novol, premium, use_vol=False)
        p_v = rollout(cfg, S, hedger_vol, premium, use_vol=True)
    st = lambda p: dict(mean=p.mean().item(), std=p.std().item(), cvar99=cvar_of(p))
    return dict(bs=st(p_bs), novol=st(p_nv), vol=st(p_v)), (p_bs, p_nv, p_v)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config()
    prem = heston_premium(cfg)
    print(f"Heston premium (MC): {prem:.4f}   BS hedger sigma = {cfg.sigma_bs:.3f}\n")

    h_nv = train(cfg, cfg.seed, use_vol=False, premium=prem, verbose=True)
    h_v = train(cfg, cfg.seed, use_vol=True, premium=prem, verbose=True)
    stats, (p_bs, p_nv, p_v) = evaluate(cfg, prem, h_v, h_nv)

    print(f"\n{'strategy':22s} {'std':>9s} {'CVaR99':>10s}")
    for key, lab in [("bs", "BS delta (const vol)"), ("novol", "Deep (no vol feat)"),
                     ("vol", "Deep (+ vol feat)")]:
        s = stats[key]
        print(f"{lab:22s} {s['std']:9.4f} {s['cvar99']:+10.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bins = np.linspace(-3, 3, 130)
    ax.hist(p_bs.numpy(), bins=bins, alpha=0.45, density=True,
            label=f"BS delta (CVaR {stats['bs']['cvar99']:+.3f})")
    ax.hist(p_nv.numpy(), bins=bins, alpha=0.45, density=True,
            label=f"Deep no-vol (CVaR {stats['novol']['cvar99']:+.3f})")
    ax.hist(p_v.numpy(), bins=bins, alpha=0.45, density=True,
            label=f"Deep +vol (CVaR {stats['vol']['cvar99']:+.3f})")
    ax.axvline(0, color="k", lw=.8, ls="--")
    ax.set_xlabel("terminal P&L (Heston, zero cost)"); ax.set_ylabel("density")
    ax.set_title("Phase 2 — incomplete market: deep hedger beats constant-vol BS delta")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("phase2_pnl.png", dpi=130)
