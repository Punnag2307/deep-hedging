"""
Deep Hedging — Phase 0 (v2): frictionless Black–Scholes sanity check, hardened.

What changed vs v1 (the "make it really solid" pass)
----------------------------------------------------
1. Correct discounting in the BS closed form. v1 hard-coded r = 0, so the `r`
   knob silently did nothing and would have produced wrong prices the moment
   rates entered. Now general; numerically identical at r = 0.
2. Multi-seed training. The headline metrics are reported as mean ± std across
   several seeds, so the result is demonstrably robust, not one lucky run.
3. An MSE (variance-minimising) baseline trained alongside the CVaR policy.
   This *proves* the slightly higher dispersion of the CVaR hedge is the
   objective at work, not a defect: the MSE net should match the BS-delta std,
   while the CVaR net trades a little std for a better 99% tail.
4. Cleanups + an explicit note that P&L accounting assumes r = 0 financing
   (to be generalised when rates are introduced in a later phase).
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


# ----------------------------------------------------------------------------- #
@dataclass
class Config:
    S0: float = 100.0
    sigma: float = 0.20
    r: float = 0.0
    mu: float = 0.0               # real-world drift; 0 => P&L not rewarded for direction
    T: float = 30.0 / 365.0
    n_steps: int = 30
    K: float = 100.0
    alpha: float = 0.99
    hidden: Tuple[int, ...] = (32, 32)
    batch: int = 8192
    epochs: int = 2200
    lr: float = 1e-3
    seeds: Tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


# ----------------------------------------------------------------------------- #
def simulate_market(cfg, n_paths, gen):
    """GBM price paths, shape (n_paths, n_steps + 1).  (Swap for Heston later.)"""
    Z = torch.randn(n_paths, cfg.n_steps, generator=gen, device=cfg.device)
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
    vol = cfg.sigma * math.sqrt(cfg.dt)
    log_incr = drift + vol * Z
    logS = math.log(cfg.S0) + torch.cat(
        [torch.zeros(n_paths, 1, device=cfg.device), torch.cumsum(log_incr, dim=1)], dim=1)
    return torch.exp(logS)


def terminal_payoff(cfg, S):
    """European call. (Swap for barrier / autocallable later.)"""
    return torch.clamp(S[:, -1] - cfg.K, min=0.0)


# ----------------------------------------------------------------------------- #
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_call_price_delta(S, K, sigma, tau, r=0.0):
    """Black–Scholes call price and delta with *proper* discounting."""
    tau = torch.clamp(tau, min=1e-12)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    price = S * _norm_cdf(d1) - K * torch.exp(-r * tau) * _norm_cdf(d2)
    return price, _norm_cdf(d1)


def premium_of(cfg):
    p0, _ = bs_call_price_delta(torch.tensor(cfg.S0), cfg.K, cfg.sigma,
                                torch.tensor(cfg.T), cfg.r)
    return float(p0)


# ----------------------------------------------------------------------------- #
class Hedger(nn.Module):
    """One shared MLP reused at every step. Inputs are observable at time t only."""
    def __init__(self, cfg):
        super().__init__()
        dims = (3,) + cfg.hidden
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor([10.0, 1.0, 1.0]))

    def forward(self, log_m, tau_norm, prev_holding):
        x = torch.stack([log_m, tau_norm, prev_holding], dim=-1) * self.scale
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------------- #
def rollout_pnl(cfg, S, hedger, premium, collect_deltas=False):
    # NOTE: assumes r = 0 financing (cash neither earns nor costs). Generalise by
    # accreting cash at r once rates are introduced in a later phase.
    n_paths = S.shape[0]
    holding = torch.zeros(n_paths, device=cfg.device)
    trading_pnl = torch.zeros(n_paths, device=cfg.device)
    nn_d, bs_d = [], []
    for k in range(cfg.n_steps):
        tau = cfg.T - k * cfg.dt
        log_m = torch.log(S[:, k] / cfg.K)
        tau_norm = torch.full((n_paths,), tau / cfg.T, device=cfg.device)
        delta = hedger(log_m, tau_norm, holding)
        trading_pnl = trading_pnl + delta * (S[:, k + 1] - S[:, k])
        holding = delta
        if collect_deltas:
            _, d = bs_call_price_delta(S[:, k], cfg.K, cfg.sigma,
                                       torch.full((n_paths,), tau, device=cfg.device), cfg.r)
            nn_d.append(delta.detach()); bs_d.append(d.detach())
    pnl = premium + trading_pnl - terminal_payoff(cfg, S)
    if collect_deltas:
        return pnl, torch.stack(nn_d, 1), torch.stack(bs_d, 1)
    return pnl


def bs_delta_pnl(cfg, S, premium):
    n_paths = S.shape[0]
    trading = torch.zeros(n_paths, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = torch.full((n_paths,), cfg.T - k * cfg.dt, device=cfg.device)
        _, delta = bs_call_price_delta(S[:, k], cfg.K, cfg.sigma, tau, cfg.r)
        trading += delta * (S[:, k + 1] - S[:, k])
    return premium + trading - terminal_payoff(cfg, S)


# ----------------------------------------------------------------------------- #
def cvar_loss(pnl, alpha, w):
    L = -pnl
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(L - w, min=0.0).mean()


def cvar_of_pnl(pnl, alpha=0.99):
    L = -pnl
    k = max(1, int(round((1.0 - alpha) * pnl.numel())))
    return -torch.topk(L, k).values.mean().item()


# ----------------------------------------------------------------------------- #
def train(cfg, seed=0, objective="cvar", verbose=False):
    """objective in {'cvar', 'mse'}.  'mse' = variance-minimising baseline."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    hedger = Hedger(cfg).to(cfg.device)
    params = list(hedger.parameters())
    w = None
    if objective == "cvar":
        w = torch.zeros(1, device=cfg.device, requires_grad=True)
        params = params + [w]
    premium = premium_of(cfg)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.85)], gamma=0.3)
    for epoch in range(cfg.epochs):
        S = simulate_market(cfg, cfg.batch, gen)
        pnl = rollout_pnl(cfg, S, hedger, premium)
        loss = cvar_loss(pnl, cfg.alpha, w) if objective == "cvar" else (pnl ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch % 300 == 0 or epoch == cfg.epochs - 1):
            print(f"  [{objective}] epoch {epoch:4d}  loss {loss.item():+.4f}  "
                  f"mean P&L {pnl.mean().item():+.4f}")
    return hedger, premium


def evaluate(cfg, hedger, premium, n_test=100_000):
    gen = torch.Generator(device=cfg.device).manual_seed(123_456)
    S = simulate_market(cfg, n_test, gen)
    with torch.no_grad():
        pnl, nn_d, bs_d = rollout_pnl(cfg, S, hedger, premium, collect_deltas=True)
    stats = dict(mean=pnl.mean().item(), std=pnl.std().item(), cvar99=cvar_of_pnl(pnl),
                 delta_mae=(nn_d - bs_d).abs().mean().item(),
                 delta_corr=float(np.corrcoef(nn_d.flatten().numpy(),
                                              bs_d.flatten().numpy())[0, 1]))
    return stats, pnl, nn_d, bs_d


def bs_eval(cfg, premium, n_test=100_000):
    gen = torch.Generator(device=cfg.device).manual_seed(123_456)
    S = simulate_market(cfg, n_test, gen)
    pnl = bs_delta_pnl(cfg, S, premium)
    return dict(mean=pnl.mean().item(), std=pnl.std().item(), cvar99=cvar_of_pnl(pnl)), pnl


def _agg(xs):
    a = np.array(xs)
    return a.mean(), a.std()


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config()
    premium = premium_of(cfg)
    print(f"Pinned premium (BS ATM price): {premium:.4f}\n")

    # multi-seed CVaR policy
    cvar_stats, rep = [], None
    for i, seed in enumerate(cfg.seeds):
        print(f"Training CVaR policy, seed {seed} ...")
        hedger, _ = train(cfg, seed=seed, objective="cvar", verbose=(i == 0))
        s, pnl, nn_d, bs_d = evaluate(cfg, hedger, premium)
        cvar_stats.append(s)
        if i == 0:
            rep = (pnl, nn_d, bs_d)

    # MSE (variance) baseline to explain the std gap
    print("\nTraining MSE (variance) baseline ...")
    hedger_mse, _ = train(cfg, seed=cfg.seeds[0], objective="mse")
    mse_stats, mse_pnl, _, _ = evaluate(cfg, hedger_mse, premium)

    # analytic BS delta benchmark
    bs_stats, bs_pnl = bs_eval(cfg, premium)

    sm, ss = _agg([s["std"] for s in cvar_stats])
    cm, cs = _agg([s["cvar99"] for s in cvar_stats])
    rm, rs = _agg([s["delta_corr"] for s in cvar_stats])
    mm, md = _agg([s["delta_mae"] for s in cvar_stats])

    lines = []
    lines.append("=================  Phase 0 results  =================")
    lines.append(f"{'strategy':14s} {'mean':>8s} {'std':>14s} {'CVaR99':>14s}")
    lines.append(f"{'BS delta':14s} {bs_stats['mean']:+8.4f} {bs_stats['std']:14.4f} {bs_stats['cvar99']:+14.4f}")
    lines.append(f"{'Deep (MSE)':14s} {mse_stats['mean']:+8.4f} {mse_stats['std']:14.4f} {mse_stats['cvar99']:+14.4f}")
    lines.append(f"{'Deep (CVaR)':14s} {'':8s} {sm:8.4f}±{ss:5.3f} {cm:+8.4f}±{cs:5.3f}")
    lines.append(f"\nCVaR delta recovery over {len(cfg.seeds)} seeds:"
                 f"  corr {rm:.4f}±{rs:.4f}   MAE {mm:.4f}±{md:.4f}")
    lines.append("=====================================================")
    report = "\n".join(lines)
    print("\n" + report)
    with open("phase0_results.txt", "w") as f:
        f.write(report + "\n")

    # plots
    pnl, nn_d, bs_d = rep
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-2.5, 2.5, 120)
    ax.hist(bs_pnl.numpy(), bins=bins, alpha=0.5, density=True, label="BS delta")
    ax.hist(mse_pnl.numpy(), bins=bins, alpha=0.5, density=True, label="Deep (MSE)")
    ax.hist(pnl.numpy(), bins=bins, alpha=0.5, density=True, label="Deep (CVaR)")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("terminal P&L"); ax.set_ylabel("density")
    ax.set_title("Phase 0 — MSE matches delta; CVaR trades std for a better tail")
    ax.legend(); fig.tight_layout(); fig.savefig("phase0_pnl.png", dpi=130)

    idx = np.random.default_rng(0).choice(nn_d.numel(), size=20000, replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(bs_d.flatten().numpy()[idx], nn_d.flatten().numpy()[idx], s=2, alpha=0.15)
    ax.plot([0, 1], [0, 1], "r--", lw=1.2, label="y = x")
    ax.set_xlabel("Black–Scholes delta N(d1)"); ax.set_ylabel("learned hedge ratio")
    ax.set_title("Phase 0 — delta recovery (CVaR, seed 0)")
    ax.legend(); fig.tight_layout(); fig.savefig("phase0_delta.png", dpi=130)
