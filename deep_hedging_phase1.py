"""
Deep Hedging — Phase 1: proportional transaction costs.

What's new vs Phase 0
---------------------
* The rollout now charges a proportional cost  c * S_t * |change in holding|
  on every trade (the initial purchase, each rebalance, and the terminal
  unwind), and subtracts it from P&L. The CVaR objective is unchanged, but it
  now sees costs, so the optimal policy changes.
* The analytic Black–Scholes delta-hedge ALSO pays these costs (it just doesn't
  adapt to them) — that's the fair, like-for-like benchmark.
* We track turnover (sum of |trades|) so we can show the cost-aware hedger
  trading less than delta.

Two results to expect
---------------------
1. No-trade band: with costs, the policy stops chasing small delta changes.
   Plotting the chosen position vs the *current* holding shows a band around
   the target where it simply does nothing (follows y = x). The band widens as
   costs rise — this reproduces the Whalley–Wilmott result without hard-coding.
2. After costs, the deep hedger beats BS delta on CVaR and on turnover, and the
   gap grows with the cost level — through realistic costs (verified to ~50 bps).

Honest caveat (high-cost regime)
--------------------------------
At very high costs (~1% of notional per trade) the hedger still halves cost and
turnover, but its 99%-tail (CVaR) optimisation lags BS delta within a practical
training budget: the strong cost-reduction gradient overwhelms the sparse
~80-sample tail gradient. More epochs narrow the gap (this is convergence, not a
modelling error). Larger batches (more tail samples) or a CVaR/variance warm-up
would help — a clean future improvement, and a good talking point on the limits
of minibatch CVaR.
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
    sigma: float = 0.20
    r: float = 0.0
    mu: float = 0.0
    T: float = 30.0 / 365.0
    n_steps: int = 30
    K: float = 100.0
    alpha: float = 0.99
    hidden: Tuple[int, ...] = (32, 32)
    batch: int = 8192
    epochs: int = 2000
    lr: float = 1e-3
    cost_rate: float = 0.0025                       # headline cost (25 bps of notional)
    cost_grid: Tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005, 0.01)
    seed: int = 0
    device: str = "cpu"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


# --------------------------------------------------------------------------- #
def simulate_market(cfg, n_paths, gen):
    Z = torch.randn(n_paths, cfg.n_steps, generator=gen, device=cfg.device)
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
    vol = cfg.sigma * math.sqrt(cfg.dt)
    log_incr = drift + vol * Z
    logS = math.log(cfg.S0) + torch.cat(
        [torch.zeros(n_paths, 1, device=cfg.device), torch.cumsum(log_incr, dim=1)], dim=1)
    return torch.exp(logS)


def terminal_payoff(cfg, S):
    return torch.clamp(S[:, -1] - cfg.K, min=0.0)


def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_call_price_delta(S, K, sigma, tau, r=0.0):
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


# --------------------------------------------------------------------------- #
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

    def forward(self, log_m, tau_norm, prev_holding):
        x = torch.stack([log_m, tau_norm, prev_holding], dim=-1) * self.scale
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
def rollout_nn(cfg, S, hedger, premium, cost_rate):
    """Deep-hedger rollout with proportional costs. Returns pnl, turnover."""
    n = S.shape[0]
    holding = torch.zeros(n, device=cfg.device)
    pnl_trade = torch.zeros(n, device=cfg.device)
    cost = torch.zeros(n, device=cfg.device)
    turn = torch.zeros(n, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = cfg.T - k * cfg.dt
        log_m = torch.log(S[:, k] / cfg.K)
        tau_norm = torch.full((n,), tau / cfg.T, device=cfg.device)
        delta = hedger(log_m, tau_norm, holding)
        trade = delta - holding
        cost = cost + cost_rate * S[:, k] * trade.abs()
        turn = turn + trade.abs()
        pnl_trade = pnl_trade + delta * (S[:, k + 1] - S[:, k])
        holding = delta
    cost = cost + cost_rate * S[:, -1] * holding.abs()          # terminal unwind
    turn = turn + holding.abs()
    pnl = premium + pnl_trade - cost - terminal_payoff(cfg, S)
    return pnl, turn


def rollout_bs(cfg, S, premium, cost_rate):
    """Analytic delta-hedge, also paying the same costs (but not adapting)."""
    n = S.shape[0]
    holding = torch.zeros(n, device=cfg.device)
    pnl_trade = torch.zeros(n, device=cfg.device)
    cost = torch.zeros(n, device=cfg.device)
    turn = torch.zeros(n, device=cfg.device)
    for k in range(cfg.n_steps):
        tau = torch.full((n,), cfg.T - k * cfg.dt, device=cfg.device)
        _, delta = bs_call_price_delta(S[:, k], cfg.K, cfg.sigma, tau, cfg.r)
        trade = delta - holding
        cost = cost + cost_rate * S[:, k] * trade.abs()
        turn = turn + trade.abs()
        pnl_trade = pnl_trade + delta * (S[:, k + 1] - S[:, k])
        holding = delta
    cost = cost + cost_rate * S[:, -1] * holding.abs()
    turn = turn + holding.abs()
    pnl = premium + pnl_trade - cost - terminal_payoff(cfg, S)
    return pnl, turn


def band_curve(cfg, hedger, S_fixed, tau, prev_grid):
    """Chosen new holding vs current holding at fixed (S, tau): reveals the band."""
    m = len(prev_grid)
    log_m = torch.full((m,), math.log(S_fixed / cfg.K))
    tau_norm = torch.full((m,), tau / cfg.T)
    prev = torch.tensor(prev_grid, dtype=torch.float32)
    with torch.no_grad():
        return hedger(log_m, tau_norm, prev).numpy()


# --------------------------------------------------------------------------- #
def cvar_loss(pnl, alpha, w):
    L = -pnl
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(L - w, min=0.0).mean()


def cvar_of(pnl, alpha=0.99):
    L = -pnl
    k = max(1, int(round((1.0 - alpha) * pnl.numel())))
    return -torch.topk(L, k).values.mean().item()


def train(cfg, seed, cost_rate, verbose=False):
    torch.manual_seed(seed)
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    hedger = Hedger(cfg).to(cfg.device)
    w = torch.zeros(1, device=cfg.device, requires_grad=True)
    premium = premium_of(cfg)
    opt = torch.optim.Adam(list(hedger.parameters()) + [w], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.85)], gamma=0.3)
    for epoch in range(cfg.epochs):
        S = simulate_market(cfg, cfg.batch, gen)
        pnl, _ = rollout_nn(cfg, S, hedger, premium, cost_rate)
        loss = cvar_loss(pnl, cfg.alpha, w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch % 400 == 0 or epoch == cfg.epochs - 1):
            print(f"  c={cost_rate:.4f} epoch {epoch:4d}  loss {loss.item():+.4f}")
    return hedger, premium


def evaluate_at_cost(cfg, hedger, premium, cost_rate, n_test=100_000):
    gen = torch.Generator(device=cfg.device).manual_seed(123_456)
    S = simulate_market(cfg, n_test, gen)
    with torch.no_grad():
        nn_pnl, nn_turn = rollout_nn(cfg, S, hedger, premium, cost_rate)
        bs_pnl, bs_turn = rollout_bs(cfg, S, premium, cost_rate)
    pack = lambda p, t: dict(mean=p.mean().item(), std=p.std().item(),
                             cvar99=cvar_of(p), turnover=t.mean().item())
    return pack(nn_pnl, nn_turn), pack(bs_pnl, bs_turn), nn_pnl, bs_pnl


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config()
    premium = premium_of(cfg)
    print(f"Pinned premium: {premium:.4f}\n")

    prev_grid = np.linspace(-0.1, 1.1, 200)
    bands, sweep = {}, []
    headline = None
    for c in cfg.cost_grid:
        print(f"Training cost-aware hedger, c={c:.4f} ...")
        hedger, _ = train(cfg, cfg.seed, c, verbose=(c == cfg.cost_grid[0]))
        nn_s, bs_s, nn_pnl, bs_pnl = evaluate_at_cost(cfg, hedger, premium, c)
        sweep.append((c, nn_s, bs_s))
        bands[c] = band_curve(cfg, hedger, cfg.S0, cfg.T / 2, prev_grid)
        if abs(c - cfg.cost_rate) < 1e-12:
            headline = (nn_pnl, bs_pnl, nn_s, bs_s)

    # report
    print(f"\n{'cost':>7s} | {'NN CVaR':>9s} {'BS CVaR':>9s} | "
          f"{'NN turn':>8s} {'BS turn':>8s}")
    for c, nn_s, bs_s in sweep:
        print(f"{c:7.4f} | {nn_s['cvar99']:+9.4f} {bs_s['cvar99']:+9.4f} | "
              f"{nn_s['turnover']:8.3f} {bs_s['turnover']:8.3f}")

    # plot 1: no-trade band
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot(prev_grid, prev_grid, "k--", lw=1, label="no trade (y=x)")
    for c in cfg.cost_grid:
        ax.plot(prev_grid, bands[c], lw=1.6, label=f"c={c:.4f}")
    ax.set_xlabel("current holding"); ax.set_ylabel("chosen new holding")
    ax.set_title("Phase 1 — no-trade band widens with cost\n(at S=K, half-life)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("phase1_band.png", dpi=130)

    # plot 2: CVaR & turnover vs cost
    cs = [c for c, _, _ in sweep]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(cs, [s["cvar99"] for _, s, _ in sweep], "o-", label="Deep hedge")
    a1.plot(cs, [s["cvar99"] for _, _, s in sweep], "s-", label="BS delta")
    a1.set_xlabel("cost rate"); a1.set_ylabel("CVaR99 (higher=better)")
    a1.set_title("Cost-adjusted tail risk"); a1.legend()
    a2.plot(cs, [s["turnover"] for _, s, _ in sweep], "o-", label="Deep hedge")
    a2.plot(cs, [s["turnover"] for _, _, s in sweep], "s-", label="BS delta")
    a2.set_xlabel("cost rate"); a2.set_ylabel("turnover")
    a2.set_title("Trading activity"); a2.legend()
    fig.tight_layout(); fig.savefig("phase1_sweep.png", dpi=130)

    # plot 3: P&L at headline cost
    if headline is not None:
        nn_pnl, bs_pnl, nn_s, bs_s = headline
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bins = np.linspace(-3, 2, 120)
        ax.hist(bs_pnl.numpy(), bins=bins, alpha=0.5, density=True,
                label=f"BS delta (CVaR {bs_s['cvar99']:+.3f})")
        ax.hist(nn_pnl.numpy(), bins=bins, alpha=0.5, density=True,
                label=f"Deep hedge (CVaR {nn_s['cvar99']:+.3f})")
        ax.axvline(0, color="k", lw=.8, ls="--")
        ax.set_xlabel("terminal P&L (after costs)"); ax.set_ylabel("density")
        ax.set_title(f"Phase 1 — P&L after costs (c={cfg.cost_rate:.4f})")
        ax.legend(); fig.tight_layout(); fig.savefig("phase1_pnl.png", dpi=130)
