"""
Deep Hedging — Phase 4 (headline): autocallable note + a second instrument.

The product (single-underlying autocallable, scaled to principal 100)
--------------------------------------------------------------------
* Quarterly observation dates. At an early obs date, if S >= autocall level AC,
  the note redeems early paying 100 + (accrued coupon) and terminates.
* If it survives to maturity:  S>=AC -> 100 + full coupon ;  KI<=S<AC -> 100 ;
  S<KI -> S (downside participation). The jump at the KI barrier is a digital
  cliff, and the early-redemption feature adds further discontinuities.
The dealer is SHORT this note (received the premium) and must hedge the payout.

Why a second instrument
-----------------------
The note's value has sharp gamma at the barrier cliff and the autocall level.
The underlying alone cannot manage a (near-)digital under discrete rebalancing.
A vanilla option supplies the missing convexity. We therefore compare a deep
hedger with the UNDERLYING ONLY against one with UNDERLYING + a vanilla put,
both CVaR-trained — isolating the value of the second instrument.

Setup: GBM, zero cost (so the hedging put is priced in closed form and the whole
thing is exactly reproducible). Combining this with Phase 2's Heston (true vega
risk) is the natural further extension.

Finding (verified, seed 0)
--------------------------
The CVaR tail is dominated by paths ending near the downside barrier (the digital
cliff) — none of the worst paths autocall early. A deep hedger with the UNDERLYING
ALONE gets std 5.08 / CVaR99 -6.09. Adding a vanilla put struck AT the barrier (70),
where that cliff lives, gives std 3.59 / CVaR99 -5.33: ~29% lower dispersion and a
better tail. The strike matters — a put at 80 lowered std but *worsened* CVaR because
its convexity sat away from the tail; moving it to the barrier fixed both. The second
instrument supplies convexity the underlying cannot replicate under discrete trading.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
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
    T: float = 1.0
    n_steps: int = 12                       # monthly grid
    obs_early: Tuple[int, ...] = (3, 6, 9)  # quarterly early-autocall obs (maturity=12)
    AC: float = 100.0                       # autocall level
    KI: float = 70.0                        # downside barrier
    coupon: float = 2.0                     # per-period coupon (cash, principal=100)
    K_put: float = 70.0                     # strike of the hedging put
    alpha: float = 0.99
    hidden: Tuple[int, ...] = (48, 48)
    batch: int = 8192
    epochs: int = 2000
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def ttno_grid(self):
        obs_all = list(self.obs_early) + [self.n_steps]
        g = []
        for k in range(self.n_steps):
            nxt = min(o for o in obs_all if o > k)
            g.append((nxt - k) / self.n_steps)
        return g


# --------------------------------------------------------------------------- #
def simulate_gbm(cfg, n, gen):
    Z = torch.randn(n, cfg.n_steps, generator=gen, device=cfg.device)
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
    vol = cfg.sigma * math.sqrt(cfg.dt)
    logS = math.log(cfg.S0) + torch.cat(
        [torch.zeros(n, 1, device=cfg.device), torch.cumsum(drift + vol * Z, 1)], 1)
    return torch.exp(logS)


def autocall_dynamics(cfg, S):
    """Return active mask (n, n+1), note payment (n,), early-autocall flag (n,)."""
    n = S.shape[0]
    autocall_step = torch.full((n,), cfg.n_steps, dtype=torch.long, device=cfg.device)
    early = torch.zeros(n, dtype=torch.bool, device=cfg.device)
    amount = torch.zeros(n, device=cfg.device)
    for i, e in enumerate(cfg.obs_early):
        trig = (S[:, e] >= cfg.AC) & (~early)
        autocall_step[trig] = e
        amount[trig] = 100.0 + cfg.coupon * (i + 1)
        early = early | (S[:, e] >= cfg.AC)
    ST = S[:, -1]
    full_cpn = 100.0 + cfg.coupon * (len(cfg.obs_early) + 1)
    mat = torch.where(ST >= cfg.AC, torch.full_like(ST, full_cpn),
                      torch.where(ST >= cfg.KI, torch.full_like(ST, 100.0), ST))
    payment = torch.where(early, amount, mat)
    ks = torch.arange(cfg.n_steps + 1, device=cfg.device)
    autocalled_by = (ks.unsqueeze(0) >= autocall_step.unsqueeze(1)) & early.unsqueeze(1)
    return (~autocalled_by).float(), payment, early


def premium_of(cfg, n=400_000):
    gen = torch.Generator(device=cfg.device).manual_seed(999)
    S = simulate_gbm(cfg, n, gen)
    _, payment, early = autocall_dynamics(cfg, S)
    return payment.mean().item(), early.float().mean().item()


# --------------------------------------------------------------------------- #
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_put(S, K, sigma, tau):
    tau = torch.clamp(tau, min=1e-12)
    sq = torch.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * sq)
    d2 = d1 - sigma * sq
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)        # r = 0


def put_prices(cfg, S):
    n = S.shape[0]
    tau = torch.tensor([max(cfg.T - k * cfg.dt, 0.0) for k in range(cfg.n_steps + 1)],
                       device=cfg.device)
    O = bs_put(S, cfg.K_put, cfg.sigma, tau.unsqueeze(0).expand(n, -1))
    O = O.clone(); O[:, -1] = torch.clamp(cfg.K_put - S[:, -1], min=0.0)
    return O


# --------------------------------------------------------------------------- #
class Hedger(nn.Module):
    def __init__(self, n_in, n_out, hidden, scale):
        super().__init__()
        dims = (n_in,) + tuple(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers += [nn.Linear(dims[-1], n_out)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, feats):
        return self.net(feats * self.scale)


def rollout(cfg, S, premium, hedger, two_inst, collect=False):
    n = S.shape[0]
    active, payment, _ = autocall_dynamics(cfg, S)
    O = put_prices(cfg, S) if two_inst else None
    ttno = cfg.ttno_grid
    hS = torch.zeros(n, device=cfg.device)
    hO = torch.zeros(n, device=cfg.device)
    pnl = torch.zeros(n, device=cfg.device)
    recO = []
    for k in range(cfg.n_steps):
        tau = cfg.T - k * cfg.dt
        log_m = torch.log(S[:, k] / cfg.S0)
        tau_n = torch.full((n,), tau / cfg.T, device=cfg.device)
        ttn = torch.full((n,), ttno[k], device=cfg.device)
        if two_inst:
            feats = torch.stack([log_m, tau_n, ttn, hS, hO], dim=-1)
            out = hedger(feats)
            dS = out[:, 0] * active[:, k]; dO = out[:, 1] * active[:, k]
            pnl = pnl + dS * (S[:, k + 1] - S[:, k]) + dO * (O[:, k + 1] - O[:, k])
            hS, hO = dS, dO
            if collect: recO.append(dO.detach())
        else:
            feats = torch.stack([log_m, tau_n, ttn, hS], dim=-1)
            dS = hedger(feats)[:, 0] * active[:, k]
            pnl = pnl + dS * (S[:, k + 1] - S[:, k])
            hS = dS
    pnl = premium + pnl - payment
    if collect and two_inst:
        return pnl, torch.stack(recO, 1)
    return pnl


# --------------------------------------------------------------------------- #
def cvar_loss(pnl, alpha, w):
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(-pnl - w, min=0.0).mean()


def cvar_of(pnl, alpha=0.99):
    k = max(1, int(round((1.0 - alpha) * pnl.numel())))
    return -torch.topk(-pnl, k).values.mean().item()


def train(cfg, seed, two_inst, premium, verbose=False):
    torch.manual_seed(seed)
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    if two_inst:
        hedger = Hedger(5, 2, cfg.hidden, [10., 1., 4., 1., 1.]).to(cfg.device)
    else:
        hedger = Hedger(4, 1, cfg.hidden, [10., 1., 4., 1.]).to(cfg.device)
    w = torch.zeros(1, device=cfg.device, requires_grad=True)
    opt = torch.optim.Adam(list(hedger.parameters()) + [w], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(cfg.epochs * 0.6), int(cfg.epochs * 0.85)], gamma=0.3)
    for epoch in range(cfg.epochs):
        S = simulate_gbm(cfg, cfg.batch, gen)
        pnl = rollout(cfg, S, premium, hedger, two_inst)
        loss = cvar_loss(pnl, cfg.alpha, w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (epoch % 400 == 0 or epoch == cfg.epochs - 1):
            tag = "2-inst" if two_inst else "1-inst"
            print(f"  [{tag}] epoch {epoch:4d}  loss {loss.item():+.4f}")
    return hedger


if __name__ == "__main__":
    cfg = Config()
    prem, acp = premium_of(cfg)
    print(f"Autocallable premium (MC): {prem:.4f}   early-autocall prob: {acp:.1%}\n")

    h1 = train(cfg, cfg.seed, two_inst=False, premium=prem, verbose=True)
    h2 = train(cfg, cfg.seed, two_inst=True, premium=prem, verbose=True)

    gen = torch.Generator(device=cfg.device).manual_seed(123_456)
    S = simulate_gbm(cfg, 100_000, gen)
    with torch.no_grad():
        p1 = rollout(cfg, S, prem, h1, two_inst=False)
        p2, dO = rollout(cfg, S, prem, h2, two_inst=True, collect=True)

    st = lambda p: (p.mean().item(), p.std().item(), cvar_of(p))
    print(f"\n{'strategy':28s} {'mean':>9s} {'std':>9s} {'CVaR99':>10s}")
    for lab, p in [("Deep: underlying only", p1), ("Deep: underlying + put", p2)]:
        m, s, c = st(p)
        print(f"{lab:28s} {m:+9.4f} {s:9.4f} {c:+10.4f}")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    lo = float(min(p1.min(), p2.min())); hi = float(max(p1.max(), p2.max()))
    bins = np.linspace(max(lo, -25), min(hi, 15), 140)
    ax.hist(p1.numpy(), bins=bins, alpha=0.5, density=True,
            label=f"underlying only (std {p1.std():.2f}, CVaR {cvar_of(p1):+.2f})")
    ax.hist(p2.numpy(), bins=bins, alpha=0.5, density=True,
            label=f"+ put (std {p2.std():.2f}, CVaR {cvar_of(p2):+.2f})")
    ax.axvline(0, color="k", lw=.8, ls="--")
    ax.set_xlabel("dealer P&L"); ax.set_ylabel("density")
    ax.set_title("Phase 4 — autocallable: a 2nd instrument tightens the hedge")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("phase4_pnl.png", dpi=130)
