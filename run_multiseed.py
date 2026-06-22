"""
Multi-seed rigor pass for the Deep Hedging project.

Re-runs each phase's *headline* comparison across several random seeds and reports
the result as mean +/- std, giving cross-seed error bars for Phases 1-5 (Phase 0
already reports its own multi-seed numbers). This does NOT modify the phase files;
it imports them and calls their train/eval functions.

Usage:
    python run_multiseed.py
Environment overrides (handy for a quick smoke run):
    MS_SEEDS=2  MS_EPOCHS=60  python run_multiseed.py

NOTE: the full run trains ~18 networks (a few seeds x several phases), so expect
~30-45 min on a laptop CPU. Reduce MS_SEEDS or MS_EPOCHS to go faster.
"""
import os
import numpy as np
import torch

SEEDS = tuple(range(int(os.environ.get("MS_SEEDS", "3"))))
EPOCHS = int(os.environ["MS_EPOCHS"]) if "MS_EPOCHS" in os.environ else None

import deep_hedging_phase1 as p1
import deep_hedging_phase2 as p2
import deep_hedging_phase3 as p3
import deep_hedging_phase4 as p4
import deep_hedging_phase5 as p5


def cfg_of(mod):
    c = mod.Config()
    if EPOCHS is not None:
        c.epochs = EPOCHS
    return c


def ms(xs):
    a = np.array(xs, dtype=float)
    return a.mean(), a.std()


def f(xs):
    m, s = ms(xs)
    return f"{m:+7.3f} +/- {s:5.3f}"


lines = [f"=========  Multi-seed results (seeds={SEEDS}"
         + (f", epochs={EPOCHS}" if EPOCHS else "") + ")  ========="]

# ---------------- Phase 1: cost-aware hedger at the headline cost ----------- #
print("Phase 1 (transaction costs) ...")
c = cfg_of(p1); prem = p1.premium_of(c); cst = c.cost_rate
nn_cv, bs_cv, nn_tn = [], [], []
for s in SEEDS:
    h, prem = p1.train(c, s, cst)
    nn_s, bs_s, _, _ = p1.evaluate_at_cost(c, h, prem, cst)
    nn_cv.append(nn_s["cvar99"]); bs_cv.append(bs_s["cvar99"]); nn_tn.append(nn_s["turnover"])
lines.append(f"P1 cost={cst:.4f}  Deep CVaR {f(nn_cv)}  | BS CVaR {bs_cv[0]:+.3f}"
             f"  | Deep turnover {f(nn_tn)} (BS {bs_s['turnover']:.3f})")

# ---------------- Phase 2: Heston, no-vol deep hedger ----------------------- #
print("Phase 2 (Heston) ...")
c = cfg_of(p2); prem = p2.heston_premium(c)
gen = torch.Generator().manual_seed(123456); S, _ = p2.simulate_heston(c, 100000, gen)
bs_pnl = p2.bs_pnl(c, S, prem); bs_c2 = p2.cvar_of(bs_pnl)
d_cv = []
for s in SEEDS:
    h = p2.train(c, s, False, prem)
    with torch.no_grad():
        d_cv.append(p2.cvar_of(p2.rollout(c, S, h, prem, use_vol=False)))
lines.append(f"P2 Heston      Deep CVaR {f(d_cv)}  | BS CVaR {bs_c2:+.3f}")

# ---------------- Phase 3: knock-out barrier -------------------------------- #
print("Phase 3 (barrier) ...")
c = cfg_of(p3); prem, _ = p3.barrier_premium(c)
gen = torch.Generator().manual_seed(123456); S = p3.simulate_gbm(c, 100000, gen)
van = p3.rollout(c, S, prem, None); van_std = van.std().item(); van_cv = p3.cvar_of(van)
d_std, d_cv = [], []
for s in SEEDS:
    h = p3.train(c, s, prem)
    with torch.no_grad():
        pnl = p3.rollout(c, S, prem, h)
    d_std.append(pnl.std().item()); d_cv.append(p3.cvar_of(pnl))
lines.append(f"P3 barrier     Deep std {f(d_std)} (vanilla {van_std:.3f})"
             f"  | Deep CVaR {f(d_cv)} (vanilla {van_cv:+.3f})")

# ---------------- Phase 4: autocallable, 1- vs 2-instrument ----------------- #
print("Phase 4 (autocallable) ...")
c = cfg_of(p4); prem, _ = p4.premium_of(c)
gen = torch.Generator().manual_seed(123456); S = p4.simulate_gbm(c, 100000, gen)
o_std, t_std, t_cv = [], [], []
for s in SEEDS:
    h1 = p4.train(c, s, False, prem); h2 = p4.train(c, s, True, prem)
    with torch.no_grad():
        p_1 = p4.rollout(c, S, prem, h1, two_inst=False)
        p_2 = p4.rollout(c, S, prem, h2, two_inst=True)
    o_std.append(p_1.std().item()); t_std.append(p_2.std().item()); t_cv.append(p4.cvar_of(p_2))
lines.append(f"P4 autocall    +put std {f(t_std)} (underlying-only {f(o_std)})"
             f"  | +put CVaR {f(t_cv)}")

# ---------------- Phase 5: robustness (base + jumps) ------------------------ #
print("Phase 5 (robustness) ...")
c = cfg_of(p5)
base_cv, jump_cv = [], []
bbc = jbc = None
for s in SEEDS:
    h = p5.train(c, s)
    pb = c.base_params()
    (_, dc), (_, bc) = p5.evaluate_on(c, h, "heston", pb); bbc = bc
    (_, jdc), (_, jbc2) = p5.evaluate_on(c, h, "merton", pb); jbc = jbc2
    base_cv.append(dc); jump_cv.append(jdc)
lines.append(f"P5 base world  Deep CVaR {f(base_cv)}  | BS CVaR {bbc:+.3f}")
lines.append(f"P5 jumps world Deep CVaR {f(jump_cv)}  | BS CVaR {jbc:+.3f}")

lines.append("=" * 62)
report = "\n".join(lines)
print("\n" + report)
with open("multiseed_results.txt", "w") as fh:
    fh.write(report + "\n")
