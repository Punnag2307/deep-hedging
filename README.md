# Deep Hedging — Learning to Hedge Derivatives Under Real-World Frictions

A from-scratch implementation of the **deep hedging** framework (Buehler, Gonon,
Teichmann & Wood), built in PyTorch. Instead of computing a hedge from a pricing
formula, the hedging strategy is treated as an **optimal control problem**: a
neural network maps the observable market state to a position, and is trained by
differentiating the *risk* of the terminal hedged P&L straight through a
simulated trading trajectory (backprop-through-time). The objective throughout is
**CVaR₉₉** (Expected Shortfall), made smooth via the Rockafellar–Uryasev
formulation.

The project is organised as seven phases, each changing exactly one thing from the
previous, so every result is an isolated, interpretable claim — and each phase
includes an honest account of where the method strains. The first six build and
stress the framework on simulated models; the capstone (Phase 6) drops the
simulator entirely and hedges on **real S&P 500 data under nonlinear market
impact**, evaluated out-of-sample and through the 2008 and 2020 crises.

---

## Why this exists

The textbook Black–Scholes hedge (hold Δ = ∂V/∂S, rebalance continuously) is
optimal only in an idealised world: no transaction costs, constant volatility, a
complete market, continuous trading. Every one of those assumptions fails in
practice. Deep hedging asks instead: *given realistic frictions and a chosen risk
measure, what trading strategy minimises risk?* — and learns it directly from
simulated (or historical) scenarios, with no closed-form pricing formula required.

---

## Results at a glance

| Phase | Setting | Headline result |
|------:|---------|-----------------|
| **0** | GBM, frictionless, European call | Recovers the BS delta from a *pure CVaR objective* (corr **0.99** with N(d1)); an MSE-vs-CVaR comparison proves the objective shapes the hedge |
| **1** | + proportional transaction costs | A **no-trade band** emerges and widens with cost; beats BS delta on cost-adjusted CVaR through realistic costs |
| **2** | Heston stochastic vol (incomplete market) | Beats constant-vol BS delta on CVaR (**−2.95 vs −3.38**); ablation shows the edge is a better hedge ratio, not vol-timing |
| **3** | Knock-out barrier (up-&-out call) | Learns the *sign-flipping barrier delta* from scratch; cuts P&L std **2.7×** (1.61 vs 4.30) |
| **4** | Autocallable note + 2nd instrument | Adding a vanilla put at the barrier cuts std **29%** and improves CVaR; diagnostic-driven strike choice |
| **5** | Model misspecification | The Heston-trained hedger keeps its edge over BS delta across parameter shifts, a vol-regime shift, *and* jumps it never saw |
| **6** | **Real S&P 500 data + market impact** | Trained on 2004–2018, cuts 99% Expected Shortfall (= FRTB capital) **~30% out-of-sample** with **half the turnover**; honest tie on the extreme crisis tail |

All results are seeded and reproducible. The common engine: CVaR₉₉ objective,
premium pinned to the model price, a single shared MLP reused at every step,
**observable-only features** (the latent variance is never fed in), and fresh
simulated paths every epoch. Phase 6 keeps the same engine but replaces the
simulator with block-bootstrapped real returns.

---

## The phases in detail

### Phase 0 — Correctness anchor (`deep_hedging_phase0.py`)
Frictionless Black–Scholes, European call. Trained only to minimise CVaR — with
no knowledge of the BS formula — the network **recovers the delta hedge**
(correlation 0.99 ± 0.0006 across seeds, MAE 0.035). A side-by-side
*MSE-trained* baseline makes the objective concrete: the MSE (variance-minimising)
net matches the BS-delta std but has the worst tail, while the CVaR net accepts
slightly more variance to buy the best 99% tail. The "gap" is the objective, not a
bug. *Plots: `phase0_delta.png`, `phase0_pnl.png`.*

### Phase 1 — Transaction costs and the no-trade band (`deep_hedging_phase1.py`)
A proportional cost `c·S·|Δposition|` enters the P&L. Two clean results: turnover
falls monotonically as cost rises (the hedger trades less), and the learned policy
develops a **no-trade band** around the target delta — it keeps its position
rather than paying to rebalance — reproducing the Whalley–Wilmott result without
hard-coding it. The deep hedger beats BS delta on cost-adjusted CVaR through ~50
bps. *Honest caveat:* at an extreme 1% cost the 99%-tail optimisation lags BS
delta within a practical budget (the strong cost-reduction gradient overwhelms the
sparse tail gradient) — a real property of minibatch-CVaR, documented rather than
hidden. *Plots: `phase1_band.png`, `phase1_sweep.png`, `phase1_pnl.png`.*

### Phase 2 — Incomplete market under Heston (`deep_hedging_phase2.py`)
With stochastic volatility the market is incomplete: the option carries vol risk
the underlying alone cannot remove, and the constant-vol BS delta is the *wrong*
hedge ratio. At zero cost the deep hedger beats it on CVaR (−2.95 vs −3.38).
*Honest ablation:* an observable realized-vol proxy feature did **not** help — the
edge is structural (the network learns the Heston-appropriate hedge ratio), not
volatility-regime timing. Both stds (~0.88) dwarf the complete-market 0.36 because
vol risk is un-hedgeable with the underlying alone — motivating Phase 4's second
instrument. *Plot: `phase2_pnl.png`.*

### Phase 3 — Knock-out barrier (`deep_hedging_phase3.py`)
An up-&-out call: near the barrier the true delta flips sign and explodes. A
hedger using the plain vanilla delta piles into long stock right where the option
is about to vanish; its P&L is bimodal (it gambles, std 4.30). The deep hedger,
trained on the knock-out payoff, learns to **de-risk and go short** as the spot
approaches the barrier — the sign-flipping barrier delta, learned from scratch —
cutting std 2.7× to 1.61 and improving CVaR. *Plots: `phase3_policy.png`,
`phase3_pnl.png`.*

### Phase 4 — Autocallable with a second instrument (`deep_hedging_phase4.py`)
A single-underlying autocallable note (quarterly early redemption + a downside
barrier cliff) — the kind of structured product real desks struggle to hedge.
With the **underlying alone** the deep hedger gets std 5.08 / CVaR −6.09. Adding a
vanilla put as a second instrument cuts std to 3.59 (−29%) and improves CVaR. The
strike matters: a put away from the tail lowered std but *worsened* CVaR;
diagnosing that the CVaR tail lives at the downside barrier and moving the put
there fixed both. The hedger actively trades the option, with a position that
jumps at the autocall level. *Plots: `phase4_putpos.png`, `phase4_pnl.png`.*

### Phase 5 — Robustness to model misspecification (`deep_hedging_phase5.py`)
The Phase 2 hedger is trained on a single Heston calibration, then its *fixed*
policy is stress-tested on worlds it never saw — Heston with higher vol-of-vol,
stronger skew, a higher vol regime, and a Merton jump-diffusion. With the premium
re-pinned per world (so we measure hedging, not pricing), the deep hedger keeps
its CVaR *and* std edge over constant-vol BS delta in **every** world, including
the jump crashes. The learned hedge generalises rather than overfitting to one
calibration. *Plot: `phase5_robustness.png`.*

### Phase 6 — Empirical tail-risk hedging under market impact (`deep_hedging_phase6.py`)
The production-shaped capstone. It drops three simulator-era assumptions at once:
paths are **block-bootstrapped from 21 years of real S&P 500 daily returns** (so
they carry true volatility clustering and fat tails — sample kurtosis ~13 vs 0 for
a Gaussian); trading cost is **nonlinear market impact** (a square-root-law term,
so large rebalances are punished); and the objective, CVaR₉₉, is read as the **FRTB
99% Expected Shortfall capital charge**. Trained only on 2004–2018 and evaluated
out-of-sample, the impact-aware hedger cuts that capital metric ~30% (−10.50 →
−7.32) with lower P&L std *and roughly half the turnover* — it hedges better while
trading far less notional. *Honest result:* in the 2008/2020 crises (≈3× the
training volatility, never seen) it ties the constant-vol BS delta on the extreme
1% tail — daily underlying-only hedging hits a wall when the market gaps — though it
still does so at half the turnover and with lower std. Closing that crisis tail is
exactly what gamma/options hedging (Phase 4) on real data would address.
Evaluation is walk-forward (train/test split by date), with a hedge-latency
benchmark (~0.2 µs/decision on CPU). *Plot: `phase6_pnl.png`.*

---

## Design decisions (and why)

- **Risk measure: CVaR₉₉** via Rockafellar–Uryasev — coherent, tail-focused, and
  differentiable with one auxiliary variable.
- **Premium pinned to the model price** so the deep hedger and the analytic
  benchmark start from the same premium — an apples-to-apples P&L comparison.
- **One shared MLP** reused at every step (time-to-maturity is an input), trained
  by backprop-through-time over the full simulated trajectory.
- **Observable-only features** — log-moneyness, time-to-maturity, current holding
  (and instrument prices where relevant). The latent Heston variance is *never*
  fed in, because it is unobservable in practice.
- **Fresh simulated paths every epoch** — a luxury of working in a simulator that
  removes any fixed-dataset overfitting; evaluation always uses held-out paths.

## Honest limitations

- Phase 1's edge erodes at extreme (1%) transaction costs (minibatch-CVaR tail
  gradient vs cost gradient).
- Phase 2's realized-vol proxy did not add value at this horizon.
- Phases 3–4 use GBM to isolate the barrier / second-instrument effects; combining
  them with Heston (true vega risk) is the natural extension.
- Single-instrument hedging in an incomplete market has an irreducible floor —
  some risk simply cannot be removed by trading the underlying alone.
- Phase 6's edge concentrates in normal regimes; in extreme crises, daily
  underlying-only hedging only ties the benchmark on the worst-1% tail. Real
  option-chain data and gamma hedging would be needed to push past that.

## Possible extensions

- Combine Phase 4's gamma/second-instrument hedging with Phase 6's real-data +
  market-impact setting — the synthesis that would attack the crisis tail.
- Implied-volatility-surface features with real option-chain data.
- Multi-asset / portfolio hedging; a reinforcement-learning (actor–critic)
  formulation for online, non-stationary settings.

---

## Repository layout & how to run

```
deep_hedging/
├── deep_hedging_phase0.py   # frictionless BSM — delta recovery
├── deep_hedging_phase1.py   # transaction costs — no-trade band
├── deep_hedging_phase2.py   # Heston — incomplete market
├── deep_hedging_phase3.py   # knock-out barrier
├── deep_hedging_phase4.py   # autocallable + 2nd instrument
├── deep_hedging_phase5.py   # robustness / misspecification
├── deep_hedging_phase6.py   # real S&P 500 data + market impact (capstone)
├── run_multiseed.py         # cross-seed error bars for all phases
├── requirements.txt
└── results/                 # plots + results text per phase
```

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # torch, numpy, matplotlib, yfinance
python deep_hedging_phase0.py      # ...through phase6
```

Phases 0–5 are self-contained and train on CPU in 1–5 minutes. Phase 6 pulls S&P
500 history once via `yfinance` (cached to `sp500_returns.npz`), then trains the
same way. All runs are seeded and reproduce the numbers above.

For cross-seed error bars on every phase's headline metric, run
`python run_multiseed.py` — it re-runs each phase across seeds {0,1,2} and reports
mean ± std (≈30–45 min on CPU; reduce via `MS_SEEDS` / `MS_EPOCHS`). Phase 0
already reports its own multi-seed numbers in-script.

*Stack: Python, PyTorch, NumPy, Matplotlib, yfinance (CPU).*
