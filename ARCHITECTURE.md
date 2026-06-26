# Architecture

This document maps the full system. It is the entry point for an engineer new to the
codebase. For *what* the system is and the honesty boundaries, read `SPEC.md` first; for
execution safety, read `src/options_engine/execution/SAFETY.md`.

## One-paragraph overview

The engine sells out-of-the-money **iron condors** on liquid index/ETF options. It models
the underlying with a **rough-volatility (rBergomi)** simulator, turns simulated paths into a
**terminal price distribution**, prices and risk-scores condors against a **synthetic,
arbitrage-free options chain** quoted by a **market-maker simulator**, and gates trading on a
**calibrated volatility-regime detector** and a **news/event cool-off**. A risk-sensitive
**PPO reinforcement-learning agent** is the decision brain, operating *inside* a deterministic
**hard risk supervisor** (caps, leverage, drawdown kill-switch). Everything is validated by
**backtest engines with deflated-Sharpe and purged walk-forward CV**, promoted through
**HISTORY → PAPER → LIVE gates**, and executed through a **fail-closed OMS** where routing a
real order requires five independent locks.

## Layered dependency graph

Layers depend only downward; the graph is acyclic.

```
                         services/        (operational-mode broker factory, fail-closed)
                            │
        ┌───────────────────┼────────────────────┐
   backtest/           execution/            (validation + order routing)
   (metrics, gates)    (OMS, broker, IBKR)
        │                   │
        └─────────┬─────────┘
              agent/        (PPO + distributional critic — the brain)
                │
               rl/          (Gymnasium POMDP env, reward, scenarios)
                │
            strategy/       (condor selection, Kelly sizing, entry/exit, risk supervisor)
        ┌───────┼───────┬──────────┬───────────┐
     market/  regime/  news/   surrogate/   pricing/
   (MM sim,  (HMM +   (gate,  (quantile    (BS, MC,
    chain)   calib)   calib)   net + MC     condor
                               fallback)    payoff/greeks)
        └───────┴───────┴──────────┴───────────┘
                         │
                    calibration/      (rBergomi parameter estimation, walk-forward)
                         │
                     models/          (rBergomi simulator: hybrid + exact Cholesky)
                         │
                      core/           (config, errors, validation, RNG, logging, enums, time)
```

## Package-by-package

| Package | Responsibility | Key public types |
|---|---|---|
| `core/` | Cross-cutting primitives: typed config (pydantic), structured errors, fail-fast validation, reproducible RNG sub-streams, JSON logging, domain enums, time grids. | `EngineConfig`, `RiskConfig`, `SecretRef`, `RandomFactory`, `TimeGrid` |
| `models/rbergomi/` | Rough-Bergomi price/vol simulation. Production **hybrid scheme** (O(N log N), FFT) + exact **Cholesky** scheme for validation; variance reduction; MC convergence diagnostics. | `RBergomiParams`, `HybridSimulator`, `TerminalDistribution` |
| `pricing/` | Analytic Black–Scholes (prices + full Greeks + implied vol), Monte-Carlo pricing off the rBergomi distribution, iron-condor payoff/credit, net-Greeks aggregation. | `EuropeanOption`, `IronCondor`, `price_iron_condor`, `iron_condor_greeks` |
| `calibration/` | Estimate rBergomi params from data: Hurst (log-moment scaling), η/ρ (simulation moment-matching), ξ₀ (IV term structure), jumps (BNS test), all **walk-forward**. | `calibrate_rbergomi`, `CalibrationResult` |
| `surrogate/` | Neural **monotone quantile network** that approximates the MC terminal distribution for speed, with a **Wasserstein MC-fallback guardrail**. | `DistributionSurrogate`, `SurrogateGuardrail` |
| `regime/` | Gaussian **HMM** with log-space EM/Viterbi; LOW/MID/HIGH labeling; temperature calibration; Brier/log-loss/ECE; the **trade gate**. | `RegimeDetector`, `evaluate_regime_gate` |
| `market/` | **Avellaneda–Stoikov** market-maker bounded by exchange obligations; fill/slippage simulator; **synthetic chain** with no-arbitrage repair. | `AvellanedaStoikovMaker`, `build_synthetic_chain`, `simulate_fill` |
| `news/` | Scheduled-event blackout + breaking-news cool-off (default 5 trading days); keyword classifier; provider interfaces (replay + REST skeleton). | `NewsGate`, `KeywordNewsClassifier` |
| `strategy/` | Condor strike selection from the distribution; **fractional + empirical Kelly** sizing; multi-filter entry / multi-reason exit; the **deterministic risk supervisor + kill-switch**. | `EntryEvaluator`, `evaluate_exit`, `RiskSupervisor`, `Account` |
| `rl/` | Gymnasium **POMDP** wrapping the whole stack; parameterized action; leakage-free observation; risk-sensitive reward; domain-randomized scenarios; safety overlay always on. | `IronCondorTradingEnv`, `compute_reward` |
| `agent/` | **PPO from scratch** + diagonal-Gaussian actor + **distributional (quantile/CVaR) critic**; GAE; quantile-Huber loss; anti-collapse training monitors. | `PPOAgent`, `Trainer`, `compute_gae` |
| `backtest/` | Performance/risk metrics incl. **deflated Sharpe**; backtest engine (agent or rule); **purged walk-forward CV**; **promotion gates**. | `run_backtest`, `evaluate_promotion`, `purged_walk_forward_splits` |
| `execution/` | Idempotent, risk-gated **OMS**; broker interface + simulated broker; **fail-closed IBKR adapter**; the typed live-trading arming lock. See `SAFETY.md`. | `OrderManagementSystem`, `SimulatedBroker`, `LiveTradingArming` |
| `services/` | The single **fail-closed broker factory** keyed off operational mode. | `create_broker` |

## The three operational modes (single source of truth)

The *same* decision pipeline runs in every mode; only the data and broker wiring differ
(dependency injection), which guarantees backtest ↔ live parity:

- **HISTORY_BACKTEST** — historical/synthetic data, simulated broker. Offline, reproducible.
- **LIVE_BACKTEST** — real-time data, simulated fills (paper). No real orders possible.
- **ACCOUNT_TRADING** — real broker, real capital. Reachable only through the five execution
  safety locks (see `execution/SAFETY.md`).

## Defense-in-depth around capital

1. **Risk supervisor** vetoes any order breaching per-trade/day caps, leverage, margin, or
   the trailing-drawdown kill-switch — in *every* mode, on *every* order.
2. **Defined-risk only** — every position is an iron condor; max loss is bounded by
   construction.
3. **Fail-closed execution** — the simulated broker is the default; a live order needs
   ACCOUNT_TRADING mode + credentials + an enable flag + a typed confirmation phrase + risk
   approval, all at once. The final IBKR placement seam is deliberately unimplemented until
   an operator wires and signs off on it.
4. **Idempotent orders** — duplicate client-order-ids can never create duplicate positions.

## Reproducibility

All randomness flows through `core.RandomFactory`, which derives independent, named,
deterministic sub-streams from one master seed. Configs are frozen, validated, and
hashable (`EngineConfig.fingerprint()`), so a run is reproducible from its seed and config.

## Quality bar (enforced)

- `ruff` (lint), `black` (format), `mypy --strict` (types) — all green, 93 source files.
- 736 tests (695 fast + 41 slow), all passing; **93% line/branch coverage**.
- Key numerical components are **proven against known math**, not just asserted: GAE vs.
  hand-computed advantages, PPO learning a known-optimum bandit, rBergomi vs. analytic
  limits (Black–Scholes as η→0), Black–Scholes Greeks vs. finite differences, the quantile
  loss recovering true distribution quantiles, and the deflated Sharpe's multiple-testing
  monotonicity.

See `SPEC.md` §13 for the full institutional-grade definition-of-done and §14 for the risk
disclaimer.
