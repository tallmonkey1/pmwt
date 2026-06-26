# `options-engine` — System Architecture

> Auto-derived from the source code in `tallmonkey1/hutrh → unpacked/options-engine/`.
> Read [`SPEC.md`](../extracted/repo/unpacked/options-engine/SPEC.md) for the authoritative
> specification (math, RL design, honesty boundaries, risk disclaimer) and
> [`src/options_engine/execution/SAFETY.md`](../extracted/repo/unpacked/options-engine/src/options_engine/execution/SAFETY.md)
> for the five locks guarding real orders.
>
> Companion machine-readable view: [`modules.json`](modules.json) — every module's
> responsibility, public API, internal dependencies, and external third-party usage.

## One-paragraph summary

`options-engine` is a **research-and-execution platform that sells out-of-the-money (OTM)
iron condors** on liquid index/ETF options. It models the underlying with a **rough-volatility
(rBergomi)** simulator (ground-truth HybridSimulator + a fast **neural-rBergomi surrogate**
trained on TRUE paths), turns simulated paths into a **terminal price distribution**, prices
and risk-scores condors against a **synthetic, arbitrage-free options chain** quoted by an
**Avellaneda–Stoikov market-maker simulator with alpha-driven microstructure noise**,
gates trading on a **calibrated volatility-regime HMM** and a **news/event cool-off**, and
decides entries/exits with a **risk-sensitive PPO-Transformer agent** whose actor and critic
share a **causal Transformer backbone** that conditions on the last ``seq_len`` observations
(so the non-Markovian structure of the rBergomi Volterra kernel can be exploited). A
deterministic **hard risk supervisor** sits outside the policy and can veto any order
(per-trade/per-day caps, leverage, trailing-drawdown kill-switch). A **helper-critic agent**
(meta-controller) learns the **MarketAlpha** vector — a 5-feature unit-interval calmness
signal — that drives the main agent's internal diagnostics toward ``1`` (a calm, fully
profitable, stable market). Everything is validated by **backtest engines with deflated-Sharpe
and purged walk-forward CV**, promoted through **HISTORY → PAPER → LIVE gates**, and executed
through a **fail-closed OMS** where sending a real order requires five independent locks.

## Stack at a glance

| Item | Value |
|---|---|
| Language | Python ≥ 3.11 (developed/tested on 3.13) |
| Lines of source code | ~15,800 across **103 modules** in `src/options_engine/` |
| Lines of test code | ~8,500 across 91 test files |
| Internal module-level edges | **402** (extracted by AST) |
| Package-level edges | **33** (the dependency graph is **acyclic** at the package layer — verified by automated DFS) |
| External third-party packages | **25 distinct** (numpy / scipy / pydantic / torch / gymnasium / ib_insync / …) |
| License | Proprietary |
| Test framework | pytest + Hypothesis (property-based) |
| Lint / format / types | `ruff` + `black` + `mypy --strict` |

## Layered dependency graph (verified acyclic)

```
                  services/    (operational-mode broker factory, fail-closed)
                       │
        ┌──────────────┼──────────────┐
   backtest/       execution/        (validation + order routing)
   (metrics, gates) (OMS, broker, IBKR)
        │              │
        └──────┬───────┘
            agent/    (PPO + transformer backbone + helper critic — the brain)
              │
             rl/      (Gymnasium POMDP env, alpha-aware, reward, scenarios)
              │
          strategy/   (condor selection, Kelly sizing, entry/exit, risk supervisor)
       ┌─────┼─────┬─────────┬───────────┐
    market/ regime/ news/  surrogate/  pricing/
    (MM sim +  (HMM +  (gate,  (quantile   (BS, MC,
     alpha     calib)  cooloff) net + MC    condor
     noise)                       fallback) payoff/greeks)
       └─────┴─────┴─────────┴───────────┘
                  │
              calibration/  (rBergomi parameter estimation, alpha-driven)
                  │
              models/      (rBergomi ground-truth + NEURAL rBergomi surrogate)
                  │
               core/        (config, errors, validation, RNG, logging, enums,
                              time, MARKETALPHA framework)
```

The arrow direction in this diagram is **"depends on"**: every layer depends only on layers
below it. The automatic extractor confirms this — see the `totals.package_level_acyclic`
flag in [`modules.json`](modules.json).

## Package map (16 packages, 103 modules)

| Package | Mods | Responsibility | Key public types |
|---|---:|---|---|
| **`core`** | 9 | Cross-cutting primitives: pydantic-validated config, structured exceptions, fail-fast scalar validators, reproducible named RNG sub-streams, JSON correlation logging, domain enums, time grids in years, **the MarketAlpha framework** (unit-interval calmness vector, mappings to Hurst / eta / jump / shock / AS-noise knobs). | `EngineConfig`, `RiskConfig`, `SecretRef`, `RandomFactory`, `TimeGrid`, `OperationalMode`, `StrategicAction`, `MarketAlpha`, `alpha_to_hurst`, `alpha_to_stoikov_noise`, … |
| **`models.rbergomi`** | 9 | Rough-Bergomi price/vol simulation. Production **HybridSimulator** (`O(N log N)` via FFT) + exact **CholeskySimulator** for validation + a **fast neural surrogate** (`NeuralRBergomiSimulator`) trained on TRUE paths + the **alpha calibration** helper that turns a MarketAlpha into a fully-specified `RBergomiParams`. | `RBergomiParams`, `ForwardVariance`, `HybridSimulator`, `CholeskySimulator`, `NeuralRBergomiSimulator`, `build_rbergomi_params_from_alpha`, `TerminalDistribution`, `SimulationPaths` |
| **`models`** | 3 | Aggregates the rBergomi subpackage and re-exports drift/jumps utilities for outside callers. | — |
| **`pricing`** | 6 | Analytic Black–Scholes (prices + full Greeks + IV via safeguarded Newton/bisection), MC pricing off the rBergomi distribution, instrument specifications (`EuropeanOption` / `IronCondor` / `BullPutSpread`), payoff functions, portfolio aggregation. | `EuropeanOption`, `IronCondor`, `BullPutSpread`, `price`, `greeks`, `implied_volatility`, `iron_condor_pnl` |
| **`calibration`** | 9 | Estimate rBergomi parameters from data: Hurst via log-moment scaling, η by structure-function matching, ρ by leverage-curve inversion, ξ₀ term structure, BNS jump test. All **walk-forward**. | `CalibrationConfig`, `calibrate_rbergomi`, `CalibrationResult`, `ParameterEstimate` |
| **`surrogate`** | 9 | Neural **monotone quantile network** that approximates the MC terminal distribution for speed, with a **Wasserstein MC-fallback guardrail**. | `DistributionSurrogate`, `SurrogateDistribution`, `MonotoneQuantileNetwork`, `SurrogateGuardrail` |
| **`regime`** | 7 | Gaussian **HMM** (log-space EM/Viterbi, multi-restart best-LL), LOW/MID/HIGH labeling by vol tertiles, **temperature calibration**, Brier/log-loss/ECE metrics, the **trade gate**. | `GaussianHMM`, `RegimeDetector`, `RegimeNowcast`, `TemperatureScaler`, `evaluate_regime_gate` |
| **`market`** | 6 | **Avellaneda–Stoikov** market-maker bounded by exchange obligations (max spread, min size) with wing-liquidity decay **and alpha-driven microstructure noise** (`apply_alpha_noise`). Fill simulator (slippage + book walk + impact) and synthetic options chain builder with no-arbitrage repair. | `AvellanedaStoikovMaker`, `MarketMakerConfig`, `ObligationConfig`, `apply_alpha_noise`, `simulate_fill`, `OptionChain`, `build_synthetic_chain` |
| **`news`** | 6 | Scheduled-event blackout + breaking-news classifier + configurable **trading-day cool-off** (default 5). Provider interfaces (replay + REST skeleton) so backtest and live share a contract. | `NewsGate`, `NewsGateConfig`, `KeywordNewsClassifier`, `EventProvider`, `NewsProvider` |
| **`strategy`** | 7 | Condor/spread strike selection from the terminal distribution, **fractional + empirical-Kelly** sizing, multi-filter entry / multi-reason exit, the **deterministic risk supervisor + kill-switch**. | `EntryEvaluator`, `evaluate_exit`, `RiskSupervisor`, `Account`, `OpenPosition`, `select_iron_condor`, `size_position`, `empirical_kelly_fraction` |
| **`rl`** | 7 | Gymnasium **POMDP** wrapping the entire engine: leakage-free 15-d observation, parameterized hybrid action (3 strategic + 3 continuous knobs), risk-sensitive reward (or growth-reward alternative), **alpha-aware scenario generator**, domain-randomized episodes. | `IronCondorTradingEnv`, `ObservationInputs`, `build_observation`, `DecodedAction`, `compute_reward`, `Episode`, `generate_episode` |
| **`agent`** | 10 | **PPO from scratch** (MLP variant) + **PPO-Transformer** (shared causal Transformer backbone for memory over past observations) + distributional (quantile/CVaR) critic + GAE + quantile-Huber loss + anti-collapse training monitors + **helper-critic** (tabular bandit over the alpha lattice that drives internal diagnostics to 1) + on-policy rollout buffer + Transformer backbone + sinusoidal positional encoding. | `PPOAgent`, `PPOTransformerAgent`, `TransformerBackbone`, `HelperCritic`, `compute_gae`, `quantile_huber_loss`, `Trainer`, `CollapseMonitor`, `alpha_lattice` |
| **`backtest`** | 6 | Performance/risk metrics incl. **deflated Sharpe**, backtest engine shared across RL/rule policies, purged walk-forward CV, and HISTORY→PAPER→LIVE **promotion gates**. | `run_backtest`, `evaluate_promotion`, `purged_walk_forward_splits`, `deflated_sharpe_ratio`, `PerformanceMetrics` |
| **`execution`** | 6 | **Idempotent, risk-gated OMS**; `Broker` interface + simulated broker; **fail-closed IBKR adapter** with five locks; **typed-confirmation live-trading arming**. `Order` is a full iron-condor ticket (four legs, traded as one). | `OrderManagementSystem`, `SimulatedBroker`, `IBKRBroker`, `LiveTradingArming`, `Order`, `Fill`, `OrderState` |
| **`services`** | 2 | The single **fail-closed broker factory** keyed off operational mode. Refuses rather than silently simulating if a live run isn't fully armed and configured. | `create_broker` |
| **`options_engine`** | 1 | Root package init (`__version__ = "0.1.0"`). | — |

## The two-layer learning architecture (NEW)

The engine learns in **two nested loops**, both reproducible and deterministic:

```
Outer loop (helper critic)               Inner loop (main PPO-Transformer agent)
─────────────────────────                 ──────────────────────────────────────
select alpha from lattice                 observe sequence (last seq_len obs)
       │                                          │
       ▼                                          ▼
build rBergomi params from alpha          Transformer encoder -> context vector
       │                                          │
       ▼                                          ▼
generate episode (alpha-aware)             Actor: diagonal Gaussian -> action
       │                                          │
       ▼                                          ▼
build synthetic chain (alpha-aware)        Critic: quantile head -> value, CVaR
       │                                          │
       ▼                                          ▼
main agent learns under this alpha         environment -> reward -> PPO update
       │                                          │
       ▼                                          ▼
observe internal diagnostics               training step
       │                                          │
       ▼                                          ▼
helper critic updates Q[alpha]             repeat until KL early-stop
```

* **Inner loop** (PPO-Transformer): the main decision-making agent. The Transformer
  backbone gives it memory over past observations, which is essential because
  rBergomi's Volterra kernel makes future variance depend on the path of past variance
  (the MLP PPO cannot exploit this).
* **Outer loop** (helper critic): the meta-controller. Its only job is to find the
  MarketAlpha that maximises a "stability score" = sum of internal diagnostics (each in
  `[0, 1]`) produced by the main agent. At the optimum, every diagnostic is `~= 1`,
  meaning the market is fully calm, fully predictable, and the main agent is fully
  profitable and stable.

The helper critic's tabular bandit over a discrete alpha lattice (default 8 alphas
spanning `zeros()` to `ones()`) converges quickly and keeps the outer loop cheap.

## External dependencies (the import graph's leaves)

| Package | Files | Purpose |
|---|---:|---|
| `numpy` (+ `numpy.typing`) | most | All numerical primitives, vectorized MC, distributions, FFT. |
| `dataclasses` | ~55 | Immutable-by-update state (RiskConfig, Order, MarketAlpha, …) — `@dataclass(frozen=True, slots=True)` pervasively. |
| `datetime` | ~14 | Timezone-aware timestamps throughout (UTC enforced at every boundary). |
| `scipy` (+ `linalg` + `signal` + `special` + `stats`) | ~7 | Cholesky factorization, FFT convolution (hybrid scheme), `hyp2f1` (Volterra kernel), `norm.cdf` / `norm.ppf` (BS / DSR), `logsumexp` (HMM forward-backward). |
| `pydantic` | 1 | Typed, validated, frozen config models. |
| `torch` (+ `nn` + `optim`) | ~9 | PPO actor + distributional critic, **Transformer backbone**, quantile network, surrogate net, **neural rBergomi surrogate**. |
| `gymnasium` | 1 | RL environment base class and space definitions. |
| `ib_insync` | 1 | IBKR broker SDK (live broker adapter, optional extra). |
| `abc` | 4 | Abstract base classes. |
| Standard library | various | Scalar primitives, env-var resolution, correlation IDs, etc. |

## The three operational modes (single source of truth)

The *same* decision pipeline runs in every mode; only the data and broker wiring differ
(dependency injection):

- **`HISTORY_BACKTEST`** — historical/synthetic data, `SimulatedBroker`. Offline, reproducible.
- **`LIVE_BACKTEST`** — real-time data, simulated fills (paper). No real orders possible.
- **`ACCOUNT_TRADING`** — real broker, real capital. **Reachable only through the five execution safety locks**.

## Defense-in-depth around capital

1. **Risk supervisor** vetoes any order breaching per-trade/day caps, leverage, margin, or
   the trailing-drawdown kill-switch — in *every* mode, on *every* order.
2. **Defined-risk only** — every position is an iron condor; max loss is bounded by construction.
3. **Fail-closed execution** — the simulated broker is the default; a live order needs
   `ACCOUNT_TRADING` mode + credentials + `enable_live_trading=True` + a typed confirmation
   phrase equal to `"I UNDERSTAND THIS TRADES REAL MONEY"` + risk approval, all at once.
4. **Idempotent orders** — duplicate `client_order_id`s can never create duplicate positions.
5. **Reconciliation** — the OMS can compare its ledger against the broker's reported positions;
   any divergence halts new trading until investigated.

## Reproducibility

All randomness flows through `core.RandomFactory`, which derives independent, named,
deterministic sub-streams from one master seed using a stable FNV-1a hash and NumPy's
`SeedSequence` / `PCG64` machinery. Configs are frozen, validated, and hashable
(`EngineConfig.fingerprint()` returns a SHA-256 digest of the canonical JSON), so a run is
reproducible from its seed and config.

## File layout

```
options-engine/
├─ SPEC.md                       # authoritative spec (read first)
├─ ARCHITECTURE.md               # this file (read second)
├─ README.md
├─ pyproject.toml                # build, deps, tooling config
├─ Makefile                      # quality-gate shortcuts (lint/format/typecheck/test)
├─ .github/workflows/ci.yml      # CI: lint, types, fast + slow tests on 3.11–3.13
├─ src/options_engine/
│  ├─ core/                      # config, errors, validation, RNG, logging, enums,
│  │                              # time, MARKETALPHA framework
│  ├─ models/rbergomi/           # rough-volatility simulator (ground truth)
│  │                              # + NEURAL rBergomi surrogate + alpha calibration
│  ├─ pricing/                   # Black–Scholes, MC, condor payoff/greeks
│  ├─ calibration/               # rBergomi parameter estimation (walk-forward)
│  ├─ surrogate/                 # neural quantile distribution + MC fallback guardrail
│  ├─ regime/                    # HMM regime detection + trade gate
│  ├─ market/                    # MM simulator (with alpha noise) + synthetic chain
│  ├─ news/                      # scheduled-event + breaking-news gate
│  ├─ strategy/                  # condor selection, sizing, entry/exit, risk supervisor
│  ├─ rl/                        # Gymnasium POMDP env (alpha-aware) + reward + scenarios
│  ├─ agent/                     # PPO + PPO-Transformer + helper critic (the brain)
│  ├─ backtest/                  # metrics, engine, walk-forward CV, promotion gates
│  ├─ execution/                 # OMS, brokers, IBKR adapter, live-trading locks (SAFETY.md)
│  └─ services/                  # fail-closed broker factory
└─ tests/                        # 91 test files mirroring src/ (with new alpha/transformer tests)
```

## Where to start reading

1. **`SPEC.md`** — the authoritative specification including honesty boundaries and risk disclaimer.
2. **`ARCHITECTURE.md`** (this file) — the system map, including the new two-loop learning architecture.
3. **`src/options_engine/execution/SAFETY.md`** — the five locks before any real order can reach a broker.
4. **`src/options_engine/core/market_alpha.py`** — the MarketAlpha framework that drives the helper-critic / PPO-Transformer interaction.
5. **`src/options_engine/agent/ppo_transformer.py`** — the memory-enabled PPO agent (the main decision brain).
6. **`src/options_engine/agent/helper_critic.py`** — the meta-controller that learns the MarketAlpha.
7. **`src/options_engine/models/rbergomi/neural_simulator.py`** — the fast neural rBergomi trained on TRUE paths.
8. **`src/options_engine/strategy/risk_supervisor.py`** — the deterministic risk overlay that sits outside the policy.
9. **`src/options_engine/agent/ppo.py`** — the simpler MLP PPO agent (still used by the original analysis scripts).
10. **`src/options_engine/models/rbergomi/simulator.py`** — the rough-volatility price model (production hybrid scheme + exact Cholesky for validation).
