# Institutional-Grade OTM Options-Selling Engine — Master Specification (v1.0)

> **Purpose of this document.** This is the rewritten, hardened version of the original
> project brief. It is written so that an AI agent *or* a human engineering team can build
> the system from it without re-interpreting ambiguous intent. It defines scope, the math,
> the architecture, the RL design, the data contracts, the test strategy, and — critically
> — the **honesty boundaries**: which of the original requirements are achievable as stated,
> which were impossible and have been re-specified, and what the realistic performance
> envelope is.
>
> **Non-negotiable quality bar.** Every module shipped against this spec must satisfy the
> 18-point institutional-grade checklist in §13. Code with `TODO`, fake logic, silent
> failures, or untested critical paths is considered *defective* and must not be merged.

---

## 0. Disclosure & honesty preface (read first)

A spec is only institutional-grade if it does not promise things that cannot be delivered.
The following claims from the original brief are **rewritten** because the literal version
is mathematically or physically impossible. Building toward the literal version would
produce a system that looks impressive in a backtest and blows up with real capital — the
opposite of prop-firm-grade.

| Original claim | Status | Re-specified as |
|---|---|---|
| "close the position before loss comes, open right before it starts to win, so nothing is lost" | **Impossible** (requires lookahead / future information) | Policy that **maximizes risk-adjusted expected value** given only information available at decision time; minimizes *expected* drawdown, not actual loss. Zero-loss is not a target; it is a red flag. |
| "predict the next regime accurately / with high accuracy" | **Overstated** | Probabilistic regime *nowcast + short-horizon forecast* with **calibrated** probabilities and honestly reported skill (Brier score, AUC, log-loss). We trade only when calibrated confidence clears a threshold. |
| "AI learns to predict the distribution very accurately" | **Conditional** | A neural surrogate is trained to approximate the **Monte-Carlo terminal distribution** of the simulator. Accuracy is bounded by the simulator's fidelity; we measure surrogate-vs-MC error explicitly and fall back to direct MC when error exceeds tolerance. |
| "indices where market makers are REQUIRED to quote optimally even the most OTM options" | **Misconception** | Designated Market Makers have quoting *obligations* (max spread, min size, % of time) on listed option classes (e.g. SPX, major ETFs). They are **not** required to quote *tight* on deep wings. The MM simulator models realistic obligation-bounded — not optimal — quoting. |
| "leverage if very confident" | **Allowed but hard-capped** | Leverage is permitted only inside a strict risk budget with margin-aware sizing; confidence alone never unlocks unlimited leverage. |
| "aggressively compound all profits" | **Tempered** | Fractional-Kelly compounding with a hard per-trade and per-day risk cap and a global drawdown kill-switch. |
| "simulate a whole prop firm so the AI learns spread friction perfectly" | **Approximated** | We simulate a *credible* market-making/microstructure environment (Avellaneda–Stoikov + obligation constraints + queue/impact). "Perfectly" is replaced by "calibrated to public microstructure data and stress-tested." |

**Bottom line:** This system is designed to have **positive expected edge with controlled,
quantified tail risk** — not to be loss-free. Selling OTM options is *short volatility /
short gamma*: it wins often and loses rarely-but-large. The entire architecture is built
around surviving the rare large loss. Anyone who tells you otherwise should not manage money.

---

## 1. Product summary

A research-and-execution platform that **sells out-of-the-money options as iron condors**
on high-volatility index products, gated by a **rough-volatility price model**, a
**regime-detection layer**, and a **reinforcement-learning decision core** that chooses
between *theta harvesting*, *gamma harvesting*, and *no action*, and manages entries/exits.

### 1.1 Trading cadence ("Mode")
- **NORMAL** — decision/holding horizon **1 hour to 1 day** (daily-driven).
- **MFD** (Medium-Frequency Decisioning) — decision/holding horizon **1 minute to 1 hour**.

Mode is a first-class config parameter that changes the model time grid, feature windows,
RL step frequency, and cost assumptions. It is **not** "HFT" — we do not compete on latency.

### 1.2 Operational modes
1. **HISTORY_BACKTEST** — historical underlying minute data + a **synthetic options chain**
   priced by our volatility model, with **bid/ask, depth, and liquidity produced by the
   market-maker simulator**. Fully offline, fully reproducible.
2. **LIVE_BACKTEST (paper)** — real-time market data, simulated fills, no real orders.
3. **ACCOUNT_TRADING (live)** — real broker keys, real orders, identical decision logic to
   the backtests (single source of truth for strategy code).

### 1.3 Strategy nucleus
- Instrument: **iron condor** on every trade (mandatory, per brief).
- Direction: **net short premium** (OTM both wings).
- Edge source: overpriced wing IV vs. model-implied terminal distribution, harvested via
  theta decay and managed gamma, **only in calibrated low-vol / high-confidence regimes**.
- Universe: multiple liquid, MM-obligated index option classes chosen so that at least one
  pair is **negatively correlated in volatility regime**, guaranteeing a tradable book in
  most low-vol windows.

---

## 2. The quantitative core

### 2.1 Underlying price model
Spot dynamics under the physical measure:

```
dS_t / S_t = mu_t dt + sqrt(v_t) dW_t^S + (jump term)
```

- **Volatility `v_t`: rough Bergomi (rBergomi).** Variance driven by a fractional
  (Riemann–Liouville) kernel with Hurst exponent `H ∈ (0, 0.5)`:

  ```
  v_t = xi_0(t) * exp( eta * sqrt(2H) * ∫_0^t (t-s)^{H-1/2} dW_s^v  -  (eta^2 / 2) * t^{2H} )
  ```

  Parameters: forward variance curve `xi_0(t)`, vol-of-vol `eta`, Hurst `H`,
  spot–vol correlation `rho` (leverage effect, `rho < 0`).

- **Jumps (optional, regime-gated):** compound Poisson / Merton-style jumps in log-price
  with intensity `lambda` and jump-size distribution `N(mu_J, sigma_J^2)`. Enabled when the
  jump-likelihood-ratio test (§2.4) favors the jump model.

- **Dynamic drift `mu_t`:** a state-dependent drift model (not a constant). Implemented as
  a small regularized model over slow features (carry, trend, term-structure of variance,
  regime posterior). Drift is deliberately *low-weight* — for short-dated premium selling
  the distribution's **shape and tails dominate**, and over-confident drift is a classic
  blow-up cause. Drift estimation must be regularized and walk-forward validated.

### 2.2 What we actually produce
The **terminal (and path) probability distribution of the underlying** over the trade
horizon — represented as:
- a Monte-Carlo sample set (ground truth), and
- a **learned distributional surrogate** (see §2.5) for fast, in-the-loop evaluation.

This distribution feeds: condor strike selection, win-probability, expected P&L,
CVaR/tail metrics, and the RL state.

### 2.3 Simulation engine
- **Exact / hybrid rBergomi simulation** (Cholesky for small grids; **hybrid scheme**,
  Bennedsen–Lunde–Pakkanen, for performance at scale). Antithetic + Sobol QMC variates,
  control variates for variance reduction. Deterministic seeding for reproducibility.
- Vectorized (NumPy) with an optional Numba/torch path for batch MC.
- Strict convergence diagnostics: standard error reported on every estimated quantity;
  the engine refuses to return a quantity whose MC standard error exceeds tolerance.

### 2.4 Parameter estimation (calibration)
- **MLE** of (`H, eta, rho, xi_0(·)`, jump params) — but rough-vol likelihoods are not
  closed-form, so we specify a **layered, honest** approach:
  1. **Realized-volatility / spectral estimator** for `H` (e.g. estimating the scaling of
     `m(q, Δ)` log-moments — the Gatheral–Jaisson–Rosenbaum method).
  2. **Simulated / quasi-MLE** for `eta, rho` against realized-variance moments.
  3. **Implied-vol surface calibration** of `xi_0` and surface params when option data is
     available (weighted least squares on the IV surface).
  4. **Likelihood-ratio test** to decide jumps on/off.
- All calibration is **walk-forward**; parameters carry timestamps and confidence
  intervals; stale calibrations are rejected.

### 2.5 Distributional surrogate (the "AI distribution")
- Train a neural model to map `(model params, market state, horizon, mode)` →
  **distribution of terminal log-return**.
- Output a *proper distribution*, not a point estimate. Recommended:
  **distributional output via a monotonic quantile network (non-crossing quantiles)** or a
  **normalizing flow** conditioned on the state vector.
- **Loss:** quantile / CRPS loss; calibration enforced via PIT histograms.
- **Guardrail:** at inference we periodically re-run full MC and measure surrogate error
  (Wasserstein / CRPS). If error > tolerance, the system **falls back to direct MC** and
  flags the surrogate for retraining. No silent degradation.

### 2.6 Regime detection & short-horizon forecast
- Label volatility regimes (low / mid / high) without lookahead leakage.
- Models to evaluate (pick by validated skill, not by novelty):
  - **Hidden Markov Model / Markov-switching** baseline (interpretable, strong prior).
  - **Hidden semi-Markov / change-point** for duration awareness.
  - **Sequence model** (Temporal Fusion Transformer or a compact TCN) for the nowcast +
    next-step regime *posterior*.
- **Outputs are calibrated probabilities** (isotonic / temperature scaling). We report
  Brier score, log-loss, reliability diagrams. **No "high accuracy" claim ships without
  these numbers.**
- **Trade gate:** trade only when `P(low-vol now) AND P(low-vol next step)` both exceed
  calibrated thresholds, with the threshold itself tuned on out-of-sample data.

### 2.7 News / event risk gate
- Pull from a news/economic-calendar API (key in env; provider abstracted behind an
  interface).
- Two-layer filter: **(a)** scheduled-event calendar (FOMC, CPI, earnings of constituents,
  OPEX) → no new risk before known events; **(b)** breaking-news classifier (keyword rules
  **and** a lightweight transformer sentiment/relevance classifier).
- On a material event: **configurable cool-off (default 5 trading days)** during which no
  new positions are opened; existing positions move to *defensive management only*.
- The 5-day number is a **parameter**, not a hard-coded constant.

---

## 3. The market-maker / microstructure simulator

This is what makes the synthetic chain credible and teaches the RL agent **spread friction**.

- **Quoting model:** Avellaneda–Stoikov optimal market-making (inventory-aware reservation
  price + spread) **bounded by exchange MM obligations** (max quoted spread, min size,
  minimum quoting uptime per class).
- **Microstructure:** queue position model, fill probability as a function of distance to
  touch, **price impact** for marketable size, and **wing liquidity decay** (deep OTM =
  wider, thinner — realistically).
- **Greeks-consistent pricing:** option mid is produced from the *same* rough-vol model and
  IV surface, so the chain is internally arbitrage-checked (no calendar/butterfly
  violations after construction; we run a no-arb repair pass).
- **Calibration:** parameters tuned to public spread/quote statistics per class and
  stress-tested (widen spreads, pull liquidity) to ensure the strategy survives bad
  microstructure, not just average microstructure.
- **Honesty:** labelled an *approximation*. We never claim to reproduce any real firm's
  proprietary model.

---

## 4. The RL decision core (the "brain")

Per the brief, the **theta-vs-gamma-vs-nothing** decision **and** the **entry/exit
management** are a single cutting-edge RL agent. It always trades **iron condors**.

### 4.1 Framing
- **POMDP** (the true vol state is latent → partial observability).
- **Two coordinated heads / hierarchical policy:**
  1. **Strategic head** — discrete: `{HARVEST_THETA, HARVEST_GAMMA, FLAT/NO_TRADE}`.
  2. **Tactical head** — parameterized actions: condor wing distances (in std/delta),
     width, size (risk fraction), and roll/close/adjust decisions.
- Action space is therefore **parameterized / hybrid** (discrete strategic + continuous
  tactical).

### 4.2 Algorithm
- Baseline: **PPO** (stable, well-understood) for the parameterized policy.
- Risk-sensitive upgrade: **distributional RL (e.g. QR-DQN / IQN-style critic)** so the
  agent optimizes a **risk measure (CVaR), not just mean return** — essential for a short-
  premium book.
- **Offline RL pretraining** (CQL/IQL) on simulator-generated and historical-replay data,
  then fine-tune online in LIVE_BACKTEST before any capital. No agent touches real money
  until it passes the promotion gate (§9).

### 4.3 State (observation) — must be point-in-time, leakage-free
- Underlying features: returns at multiple scales, realized vol, RV term structure.
- Model features: calibrated `H, eta, rho, xi_0` snapshot + calibration confidence.
- Distribution features: model win-prob, expected credit, CVaR, tail mass, distance of
  wings to expected move.
- Regime features: calibrated regime posteriors (now + next step) + reliability flag.
- Microstructure features: current condor bid/ask, spread cost, depth, est. slippage.
- Portfolio Greeks: net delta, gamma, vega, theta, margin usage, current drawdown.
- Event features: time-to-next-scheduled-event, cool-off flag, news-risk score.
- Calendar: time-to-expiry, time-of-day, mode.

### 4.4 Reward — designed against reward hacking
```
reward_t = realized_dPnL_t
         - lambda_risk   * incremental_CVaR / drawdown penalty
         - lambda_cost   * spread + slippage + commissions actually paid
         - lambda_margin * margin/leverage utilization
         - lambda_tail   * penalty for breaching Greek/exposure limits
         + small theta-capture credit (shaping, decayed over training)
```
- Optimizes **risk-adjusted** P&L, charges **real** transaction costs every step (so the
  agent learns spread friction), and is penalized for tail/limit breaches.
- **Anti-degeneracy:** randomized episodes, regime-stratified sampling, domain
  randomization over MM params, reward normalization, and explicit checks against
  *collapse-to-no-trade* and *collapse-to-always-trade*. Mode-collapse and
  regression-to-mean are monitored as first-class training metrics.

### 4.5 Hard safety overlay (non-RL, always on)
The RL agent is **never** the last line of defense. A deterministic risk supervisor sits
*outside* the policy and can veto/override:
- Per-trade max loss = defined-risk condor width × size (condors are inherently capped —
  good).
- Per-day loss limit, global trailing-drawdown **kill-switch** (flatten + halt).
- Max net Greeks, max margin/leverage, max correlated exposure across the universe.
- **Black-swan guards:** gap/halt detection, vol-spike circuit breaker, liquidity-vanish
  detection → defensive close only.
- Every override is logged with reason codes. The agent is trained *with this overlay
  active* so it learns within the real constraint set.

---

## 5. Position construction & management

- **Iron condor sizing:** wings chosen by joint optimization of (model win-prob, credit
  net of spread, CVaR, margin). Tightness vs. risk is a tunable objective, **not** a fixed
  rule. Avoid degenerate over-tight (no trades) and over-wide (no edge) via the RL tactical
  head plus rule-based sanity bounds.
- **Entry logic** (beyond the basics in the brief): regime gate + distribution edge +
  IV-rank/term-structure filter + spread-cost filter + event gate + portfolio-fit
  (correlation/Greek budget) + RL approval.
- **Exit / management:** profit-target (% of max credit), defensive roll, mechanical
  stop on Greek/loss breach, time-stop near expiry/gamma cliff, and RL-driven dynamic
  exit. Multiple, redundant exit reasons — never a single point of failure.
- **Dynamic hedging (optional module):** delta/vega hedging with the underlying or
  offsetting options when net Greeks exceed budget; cost-aware (only hedge when expected
  risk reduction > hedge cost).

---

## 6. Capital, sizing, leverage

- **Fractional Kelly** (default cap e.g. ¼-Kelly) on the model edge, **further clamped** by
  per-trade and per-day risk budgets and margin availability.
- Compounding: profits increase the risk *base*, but each unit of risk stays inside the
  same fractional caps — "aggressive" within a hard envelope.
- **Leverage:** permitted only when (a) calibrated win-prob confidence is high, (b) margin
  headroom exists, and (c) global risk budget allows — with an absolute leverage ceiling.

---

## 7. System architecture (modules)

```
options_engine/
├─ core/            config, types, time-grid, RNG, logging, errors
├─ models/
│  ├─ rbergomi/     simulation (cholesky + hybrid), variance reduction
│  ├─ jumps/        merton/compound-poisson + LR test
│  ├─ drift/        regularized dynamic drift
│  ├─ calibration/  H/eta/rho/xi0 estimators, IV-surface fit, walk-forward
│  └─ distribution/ MC aggregation + neural surrogate (quantile/flow) + fallback
├─ regime/          HMM/MS-VAR baseline + sequence model + calibration
├─ news/            calendar + breaking-news classifier + cool-off gate
├─ market/
│  ├─ chain/        synthetic chain builder + no-arb repair
│  └─ mm_sim/       Avellaneda-Stoikov + obligations + queue/impact
├─ rl/
│  ├─ env/          Gymnasium env (POMDP), reward, domain randomization
│  ├─ agents/       PPO + distributional critic; offline pretrain (CQL/IQL)
│  └─ safety/       deterministic risk supervisor / overlay
├─ strategy/        condor construction, entry/exit, sizing/Kelly, hedging
├─ execution/
│  ├─ brokers/      broker adapter interface + implementations
│  └─ oms/          order/position/risk state machine, idempotent orders
├─ backtest/        history + live(paper) engines (shared strategy core)
├─ data/            providers (market, options, news) behind interfaces
├─ services/        runners for the 3 operational modes
├─ observability/   structured logging, metrics, tracing, audit trail
└─ tests/           unit / integration / property / regression / sims
```

**Single source of truth:** strategy + RL + risk code is identical across the three
operational modes; only the *data* and *fill* layers differ (dependency injection). This
guarantees backtest ↔ live parity.

---

## 8. Data contracts & interfaces

- All external dependencies (market data, options data, news, broker) sit behind **typed
  interfaces** with at least: a real provider, a recorded/replay provider, and a simulated
  provider. **API keys/secrets/endpoints are the only allowed placeholders**, injected via
  environment / secrets manager and validated at startup (fail fast if missing).
- Strict schemas (pydantic / dataclasses) on every boundary; reject malformed data early
  with actionable errors.
- Reproducibility: every run writes a manifest (config hash, data range, seeds, model
  versions, git SHA).

---

## 9. Validation, promotion & live-readiness gates

A strategy/agent build may **only** advance HISTORY → PAPER → LIVE if it passes:
1. **Backtest integrity:** purged & embargoed walk-forward CV (Lopez de Prado), no
   lookahead, costs included, multiple-testing/deflated-Sharpe correction.
2. **Robustness:** parameter perturbation, MM-stress scenarios, regime-stratified results,
   historical crash replays (2018 vol-mageddon, 2020 COVID, 2022).
3. **Calibration evidence:** regime/distribution calibration plots within tolerance.
4. **Risk:** max drawdown, CVaR, tail behavior within mandate; kill-switch verified.
5. **Paper parity:** live-backtest tracking error vs. history-backtest within tolerance.
6. **Sign-off artifact:** an auto-generated tear-sheet + risk report.

> If it fails a gate, it does not advance. No exceptions. This gate *is* the institutional
> discipline.

---

## 10. Testing strategy (mandatory)

- **Unit tests** for every pricing/sim/feature function (known analytic limits: e.g.
  rBergomi → Black-Scholes as `eta→0`, put-call parity, condor payoff identities).
- **Property-based tests** (Hypothesis): no-arbitrage of generated chains, monotonic
  quantiles, distribution sums to 1, Greeks signs.
- **Statistical tests:** MC convergence/standard-error bounds, surrogate-vs-MC error,
  calibration (PIT, reliability).
- **Integration tests:** full pipeline on a fixed seed produces a known tear-sheet
  (golden-file regression).
- **RL tests:** env API conformance, reward correctness, safety-overlay veto tests,
  determinism with fixed seeds, anti-collapse monitors.
- **Execution tests:** OMS state machine, idempotency, partial fills, disconnect/reconnect,
  reconciliation.
- **CI:** lint (ruff), format (black), type-check (mypy --strict), tests with coverage
  gate on critical paths, static security scan. Red CI = unmergeable.

---

## 11. Observability & operations

- Structured JSON logging with correlation IDs per decision and per order.
- Metrics: P&L, Greeks, drawdown, win-rate, regime posteriors, surrogate error, fill
  quality, latency.
- Full **audit trail**: every decision records its inputs, the model versions, the action,
  the safety verdict, and the outcome. Reconstructable after the fact.
- Alerting on kill-switch, calibration drift, data outages, broker errors.

---

## 12. Build order (phased — each phase ends green-tested)

1. **Foundation:** core (config, types, RNG, logging, errors) + test harness + CI.
2. **rBergomi sim + variance reduction** (+ analytic-limit tests).
3. **Pricing + Greeks + distribution aggregation** (MC ground truth).
4. **Calibration** (H/eta/rho/xi0 + jumps LR + walk-forward).
5. **Distribution surrogate** (+ MC-fallback guardrail).
6. **Regime layer** (+ calibration metrics).
7. **MM simulator + synthetic chain** (+ no-arb tests).
8. **News/event gate.**
9. **Strategy layer** (condor construction, sizing/Kelly, entry/exit rules) + risk
   supervisor.
10. **RL env + reward + safety overlay** (+ env conformance tests).
11. **RL agent** (offline pretrain → PPO + distributional critic) + anti-collapse monitors.
12. **Backtest engines** (history + paper) + promotion gates + tear-sheets.
13. **Execution/OMS + broker adapter interface** (paper first).
14. **Account trading runner** (behind explicit, audited enable flag).

> Nothing in a later phase is allowed to weaken the tests of an earlier phase.

---

## 13. Definition of Done — institutional-grade checklist (applies to EVERY file)

1. **Correctness** — verified against analytic limits / golden files; assumptions explicit.
2. **Reliability** — graceful on bad input; predictable, logged failure modes.
3. **Readability** — single-purpose functions, clear names, minimal complexity.
4. **Maintainability** — modular, low coupling, high cohesion, consistent style.
5. **Testability** — unit + integration + regression; deterministic; runnable via one cmd.
6. **Documentation** — docstrings, design notes, usage examples, known limitations.
7. **Input validation** — types/ranges/formats checked; invalid state rejected early.
8. **Error handling** — structured exceptions; no silent failures; actionable messages.
9. **Performance** — efficient algorithms; profiled hot paths; no premature optimization.
10. **Reproducibility** — deterministic with seeds; explicit config; run manifests.
11. **Consistency** — uniform architecture, naming, error handling, interfaces.
12. **Extensibility** — open/closed where practical; reusable components.
13. **Observability** — meaningful logging/metrics; decisions and failures visible.
14. **Security** — validate untrusted input; secrets via env; secure defaults.
15. **Dependency mgmt** — justified, pinned deps; risks understood.
16. **Code-quality practices** — lint, format, static analysis, type hints, review.
17. **Scalability** — predictable resource use; no redesign needed to grow.
18. **Domain soundness** — correct quant finance: no lookahead, costs modeled, tails
    respected, no zero-loss fantasy.

> Prototype answers "can this work?" Production answers "will this work reliably?"
> Institutional answers "will this still be correct, maintainable, and trustworthy when
> someone else changes it years from now?" **We build for the third question.**

---

## 14. Explicit risk disclaimer (must ship with the system)

Selling OTM options is a **short-volatility, negative-skew** strategy: frequent small gains,
infrequent large losses. No model — rough-vol, RL, or otherwise — eliminates tail risk.
This system targets **positive risk-adjusted expectancy with quantified, capped downside**,
enforced by defined-risk condors and a hard risk supervisor. It is **not** a guaranteed or
loss-free system, and must be validated on the user's own data and broker before any real
capital is deployed. Past/backtested performance does not guarantee future results.
```
```
```
```

---

## 15. Open decisions I need from you before coding

These genuinely change the architecture, so I'd rather ask than guess (see chat).
