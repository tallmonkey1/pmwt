"""Extract the module dependency graph for options_engine (schema v2.0).

Parses every Python file under ``src/options_engine/`` and produces a rich
dependency manifest:

* Per-module: full docstring, every public name (class, function, method,
  property, constant), each with parameters (name / default / annotation),
  return annotation, docstring summary, ``Raises`` clauses extracted from the
  docstring, and any side-effects the docstring mentions.
* Per-edge: the *specific* symbols imported across each internal/external edge
  (not just the source module), so a reader can see exactly which names flow
  where.
* Per-package: aggregated fan-in / fan-out, the responsibility summary, and a
  topological-layer assignment (foundation / quant-core / inference /
  decision / learning / evaluation / execution).

The output is written to ``modules.json`` alongside this script.

Usage: python extract_deps.py <repo_root>
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Package-level metadata
# ---------------------------------------------------------------------------

SUBPACKAGES: tuple[str, ...] = (
    "core",
    "models.rbergomi",
    "pricing",
    "calibration",
    "surrogate",
    "regime",
    "market",
    "news",
    "strategy",
    "rl",
    "agent",
    "backtest",
    "execution",
    "services",
    "models",
)

PACKAGE_DESCRIPTIONS: dict[str, str] = {
    "core": "Cross-cutting primitives: typed pydantic config, structured errors, fail-fast "
    "validation helpers, reproducible RNG sub-streams, JSON logging, domain enums, "
    "time grids, MarketAlpha (the unit-interval calmness vector the helper-critic "
    "optimises). The dependency root of every other package.",
    "models.rbergomi": "Rough-Bergomi price/vol simulation: the production HybridSimulator "
    "(O(N log N) via FFT) + the exact CholeskySimulator for validation + a fast "
    "NeuralRBergomiSimulator (Transformer trained on TRUE paths) + alpha-driven "
    "parameter construction + terminal-distribution aggregation with MC error control.",
    "models": "Aggregates the rBergomi subpackage and re-exports drift/jumps utilities "
    "for outside callers.",
    "pricing": "Analytic Black-Scholes (prices + full Greeks + IV via safeguarded "
    "Newton/bisection), Monte-Carlo pricing off the rBergomi distribution, instrument "
    "specifications (EuropeanOption / IronCondor / BullPutSpread), payoff functions, "
    "and portfolio aggregation.",
    "calibration": "rBergomi parameter estimation from data: Hurst (log-moment scaling), "
    "vol-of-vol eta (structure-function matching), spot-vol correlation rho "
    "(leverage-curve inversion), forward-variance xi0, and a BNS jump test. All "
    "walk-forward.",
    "surrogate": "Neural monotone-quantile-network approximation of the MC terminal "
    "distribution for speed, with a Wasserstein-distance MC-fallback guardrail and "
    "a full fit / predict / save / load lifecycle.",
    "regime": "Gaussian HMM regime detector (LOW/MID/HIGH): log-space EM/Viterbi, "
    "temperature calibration, Brier/log-loss/ECE metrics, and a calibrated trade gate.",
    "market": "Avellaneda-Stoikov market-maker bounded by exchange obligations (max "
    "spread, min size) with wing-liquidity decay + alpha-driven microstructure noise "
    "(`apply_alpha_noise`). Fill simulator (slippage + book walk + impact) and a "
    "no-arbitrage synthetic chain builder.",
    "news": "Scheduled-event blackout + breaking-news classifier + configurable "
    "trading-day cool-off gate. Provider interfaces (replay + REST skeleton) so the "
    "backtest and live modes share a contract.",
    "strategy": "Condor/spread strike selection from the terminal distribution, "
    "fractional + empirical-Kelly sizing, multi-filter entry / multi-reason exit, and "
    "the deterministic risk supervisor + trailing-drawdown kill-switch.",
    "rl": "Gymnasium POMDP wrapping the entire engine: leakage-free 15-d observation, "
    "parameterized hybrid action (discrete strategic + 3 continuous knobs), "
    "risk-sensitive reward (or growth-reward alternative), and domain-randomized "
    "alpha-aware scenario generation.",
    "agent": "PPO from scratch (MLP variant) + PPO-Transformer (shared causal Transformer "
    "backbone for memory over past observations) + distributional (quantile/CVaR) "
    "critic + GAE + quantile-Huber loss + anti-collapse training monitors + "
    "helper-critic (tabular bandit over the alpha lattice that drives internal "
    "diagnostics to 1) + on-policy rollout buffer + Transformer backbone + "
    "sinusoidal positional encoding.",
    "backtest": "Performance/risk metrics including the deflated Sharpe, backtest engine "
    "shared across RL/rule policies, purged walk-forward CV, and the HISTORY -> "
    "PAPER -> LIVE promotion gates.",
    "execution": "Idempotent, risk-gated OMS; Broker interface + simulated broker; "
    "fail-closed IBKR adapter (typed-confirmation arming); live-trading safety "
    "documentation. Five independent locks guard any real order.",
    "services": "Operational-mode runners and the single fail-closed broker factory "
    "(HISTORY_BACKTEST/LIVE_BACKTEST -> simulated, ACCOUNT_TRADING -> live, "
    "refuses rather than silently degrading).",
}

PACKAGE_LAYERS: dict[str, str] = {
    "core": "foundation",
    "models.rbergomi": "quantitative_core",
    "models": "quantitative_core",
    "pricing": "quantitative_core",
    "calibration": "calibration_layer",
    "surrogate": "inference_layer",
    "regime": "inference_layer",
    "market": "inference_layer",
    "news": "inference_layer",
    "strategy": "decision_layer",
    "rl": "learning_layer",
    "agent": "learning_layer",
    "backtest": "evaluation_layer",
    "execution": "execution_layer",
    "services": "execution_layer",
}

EXTERNAL_PURPOSES: dict[str, str] = {
    "__future__": "Python 2/3 compatibility shim (annotations).",
    "abc": "Abstract base classes (Broker, HMM, providers).",
    "collections": "Container / iteration helpers (deque, namedtuple-like).",
    "contextlib": "Context manager utilities.",
    "contextvars": "Correlation-ID context propagation in logging.",
    "copy": "Deep-copy for best-surrogate checkpointing.",
    "dataclasses": "Immutable-by-update state (RiskConfig, Order, MarketAlpha, ...).",
    "datetime": "Timezone-aware timestamps throughout.",
    "enum": "Domain enums (OperationalMode, OrderSide, ...).",
    "math": "isfinite / log / sqrt scalar primitives.",
    "os": "Environment variable resolution for SecretRef.",
    "pathlib": "Filesystem paths.",
    "typing": "Type hints (Protocol, runtime_checkable).",
    "numpy": "All numerical primitives, vectorized MC, distributions.",
    "scipy": "Cholesky factorization (rBergomi covariance), FFT convolution (hybrid scheme), implied-vol Newton/bisection, hypergeometric 2F1 (Volterra kernel).",
    "scipy.linalg": "Cholesky factorization (CholeskySimulator).",
    "scipy.signal": "fftconvolve for the hybrid-scheme weights.",
    "scipy.special": "logsumexp (HMM forward-backward), hyp2f1, norm.",
    "scipy.stats": "norm.cdf / norm.ppf for Black-Scholes and DSR.",
    "pydantic": "Typed, validated, frozen config models (EngineConfig, ...).",
    "torch": "PPO actor + distributional critic, quantile network, surrogate net, "
    "Transformer backbone, neural rBergomi simulator.",
    "torch.nn": "Neural network modules (MLP trunk, quantile head, LayerNorm).",
    "torch.optim": "Adam optimizer for PPO and the surrogate.",
    "gymnasium": "RL environment base class and space definitions.",
    "hypothesis": "Property-based test data generator (used in tests).",
    "ib_insync": "IBKR broker SDK (live broker adapter, optional).",
    "json": "Run-manifest / config fingerprint serialization.",
    "hashlib": "SHA-256 config fingerprint.",
    "numpy.typing": "Type-annotated numpy arrays (NDArray[np.float64]).",
    "pytest": "Test framework (declared in dev dependencies).",
    "logging": "Structured logging (used in core.logging and other modules).",
    "string": "String utilities.",
    "math": "isfinite / log / sqrt scalar primitives.",
}

# Project's own subpackage names (used to classify imports as internal/external).
PACKAGE_NAMES: frozenset[str] = frozenset({
    "options_engine",
    "core", "models", "pricing", "calibration", "surrogate",
    "regime", "market", "news", "strategy", "rl", "agent",
    "backtest", "execution", "services",
})


# ---------------------------------------------------------------------------
# Per-module criticality ranking
# ---------------------------------------------------------------------------
#
# Each module is scored on TWO dimensions:
#
# * simulation_accuracy (1-5): how much does this module affect the numerical
#   correctness of the simulation (price distributions, Greeks, MC estimates)?
#     5 = closed-form math, exact distribution correctness (rBergomi kernel, BS,
#         Volterra covariance, payoff identities)
#     4 = heavy numerical simulation primitives (HybridSimulator, MC, Cholesky)
#     3 = aggregate / distribution / payoff / market-microstructure
#     2 = chain construction, regime features
#     1 = infrastructure (config, logging, errors, validation helpers)
#
# * training_accuracy (1-5): how much does this module affect the training loop
#   (gradient quality, convergence, anti-collapse, outer-loop learning)?
#     5 = loss functions, PPO core, gradient computation (losses, ppo,
#         ppo_transformer, networks, reward, env)
#     4 = reward shaping, GAE, advantage computation, value targets, fast surrogate
#     3 = buffer / storage, network architecture
#     2 = training loop plumbing, monitoring, anti-collapse
#     1 = infrastructure
#
# Plus:
# * regression_sensitivity: catastrophic / high / medium / low -- what
#   happens if this module has a bug.
# * fallback_strategy: how to mitigate a regression
#     mc_fallback       - fall back to direct Monte Carlo ground truth
#     theory_check      - verify against analytical limit
#     circuit_breaker   - kill switch, stop the engine
#     manual_review     - requires operator / human review
#     none              - nothing critical to fail
# * rationale: why this scoring
# * role: short tag categorising the module

CRITICALITY: dict[str, dict[str, object]] = {
    # ------------------------------- core -------------------------------
    "core.config": {
        "simulation_accuracy": 3,
        "training_accuracy": 2,
        "role": "typed_pydantic_config",
        "rationale": "EngineConfig/RiskConfig/MonteCarloConfig seed every simulation parameter. "
        "Wrong validation here lets out-of-domain parameters reach the rBergomi simulator and "
        "silently corrupt the price distribution.",
        "regression_sensitivity": "high",
        "fallback_strategy": "manual_review",
    },
    "core.enums": {
        "simulation_accuracy": 3,
        "training_accuracy": 2,
        "role": "domain_enums",
        "rationale": "StrategicAction / OrderSide / OperationalMode drive every decision branch in "
        "the env, OMS, and IBKR adapter. Wrong enum value causes silent wrong-side orders.",
        "regression_sensitivity": "high",
        "fallback_strategy": "manual_review",
    },
    "core.errors": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "structured_exceptions",
        "rationale": "Fail-loud exceptions are how out-of-domain inputs and MC-convergence failures "
        "are surfaced. Swallowing or mis-classifying errors here turns numerical problems into "
        "silent wrong answers.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "core.logging": {
        "simulation_accuracy": 1,
        "training_accuracy": 1,
        "role": "structured_logging",
        "rationale": "JSON correlation logging. Not in the inner simulation or training path; "
        "observability layer only.",
        "regression_sensitivity": "low",
        "fallback_strategy": "none",
    },
    "core.market_alpha": {
        "simulation_accuracy": 5,
        "training_accuracy": 5,
        "role": "alpha_to_model_mapping",
        "rationale": "CRITICAL: every component of MarketAlpha drives a model knob (Hurst, eta, "
        "AS noise, jumps, shocks). The helper-critic learns alpha to drive diagnostics to 1; "
        "wrong mappings here silently invert the curriculum. Wrong alpha -> wrong simulator "
        "params -> wrong distributions -> wrong agent.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "core.random": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "reproducible_rng",
        "rationale": "RandomFactory with FNV-1a name hashing + SeedSequence-derived sub-streams "
        "guarantees a (seed, name) pair always produces the same stream. A regression here "
        "breaks reproducibility across the entire engine, invalidates backtest tear-sheets, "
        "and makes PPO on-policy updates non-deterministic.",
        "regression_sensitivity": "high",
        "fallback_strategy": "manual_review",
    },
    "core.timegrid": {
        "simulation_accuracy": 4,
        "training_accuracy": 2,
        "role": "time_grid_in_years",
        "rationale": "Day-count conventions and linspace endpoints drive every dt in the "
        "rBergomi simulator and the IBKR expiry conversion. An off-by-one here shifts "
        "every simulated path.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "core.validation": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "scalar_validators",
        "rationale": "Every numerical entry point uses these to fail-fast on bad inputs. "
        "A bug (e.g. wrong range) lets out-of-domain values reach the rBergomi simulator "
        "and produces NaN/Inf paths.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    # ------------------------- models.rbergomi -------------------------
    "models.rbergomi.simulator": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "ground_truth_rbergomi_simulator",
        "rationale": "THE ground-truth price simulator. HybridSimulator's FFT convolution "
        "weights (Bennedsen-Lunde-Pakkanen) and CholeskySimulator's exact covariance are "
        "mathematically verified against analytical limits. Any regression silently "
        "corrupts every downstream price/distribution/training distribution.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "models.rbergomi.params": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "rbergomi_params_container",
        "rationale": "RBergomiParams validation enforces the model's valid domain "
        "(H in (0, 0.5), eta > 0, rho in [-1, 1], xi0 > 0). Bad validation lets "
        "mathematically degenerate parameters reach the simulator.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "models.rbergomi.kernel": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "volterra_kernel_math",
        "rationale": "Closed-form Volterra autocovariance + cross-covariance + hybrid "
        "discrete covariance + hybrid weights. These are the analytical limits the "
        "test suite verifies the simulator against. Wrong kernel math = wrong "
        "simulator AND wrong analytical oracle.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "models.rbergomi.noise": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "standard_normal_draws",
        "rationale": "Antithetic + Sobol QMC drivers for variance reduction. Wrong "
        "antithetic sign flip or wrong Sobol inverse-CDF clip = biased MC estimates.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "models.rbergomi.results": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "result_containers",
        "rationale": "SimulationPaths / TerminalDistribution invariants (finite values, "
        "matching shapes, monotonic arrays) prevent silent corruption propagating "
        "from sim to pricing to training.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "models.rbergomi.diagnostics": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "mc_convergence_control",
        "rationale": "MonteCarloSummary + relative-SE convergence gate (refuses under-converged "
        "distributions). Without this gate, noisy distributions silently reach training "
        "data and bias PPO updates.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "models.rbergomi.neural_simulator": {
        "simulation_accuracy": 4,
        "training_accuracy": 5,
        "role": "fast_neural_rbergomi_surrogate",
        "rationale": "Trained Transformer that replaces the O(N log N) hybrid simulator "
        "in the inner loop. Bad training = wrong fast distributions. Pairs with "
        "SurrogateGuardrail which falls back to MC if Wasserstein error exceeds tolerance.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "models.rbergomi.alpha_calibration": {
        "simulation_accuracy": 5,
        "training_accuracy": 4,
        "role": "alpha_to_rbergomi_params",
        "rationale": "Single source of truth for turning MarketAlpha into a full "
        "RBergomiParams (hurst, eta, rho, forward_variance, rate). Wrong mapping = "
        "wrong market for the helper-critic's curriculum.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    # --------------------------- models ---------------------------
    "models.jumps": {
        "simulation_accuracy": 2,
        "training_accuracy": 2,
        "role": "merton_jump_placeholder",
        "rationale": "Standalone Merton jump utilities; not wired into the main "
        "HybridSimulator path yet. Low impact until integrated.",
        "regression_sensitivity": "low",
        "fallback_strategy": "none",
    },
    "models.drift": {
        "simulation_accuracy": 2,
        "training_accuracy": 2,
        "role": "drift_estimator_placeholder",
        "rationale": "Standalone ridge drift estimator; not wired into the main "
        "HybridSimulator path yet. Low impact until integrated.",
        "regression_sensitivity": "low",
        "fallback_strategy": "none",
    },
    # --------------------------- pricing --------------------------
    "pricing.black_scholes": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "analytical_bs_pricing_oracle",
        "rationale": "Closed-form BS + Greeks + implied vol. This IS the analytical "
        "oracle that the rBergomi MC prices are tested against in the eta->0 limit. "
        "Wrong BS = wrong validation oracle = wrong sign-off on the simulator.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "pricing.instruments": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "instrument_specifications",
        "rationale": "EuropeanOption / IronCondor / BullPutSpread with strict strike "
        "ordering validation. Wrong structure -> wrong payoff -> wrong P&L.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "pricing.payoff": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "payoff_functions",
        "rationale": "Payoff identities (max(S-K, 0), IronCondor tent shape). "
        "Tested against known closed-form limits. Wrong payoff = every reward/P&L "
        "downstream is wrong.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "pricing.monte_carlo": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "mc_pricing_off_rbergomi",
        "rationale": "MC pricing off the TerminalDistribution, with convergence "
        "gate (relative SE). Wrong MC = wrong model-fair credit = wrong edge estimate "
        "= wrong entry decision.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "pricing.portfolio": {
        "simulation_accuracy": 3,
        "training_accuracy": 2,
        "role": "portfolio_greeks_aggregation",
        "rationale": "Aggregates per-leg Greeks with signed contributions. Wrong sign = "
        "wrong net Greek for hedging decisions.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "theory_check",
    },
    # ------------------------- calibration -------------------------
    "calibration.calibrator": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "end_to_end_calibration",
        "rationale": "Orchestrates Hurst/eta/rho/xi0 estimation + jump test. "
        "Bad orchestration = bad parameters for every subsequent simulation.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "calibration.hurst": {
        "simulation_accuracy": 5,
        "training_accuracy": 4,
        "role": "hurst_estimator",
        "rationale": "Log-moment scaling estimator of H. H is the dominant "
        "rough-volatility signature; wrong H = wrong rBergomi dynamics = "
        "wrong agent perception of the market.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "calibration.realized": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "realized_variance_estimators",
        "rationale": "Realized variance + bipower variation (jump-robust). "
        "Input to Hurst and vol-of-vol estimators.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "mc_fallback",
    },
    "calibration.vol_of_vol": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "eta_and_rho_estimators",
        "rationale": "Structure-function matching for eta, leverage-curve inversion "
        "for rho. Wrong eta/rho = wrong vol-of-vol and wrong leverage = wrong paths.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "calibration.forward_variance": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "xi0_estimator",
        "rationale": "Forward-variance curve from realised vol (level) or "
        "ATM IV term structure (curve). Bad xi0 = wrong overall variance scale.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "theory_check",
    },
    "calibration.jumps": {
        "simulation_accuracy": 3,
        "training_accuracy": 2,
        "role": "bns_jump_test",
        "rationale": "BNS relative-jump test decides whether to enable jumps. "
        "Wrong test = wrong Merton regime in the simulator.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "theory_check",
    },
    "calibration.walk_forward": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "walk_forward_discipline",
        "rationale": "Rolling/anchored walk-forward windows. The walk-forward "
        "discipline is what prevents look-ahead bias from contaminating the engine.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "manual_review",
    },
    "calibration.results": {
        "simulation_accuracy": 2,
        "training_accuracy": 2,
        "role": "calibration_result_containers",
        "rationale": "ParameterEstimate / CalibrationResult containers + staleness "
        "checks. Wrong invariants = silently stale parameters.",
        "regression_sensitivity": "low",
        "fallback_strategy": "manual_review",
    },
    # -------------------------- surrogate --------------------------
    "surrogate.surrogate": {
        "simulation_accuracy": 4,
        "training_accuracy": 5,
        "role": "neural_distribution_surrogate",
        "rationale": "Trained quantile network approximating MC terminal distribution. "
        "Pinned to ground truth by the guardrail. Bad training = wrong fast "
        "distributions in the inner loop.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "surrogate.distribution": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "distribution_representation",
        "rationale": "SurrogateDistribution (quantile-based) + interpolation. "
        "Wrong interpolation = non-monotone quantiles = silent wrong CDF.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "mc_fallback",
    },
    "surrogate.features": {
        "simulation_accuracy": 3,
        "training_accuracy": 4,
        "role": "feature_scaling_for_nn",
        "rationale": "FeatureScaler + RawInputs build the network's input vector. "
        "Wrong scale = badly-conditioned NN training = bad convergence.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "manual_review",
    },
    "surrogate.losses": {
        "simulation_accuracy": 3,
        "training_accuracy": 5,
        "role": "pinball_quantile_loss",
        "rationale": "Pinball loss is THE training objective of the surrogate. "
        "Wrong loss = wrong gradient = wrong learned quantiles.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "surrogate.metrics": {
        "simulation_accuracy": 3,
        "training_accuracy": 4,
        "role": "surrogate_calibration_metrics",
        "rationale": "CRPS / Wasserstein / ECE / PIT metrics. The guardrail "
        "consumes Wasserstein1 to decide whether to fall back to MC.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "theory_check",
    },
    "surrogate.guardrail": {
        "simulation_accuracy": 5,
        "training_accuracy": 4,
        "role": "mc_fallback_decision",
        "rationale": "CRITICAL SAFETY: this is the single decision point that "
        "falls back to direct MC when the surrogate error exceeds tolerance. "
        "A regression here lets a wrong surrogate silently reach the engine.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "surrogate.quantile_network": {
        "simulation_accuracy": 3,
        "training_accuracy": 5,
        "role": "monotone_quantile_network_arch",
        "rationale": "Monotone quant network (no-crossing). Wrong architecture "
        "(e.g. non-monotone outputs) = non-CDF quantiles.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "surrogate.dataset": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "surrogate_training_data",
        "rationale": "TrainingData builder + ScenarioRanges + quantile levels. "
        "Wrong scenarios = unrepresentative training data.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "mc_fallback",
    },
    # --------------------------- regime -----------------------------
    "regime.hmm": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "gaussian_hmm",
        "rationale": "Log-space EM/Viterbi with multi-restart. Wrong EM = "
        "wrong regime detection = wrong gate decisions = wrong trading.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "regime.detector": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "regime_detector_orchestrator",
        "rationale": "RegimeDetector wraps HMM + temperature calibration. "
        "Wrong nowcast = wrong gate.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "regime.gate": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "trade_gate_low_vol_only",
        "rationale": "CRITICAL SAFETY: trade gate that only opens new risk "
        "when P(low-vol now AND next) both clear thresholds + high-vol is "
        "below the caps. Wrong gate = trades in dangerous regimes.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "regime.metrics": {
        "simulation_accuracy": 2,
        "training_accuracy": 3,
        "role": "regime_calibration_metrics",
        "rationale": "Brier / log-loss / ECE / reliability-curve. Diagnostics "
        "for whether the gate's probabilities can be trusted.",
        "regression_sensitivity": "low",
        "fallback_strategy": "manual_review",
    },
    "regime.calibration": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "temperature_scaling",
        "rationale": "Temperature scaling for calibrated probabilities. "
        "Wrong T = wrong gate thresholds.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "theory_check",
    },
    "regime.features": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "leakage_free_regime_features",
        "rationale": "Point-in-time feature builder (log_rv, rv_change, "
        "abs_return, downside_ratio). Wrong features = wrong regime posterior.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "manual_review",
    },
    # --------------------------- market --------------------------
    "market.market_maker": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "as_market_maker",
        "rationale": "AS quotes + wing decay + obligation bounds + (optional) "
        "alpha-driven noise. Wrong spread = wrong fill cost = wrong P&L.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "mc_fallback",
    },
    "market.alpha_noise": {
        "simulation_accuracy": 3,
        "training_accuracy": 4,
        "role": "alpha_driven_quote_noise",
        "rationale": "Bounded asymmetric AS-noise injection driven by "
        "MarketAlpha[1]. The helper-critic tunes alpha to drive diagnostics "
        "to 1; wrong noise mapping inverts the curriculum.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "theory_check",
    },
    "market.execution_sim": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "fill_simulator",
        "rationale": "Slippage + book walk + impact + partial fills. "
        "The agent learns from realised slippage; wrong fills = wrong gradient.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "mc_fallback",
    },
    "market.quotes": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "quote_data_structures",
        "rationale": "Quote / QuotedOption invariants (no crossed market, "
        "positive sizes). Wrong invariants = silent bad quotes.",
        "regression_sensitivity": "low",
        "fallback_strategy": "circuit_breaker",
    },
    "market.chain": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "synthetic_options_chain_builder",
        "rationale": "No-arbitrage repair on the call curve + put-call parity. "
        "An arbitrable chain lets the strategy 'win' on simulator artefacts.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    # --------------------------- news -----------------------------
    "news.gate": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "news_event_trade_gate",
        "rationale": "CRITICAL SAFETY: scheduled-event blackout + breaking-news "
        "trading-day cool-off. Wrong gate = trades through FOMC / CPI / "
        "material-news windows.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "news.classifier": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "keyword_news_classifier",
        "rationale": "Lexicon-based relevance + severity classification. "
        "Misses a CRITICAL keyword = misses a material event = wrong gate decision.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "manual_review",
    },
    "news.events": {
        "simulation_accuracy": 3,
        "training_accuracy": 2,
        "role": "news_event_data_structures",
        "rationale": "NewsItem / ScheduledEvent + EventSeverity enum. "
        "Wrong timezone-naive datetime = wrong blackout window.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "circuit_breaker",
    },
    "news.providers": {
        "simulation_accuracy": 2,
        "training_accuracy": 2,
        "role": "news_provider_interfaces",
        "rationale": "Replay / REST provider abstractions. Wrong interface = "
        "missing items in the gate window.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "manual_review",
    },
    "news.calendar": {
        "simulation_accuracy": 3,
        "training_accuracy": 2,
        "role": "calendar_blackout_windowing",
        "rationale": "Scheduled-event blackout windowing (lead time + cool-off). "
        "Wrong window = gate fires at wrong times.",
        "regression_sensitivity": "medium",
        "fallback_strategy": "manual_review",
    },
    # -------------------------- strategy --------------------------
    "strategy.condor_selection": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "condor_strike_selection",
        "rationale": "Strike selection from the terminal distribution (joint "
        "objective on win_prob, CVaR, spread cost). Wrong selection = wrong "
        "structure = wrong payoff distribution.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "strategy.sizing": {
        "simulation_accuracy": 5,
        "training_accuracy": 4,
        "role": "fractional_kelly_sizing",
        "rationale": "Fractional Kelly (binary + empirical). Wrong sizing = "
        "wrong dollar risk per trade = wrong drawdown behaviour. Growth-optimal "
        "sizing drives long-run wealth; wrong sizing loses money in expectation.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "strategy.entry": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "multi_filter_entry_logic",
        "rationale": "Six independent entry filters (regime, news, distribution, "
        "spread-cost, sizing, supervisor). A broken filter = trades that should "
        "be rejected get through.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "strategy.exit": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "multi_reason_exit_logic",
        "rationale": "Stop-loss / profit target / regime breach / time stop. "
        "Wrong exit = positions held through blowups.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "strategy.risk_supervisor": {
        "simulation_accuracy": 5,
        "training_accuracy": 4,
        "role": "hard_risk_supervisor_with_kill_switch",
        "rationale": "CRITICAL SAFETY: deterministic pre-trade risk checks + "
        "trailing-drawdown kill-switch. The agent is *never* the last line of "
        "defense; this is. Wrong check = real-money loss exceeds mandate.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "strategy.account": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "accounting_state",
        "rationale": "Account / OpenPosition bookkeeping (cash, P&L, high-water "
        "mark, drawdown). Wrong book-keeping = wrong equity, wrong drawdown = "
        "wrong kill-switch trigger.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    # ---------------------------- rl -----------------------------
    "rl.env": {
        "simulation_accuracy": 5,
        "training_accuracy": 5,
        "role": "gymnasium_pomdp_env",
        "rationale": "THE env. Every step: alpha-aware scenario -> observation -> "
        "action -> reward -> next state. Wrong env = wrong transitions = wrong "
        "training data = wrong agent.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "rl.action": {
        "simulation_accuracy": 4,
        "training_accuracy": 5,
        "role": "parameterized_action_decoder",
        "rationale": "Decodes raw 6-element policy vector into a bounded "
        "DecodedAction. Wrong decode = wrong strategy selection + wrong sizing.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "rl.observation": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "leakage_free_observation_builder",
        "rationale": "15-d leakage-free observation (distribution edge + "
        "regime + portfolio + calendar). Wrong features = wrong state = "
        "wrong policy.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "rl.reward": {
        "simulation_accuracy": 5,
        "training_accuracy": 5,
        "role": "risk_sensitive_reward",
        "rationale": "THE reward function: P&L - risk - cost - margin - tail + "
        "shaping. Wrong reward = wrong gradient = wrong policy (the classic "
        "do-nothing collapse is a reward-shaping defect).",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "rl.growth_reward": {
        "simulation_accuracy": 5,
        "training_accuracy": 5,
        "role": "growth_optimal_alternative_reward",
        "rationale": "Kelly-log-wealth core with opportunity-cost + edge-shaping. "
        "Designed to provably prevent the do-nothing collapse. Wrong = collapse "
        "returns (the original reward's failure mode).",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "rl.scenario": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "episode_generator",
        "rationale": "Domain-randomized episodes (alpha-aware) with Hurst/eta "
        "sampling + chain construction. Bad scenarios = unrepresentative "
        "training data = bad generalisation.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    # --------------------------- agent ----------------------------
    "agent.ppo": {
        "simulation_accuracy": 3,
        "training_accuracy": 5,
        "role": "ppo_mlp_core",
        "rationale": "THE PPO algorithm (MLP variant). GAE + clip surrogate + "
        "distributional critic. Wrong update = wrong policy.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "agent.ppo_transformer": {
        "simulation_accuracy": 3,
        "training_accuracy": 5,
        "role": "ppo_transformer_core_with_memory",
        "rationale": "THE PPO-Transformer: shared Transformer backbone for "
        "actor + critic, sequence-aware PPO. Can exploit rBergomi's non-Markovian "
        "Volterra structure (the MLP cannot). Wrong = same as agent.ppo + "
        "memory broken.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "agent.transformer_backbone": {
        "simulation_accuracy": 2,
        "training_accuracy": 5,
        "role": "causal_transformer_encoder",
        "rationale": "Causal multi-head self-attention + sinusoidal PE + "
        "norm-first pre-LN. Wrong attention mask = look-ahead = silent future-"
        "leakage in training.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "agent.gae": {
        "simulation_accuracy": 4,
        "training_accuracy": 5,
        "role": "gae_advantage_computation",
        "rationale": "GAE backward recursion with (1-done) masking + bootstrap. "
        "Wrong GAE = wrong variance-reduced advantages = wrong gradient.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "agent.losses": {
        "simulation_accuracy": 3,
        "training_accuracy": 5,
        "role": "ppo_clip_and_quantile_huber_losses",
        "rationale": "PPO clipped surrogate + quantile-Huber (Dabney 2018) for "
        "the distributional critic. Wrong loss = wrong gradient direction.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    "agent.networks": {
        "simulation_accuracy": 3,
        "training_accuracy": 5,
        "role": "actor_and_critic_networks",
        "rationale": "GaussianActor + DistributionalCritic (quantile head + CVaR). "
        "Wrong architecture = wrong log_std parameterisation = bad exploration.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "agent.rollout": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "on_policy_rollout_buffer",
        "rationale": "Pre-allocated transition store + advantage computation. "
        "Wrong buffer invariants = corrupted training data.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "agent.trainer": {
        "simulation_accuracy": 4,
        "training_accuracy": 5,
        "role": "ppo_training_loop",
        "rationale": "Training loop orchestration + shaping decay + anti-collapse "
        "monitors. Wrong loop = wrong convergence / wrong collapse detection.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "agent.helper_critic": {
        "simulation_accuracy": 5,
        "training_accuracy": 5,
        "role": "alpha_meta_learner",
        "rationale": "CRITICAL: outer-loop meta-controller that learns MarketAlpha. "
        "Drives every internal diagnostic toward 1. Wrong helper-critic = "
        "wrong alpha = wrong curriculum = wrong market for the inner agent.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "theory_check",
    },
    # -------------------------- backtest --------------------------
    "backtest.engine": {
        "simulation_accuracy": 4,
        "training_accuracy": 4,
        "role": "backtest_engine",
        "rationale": "Runs the env for a policy for n_steps; produces BacktestResult. "
        "Wrong loop bookkeeping = wrong equity curve.",
        "regression_sensitivity": "high",
        "fallback_strategy": "manual_review",
    },
    "backtest.metrics": {
        "simulation_accuracy": 5,
        "training_accuracy": 4,
        "role": "performance_and_risk_metrics",
        "rationale": "Sharpe / Sortino / DSR / max DD / CVaR. DSR is the multiple-"
        "testing-corrected Sharpe that the promotion gate uses. Wrong DSR = wrong "
        "sign-off on whether a strategy is real or a fluke of many trials.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "backtest.promotion": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "history_to_live_promotion_gates",
        "rationale": "CRITICAL SAFETY: DSR / max DD / CVaR / kill-switch / walk-"
        "forward / stress gates. A regression lets a bad strategy advance to "
        "PAPER or LIVE trading.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "backtest.validation": {
        "simulation_accuracy": 3,
        "training_accuracy": 3,
        "role": "purged_walk_forward_cv",
        "rationale": "Purged/embargoed walk-forward splits. The purge/embargo "
        "is what prevents look-ahead bias from contaminating CV scores.",
        "regression_sensitivity": "high",
        "fallback_strategy": "theory_check",
    },
    "backtest.results": {
        "simulation_accuracy": 2,
        "training_accuracy": 2,
        "role": "backtest_result_containers",
        "rationale": "StepRecord / TradeRecord / BacktestResult + invariants. "
        "Wrong invariants = silent bad audit trail.",
        "regression_sensitivity": "low",
        "fallback_strategy": "manual_review",
    },
    # ------------------------- execution --------------------------
    "execution.broker": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "broker_interface",
        "rationale": "Broker ABC + SimulatedBroker. Wrong partial-fill logic = "
        "wrong cash impact = wrong backtest.",
        "regression_sensitivity": "high",
        "fallback_strategy": "mc_fallback",
    },
    "execution.oms": {
        "simulation_accuracy": 5,
        "training_accuracy": 3,
        "role": "idempotent_risk_gated_oms",
        "rationale": "CRITICAL MONEY: idempotent OMS (client_order_id), risk "
        "gate on every opening order, reconciliation, kill-switch. Wrong OMS "
        "creates duplicate orders or accepts veto-violating orders.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "execution.orders": {
        "simulation_accuracy": 4,
        "training_accuracy": 3,
        "role": "iron_condor_order_ticket",
        "rationale": "Order / Fill / OrderState with strict validation "
        "(timezone-aware, non-empty ids, positive quantities). Wrong invariants = "
        "silent bad orders.",
        "regression_sensitivity": "high",
        "fallback_strategy": "circuit_breaker",
    },
    "execution.live_guard": {
        "simulation_accuracy": 5,
        "training_accuracy": 2,
        "role": "typed_phrase_live_trading_arming",
        "rationale": "CRITICAL MONEY: ClassVar REQUIRED_PHRASE = "
        "'I UNDERSTAND THIS TRADES REAL MONEY'. Wrong guard = real order "
        "routed by default = money loss.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    "execution.ibkr": {
        "simulation_accuracy": 5,
        "training_accuracy": 2,
        "role": "ibkr_live_adapter",
        "rationale": "CRITICAL MONEY: IBKR live broker adapter with the 5-lock "
        "fail-closed chain. The placement seam raises NotImplementedError until "
        "the operator wires it. Wrong lock order = a single typo could route a "
        "real order.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
    # -------------------------- services --------------------------
    "services.runner": {
        "simulation_accuracy": 5,
        "training_accuracy": 2,
        "role": "broker_factory",
        "rationale": "CRITICAL MONEY: single fail-closed broker factory. "
        "Wrong mode -> real order routed when operator thinks it's paper = "
        "money loss. Wrong refusal logic = paper mode uses real broker.",
        "regression_sensitivity": "catastrophic",
        "fallback_strategy": "circuit_breaker",
    },
}


# ---------------------------------------------------------------------------
# AST walking helpers
# ---------------------------------------------------------------------------

def module_path_for(path: Path, root: Path) -> str:
    """Map a source file's path to its dotted module path (bare, no ``options_engine`` prefix)."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return ""
    return ast.get_docstring(tree) or ""


def docstring_summary(doc: str) -> str:
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def docstring_extract_raises(doc: str) -> list[str]:
    """Extract ``Raises`` section entries from a docstring.

    Recognises several common formats:

    * a ``"Raises"`` / ``"Raises:"`` section header followed by indented lines,
    * Google-style ``"Raises:"`` followed by indented bullets,
    * Numpy-style ``"Raises"`` underline followed by indented lines,
    * Sphinx-style ``":raises ExceptionName:"`` field lists.

    Returns the bullet / line text, de-duplicated.
    """
    if not doc:
        return []
    out: list[str] = []
    in_raises = False
    next_section_markers = (
        "returns",
        "parameters",
        "notes",
        "see also",
        "examples",
        "references",
        "yields",
        "side effects",
        "validation",
    )
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower().rstrip(":")
        # Sphinx-style field: ``:raises ExceptionName:`` -- inline.
        if low.startswith(":raises"):
            content = stripped.split(":", 2)[-1].strip()
            if content:
                out.append(content)
            continue
        # Section header: ``Raises`` / ``Raises:`` followed by other content
        # on the same line counts as a section start.
        if low == "raises" or low.startswith("raises "):
            in_raises = True
            # Pull any inline text after the header keyword.
            tail = stripped.split(" ", 1)[1].strip() if " " in stripped else ""
            tail = tail.rstrip(":").strip()
            if tail:
                out.append(tail)
            continue
        # Numpy-style underline (``----``) following a ``Raises`` line.
        if in_raises:
            if any(low == m for m in next_section_markers):
                in_raises = False
                continue
            # Drop bullet markers and the trailing ``:`` from lines like
            # ``ValidationError: blah``.
            clean = stripped.lstrip("-* ").rstrip(":").strip()
            if clean:
                out.append(clean)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _format_param_default(d: ast.AST) -> str | None:
    """Return the default value of a function argument as a code-like string."""
    try:
        return ast.unparse(d)
    except Exception:  # pragma: no cover - defensive
        return None


def _format_annotation(a: ast.AST | None) -> str | None:
    if a is None:
        return None
    try:
        return ast.unparse(a)
    except Exception:  # pragma: no cover - defensive
        return None


def _collect_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    out: list[str] = []
    for dec in node.decorator_list:
        try:
            out.append(ast.unparse(dec))
        except Exception:  # pragma: no cover - defensive
            pass
    return out


def _extract_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, object]:
    """Extract a structured signature from a function/method node."""
    args = node.args
    pos = [
        {"name": a.arg, "annotation": _format_annotation(a.annotation)}
        for a in args.posonlyargs + args.args
    ]
    vararg = (
        {"name": args.vararg.arg, "annotation": _format_annotation(args.vararg.annotation)}
        if args.vararg
        else None
    )
    kwonly: list[dict[str, object]] = []
    # Python AST stores kwonly defaults in ``arguments.kw_defaults`` (parallel list
    # of length len(kwonlyargs) -- entries may be None where no default was given).
    kw_defaults: list[ast.AST | None] = list(args.kw_defaults)  # type: ignore[attr-defined]
    while len(kw_defaults) < len(args.kwonlyargs):
        kw_defaults.append(None)
    for a, default in zip(args.kwonlyargs, kw_defaults):
        kwonly.append(
            {
                "name": a.arg,
                "annotation": _format_annotation(a.annotation),
                "default": _format_param_default(default) if default is not None else None,
            }
        )
    kwarg = (
        {"name": args.kwarg.arg, "annotation": _format_annotation(args.kwarg.annotation)}
        if args.kwarg
        else None
    )
    defaults_offset = len(args.args) - len(args.defaults)
    for i, default in enumerate(args.defaults):
        if i + defaults_offset < len(pos):
            pos[i + defaults_offset]["default"] = _format_param_default(default)
    return {
        "positional": pos,
        "vararg": vararg,
        "keyword_only": kwonly,
        "kwarg": kwarg,
        "return_annotation": _format_annotation(node.returns),
        "decorators": _collect_decorators(node),
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "is_classmethod": any(d.startswith("classmethod") for d in _collect_decorators(node)),
        "is_staticmethod": any(d.startswith("staticmethod") for d in _collect_decorators(node)),
        "is_property": any(d.startswith("property") for d in _collect_decorators(node)),
    }


def _extract_class(node: ast.ClassDef) -> dict[str, object]:
    """Extract a structured class record: bases, fields, methods, properties."""
    bases = [_format_annotation(b) for b in node.bases]
    decorators = _collect_decorators(node)
    raw_doc = ast.get_docstring(node) or ""
    fields: list[dict[str, object]] = []
    methods: list[dict[str, object]] = []
    properties: list[dict[str, object]] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _extract_function_signature(item)
            entry = {
                "name": item.name,
                "kind": "method",
                "summary": docstring_summary(ast.get_docstring(item) or ""),
                "signature": sig,
                "raises": docstring_extract_raises(ast.get_docstring(item) or ""),
                "is_property": any(d.startswith("property") for d in sig["decorators"]),
            }
            if entry["is_property"]:
                properties.append(entry)
            else:
                methods.append(entry)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            # ``x: int = 5`` style field declaration.
            fields.append(
                {
                    "name": item.target.id,
                    "annotation": _format_annotation(item.annotation),
                    "default": _format_param_default(item.value) if item.value is not None else None,
                }
            )
        elif isinstance(item, ast.Assign):
            # ``x = 5`` style module-level assignment inside a class (rare).
            for target in item.targets:
                if isinstance(target, ast.Name):
                    fields.append(
                        {
                            "name": target.id,
                            "annotation": None,
                            "default": _format_param_default(item.value),
                        }
                    )
    return {
        "name": node.name,
        "kind": "class",
        "summary": docstring_summary(raw_doc),
        "bases": bases,
        "decorators": decorators,
        "fields": fields,
        "methods": methods,
        "properties": properties,
        "raises": docstring_extract_raises(raw_doc),
    }


def _extract_module_constants(tree: ast.Module) -> list[dict[str, object]]:
    """Module-level constants (UPPER_CASE assignments)."""
    out: list[dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    out.append(
                        {
                            "name": target.id,
                            "value": _format_param_default(node.value),
                        }
                    )
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id.isupper()
                and node.value is not None
            ):
                out.append(
                    {
                        "name": node.target.id,
                        "value": _format_param_default(node.value),
                    }
                )
    return out


def extract_public_api(path: Path) -> list[dict[str, object]]:
    """Walk the module AST and return the public surface as a structured list."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    public: list[dict[str, object]] = []
    public_names: set[str] = set()
    # Honour ``__all__`` if present.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        public_names = {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
                        break
    if not public_names:
        # Fall back to anything that doesn't start with an underscore.
        for node in tree.body:
            name = getattr(node, "name", None)
            if isinstance(name, str) and not name.startswith("_"):
                public_names.add(name)
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(name, str) and name in public_names:
            doc = ast.get_docstring(node) or ""
            if isinstance(node, ast.ClassDef):
                public.append(_extract_class(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = _extract_function_signature(node)
                public.append(
                    {
                        "name": node.name,
                        "kind": "function",
                        "summary": docstring_summary(doc),
                        "signature": sig,
                        "raises": docstring_extract_raises(doc),
                    }
                )
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    public.append(
                        {
                            "name": node.target.id,
                            "kind": "constant",
                            "annotation": _format_annotation(node.annotation),
                            "default": _format_param_default(node.value) if node.value is not None else None,
                        }
                    )
    return public


# ---------------------------------------------------------------------------
# Import extraction (internal vs external + per-symbol)
# ---------------------------------------------------------------------------

def _classify(modname: str) -> tuple[str, bool]:
    """Return ``(canonical, is_internal)`` for a dotted module name.

    ``canonical`` is the form stored in the manifest. For internal modules it's
    the bare form (e.g. ``core.errors``); for external modules it's just the
    top-level package name (e.g. ``numpy``).
    """
    top = modname.split(".")[0]
    if top in PACKAGE_NAMES:
        return modname, True
    return top, False


def extract_imports(
    path: Path, file_to_module: dict[Path, str], package_root: Path
) -> tuple[
    dict[str, set[str]],  # internal: canonical -> {symbols}
    dict[str, set[str]],  # external: top-level -> {symbols}
]:
    """Return ``(internal, external)`` as maps of module -> {imported symbols}."""
    internal: dict[str, set[str]] = defaultdict(set)
    external: dict[str, set[str]] = defaultdict(set)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}, {}

    current_module = file_to_module[path]
    rel = path.relative_to(package_root)
    is_package_init = path.name == "__init__.py"
    pkg_parts = list(rel.parent.parts)
    current_package = ".".join(pkg_parts)

    def resolve_relative(level: int, module: str | None) -> str | None:
        if level == 0:
            return module
        parts = current_package.split(".") if current_package else []
        if level - 1 > len(parts):
            return None
        base_parts = parts[: len(parts) - (level - 1)]
        if module:
            base_parts = base_parts + module.split(".")
        return ".".join(base_parts) or None

    def add(module: str | None, names: list[str], is_relative: bool, level: int) -> None:
        if module is None and names:
            # `import X` form.
            for name in names:
                key, is_internal = _classify(name)
                (internal if is_internal else external)[key].add(name)
            return
        if module is None:
            return
        key, is_internal = _classify(module)
        target = internal if is_internal else external
        # Track the module itself (so we know the dependency exists) plus every
        # imported symbol (so downstream callers can see what flows through).
        target[key].add(module)
        for name in names:
            target[key].add(f"{module}.{name}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(None, [alias.name], False, 0)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and not node.level:
                continue
            resolved = resolve_relative(node.level, node.module)
            if resolved is None:
                continue
            names = [alias.name for alias in node.names]
            add(resolved, names, True, node.level)
    return dict(internal), dict(external)


# ---------------------------------------------------------------------------
# Package / graph aggregation
# ---------------------------------------------------------------------------

def package_of(module: str) -> str:
    parts = module.split(".")
    if parts and parts[0] == "options_engine":
        parts = parts[1:]
    if not parts:
        return "<root>"
    sub = parts[0]
    if sub == "models" and len(parts) > 1 and parts[1] == "rbergomi":
        return "models.rbergomi"
    return sub


def find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one back-edge ``[u, ..., v]`` in ``graph`` if a cycle exists, else ``None``."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in graph}
    parent: dict[str, str] = {}
    stack: list[tuple[str, list[str]]] = [(n, list(graph[n])) for n in graph]
    while stack:
        node, children = stack[-1]
        if color.get(node) == WHITE:
            color[node] = GRAY
        if not children:
            color[node] = BLACK
            stack.pop()
            continue
        nxt = children.pop()
        if color.get(nxt) == GRAY:
            cycle = [nxt, node]
            cur = node
            while cur in parent and parent[cur] != nxt:
                cur = parent[cur]
                cycle.append(cur)
            return list(reversed(cycle))
        if color.get(nxt) == WHITE:
            parent[nxt] = node
            stack.append((nxt, list(graph[nxt])))
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _criticality_totals(modules: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate criticality counts and the top-criticality modules."""
    sim_hist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    train_hist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    sensitivity = {"low": 0, "medium": 0, "high": 0, "catastrophic": 0}
    fallback = {"none": 0, "manual_review": 0, "mc_fallback": 0, "theory_check": 0, "circuit_breaker": 0}
    catastrophic: list[dict[str, object]] = []
    high_sim: list[dict[str, object]] = []
    high_train: list[dict[str, object]] = []
    for m in modules:
        c = m["criticality"]  # type: ignore[assignment]
        sim_hist[c["simulation_accuracy"]] += 1  # type: ignore[index]
        train_hist[c["training_accuracy"]] += 1  # type: ignore[index]
        sensitivity[c["regression_sensitivity"]] += 1  # type: ignore[index]
        fallback[c["fallback_strategy"]] += 1  # type: ignore[index]
        if c["regression_sensitivity"] == "catastrophic":  # type: ignore[comparison]
            catastrophic.append(
                {"module": m["module"], "package": m["package"], "role": c["role"]}  # type: ignore[index]
            )
        if c["simulation_accuracy"] >= 5:  # type: ignore[comparison]
            high_sim.append(m["module"])  # type: ignore[arg-type]
        if c["training_accuracy"] >= 5:  # type: ignore[comparison]
            high_train.append(m["module"])  # type: ignore[arg-type]
    catastrophic.sort(key=lambda d: d["module"])  # type: ignore[arg-type]
    high_sim.sort()
    high_train.sort()
    return {
        "simulation_accuracy_histogram": sim_hist,
        "training_accuracy_histogram": train_hist,
        "regression_sensitivity_histogram": sensitivity,
        "fallback_strategy_histogram": fallback,
        "modules_with_catastrophic_regression_sensitivity": catastrophic,
        "modules_with_max_simulation_accuracy_5": high_sim,
        "modules_with_max_training_accuracy_5": high_train,
    }


def main(repo_root: Path) -> None:
    src_dir = repo_root / "src" / "options_engine"
    files = sorted(p for p in src_dir.rglob("*.py") if "__pycache__" not in p.parts)
    file_to_module: dict[Path, str] = {}
    for path in files:
        file_to_module[path] = module_path_for(path, src_dir)

    modules: list[dict[str, object]] = []
    internal_edge_symbols: dict[tuple[str, str], set[str]] = defaultdict(set)
    external_edge_symbols: dict[str, set[str]] = defaultdict(set)
    external_top_modules: dict[str, set[str]] = defaultdict(set)

    bare_modules = set(file_to_module.values())
    canonical_to_bare = {f"options_engine.{b}": b for b in bare_modules}

    for path in files:
        mod = file_to_module[path]
        internal, external = extract_imports(path, file_to_module, src_dir.parent)
        doc_full = module_docstring(path)
        api = extract_public_api(path)
        summary = docstring_summary(doc_full)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            constants = _extract_module_constants(tree)
        except SyntaxError:
            constants = []
        # Convert internal-symbol maps to a list of {module -> symbols} edges.
        # Translate canonical forms back to bare for compactness in the output.
        internal_edges_for_module: list[dict[str, object]] = []
        for canonical_target, symbols in internal.items():
            bare_target = canonical_to_bare.get(canonical_target, canonical_target)
            if bare_target == mod:
                continue
            sorted_symbols = sorted(symbols)
            internal_edge_symbols[(mod, bare_target)].update(symbols)
            internal_edges_for_module.append(
                {
                    "module": bare_target,
                    "symbols": sorted_symbols,
                }
            )
        # External.
        external_entries: list[dict[str, object]] = []
        for top, symbols in external.items():
            sorted_symbols = sorted(symbols)
            external_top_modules[top].update(symbols)
            external_edge_symbols[top].update(symbols)
            external_entries.append(
                {
                    "module": top,
                    "purpose": EXTERNAL_PURPOSES.get(top, ""),
                    "imported_symbols": sorted_symbols,
                }
            )
        modules.append(
            {
                "module": mod,
                "package": package_of(mod),
                "layer": PACKAGE_LAYERS.get(package_of(mod), "unknown"),
                "path": str(path.relative_to(repo_root)),
                "docstring_summary": summary,
                "docstring_full": doc_full,
                "loc": path.read_text(encoding="utf-8").count("\n") + 1,
                "constants": constants,
                "public_api": api,
                "validation_rules": _extract_validation_rules(doc_full),
                "internal_dependencies": sorted(
                    internal_edges_for_module,
                    key=lambda d: d["module"],
                ),
                "external_dependencies": sorted(
                    external_entries,
                    key=lambda d: d["module"],
                ),
                "criticality": _criticality_record(mod),
            }
        )

    # Package-level summary.
    package_modules: dict[str, set[str]] = defaultdict(set)
    for m in modules:
        package_modules[m["package"]].add(m["module"])  # type: ignore[arg-type]
    packages_out: list[dict[str, object]] = []
    for pkg, mods in sorted(package_modules.items()):
        packages_out.append(
            {
                "package": pkg,
                "layer": PACKAGE_LAYERS.get(pkg, "unknown"),
                "responsibility": PACKAGE_DESCRIPTIONS.get(pkg, ""),
                "module_count": len(mods),
                "modules": sorted(mods),
            }
        )

    # Package edges (with the number of underlying module-level edges).
    pkg_pairs: dict[tuple[str, str], int] = defaultdict(int)
    pkg_fan_out: dict[str, set[str]] = defaultdict(set)
    pkg_fan_in: dict[str, set[str]] = defaultdict(set)
    for a, b in internal_edge_symbols:
        pa = package_of(a)
        pb = package_of(b)
        if pa == pb or pa == "<root>" or pb == "<root>":
            continue
        pkg_pairs[(pa, pb)] += 1
        pkg_fan_out[pa].add(pb)
        pkg_fan_in[pb].add(pa)
    package_edges: list[dict[str, object]] = []
    for (a, b), c in sorted(pkg_pairs.items()):
        package_edges.append({"from": a, "to": b, "module_edges": c})

    # Internal edges with the full symbol list.
    internal_edges_full = [
        {
            "from": a,
            "to": b,
            "symbols": sorted(symbols),
        }
        for (a, b), symbols in sorted(internal_edge_symbols.items())
    ]

    # External edges with the full symbol list.
    external_edges_full = [
        {
            "module": top,
            "purpose": EXTERNAL_PURPOSES.get(top, ""),
            "imported_symbols": sorted(symbols),
        }
        for top, symbols in sorted(external_top_modules.items())
    ]

    graph_pkg = {p: set(pkg_fan_out[p]) for p in pkg_fan_out}
    cycle = find_cycle(graph_pkg)

    out = {
        "schema_version": "2.0",
        "generated_by": "extract_deps.py",
        "project": {
            "name": "options-engine",
            "version": "0.1.0",
            "language": "Python",
            "requires_python": ">=3.11",
            "license": "Proprietary",
            "description": (
                "Institutional-grade OTM iron-condor options-selling engine "
                "(rough-volatility rBergomi + transformer-PPO with memory + "
                "alpha-driven curriculum + neural-rBergomi surrogate + "
                "fail-closed execution)."
            ),
            "design_principles": [
                "Honest re-specification of impossible requirements (SPEC §0).",
                "Two-loop learning: helper-critic tunes alpha, PPO-Transformer learns under it.",
                "Causal Transformer backbone gives memory over past observations, "
                "exploiting rBergomi's non-Markovian Volterra structure.",
                "Alpha-driven microstructure noise injected into Avellaneda-Stoikov quotes.",
                "Neural rBergomi surrogate trained on TRUE paths for fast inference.",
                "Five independent locks before any real order can reach a broker.",
                "Layered, acyclic dependency graph (verified by automated DFS).",
            ],
        },
        "totals": {
            "source_modules": len(modules),
            "internal_edges": len(internal_edges_full),
            "packages": len(packages_out),
            "package_edges": len(package_edges),
            "external_dependencies": len(external_edges_full),
            "package_level_acyclic": cycle is None,
            "package_level_cycle_detected": cycle,
            "criticality": _criticality_totals(modules),
        },
        "package_dependency_layers": {
            "foundation": ["core"],
            "quantitative_core": ["models", "models.rbergomi", "pricing"],
            "calibration_layer": ["calibration"],
            "inference_layers": ["market", "news", "regime", "surrogate"],
            "decision_layer": ["strategy"],
            "learning_layer": ["agent", "rl"],
            "evaluation_layer": ["backtest"],
            "execution_layer": ["execution", "services"],
        },
        "packages": packages_out,
        "package_edges": package_edges,
        "package_fan_out": {p: sorted(v) for p, v in pkg_fan_out.items()},
        "package_fan_in": {p: sorted(v) for p, v in pkg_fan_in.items()},
        "external_dependencies": external_edges_full,
        "modules": modules,
        "internal_edges": internal_edges_full,
    }
    out_path = Path(__file__).parent / "modules.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
    print(
        f"Wrote {out_path}: {len(modules)} modules, "
        f"{len(internal_edges_full)} internal edges ({sum(len(e['symbols']) for e in internal_edges_full)} symbol imports), "
        f"{len(packages_out)} packages, {len(external_edges_full)} external deps."
    )


def _extract_validation_rules(doc: str) -> list[str]:
    """Extract ``Validation`` / ``Rules`` / invariant descriptions from a docstring."""
    if not doc:
        return []
    out: list[str] = []
    in_section = False
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if (
            low.startswith("validation")
            or low.startswith("rules")
            or low.startswith("invariants")
            or low.startswith("preconditions")
        ) and (stripped.endswith(":") or stripped.endswith("----")):
            in_section = True
            continue
        if in_section:
            if (
                low.startswith("parameters")
                or low.startswith("returns")
                or low.startswith("notes")
                or low.startswith("examples")
                or low.startswith("references")
                or low.startswith("see also")
                or low.startswith("side effects")
            ):
                in_section = False
                continue
            clean = stripped.lstrip("-* ").strip()
            if clean:
                out.append(clean)
    return out


def _criticality_record(module: str) -> dict[str, object]:
    """Return the criticality record for ``module``.

    If no manual entry exists (e.g. for ``__init__`` files or future modules), a
    conservative default is returned so the manifest always contains a record.
    """
    entry = CRITICALITY.get(module)
    if entry is None:
        return {
            "simulation_accuracy": 2,
            "training_accuracy": 2,
            "overall_criticality": 2,
            "role": "package_init_or_unscored",
            "rationale": (
                "No explicit criticality assigned; usually an `__init__.py` "
                "re-exporting public names or a future module. Errors here are "
                "low-impact unless the package re-exports core numerical primitives."
            ),
            "regression_sensitivity": "low",
            "fallback_strategy": "none",
        }
    sim = int(entry["simulation_accuracy"])  # type: ignore[arg-type]
    train = int(entry["training_accuracy"])  # type: ignore[arg-type]
    return {
        "simulation_accuracy": sim,
        "training_accuracy": train,
        "overall_criticality": max(sim, train),
        "role": entry["role"],
        "rationale": entry["rationale"],
        "regression_sensitivity": entry["regression_sensitivity"],
        "fallback_strategy": entry["fallback_strategy"],
    }


if __name__ == "__main__":
    repo = Path(sys.argv[1])
    main(repo)
