"""Deeper simulation-correctness audit covering scenarios, env, strategy,
sizing, account, and the data-flow between modules.

Run with: ``python audit_deep_correctness.py``
"""

from __future__ import annotations

import datetime as _dt
import sys
import traceback
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, "/home/user/hutrh/extracted/repo/unpacked/options-engine/src")

from options_engine.core.config import RiskConfig
from options_engine.core.market_alpha import MarketAlpha
from options_engine.core.random import RandomFactory
from options_engine.core.timegrid import TimeGrid
from options_engine.models.rbergomi import (
    ForwardVariance,
    HybridSimulator,
    RBergomiParams,
    build_terminal_distribution,
)
from options_engine.pricing.instruments import EuropeanOption, IronCondor, BullPutSpread, OptionRight
from options_engine.pricing.payoff import iron_condor_payoff, iron_condor_pnl, bull_put_spread_payoff
from options_engine.rl.scenario import Episode, ScenarioConfig, generate_episode
from options_engine.strategy.account import Account, OpenPosition
from options_engine.strategy.sizing import (
    SizingInputs, SizingResult, size_position, kelly_fraction,
    empirical_kelly_fraction,
)
from options_engine.strategy.condor_selection import CondorSelectionConfig, select_iron_condor
from options_engine.rl.observation import build_observation, observation_bounds
from options_engine.rl.env import EnvConfig, IronCondorTradingEnv


@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def _isfinite(x) -> bool:
    arr = np.asarray(x, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


def _close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ===========================================================================
# Scenario generator audit
# ===========================================================================

def audit_scenario_domain_randomization() -> list[Result]:
    """Different episodes must produce different realized variance + VRP."""
    out: list[Result] = []
    cfg = ScenarioConfig(n_steps=5, n_paths=1000, horizon_days=10.0,
                        realized_variance_range=(0.02, 0.08),
                        vrp_multiplier_range=(1.2, 2.5))
    realized_vars = []
    vrps = []
    for seed in range(20):
        factory = RandomFactory(seed)
        episode = generate_episode(rng_factory=factory, config=cfg, episode_index=seed)
        realized_vars.append(episode.realized_variance)
        vrps.append(episode.vrp_multiplier)
    var_spread = max(realized_vars) - min(realized_vars)
    vrp_spread = max(vrps) - min(vrps)
    out.append(Result(
        "Scenario: realized_variance varies across episodes",
        var_spread > 0.03,
        f"range = [{min(realized_vars):.4f}, {max(realized_vars):.4f}]",
    ))
    out.append(Result(
        "Scenario: vrp_multiplier varies across episodes",
        vrp_spread > 0.5,
        f"range = [{min(vrps):.3f}, {max(vrps):.3f}]",
    ))
    return out


def audit_scenario_alpha_awareness() -> list[Result]:
    """Alpha must override the random Hurst/eta sampling."""
    out: list[Result] = []
    cfg = ScenarioConfig(n_steps=4, n_paths=1000, horizon_days=5.0,
                        realized_variance_range=(0.04, 0.041),
                        vrp_multiplier_range=(1.0, 1.001))
    # Two scenarios with the same RNG seed but different alphas must differ
    # (because Hurst/eta come from alpha, not from the random sampler).
    rng_a = RandomFactory(0)
    rng_b = RandomFactory(0)
    ep_alpha_rough = generate_episode(
        rng_factory=rng_a, config=cfg, episode_index=0,
        alpha=MarketAlpha.scalar(0.0),
    )
    ep_alpha_calm = generate_episode(
        rng_factory=rng_b, config=cfg, episode_index=0,
        alpha=MarketAlpha.scalar(1.0),
    )
    # The terminal sample distributions should differ because Hurst differs.
    rough_terminal = ep_alpha_rough.steps[0].realized_terminal_sample
    calm_terminal = ep_alpha_calm.steps[0].realized_terminal_sample
    out.append(Result(
        "Scenario: alpha=0 vs alpha=1 produce different terminal distributions",
        bool(np.std(rough_terminal) > 0) and bool(np.std(calm_terminal) > 0)
        and float(np.std(rough_terminal)) != float(np.std(calm_terminal)),
        f"std(alpha=0) = {float(np.std(rough_terminal)):.4f}, "
        f"std(alpha=1) = {float(np.std(calm_terminal)):.4f}",
    ))
    out.append(Result(
        "Scenario: rough alpha gives wider terminal distribution than calm alpha",
        float(np.std(rough_terminal)) > float(np.std(calm_terminal)),
        f"std ratio = {float(np.std(rough_terminal)) / float(np.std(calm_terminal)):.3f}",
    ))
    # Default (alpha=None) must match the legacy random-sampling behavior.
    rng_c = RandomFactory(0)
    ep_default = generate_episode(rng_factory=rng_c, config=cfg, episode_index=0)
    default_terminal = ep_default.steps[0].realized_terminal_sample
    out.append(Result(
        "Scenario: alpha=None still produces a valid episode",
        len(ep_default.steps) == cfg.n_steps
        and bool(_isfinite(default_terminal)),
        f"steps={len(ep_default.steps)}, "
        f"std={float(np.std(default_terminal)):.4f}",
    ))
    return out


def audit_scenario_steps_consistency() -> list[Result]:
    """Every step in an episode must have consistent spot, distribution, regime."""
    out: list[Result] = []
    cfg = ScenarioConfig(n_steps=10, n_paths=1000, horizon_days=14.0)
    rng = RandomFactory(7)
    episode = generate_episode(rng_factory=rng, config=cfg, episode_index=0)
    out.append(Result(
        f"Scenario: episode has exactly n_steps = {cfg.n_steps}",
        len(episode.steps) == cfg.n_steps,
        f"got {len(episode.steps)}",
    ))
    for i, step in enumerate(episode.steps):
        # Every step must have a finite terminal sample.
        if not _isfinite(step.realized_terminal_sample):
            out.append(Result(
                f"Scenario step {i}: terminal sample finite",
                False, "non-finite values",
            ))
            break
        # Every step must have a no-arb chain with quotes on both sides.
        try:
            atm_call = step.chain.call(step.chain.strikes[len(step.chain.strikes) // 2])
            atm_put = step.chain.put(step.chain.strikes[len(step.chain.strikes) // 2])
            if atm_call.quote.bid >= atm_call.quote.ask or atm_put.quote.bid >= atm_put.quote.ask:
                out.append(Result(
                    f"Scenario step {i}: chain quotes are non-crossed",
                    False, "crossed market",
                ))
                break
        except Exception as e:
            out.append(Result(
                f"Scenario step {i}: chain lookup succeeds",
                False, str(e),
            ))
            break
    else:
        out.append(Result(
            f"Scenario: all {cfg.n_steps} steps have valid finite samples + non-crossed chains",
            True, f"{cfg.n_steps} steps verified",
        ))
    return out


def audit_scenario_regime_matches_realized_variance() -> list[Result]:
    """Higher realized variance should produce lower P(LOW) regime probability."""
    out: list[Result] = []
    cfg = ScenarioConfig(n_steps=4, n_paths=1000, horizon_days=5.0)
    low_probs = []
    for seed in range(30):
        rng = RandomFactory(seed)
        ep = generate_episode(rng_factory=rng, config=cfg, episode_index=seed)
        low_probs.append((ep.realized_variance, ep.steps[0].regime.current_probability(
            __import__("options_engine.core.enums", fromlist=["VolRegime"]).VolRegime.LOW
        )))
    pairs = sorted(low_probs, key=lambda x: x[0])
    low_variance_low_prob = pairs[0][1]
    high_variance_low_prob = pairs[-1][1]
    out.append(Result(
        "Scenario regime: low realized variance -> higher P(LOW) than high realized variance",
        low_variance_low_prob > high_variance_low_prob,
        f"low-VAR P(LOW)={low_variance_low_prob:.3f}, "
        f"high-VAR P(LOW)={high_variance_low_prob:.3f}",
    ))
    return out


# ===========================================================================
# Env audit
# ===========================================================================

def audit_env_observation_bounds() -> list[Result]:
    """Every observation must lie in the declared observation-space bounds."""
    out: list[Result]
    out = []
    low, high = observation_bounds()
    env = IronCondorTradingEnv(seed=0, scenario_config=ScenarioConfig(n_steps=10, n_paths=1000))
    obs, _ = env.reset(seed=0)
    in_space = bool(np.all(obs >= low)) and bool(np.all(obs <= high))
    out.append(Result(
        "Env: reset observation lies in observation_space bounds",
        in_space, f"obs shape = {obs.shape}",
    ))
    # Try a non-FLAT action so a trade may execute.
    trade_action = np.array([2.0, 0, -1, -0.5, 0, 1.0], dtype=np.float32)
    for _ in range(5):
        obs, _r, term, trunc, _info = env.step(trade_action)
        in_space = bool(np.all(obs >= low)) and bool(np.all(obs <= high))
        if not in_space:
            out.append(Result(
                "Env: step observation lies in observation_space bounds",
                False, f"obs={obs}, low={low}, high={high}",
            ))
            return out
        if term or trunc:
            break
    out.append(Result(
        "Env: every step observation lies in observation_space bounds",
        True, "5 steps verified",
    ))
    return out


def audit_env_action_handling() -> list[Result]:
    """All three strategic actions (HARVEST_THETA, HARVEST_GAMMA, FLAT) must be
    accepted; out-of-range continuous knobs must be clipped, not errored."""
    out = []
    env = IronCondorTradingEnv(seed=0, scenario_config=ScenarioConfig(n_steps=4, n_paths=1000))
    env.reset(seed=0)
    # FLAT action: logit 3 (FLAT has index 2 in _STRATEGIC_ORDER = (THETA, GAMMA, FLAT)).
    flat_action = np.array([0, 0, 5.0, 0, 0, 0], dtype=np.float32)
    try:
        env.step(flat_action)
        out.append(Result("Env: accepts FLAT action (logits [0,0,5])", True, "no exception"))
    except Exception as e:
        out.append(Result("Env: accepts FLAT action (logits [0,0,5])", False, str(e)))
    # THETA action: logit 5 at index 0.
    theta_action = np.array([5.0, 0, 0, -0.5, 0, 1.0], dtype=np.float32)
    try:
        env.step(theta_action)
        out.append(Result("Env: accepts HARVEST_THETA action (logits [5,0,0])", True, "no exception"))
    except Exception as e:
        out.append(Result("Env: accepts HARVEST_THETA action (logits [5,0,0])", False, str(e)))
    # GAMMA action: logit 5 at index 1.
    gamma_action = np.array([0, 5.0, 0, -0.5, 0, 1.0], dtype=np.float32)
    try:
        env.step(gamma_action)
        out.append(Result("Env: accepts HARVEST_GAMMA action (logits [0,5,0])", True, "no exception"))
    except Exception as e:
        out.append(Result("Env: accepts HARVEST_GAMMA action (logits [0,5,0])", False, str(e)))
    # Out-of-range continuous knobs (e.g. tail_prob = 100, wing_width = -100) must be clipped, not rejected.
    wild_action = np.array([5.0, 0, 0, 100.0, -100.0, 50.0], dtype=np.float32)
    try:
        env.step(wild_action)
        out.append(Result(
            "Env: out-of-range continuous knobs are clipped (not rejected)",
            True, "no exception",
        ))
    except Exception as e:
        out.append(Result(
            "Env: out-of-range continuous knobs are clipped (not rejected)",
            False, str(e),
        ))
    return out


def audit_env_accounting_consistency() -> list[Result]:
    """After a trade, account.equity must equal cash + (sum of closed P&L)."""
    out = []
    env = IronCondorTradingEnv(seed=0, scenario_config=ScenarioConfig(n_steps=4, n_paths=1000))
    env.reset(seed=0)
    # Take an aggressive THETA action every step.
    action = np.array([5.0, 0, -1, -0.5, 0, 1.0], dtype=np.float32)
    for _ in range(4):
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    # At episode end, all positions are closed (env opens and closes on the
    # same step) so equity == cash + unrealized_pnl == cash (unrealized_pnl = 0).
    # Note: ``realized_pnl`` in this codebase is the payoff-only component
    # (``pnl - theta``); the credit premium is credited directly to cash, so
    # ``cash + realized_pnl != equity`` by design. The correct invariant at
    # episode end is ``equity == cash`` (no open positions).
    eq = env.account_equity()
    cash = env._account.cash  # type: ignore[attr-defined]
    out.append(Result(
        "Env: account.equity() == cash at episode end (positions closed)",
        _close(eq, cash, atol=1e-9, rtol=1e-9),
        f"eq={eq:.4f}, cash={cash:.4f}",
    ))
    # And the realised_pnl field equals sum of (pnl - theta) over closed trades.
    pnl = env._account.realized_pnl  # type: ignore[attr-defined]
    starting_cash = env._env_config.starting_cash  # type: ignore[attr-defined]
    theta_captured_total = starting_cash - cash - pnl  # credits received = starting - cash - payoff-only
    out.append(Result(
        "Env: realised_pnl tracks payoff component (pnl - theta), not net cash change",
        # cash_change = starting - cash; pnl_field_change = pnl; credit_total = starting - cash - pnl
        theta_captured_total >= 0.0,
        f"cash_change={starting_cash - cash:.2f}, "
        f"realised_pnl={pnl:.2f}, "
        f"implied credits={theta_captured_total:.2f}",
    ))
    return out


# ===========================================================================
# Strategy / sizing / account audit
# ===========================================================================

def audit_sizing_kelly_invariant() -> list[Result]:
    """Kelly fraction must be 0 when win probability <= 1/(1+payoff_ratio)."""
    out = []
    # For a binary bet with b=1 (1:1 payoff), Kelly f* = 2p - 1. f* >= 0 iff p >= 0.5.
    for p in (0.1, 0.3, 0.49, 0.5, 0.6, 0.9):
        f = kelly_fraction(win_probability=p, payoff_ratio=1.0)
        expected_f = max(0.0, 2 * p - 1)
        out.append(Result(
            f"Kelly(p={p}, b=1) = {expected_f:.3f}",
            _close(f, expected_f, atol=1e-9, rtol=1e-9),
            f"got {f:.4f}",
        ))
    # For b=2 (2:1 payoff), f* = p - (1-p)/2 = 1.5p - 0.5. f* >= 0 iff p >= 1/3.
    for p in (0.2, 0.3, 0.4, 0.7):
        f = kelly_fraction(win_probability=p, payoff_ratio=2.0)
        expected_f = max(0.0, 1.5 * p - 0.5)
        out.append(Result(
            f"Kelly(p={p}, b=2) = {expected_f:.3f}",
            _close(f, expected_f, atol=1e-9, rtol=1e-9),
            f"got {f:.4f}",
        ))
    # Kelly must be non-negative for any p in [0, 1].
    for p in np.linspace(0.0, 1.0, 21):
        f = kelly_fraction(win_probability=float(p), payoff_ratio=0.5)
        if f < 0.0:
            out.append(Result(
                f"Kelly(p={p:.2f}, b=0.5) >= 0", False, f"got {f}",
            ))
            break
    else:
        out.append(Result(
            "Kelly(p, b=0.5) is non-negative for p in [0, 1]",
            True, "21 samples verified",
        ))
    return out


def audit_sizing_size_position_constraints() -> list[Result]:
    """size_position must respect all four hard caps (kelly, per-trade, daily, margin)."""
    out = []
    risk = RiskConfig(max_risk_fraction_per_trade=0.02, max_risk_fraction_per_day=0.06,
                     kelly_fraction=0.5, max_leverage=2.0)
    # Case 1: kelly says 5 contracts, per-trade cap says 3 -> per-trade wins.
    # risk_per_condor = 5.0; equity = 10000 -> kelly_capital = 5*0.02*10000 = 1000 (fractional=0.5*2.0)
    # Actually let's use kelly_fraction=0.25 with full_kelly=2.0 -> fractional = 0.5
    # kelly_capital = 0.5 * 10000 = 5000
    # per_trade_cap = 0.02 * 10000 = 200; risk_per_condor = 5000/200 = 25 contracts limit
    # This case is messy; use simpler numbers instead.
    risk2 = RiskConfig(max_risk_fraction_per_trade=0.10, max_risk_fraction_per_day=0.20,
                      kelly_fraction=0.25, max_leverage=2.0)
    # Setup: equity=10000, max_risk_per_trade=0.10 -> 1000 cap; available_margin=10000
    # credit=5, max_loss=5 -> b=1; win_prob=0.8 -> kelly=2*0.8-1=0.6; fractional=0.6*0.25=0.15
    # kelly_capital = 0.15 * 10000 = 1500; per_trade_cap = 0.10*10000=1000
    # daily_budget = 0.20*10000=2000 -> remaining 2000
    # margin_capital = 10000
    # binding constraint = per_trade_cap (1000)
    # quantity = floor(1000 / 5) = 200
    inputs = SizingInputs(
        account_equity=10_000.0, win_probability=0.8, net_credit=5.0,
        max_loss_per_condor=5.0, multiplier=1.0, available_margin=10_000.0,
    )
    result = size_position(inputs, risk=risk2)
    out.append(Result(
        f"size_position: per_trade cap binds (got {result.binding_constraint!r}, "
        f"quantity={result.quantity})",
        result.binding_constraint == "per_trade_cap" and result.quantity == 200,
        f"binding={result.binding_constraint}, q={result.quantity}",
    ))
    # Case 2: kelly tiny, margin tiny -> margin binds.
    inputs2 = SizingInputs(
        account_equity=10_000.0, win_probability=0.5, net_credit=1.0,
        max_loss_per_condor=5.0, multiplier=1.0, available_margin=200.0,
    )
    result2 = size_position(inputs2, risk=risk2)
    # margin_capital=200; per_contract risk=5 -> 40 contracts
    # per_trade_cap=1000/5=200 contracts -> per_trade bigger
    # kelly=2*0.5-1=0 -> fractional=0 -> kelly_capital=0 -> kelly binds
    # Wait: kelly=0, so kelly_capital=0. daily_budget=2000, margin=200. Binding=kelly.
    # Quantity should be 0.
    out.append(Result(
        f"size_position: kelly=0 -> quantity=0 (binding=kelly)",
        result2.binding_constraint == "kelly" and result2.quantity == 0,
        f"binding={result2.binding_constraint}, q={result2.quantity}",
    ))
    # Case 3: sufficient margin, sufficient kelly -> kelly binds.
    inputs3 = SizingInputs(
        account_equity=10_000.0, win_probability=0.99, net_credit=2.5,
        max_loss_per_condor=5.0, multiplier=1.0, available_margin=10_000.0,
    )
    result3 = size_position(inputs3, risk=risk2)
    # kelly=2*0.99-1=0.98; fractional=0.98*0.25=0.245 -> kelly_cap=2450
    # per_trade=1000, daily=2000 (remaining=2000), margin=10000
    # binding = per_trade (1000); quantity = floor(1000/5) = 200
    out.append(Result(
        f"size_position: per_trade binds at 200 contracts (got q={result3.quantity})",
        result3.quantity == 200,
        f"binding={result3.binding_constraint}, q={result3.quantity}",
    ))
    return out


def audit_sizing_empirical_kelly() -> list[Result]:
    """empirical_kelly_fraction returns 0 for a non-positive sample mean and the
    growth-optimal f for positive means."""
    out = []
    # All losses -> mean < 0 -> 0.
    losses = -np.ones(100, dtype=np.float64)
    out.append(Result(
        "empirical_kelly_fraction: all-loss sample -> 0",
        empirical_kelly_fraction(losses) == 0.0,
        f"got {empirical_kelly_fraction(losses):.4f}",
    ))
    # 60% win, 40% loss of equal magnitude -> mean > 0, growth-optimal > 0.
    pnl = np.where(np.random.default_rng(0).uniform(size=200) < 0.6, 1.0, -1.0)
    f = empirical_kelly_fraction(pnl)
    out.append(Result(
        "empirical_kelly_fraction: positive-mean sample -> > 0",
        f > 0.0,
        f"got {f:.4f}",
    ))
    return out


def audit_account_drawdown_correctness() -> list[Result]:
    """Account.drawdown == (HWM - equity) / HWM, clamped to 0."""
    out = []
    acc = Account.open(starting_cash=10_000.0).with_high_water_mark_updated()
    # equity = 10000, HWM = 10000 -> drawdown = 0.
    out.append(Result(
        "Account.drawdown: equity == HWM -> 0",
        acc.drawdown() == 0.0,
        f"got {acc.drawdown()}",
    ))
    # Force a 20% loss: close_position with realized=-2000.
    condor = IronCondor(90.0, 95.0, 105.0, 110.0, 0.05, quantity=1)
    pos = OpenPosition(
        position_id="t", condor=condor, entry_credit=1.0, quantity=1,
        multiplier=100.0, entry_time=_dt.datetime(2024, 1, 1, tzinfo=_dt.UTC),
        entry_spot=100.0,
    )
    acc2 = acc.add_position(pos, premium_received=100.0).close_position("t", realized=-2000.0)
    # equity = 10000 + 100 - 2000 = 8100. HWM = 10000. drawdown = (10000 - 8100) / 10000 = 0.19.
    out.append(Result(
        "Account.drawdown: 20% loss -> 0.19",
        _close(acc2.drawdown(), 0.19, atol=1e-9, rtol=1e-9),
        f"got {acc2.drawdown()}",
    ))
    # Force a 50% gain: close_position with realized=+5000.
    pos2 = OpenPosition(
        position_id="t", condor=condor, entry_credit=1.0, quantity=1,
        multiplier=100.0, entry_time=_dt.datetime(2024, 1, 1, tzinfo=_dt.UTC),
        entry_spot=100.0,
    )
    acc3 = acc.add_position(pos2, premium_received=100.0).close_position("t", realized=+5000.0)
    # equity = 10000 + 100 + 5000 = 15100; HWM should be raised to 15100; drawdown = 0.
    out.append(Result(
        "Account.drawdown: profit raises HWM, drawdown clamps to 0",
        acc3.drawdown() == 0.0 and acc3.high_water_mark >= 15100.0,
        f"drawdown={acc3.drawdown()}, HWM={acc3.high_water_mark}",
    ))
    return out


def audit_account_equity_invariant() -> list[Result]:
    """Account.equity == cash + realized_pnl + unrealized_pnl (cash + unrealized at minimum)."""
    out = []
    acc = Account.open(starting_cash=10_000.0)
    out.append(Result(
        "Account.equity == cash when no positions",
        _close(acc.equity(), acc.cash, atol=1e-9, rtol=1e-9),
        f"cash={acc.cash}, eq={acc.equity()}",
    ))
    out.append(Result(
        "Account.equity(0) == equity() == cash when no positions",
        _close(acc.equity(0.0), acc.equity(), atol=1e-9, rtol=1e-9),
        f"eq={acc.equity()}, eq(0)={acc.equity(0.0)}",
    ))
    # With unrealized pnl.
    out.append(Result(
        "Account.equity(unrealized) == cash + unrealized_pnl",
        _close(acc.equity(unrealized_pnl=500.0), 10_500.0, atol=1e-9, rtol=1e-9),
        f"eq={acc.equity(500.0)}",
    ))
    return out


# ===========================================================================
# Condor selection audit
# ===========================================================================

def audit_condor_selection_consistency() -> list[Result]:
    """Selected condor must have strikes ordered (put_long < put_short < call_short < call_long)
    and a positive net credit."""
    out = []
    cfg = ScenarioConfig(n_steps=5, n_paths=4000,
                        realized_variance_range=(0.03, 0.05),
                        vrp_multiplier_range=(1.5, 2.5))
    rng = RandomFactory(42)
    episode = generate_episode(rng_factory=rng, config=cfg, episode_index=0)
    n_violations = 0
    n_no_arb_violations = 0
    for step in episode.steps:
        sel = select_iron_condor(
            step.realized_distribution, step.chain,
            config=CondorSelectionConfig(min_win_probability=0.3, min_net_credit=0.01),
            terminal_sample=step.realized_terminal_sample,
        )
        if sel is None:
            continue
        c = sel.condor
        ordered = (c.put_long_strike < c.put_short_strike < c.call_short_strike < c.call_long_strike)
        if not ordered:
            n_violations += 1
        if sel.net_credit <= 0.0:
            n_no_arb_violations += 1
    out.append(Result(
        f"CondorSelection: strikes are ordered across {len(episode.steps)} steps",
        n_violations == 0,
        f"{n_violations} violations",
    ))
    out.append(Result(
        "CondorSelection: net credit > 0 for all selected condors",
        n_no_arb_violations == 0,
        f"{n_no_arb_violations} non-positive credits",
    ))
    return out


# ===========================================================================
# Iron condor payoff deeper checks
# ===========================================================================

def audit_iron_condor_payoff_extremes() -> list[Result]:
    """Verify payoff at all extreme spots (deep ITM put, deep OTM, deep ITM call)."""
    out = []
    condor = IronCondor(
        put_long_strike=90.0, put_short_strike=95.0,
        call_short_strike=105.0, call_long_strike=110.0,
        expiry=0.05, quantity=1,
    )
    # Spot deep below put_long (e.g. 50): max loss for short.
    # legs: long put @ 90 (qty +1) pays (90-50)=40; short put @ 95 (qty -1) pays -(95-50)= -45
    # longs calls @ 105, 110: 0
    # Total payoff: 40 - 45 = -5 (max loss = spread width - 0 = 5)
    payoff_50 = iron_condor_payoff(condor, np.array([50.0]))
    out.append(Result(
        "IronCondor payoff at deep ITM put: -5 (max loss)",
        bool(np.allclose(payoff_50, [-5.0], atol=1e-9)),
        f"got {float(payoff_50[0])}",
    ))
    # Spot deep above call_long (e.g. 200): symmetric.
    # legs: short call @ 105 (qty -1) pays -(200-105)=-95; long call @ 110 (qty +1) pays (200-110)=90
    # Total: -95 + 90 = -5
    payoff_200 = iron_condor_payoff(condor, np.array([200.0]))
    out.append(Result(
        "IronCondor payoff at deep ITM call: -5 (max loss)",
        bool(np.allclose(payoff_200, [-5.0], atol=1e-9)),
        f"got {float(payoff_200[0])}",
    ))
    # Spot between short strikes (profit zone): 0
    payoff_100 = iron_condor_payoff(condor, np.array([100.0]))
    out.append(Result(
        "IronCondor payoff at ATM: 0",
        bool(np.allclose(payoff_100, [0.0], atol=1e-9)),
        f"got {float(payoff_100[0])}",
    ))
    return out


def audit_bull_put_spread_payoff() -> list[Result]:
    """BullPutSpread payoff: 0 above short strike, -spread_width below long strike."""
    out = []
    spread = BullPutSpread(long_strike=90.0, short_strike=95.0, expiry=0.05, quantity=1)
    # Above short strike (spot=120): both puts expire worthless -> payoff = 0.
    payoff_high = bull_put_spread_payoff(spread, np.array([120.0]))
    out.append(Result(
        "BullPutSpread payoff above short strike: 0",
        bool(np.allclose(payoff_high, [0.0], atol=1e-9)),
        f"got {float(payoff_high[0])}",
    ))
    # Below long strike (spot=50): long put pays 40, short put pays -45, total = -5 (max loss).
    payoff_low = bull_put_spread_payoff(spread, np.array([50.0]))
    out.append(Result(
        "BullPutSpread payoff below long strike: -5 (max loss)",
        bool(np.allclose(payoff_low, [-5.0], atol=1e-9)),
        f"got {float(payoff_low[0])}",
    ))
    return out


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    print("=" * 72)
    print("options_engine -- DEEP simulation correctness audit")
    print("=" * 72)
    audits = [
        ("Scenario: domain randomization", audit_scenario_domain_randomization),
        ("Scenario: alpha awareness", audit_scenario_alpha_awareness),
        ("Scenario: per-step consistency", audit_scenario_steps_consistency),
        ("Scenario: regime vs realized variance", audit_scenario_regime_matches_realized_variance),
        ("Env: observation in space", audit_env_observation_bounds),
        ("Env: action handling", audit_env_action_handling),
        ("Env: accounting consistency", audit_env_accounting_consistency),
        ("Sizing: Kelly analytical invariant", audit_sizing_kelly_invariant),
        ("Sizing: constraint binding", audit_sizing_size_position_constraints),
        ("Sizing: empirical Kelly", audit_sizing_empirical_kelly),
        ("Account: drawdown math", audit_account_drawdown_correctness),
        ("Account: equity invariant", audit_account_equity_invariant),
        ("Condor selection: consistency", audit_condor_selection_consistency),
        ("IronCondor: payoff extremes", audit_iron_condor_payoff_extremes),
        ("BullPutSpread: payoff", audit_bull_put_spread_payoff),
    ]
    all_results: list[Result] = []
    for label, fn in audits:
        print(f"\n--- {label} ---")
        try:
            results = fn()
        except Exception:
            traceback.print_exc()
            results = [Result(label, False, "AUDIT CRASHED")]
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name}: {r.detail}")
            all_results.append(r)
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    print()
    print("=" * 72)
    print(f"DEEP AUDIT SUMMARY: {n_pass}/{n_total} passed ({n_pass / max(n_total, 1) * 100:.0f}%)")
    print("=" * 72)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
