"""Training-integrity audit: detect exploitable artifacts in the RL simulation.

These tests verify that the simulation is internally consistent, does not leak future
information, and is free of the obvious arbitrage / scaling / reproducibility bugs that a
PPO policy could exploit. They are designed to catch the subtle corruption modes that
static unit tests can miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import OptionRight, StrategicAction, VolRegime
from options_engine.core.market_alpha import MarketAlpha
from options_engine.core.random import RandomFactory
from options_engine.pricing.black_scholes import price as _bs_price_array

def bs_price(*args, **kwargs) -> float:
    """Scalar convenience wrapper around the vectorized Black-Scholes pricer."""
    return float(_bs_price_array(*args, **kwargs)[0])
from options_engine.rl.action import ActionBounds, decode_action
from options_engine.rl.env import EnvConfig, IronCondorTradingEnv
from options_engine.rl.scenario import ScenarioConfig, generate_episode
from options_engine.rl.observation import OBSERVATION_DIM


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_env(*, seed: int = 123, scenario: ScenarioConfig | None = None) -> IronCondorTradingEnv:
    cfg = scenario or ScenarioConfig(n_steps=10, n_paths=8_000, horizon_days=7.0, steps_per_day=4)
    return IronCondorTradingEnv(
        seed=seed,
        env_config=EnvConfig(max_open_positions=5, starting_cash=1_000_000.0),
        scenario_config=cfg,
    )


def _flat_action() -> np.ndarray:
    """Return an action vector that decodes to the FLAT strategic action."""
    # Logits at index 2 (FLAT) dominate; continuous knobs are irrelevant for FLAT.
    return np.array([-1.0, -1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _trade_action(
    *, tail: float = 0.15, wing: float = 0.05, size: float = 1.0
) -> np.ndarray:
    """Return an action vector that decodes to HARVEST_THETA with the given tactical knobs."""
    # Argmax is at index 0. Knobs are in [-1, 1] and get mapped to their ranges by
    # ActionBounds, so we feed back the unit-scale values that produce the targets.
    b = ActionBounds()
    tail_unit = 2.0 * (tail - b.tail_probability[0]) / (b.tail_probability[1] - b.tail_probability[0]) - 1.0
    wing_unit = 2.0 * (wing - b.wing_width_fraction[0]) / (b.wing_width_fraction[1] - b.wing_width_fraction[0]) - 1.0
    size_unit = 2.0 * (size - b.size_fraction[0]) / (b.size_fraction[1] - b.size_fraction[0]) - 1.0
    return np.array([2.0, -1.0, -1.0, tail_unit, wing_unit, size_unit], dtype=np.float64)


# ----------------------------------------------------------------------------
# Reproducibility and determinism
# ----------------------------------------------------------------------------


def test_episode_reproducible_bit_for_bit() -> None:
    """Same seed must produce identical observations and rewards."""
    cfg = ScenarioConfig(n_steps=5, n_paths=4_000, horizon_days=5.0, steps_per_day=4)
    env_a = _make_env(seed=42, scenario=cfg)
    env_b = _make_env(seed=42, scenario=cfg)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)

    action = _trade_action()
    for _ in range(len(env_a._episode or ())):
        o_a, r_a, term_a, trunc_a, _ = env_a.step(action)
        o_b, r_b, term_b, trunc_b, _ = env_b.step(action)
        np.testing.assert_array_equal(o_a, o_b)
        assert r_a == pytest.approx(r_b, abs=1e-9)
        assert term_a == term_b
        assert trunc_a == trunc_b
        if term_a or trunc_a:
            break


# ----------------------------------------------------------------------------
# Time semantics: the simulation horizon must advance
# ----------------------------------------------------------------------------


def test_chain_expiry_advances_each_step() -> None:
    """The option chain must have a shrinking expiry as the episode progresses."""
    env = _make_env(seed=7)
    env.reset()
    episode = env._episode
    assert episode is not None

    expiries = [step.chain.expiry for step in episode.steps]
    # Strictly decreasing (each step uses the remaining horizon).
    assert all(expiries[i] > expiries[i + 1] for i in range(len(expiries) - 1))
    # First expiry is the full horizon in years; last expiry is one step in years.
    assert expiries[0] > 0.0
    assert expiries[-1] > 0.0


def test_spot_path_is_consistent_with_realized_distribution() -> None:
    """The spot path that drives observations must be a plausible path from the model."""
    env = _make_env(seed=11)
    env.reset()
    episode = env._episode
    assert episode is not None

    spots = np.array([step.spot for step in episode.steps], dtype=np.float64)
    # Spot stays strictly positive.
    assert np.all(spots > 0.0)
    # Returns are finite.
    log_returns = np.diff(np.log(spots))
    assert np.all(np.isfinite(log_returns))

    # The terminal distribution at each step should be roughly centered around the current
    # spot, not wildly inconsistent.
    for step in episode.steps:
        mean_terminal = float(np.mean(step.realized_terminal_sample))
        assert mean_terminal > 0.0
        # The expected terminal spot should be within a few standard deviations of the
        # current spot (in the rBergomi martingale case it would be exactly the spot).
        std_terminal = float(np.std(step.realized_terminal_sample))
        assert abs(mean_terminal - step.spot) < 5.0 * std_terminal + 1e-6


# ----------------------------------------------------------------------------
# Maximum-alpha (calm) world must be Black-Scholes-like and free of artifacts
# ----------------------------------------------------------------------------


def test_alpha_one_chain_prices_match_black_scholes_within_mc_error() -> None:
    """At alpha=1 the synthetic chain should agree with BS up to Monte-Carlo noise.

    This is the core realism check: a maximally calm market is a near-Black-Scholes world,
    so the chain cannot carry systematic mispricing that PPO could arbitrage.
    """
    alpha = MarketAlpha.ones()
    cfg = ScenarioConfig(n_steps=3, n_paths=20_000, horizon_days=7.0, steps_per_day=4)
    ep = generate_episode(
        rng_factory=RandomFactory(99),
        config=cfg,
        episode_index=0,
        alpha=alpha,
    )

    for step in ep.steps:
        chain = step.chain
        expiry = chain.expiry
        spot = step.spot
        # The pricing variance is realized_var * vrp; the chain was built from that.
        # We recover it from the realized_distribution's variance (approx) and the VRP,
        # but for BS comparison we use the implied vol from the pricing distribution.
        # The cleanest oracle is the model variance used for the pricing simulation.
        pricing_var = ep.realized_variance * ep.vrp_multiplier
        vol = float(np.sqrt(pricing_var))

        errors: list[float] = []
        ses: list[float] = []
        for strike in chain.strikes:
            call_quote = chain.call(float(strike)).quote
            mid = call_quote.mid
            bs = bs_price(spot, strike, expiry, vol, OptionRight.CALL, rate=0.0)
            # The chain price is an MC estimate; its noise scale is roughly the spread / 2
            # (conservative) or the theoretical standard error. Use the quoted spread as a
            # practical tolerance band.
            tol = 0.5 * call_quote.spread + 0.01
            errors.append(abs(mid - bs))
            ses.append(tol)

        max_error = max(errors)
        max_tol = max(ses)
        # Every step should be within the quoted spread tolerance; no systematic BS bias.
        assert max_error <= max_tol, (
            f"alpha=1 chain deviates from BS by {max_error:.4f} (tolerance {max_tol:.4f}) "
            f"at step with expiry={expiry:.4f}, spot={spot:.2f}"
        )


def test_alpha_one_terminal_log_returns_are_approximately_normal() -> None:
    """At alpha=1 the terminal distribution should be close to log-normal.

    We do not require a formal normality test (under-powered at 8k paths); instead we check
    that the mean and standard deviation of the log-returns are consistent with the model
    variance. Any gross artifact (e.g., all paths identical, extreme skew) would fail here.
    """
    alpha = MarketAlpha.ones()
    cfg = ScenarioConfig(n_steps=2, n_paths=20_000, horizon_days=14.0, steps_per_day=4)
    ep = generate_episode(
        rng_factory=RandomFactory(101),
        config=cfg,
        episode_index=0,
        alpha=alpha,
    )
    step = ep.steps[0]
    log_returns = step.realized_distribution.log_returns

    mean_lr = float(np.mean(log_returns))
    std_lr = float(np.std(log_returns, ddof=1))
    expected_std = float(np.sqrt(ep.realized_variance * step.chain.expiry))

    # Mean should be close to -0.5 * variance (martingale correction in log-Euler).
    assert abs(mean_lr + 0.5 * std_lr**2) < 0.05

    # Standard deviation should be close to the model variance (within 10% relative).
    if expected_std > 0.0:
        assert abs(std_lr - expected_std) / expected_std < 0.10

    # Range must be finite and non-degenerate.
    assert np.isfinite(std_lr)
    assert std_lr > 0.0


# ----------------------------------------------------------------------------
# No arbitrage in the synthetic chain
# ----------------------------------------------------------------------------


def test_chain_satisfies_static_no_arbitrage_constraints() -> None:
    """Every chain must be monotone, convex, and obey put-call parity."""
    ep = generate_episode(
        rng_factory=RandomFactory(202),
        config=ScenarioConfig(n_steps=5, n_paths=4_000, horizon_days=7.0, steps_per_day=4),
        episode_index=0,
    )

    for step in ep.steps:
        chain = step.chain
        strikes = chain.strikes
        rate = 0.0
        discount = float(np.exp(-rate * step.chain.expiry))
        spot = step.spot

        # The repaired theoretical-value curve is where no-arbitrage is enforced strictly.
        call_theory = np.array([chain.call(float(k)).theoretical_value for k in strikes])
        put_theory = np.array([chain.put(float(k)).theoretical_value for k in strikes])

        # Calls must be non-increasing in strike.
        assert np.all(np.diff(call_theory) <= 1e-9)

        # Convexity: each interior point must lie on or below the chord of its neighbours
        # (strikes are not necessarily uniformly spaced, so np.diff is not the right test).
        for i in range(1, len(strikes) - 1):
            x0, x1, x2 = strikes[i - 1], strikes[i], strikes[i + 1]
            y0, y2 = call_theory[i - 1], call_theory[i + 1]
            chord_y1 = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
            assert call_theory[i] <= chord_y1 + 1e-9

        # Lower bound: C >= max(S - K*discount, 0).
        lower = np.maximum(spot - strikes * discount, 0.0)
        assert np.all(call_theory >= lower - 1e-9)

        # Put-call parity: C - P = S - K*discount.
        parity = call_theory - put_theory
        expected = spot - strikes * discount
        assert np.all(np.abs(parity - expected) < 1e-6)

        # Quotes must not cross and must be non-negative. The theoretical curve carries the
        # no-arbitrage guarantees; the quoted two-sided market is bounded by the exchange
        # obligations and tick rounding and may be wider than the theoretical half-spread.
        for k in strikes:
            q = chain.call(float(k)).quote
            assert q.bid >= 0.0
            assert q.bid < q.ask
            q = chain.put(float(k)).quote
            assert q.bid >= 0.0
            assert q.bid < q.ask


# ----------------------------------------------------------------------------
# Observation leakage
# ----------------------------------------------------------------------------


def test_observation_has_no_lookahead_to_next_spot_return() -> None:
    """No observation feature may be a deterministic function of the next step's return.

    A PPO policy that can read the future in the observation would be catastrophic. We
    average over several episodes to wash out the spurious correlations that a single
    short episode can produce by chance.
    """
    cfg = ScenarioConfig(n_steps=30, n_paths=4_000, horizon_days=10.0, steps_per_day=4)
    all_obs: list[np.ndarray] = []
    all_returns: list[float] = []
    for seed in range(10):
        ep = generate_episode(rng_factory=RandomFactory(seed), config=cfg, episode_index=0)
        for i, step in enumerate(ep.steps[:-1]):
            dist = step.realized_distribution
            expected_move = float(np.std(dist.log_returns))
            sigma = expected_move if expected_move > 0 else 1e-6
            obs = np.array(
                [
                    dist.probability_in_range(-sigma, sigma),
                    dist.probability_below(-2.0 * sigma),
                    1.0 - dist.probability_below(2.0 * sigma),
                    min(expected_move, 1.0),
                    step.regime.current_prob(VolRegime.LOW),
                    step.regime.current_prob(VolRegime.MID),
                    step.regime.current_prob(VolRegime.HIGH),
                    step.regime.next_prob(VolRegime.LOW),
                    step.regime.next_prob(VolRegime.HIGH),
                    min(step.atm_relative_spread, 1.0),
                    0.0,  # margin_utilization
                    0.0,  # drawdown
                    0.0,  # open_fraction
                    1.0 - i / max(1, len(ep)),  # ttx
                    0.0,  # news
                ],
                dtype=np.float32,
            )
            all_obs.append(obs)
            all_returns.append(float(np.log(ep.steps[i + 1].spot / step.spot)))

    obs_matrix = np.vstack(all_obs)
    # Check maximum absolute correlation between any non-constant feature and next return.
    # Constant features (e.g., the hard-coded news flag, zero drawdown) have zero std and
    # produce NaN; we ignore them rather than treating that as leakage.
    max_corr = 0.0
    for f in range(obs_matrix.shape[1]):
        feat = obs_matrix[:, f]
        if np.std(feat) < 1e-12:
            continue
        corr = float(np.corrcoef(feat, all_returns)[0, 1])
        if np.isfinite(corr):
            max_corr = max(max_corr, abs(corr))

    # True leakage would give a correlation near 1.0. With random sampling we expect < 0.35.
    assert max_corr < 0.35, f"max observation/next-return correlation = {max_corr:.3f}"


# ----------------------------------------------------------------------------
# Reward and action-space sanity
# ----------------------------------------------------------------------------


def test_flat_action_has_zero_mean_reward_and_no_equity_drift() -> None:
    """FLAT must not be a free lunch: doing nothing should leave equity unchanged on average.

    If FLAT produced positive reward for free, PPO would collapse to never trading.
    """
    env = _make_env(seed=404)
    env.reset()
    flat = _flat_action()
    rewards: list[float] = []
    for _ in range(len(env._episode or ())):
        _, reward, term, trunc, _ = env.step(flat)
        rewards.append(float(reward))
        if term or trunc:
            break

    assert len(rewards) > 0
    mean_reward = float(np.mean(rewards))
    # With no trading, equity should stay at starting cash, so all reward terms are zero.
    # Tolerance allows tiny floating-point noise.
    assert mean_reward == pytest.approx(0.0, abs=1e-8)
    assert env.account_equity() == pytest.approx(1_000_000.0, abs=1e-6)


def test_trade_cvar_penalty_scales_with_quantity() -> None:
    """The incremental-CVaR penalty in the reward must be proportional to position size.

    This catches the corruption where the risk term was only for one condor regardless of
    how many were traded.
    """
    env = _make_env(seed=505)
    env.reset()

    # Trade with full size and record incremental_cVaR from the info breakdown.
    full_action = _trade_action(size=1.0)
    _, _, _, _, info_full = env.step(full_action)
    cvar_full = info_full["reward_breakdown"]["risk"]

    env = _make_env(seed=505)  # same seed -> same episode
    env.reset()
    half_action = _trade_action(size=0.5)
    _, _, _, _, info_half = env.step(half_action)
    cvar_half = info_half["reward_breakdown"]["risk"]

    # The risk term is negative; its magnitude should scale roughly with quantity.
    # We do not require exact proportionality because the condor selected may differ
    # slightly, but the full-size penalty must be larger in magnitude.
    assert cvar_full <= 0.0
    assert cvar_half <= 0.0
    assert abs(cvar_full) > abs(cvar_half)


def test_wild_action_is_clipped_and_does_not_crash() -> None:
    """A completely random or extreme action vector must decode safely."""
    rng = np.random.default_rng(606)
    for _ in range(100):
        raw = rng.normal(size=6)  # can be any real numbers
        decoded = decode_action(raw)
        assert decoded.tail_probability > 0.0
        assert decoded.tail_probability < 0.5
        assert decoded.wing_width_fraction > 0.0
        assert decoded.size_fraction >= 0.0
        assert decoded.size_fraction <= 1.0


# ----------------------------------------------------------------------------
# Safety overlay and risk limits
# ----------------------------------------------------------------------------


def test_kill_switch_halts_trading_after_large_loss() -> None:
    """A large enough drawdown must trigger the kill switch and stop trading."""
    from options_engine.core.config import RiskConfig

    # Use a tiny drawdown limit so any realized loss trips the switch.
    env = IronCondorTradingEnv(
        seed=1,
        risk=RiskConfig(
            max_risk_fraction_per_trade=0.10,
            max_risk_fraction_per_day=0.25,
            max_drawdown_kill_switch=0.001,
            max_leverage=4.0,
        ),
        env_config=EnvConfig(max_open_positions=5, starting_cash=1_000_000.0),
        scenario_config=ScenarioConfig(
            n_steps=12, n_paths=4_000, horizon_days=7.0, steps_per_day=4
        ),
        action_bounds=ActionBounds(
            tail_probability=(0.08, 0.18), wing_width_fraction=(0.025, 0.05)
        ),
    )
    env.reset()

    # Force a sequence of large trades to hit the drawdown limit quickly.
    aggressive = np.array([2.0, 0.0, -1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    terminated = False
    for _ in range(50):
        _, _, term, trunc, info = env.step(aggressive)
        if term or trunc:
            terminated = term
            break

    assert terminated, "kill switch should have triggered after large losses"


def test_environment_is_deterministic_given_seed_and_actions() -> None:
    """Same seed + same action sequence -> same trajectory (reproducibility)."""
    cfg = ScenarioConfig(n_steps=6, n_paths=4_000, horizon_days=6.0, steps_per_day=4)
    env_a = _make_env(seed=808, scenario=cfg)
    env_b = _make_env(seed=808, scenario=cfg)

    rng = np.random.default_rng(909)
    actions = [rng.normal(size=6).astype(np.float32) for _ in range(20)]

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)

    for action in actions:
        o_a, r_a, t_a, c_a, _ = env_a.step(action)
        o_b, r_b, t_b, c_b, _ = env_b.step(action)
        np.testing.assert_array_equal(o_a, o_b)
        assert r_a == pytest.approx(r_b, abs=1e-9)
        assert t_a == t_b
        assert c_a == c_b
        if t_a or c_a:
            break


# ----------------------------------------------------------------------------
# Strategic action decoding
# ----------------------------------------------------------------------------


def test_strategic_argmax_selects_correct_action() -> None:
    """The strategic head must reliably decode to the intended action."""
    theta = np.array([2.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    gamma = np.array([-1.0, 2.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    flat = np.array([-1.0, -1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)

    assert decode_action(theta).strategic is StrategicAction.HARVEST_THETA
    assert decode_action(gamma).strategic is StrategicAction.HARVEST_GAMMA
    assert decode_action(flat).strategic is StrategicAction.FLAT
