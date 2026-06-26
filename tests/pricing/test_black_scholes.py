"""Tests for analytic Black-Scholes pricing and Greeks (validation oracle)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ValidationError
from options_engine.pricing import black_scholes as bs


class TestPrice:
    def test_put_call_parity(self) -> None:
        s, k, t, v, r, q = 100.0, 105.0, 0.5, 0.2, 0.03, 0.01
        call = float(bs.price(s, k, t, v, OptionRight.CALL, rate=r, dividend=q)[0])
        put = float(bs.price(s, k, t, v, OptionRight.PUT, rate=r, dividend=q)[0])
        lhs = call - put
        rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_prices_non_negative(self) -> None:
        s = np.array([50.0, 100.0, 150.0])
        for right in (OptionRight.CALL, OptionRight.PUT):
            p = bs.price(s, 100.0, 1.0, 0.2, right)
            assert np.all(p >= 0.0)

    def test_zero_vol_is_discounted_intrinsic(self) -> None:
        # With zero vol, a call worth max(F - K, 0) discounted.
        s, k, t, r = 110.0, 100.0, 1.0, 0.05
        call = float(bs.price(s, k, t, 0.0, OptionRight.CALL, rate=r)[0])
        forward = s * np.exp(r * t)
        expected = np.exp(-r * t) * max(forward - k, 0.0)
        assert call == pytest.approx(expected, abs=1e-9)

    def test_deep_itm_call_approaches_forward_minus_strike(self) -> None:
        s, k, t, r = 1000.0, 100.0, 1.0, 0.0
        call = float(bs.price(s, k, t, 0.2, OptionRight.CALL, rate=r)[0])
        assert call == pytest.approx(s - k, rel=1e-6)

    def test_vectorized_broadcast(self) -> None:
        strikes = np.array([90.0, 100.0, 110.0])
        prices = bs.price(100.0, strikes, 1.0, 0.2, OptionRight.CALL)
        assert prices.shape == (3,)
        # Calls are decreasing in strike.
        assert np.all(np.diff(prices) < 0.0)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValidationError):
            bs.price(-1.0, 100.0, 1.0, 0.2, OptionRight.CALL)
        with pytest.raises(ValidationError):
            bs.price(100.0, 100.0, -1.0, 0.2, OptionRight.CALL)
        with pytest.raises(ValidationError):
            bs.price(100.0, 100.0, 1.0, -0.2, OptionRight.CALL)


class TestGreeks:
    @pytest.mark.parametrize("right", [OptionRight.CALL, OptionRight.PUT])
    def test_delta_matches_finite_difference(self, right: OptionRight) -> None:
        s, k, t, v, r, q = 100.0, 105.0, 0.5, 0.25, 0.02, 0.01
        g = bs.greeks(s, k, t, v, right, rate=r, dividend=q)
        h = 1e-4
        fd = (
            float(bs.price(s + h, k, t, v, right, rate=r, dividend=q)[0])
            - float(bs.price(s - h, k, t, v, right, rate=r, dividend=q)[0])
        ) / (2 * h)
        assert float(g.delta[0]) == pytest.approx(fd, abs=1e-5)

    def test_gamma_matches_finite_difference(self) -> None:
        s, k, t, v = 100.0, 100.0, 0.5, 0.25
        g = bs.greeks(s, k, t, v, OptionRight.CALL)
        h = 1e-3
        base = float(bs.price(s, k, t, v, OptionRight.CALL)[0])
        up = float(bs.price(s + h, k, t, v, OptionRight.CALL)[0])
        down = float(bs.price(s - h, k, t, v, OptionRight.CALL)[0])
        fd = (up - 2 * base + down) / (h * h)
        assert float(g.gamma[0]) == pytest.approx(fd, rel=1e-3)

    def test_vega_matches_finite_difference(self) -> None:
        s, k, t, v = 100.0, 105.0, 0.5, 0.25
        g = bs.greeks(s, k, t, v, OptionRight.CALL)
        h = 1e-5
        fd = (
            float(bs.price(s, k, t, v + h, OptionRight.CALL)[0])
            - float(bs.price(s, k, t, v - h, OptionRight.CALL)[0])
        ) / (2 * h)
        assert float(g.vega[0]) == pytest.approx(fd, rel=1e-4)

    def test_theta_matches_finite_difference(self) -> None:
        # Theta is the derivative w.r.t. *calendar time*, i.e. the negative of the partial
        # derivative w.r.t. time-to-expiry. Long options therefore have negative theta.
        s, k, t, v, r = 100.0, 100.0, 0.5, 0.25, 0.03
        g = bs.greeks(s, k, t, v, OptionRight.CALL, rate=r)
        h = 1e-5
        d_price_d_expiry = (
            float(bs.price(s, k, t + h, v, OptionRight.CALL, rate=r)[0])
            - float(bs.price(s, k, t - h, v, OptionRight.CALL, rate=r)[0])
        ) / (2 * h)
        assert float(g.theta[0]) == pytest.approx(-d_price_d_expiry, abs=1e-3)
        assert float(g.theta[0]) < 0.0  # long option decays

    def test_gamma_vega_equal_for_call_and_put(self) -> None:
        s, k, t, v = 100.0, 105.0, 0.5, 0.25
        gc = bs.greeks(s, k, t, v, OptionRight.CALL)
        gp = bs.greeks(s, k, t, v, OptionRight.PUT)
        assert float(gc.gamma[0]) == pytest.approx(float(gp.gamma[0]))
        assert float(gc.vega[0]) == pytest.approx(float(gp.vega[0]))

    def test_call_put_delta_relation(self) -> None:
        # delta_call - delta_put = e^{-qT}.
        s, k, t, v, q = 100.0, 105.0, 0.5, 0.25, 0.02
        dc = float(bs.greeks(s, k, t, v, OptionRight.CALL, dividend=q).delta[0])
        dp = float(bs.greeks(s, k, t, v, OptionRight.PUT, dividend=q).delta[0])
        assert dc - dp == pytest.approx(np.exp(-q * t), abs=1e-9)

    def test_degenerate_greeks_vanish(self) -> None:
        g = bs.greeks(100.0, 105.0, 0.5, 0.0, OptionRight.CALL)
        assert float(g.gamma[0]) == 0.0
        assert float(g.vega[0]) == 0.0


class TestImpliedVolatility:
    @pytest.mark.parametrize("right", [OptionRight.CALL, OptionRight.PUT])
    @pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
    def test_round_trip(self, right: OptionRight, vol: float) -> None:
        s, k, t, r, q = 100.0, 110.0, 0.5, 0.03, 0.01
        px = float(bs.price(s, k, t, vol, right, rate=r, dividend=q)[0])
        iv = bs.implied_volatility(px, s, k, t, right, rate=r, dividend=q)
        assert iv == pytest.approx(vol, abs=1e-5)

    def test_deep_otm_wing(self) -> None:
        # Deep OTM (tiny vega) is exactly where the safeguarded solver matters.
        s, k, t, vol = 100.0, 200.0, 0.1, 0.4
        px = float(bs.price(s, k, t, vol, OptionRight.CALL)[0])
        iv = bs.implied_volatility(px, s, k, t, OptionRight.CALL)
        assert iv == pytest.approx(vol, abs=1e-3)

    def test_rejects_arbitrage_violating_price(self) -> None:
        with pytest.raises(ValidationError):
            # Price above the underlying is impossible for a call.
            bs.implied_volatility(150.0, 100.0, 100.0, 1.0, OptionRight.CALL)

    @settings(max_examples=40, deadline=None)
    @given(
        vol=st.floats(0.05, 2.0),
        moneyness=st.floats(0.7, 1.4),
        t=st.floats(0.05, 2.0),
    )
    def test_round_trip_property(self, vol: float, moneyness: float, t: float) -> None:
        s = 100.0
        k = s * moneyness
        # Always invert from the OTM side: the IV problem is only well-conditioned where
        # the option has meaningful time value (vega). A deep-ITM option is ~all intrinsic,
        # so its implied vol is numerically unidentifiable -- the standard practice is to
        # use the equivalent OTM contract via put-call parity, which we replicate here.
        right = OptionRight.CALL if k >= s else OptionRight.PUT
        px = float(bs.price(s, k, t, vol, right)[0])
        vega = float(bs.greeks(s, k, t, vol, right).vega[0])
        if vega < 1e-4:  # IV not identifiable when vega is negligible
            return
        iv = bs.implied_volatility(px, s, k, t, right)
        assert iv == pytest.approx(vol, abs=1e-3)
