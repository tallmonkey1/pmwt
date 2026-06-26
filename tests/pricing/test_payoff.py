"""Tests for terminal payoff functions and iron-condor P&L."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ValidationError
from options_engine.pricing.instruments import EuropeanOption, IronCondor, OptionLeg
from options_engine.pricing.payoff import (
    iron_condor_payoff,
    iron_condor_pnl,
    leg_payoff,
    option_payoff,
)


class TestOptionPayoff:
    def test_call_payoff(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.CALL)
        spot = np.array([80.0, 100.0, 120.0])
        np.testing.assert_array_equal(option_payoff(opt, spot), [0.0, 0.0, 20.0])

    def test_put_payoff(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.PUT)
        spot = np.array([80.0, 100.0, 120.0])
        np.testing.assert_array_equal(option_payoff(opt, spot), [20.0, 0.0, 0.0])

    def test_negative_spot_rejected(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.CALL)
        with pytest.raises(ValidationError):
            option_payoff(opt, np.array([-1.0]))

    def test_leg_payoff_sign(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.CALL)
        short = OptionLeg(opt, -2)
        spot = np.array([120.0])
        np.testing.assert_array_equal(leg_payoff(short, spot), [-40.0])


class TestIronCondorPayoff:
    def _condor(self) -> IronCondor:
        return IronCondor(90.0, 95.0, 105.0, 110.0, 0.25)

    def test_payoff_zero_in_profit_zone(self) -> None:
        c = self._condor()
        spot = np.array([95.0, 100.0, 105.0])
        np.testing.assert_array_equal(iron_condor_payoff(c, spot), [0.0, 0.0, 0.0])

    def test_payoff_capped_below(self) -> None:
        c = self._condor()
        # Far below the long put: loss capped at the put spread width.
        payoff = float(iron_condor_payoff(c, np.array([50.0]))[0])
        assert payoff == pytest.approx(-c.put_spread_width)

    def test_payoff_capped_above(self) -> None:
        c = self._condor()
        payoff = float(iron_condor_payoff(c, np.array([200.0]))[0])
        assert payoff == pytest.approx(-c.call_spread_width)

    def test_payoff_partial_breach(self) -> None:
        c = self._condor()
        # At 92.5 (between long and short put), loss is 95 - 92.5 = 2.5.
        payoff = float(iron_condor_payoff(c, np.array([92.5]))[0])
        assert payoff == pytest.approx(-2.5)

    def test_payoff_is_non_positive(self) -> None:
        c = self._condor()
        spot = np.linspace(50.0, 200.0, 200)
        assert np.all(iron_condor_payoff(c, spot) <= 1e-12)


class TestIronCondorPnL:
    def _condor(self) -> IronCondor:
        return IronCondor(90.0, 95.0, 105.0, 110.0, 0.25)

    def test_max_profit_is_credit(self) -> None:
        c = self._condor()
        pnl = iron_condor_pnl(c, np.array([100.0]), entry_credit=1.2, multiplier=100)
        assert float(pnl[0]) == pytest.approx(120.0)

    def test_max_loss_is_width_minus_credit(self) -> None:
        c = self._condor()
        pnl = iron_condor_pnl(c, np.array([50.0]), entry_credit=1.2, multiplier=100)
        # (1.2 - 5.0) * 100 = -380.
        assert float(pnl[0]) == pytest.approx(-380.0)

    def test_breakeven(self) -> None:
        c = self._condor()
        # Lower breakeven = put_short - credit = 95 - 1.2 = 93.8.
        pnl = iron_condor_pnl(c, np.array([93.8]), entry_credit=1.2, multiplier=1)
        assert float(pnl[0]) == pytest.approx(0.0, abs=1e-9)

    def test_invalid_multiplier(self) -> None:
        c = self._condor()
        with pytest.raises(ValidationError):
            iron_condor_pnl(c, np.array([100.0]), entry_credit=1.0, multiplier=0.0)
