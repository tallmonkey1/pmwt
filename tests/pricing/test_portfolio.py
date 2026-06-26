"""Tests for net-Greeks aggregation of multi-leg structures."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ValidationError
from options_engine.pricing import black_scholes as bs
from options_engine.pricing.instruments import EuropeanOption, IronCondor, OptionLeg
from options_engine.pricing.portfolio import (
    NetGreeks,
    iron_condor_greeks,
    leg_net_greeks,
)


class TestNetGreeks:
    def test_addition(self) -> None:
        a = NetGreeks(price=1.0, delta=0.5, gamma=0.1, vega=2.0, theta=-0.3, rho=0.4)
        b = NetGreeks(price=2.0, delta=-0.2, gamma=0.05, vega=1.0, theta=-0.1, rho=0.2)
        c = a + b
        assert c.price == 3.0
        assert c.delta == pytest.approx(0.3)
        assert c.gamma == pytest.approx(0.15)
        assert c.theta == pytest.approx(-0.4)


class TestLegNetGreeks:
    def test_short_leg_flips_sign(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.CALL)
        long_leg = leg_net_greeks(OptionLeg(opt, 1), spot=100.0, vol=0.2, multiplier=1.0)
        short_leg = leg_net_greeks(OptionLeg(opt, -1), spot=100.0, vol=0.2, multiplier=1.0)
        assert short_leg.delta == pytest.approx(-long_leg.delta)
        assert short_leg.gamma == pytest.approx(-long_leg.gamma)
        assert short_leg.vega == pytest.approx(-long_leg.vega)

    def test_multiplier_and_quantity_scale(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.CALL)
        g1 = leg_net_greeks(OptionLeg(opt, 1), spot=100.0, vol=0.2, multiplier=1.0)
        g = leg_net_greeks(OptionLeg(opt, 3), spot=100.0, vol=0.2, multiplier=100.0)
        assert g.delta == pytest.approx(g1.delta * 300.0)

    def test_matches_single_option_greeks(self) -> None:
        opt = EuropeanOption(105.0, 0.5, OptionRight.PUT)
        leg = leg_net_greeks(OptionLeg(opt, 1), spot=100.0, vol=0.25, multiplier=1.0)
        ref = bs.greeks(100.0, 105.0, 0.5, 0.25, OptionRight.PUT)
        assert leg.delta == pytest.approx(float(ref.delta[0]))
        assert leg.theta == pytest.approx(float(ref.theta[0]))


class TestIronCondorGreeks:
    def _condor(self) -> IronCondor:
        return IronCondor(90.0, 95.0, 105.0, 110.0, 0.25)

    def test_short_condor_is_short_gamma_and_vega(self) -> None:
        # A net-short condor (sold) is short gamma and short vega near the centre.
        c = self._condor()
        vols = np.array([0.25, 0.23, 0.23, 0.25])
        g = iron_condor_greeks(c, spot=100.0, leg_vols=vols, multiplier=100.0)
        assert g.gamma < 0.0
        assert g.vega < 0.0
        # And short premium collects positive theta.
        assert g.theta > 0.0

    def test_approximately_delta_neutral_when_symmetric(self) -> None:
        c = self._condor()  # symmetric strikes around spot 100
        vols = np.array([0.22, 0.22, 0.22, 0.22])
        g = iron_condor_greeks(c, spot=100.0, leg_vols=vols, multiplier=1.0)
        # Symmetric condor centred at spot is close to delta neutral.
        assert abs(g.delta) < 0.05

    def test_wrong_number_of_vols_rejected(self) -> None:
        c = self._condor()
        with pytest.raises(ValidationError):
            iron_condor_greeks(c, spot=100.0, leg_vols=np.array([0.2, 0.2, 0.2]))

    def test_equals_sum_of_legs(self) -> None:
        c = self._condor()
        vols = [0.25, 0.23, 0.23, 0.25]
        total = iron_condor_greeks(c, spot=100.0, leg_vols=np.array(vols), multiplier=100.0)
        manual = NetGreeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for leg, vol in zip(c.legs(), vols, strict=True):
            manual = manual + leg_net_greeks(leg, spot=100.0, vol=vol, multiplier=100.0)
        assert total.delta == pytest.approx(manual.delta)
        assert total.gamma == pytest.approx(manual.gamma)
