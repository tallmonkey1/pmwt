"""Tests for option and iron-condor instrument specifications."""

from __future__ import annotations

import pytest

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ValidationError
from options_engine.pricing.instruments import EuropeanOption, IronCondor, OptionLeg


class TestEuropeanOption:
    def test_valid(self) -> None:
        opt = EuropeanOption(strike=100.0, expiry=0.5, right=OptionRight.CALL)
        assert opt.is_call and not opt.is_put

    def test_put_flags(self) -> None:
        opt = EuropeanOption(strike=100.0, expiry=0.5, right=OptionRight.PUT)
        assert opt.is_put and not opt.is_call

    @pytest.mark.parametrize("strike", [0.0, -1.0])
    def test_invalid_strike(self, strike: float) -> None:
        with pytest.raises(ValidationError):
            EuropeanOption(strike=strike, expiry=0.5, right=OptionRight.CALL)

    def test_invalid_expiry(self) -> None:
        with pytest.raises(ValidationError):
            EuropeanOption(strike=100.0, expiry=0.0, right=OptionRight.CALL)

    def test_invalid_right(self) -> None:
        with pytest.raises(ValidationError):
            EuropeanOption(strike=100.0, expiry=0.5, right="CALL")  # type: ignore[arg-type]

    def test_immutable(self) -> None:
        opt = EuropeanOption(strike=100.0, expiry=0.5, right=OptionRight.CALL)
        with pytest.raises((AttributeError, TypeError)):
            opt.strike = 1.0  # type: ignore[misc]


class TestOptionLeg:
    def test_long_short_flags(self) -> None:
        opt = EuropeanOption(100.0, 0.5, OptionRight.CALL)
        assert OptionLeg(opt, 2).is_long
        assert OptionLeg(opt, -2).is_short

    def test_zero_quantity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptionLeg(EuropeanOption(100.0, 0.5, OptionRight.CALL), 0)

    def test_non_int_quantity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptionLeg(EuropeanOption(100.0, 0.5, OptionRight.CALL), 1.5)  # type: ignore[arg-type]

    def test_bool_quantity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptionLeg(EuropeanOption(100.0, 0.5, OptionRight.CALL), True)  # type: ignore[arg-type]


class TestIronCondor:
    def _condor(self) -> IronCondor:
        return IronCondor(90.0, 95.0, 105.0, 110.0, 0.25)

    def test_valid_construction(self) -> None:
        c = self._condor()
        assert c.put_spread_width == 5.0
        assert c.call_spread_width == 5.0
        assert c.max_spread_width == 5.0
        assert c.profit_zone == (95.0, 105.0)

    def test_asymmetric_widths(self) -> None:
        c = IronCondor(80.0, 95.0, 105.0, 110.0, 0.25)
        assert c.put_spread_width == 15.0
        assert c.call_spread_width == 5.0
        assert c.max_spread_width == 15.0

    def test_strike_ordering_enforced(self) -> None:
        with pytest.raises(ValidationError):
            IronCondor(95.0, 90.0, 105.0, 110.0, 0.25)  # put_long > put_short
        with pytest.raises(ValidationError):
            IronCondor(90.0, 95.0, 95.0, 110.0, 0.25)  # short strikes equal

    def test_invalid_quantity(self) -> None:
        with pytest.raises(ValidationError):
            IronCondor(90.0, 95.0, 105.0, 110.0, 0.25, quantity=0)
        with pytest.raises(ValidationError):
            IronCondor(90.0, 95.0, 105.0, 110.0, 0.25, quantity=-1)

    def test_legs_structure(self) -> None:
        legs = self._condor().legs()
        assert len(legs) == 4
        put_long, put_short, call_short, call_long = legs
        # Wings long, inner short (the canonical short condor).
        assert put_long.is_long and put_long.option.is_put
        assert put_short.is_short and put_short.option.is_put
        assert call_short.is_short and call_short.option.is_call
        assert call_long.is_long and call_long.option.is_call

    def test_legs_scale_with_quantity(self) -> None:
        c = IronCondor(90.0, 95.0, 105.0, 110.0, 0.25, quantity=3)
        for leg in c.legs():
            assert abs(leg.quantity) == 3
