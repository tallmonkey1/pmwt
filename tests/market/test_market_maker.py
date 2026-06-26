"""Tests for the Avellaneda-Stoikov market maker."""

from __future__ import annotations

import pytest

from options_engine.core.enums import OptionRight
from options_engine.core.errors import ValidationError
from options_engine.market.market_maker import (
    AvellanedaStoikovMaker,
    MarketMakerConfig,
    ObligationConfig,
)
from options_engine.pricing.instruments import EuropeanOption


def _option() -> EuropeanOption:
    return EuropeanOption(strike=100.0, expiry=0.05, right=OptionRight.CALL)


class TestConfigValidation:
    def test_rejects_nonpositive_risk_aversion(self) -> None:
        with pytest.raises(ValidationError):
            MarketMakerConfig(risk_aversion=0.0)

    def test_rejects_small_base_size(self) -> None:
        with pytest.raises(ValidationError):
            MarketMakerConfig(base_size=0)

    def test_obligation_rejects_bad_values(self) -> None:
        with pytest.raises(ValidationError):
            ObligationConfig(max_relative_spread=0.0)
        with pytest.raises(ValidationError):
            ObligationConfig(min_size=0)


class TestQuoting:
    def test_quote_is_valid_market(self) -> None:
        mk = AvellanedaStoikovMaker(tick_size=0.05)
        qo = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.3, atm_delta_distance=0.0
        )
        assert qo.quote.bid < qo.quote.ask
        assert qo.quote.bid >= 0.0

    def test_wing_relative_spread_wider_than_atm(self) -> None:
        mk = AvellanedaStoikovMaker(tick_size=0.05)
        atm = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.3, atm_delta_distance=0.0
        )
        wing = mk.quote(
            _option(), theoretical_value=0.2, value_volatility=0.3, atm_delta_distance=0.45
        )
        assert wing.quote.relative_spread > atm.quote.relative_spread

    def test_wing_size_smaller_than_atm(self) -> None:
        mk = AvellanedaStoikovMaker(tick_size=0.05)
        atm = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.3, atm_delta_distance=0.0
        )
        wing = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.3, atm_delta_distance=0.45
        )
        assert wing.quote.bid_size < atm.quote.bid_size

    def test_long_inventory_skews_quotes_down(self) -> None:
        # A maker long inventory lowers its reservation price to encourage selling.
        mk = AvellanedaStoikovMaker(tick_size=0.01)
        flat = mk.quote(
            _option(),
            theoretical_value=2.0,
            value_volatility=0.3,
            atm_delta_distance=0.0,
            inventory=0,
        )
        long = mk.quote(
            _option(),
            theoretical_value=2.0,
            value_volatility=0.3,
            atm_delta_distance=0.0,
            inventory=30,
        )
        assert long.quote.mid < flat.quote.mid

    def test_short_inventory_skews_quotes_up(self) -> None:
        mk = AvellanedaStoikovMaker(tick_size=0.01)
        flat = mk.quote(
            _option(),
            theoretical_value=2.0,
            value_volatility=0.3,
            atm_delta_distance=0.0,
            inventory=0,
        )
        short = mk.quote(
            _option(),
            theoretical_value=2.0,
            value_volatility=0.3,
            atm_delta_distance=0.0,
            inventory=-30,
        )
        assert short.quote.mid > flat.quote.mid

    def test_obligation_caps_spread(self) -> None:
        # Even an extremely risk-averse maker cannot exceed the obligation spread cap.
        mk = AvellanedaStoikovMaker(
            config=MarketMakerConfig(risk_aversion=50.0),
            obligations=ObligationConfig(max_relative_spread=0.2, max_absolute_spread=100.0),
        )
        qo = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=1.0, atm_delta_distance=0.0
        )
        assert qo.quote.spread <= 0.2 * 2.0 + 1e-9

    def test_min_size_floor_respected(self) -> None:
        mk = AvellanedaStoikovMaker(
            config=MarketMakerConfig(base_size=5, wing_size_decay=20.0),
            obligations=ObligationConfig(min_size=2),
        )
        wing = mk.quote(
            _option(), theoretical_value=0.1, value_volatility=0.3, atm_delta_distance=0.5
        )
        assert wing.quote.bid_size >= 2

    def test_quotes_rounded_to_tick(self) -> None:
        mk = AvellanedaStoikovMaker(tick_size=0.05)
        qo = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.3, atm_delta_distance=0.0
        )
        assert abs((qo.quote.bid / 0.05) - round(qo.quote.bid / 0.05)) < 1e-9
        assert abs((qo.quote.ask / 0.05) - round(qo.quote.ask / 0.05)) < 1e-9

    def test_higher_value_vol_widens_spread(self) -> None:
        mk = AvellanedaStoikovMaker(
            tick_size=0.01, obligations=ObligationConfig(max_relative_spread=10.0)
        )
        low = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.1, atm_delta_distance=0.0
        )
        high = mk.quote(
            _option(), theoretical_value=2.0, value_volatility=0.5, atm_delta_distance=0.0
        )
        assert high.quote.spread > low.quote.spread

    def test_rejects_negative_theoretical(self) -> None:
        mk = AvellanedaStoikovMaker()
        with pytest.raises(ValidationError):
            mk.quote(
                _option(), theoretical_value=-1.0, value_volatility=0.3, atm_delta_distance=0.0
            )
