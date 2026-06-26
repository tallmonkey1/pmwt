"""Tests for iron-condor strike selection."""

from __future__ import annotations

import pytest

from options_engine.core.errors import ValidationError
from options_engine.strategy.condor_selection import (
    CondorSelectionConfig,
    select_iron_condor,
)


class TestSelectIronCondor:
    def test_selects_valid_ordered_condor(self, market_setup) -> None:
        cand = select_iron_condor(
            market_setup["dist"],
            market_setup["chain"],
            terminal_sample=market_setup["terminal_sample"],
            config=CondorSelectionConfig(min_win_probability=0.3, min_net_credit=0.01),
        )
        assert cand is not None
        c = cand.condor
        assert c.put_long_strike < c.put_short_strike < c.call_short_strike < c.call_long_strike

    def test_win_probability_reasonable(self, market_setup) -> None:
        cand = select_iron_condor(
            market_setup["dist"],
            market_setup["chain"],
            terminal_sample=market_setup["terminal_sample"],
            config=CondorSelectionConfig(
                target_tail_probability=0.15, min_win_probability=0.3, min_net_credit=0.01
            ),
        )
        assert cand is not None
        # Tail prob 0.15 per side => win prob around 0.7.
        assert 0.55 < cand.win_probability < 0.85

    def test_returns_none_below_win_threshold(self, market_setup) -> None:
        cand = select_iron_condor(
            market_setup["dist"],
            market_setup["chain"],
            terminal_sample=market_setup["terminal_sample"],
            config=CondorSelectionConfig(min_win_probability=0.999),
        )
        assert cand is None

    def test_returns_none_below_credit_threshold(self, market_setup) -> None:
        cand = select_iron_condor(
            market_setup["dist"],
            market_setup["chain"],
            terminal_sample=market_setup["terminal_sample"],
            config=CondorSelectionConfig(min_net_credit=1e6),
        )
        assert cand is None

    def test_spread_cost_nonnegative(self, market_setup) -> None:
        cand = select_iron_condor(
            market_setup["dist"],
            market_setup["chain"],
            terminal_sample=market_setup["terminal_sample"],
            config=CondorSelectionConfig(min_win_probability=0.3, min_net_credit=0.01),
        )
        assert cand is not None
        assert cand.spread_cost >= 0.0
        assert cand.cvar >= 0.0  # CVaR of a defined-risk short structure is a positive loss

    def test_works_without_terminal_sample(self, market_setup) -> None:
        # Falls back to quantile-grid sampling when no MC sample is supplied.
        cand = select_iron_condor(
            market_setup["dist"],
            market_setup["chain"],
            config=CondorSelectionConfig(min_win_probability=0.3, min_net_credit=0.01),
        )
        assert cand is not None

    def test_config_validation(self) -> None:
        with pytest.raises(ValidationError):
            CondorSelectionConfig(target_tail_probability=0.0)
        with pytest.raises(ValidationError):
            CondorSelectionConfig(wing_width_fraction=0.0)
