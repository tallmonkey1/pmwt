"""Option pricing, Greeks, and payoff aggregation (SPEC §2.2, §5).

Public surface:

* Instruments: :class:`EuropeanOption`, :class:`OptionLeg`, :class:`IronCondor`.
* Analytic Black-Scholes: :func:`bs_price`, :func:`bs_greeks`, :func:`implied_volatility`,
  :class:`BlackScholesGreeks`.
* Payoffs: :func:`option_payoff`, :func:`iron_condor_payoff`, :func:`iron_condor_pnl`.
* Monte-Carlo pricing off the rBergomi distribution: :func:`price_option`,
  :func:`price_iron_condor`, :func:`fair_iron_condor_credit`.
* Net Greeks for structures: :class:`NetGreeks`, :func:`iron_condor_greeks`.
"""

from __future__ import annotations

from .black_scholes import BlackScholesGreeks, implied_volatility
from .black_scholes import greeks as bs_greeks
from .black_scholes import price as bs_price
from .instruments import EuropeanOption, IronCondor, OptionLeg
from .monte_carlo import (
    fair_iron_condor_credit,
    price_iron_condor,
    price_option,
)
from .payoff import (
    iron_condor_payoff,
    iron_condor_pnl,
    leg_payoff,
    option_payoff,
)
from .portfolio import NetGreeks, iron_condor_greeks, leg_net_greeks

__all__ = [
    "BlackScholesGreeks",
    "EuropeanOption",
    "IronCondor",
    "NetGreeks",
    "OptionLeg",
    "bs_greeks",
    "bs_price",
    "fair_iron_condor_credit",
    "implied_volatility",
    "iron_condor_greeks",
    "iron_condor_payoff",
    "iron_condor_pnl",
    "leg_net_greeks",
    "leg_payoff",
    "option_payoff",
    "price_iron_condor",
    "price_option",
]
