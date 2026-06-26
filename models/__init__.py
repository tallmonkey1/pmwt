"""Quantitative models: price/volatility simulation, pricing, calibration, distribution.

See ``SPEC.md`` §2 and §7. This phase implements the rough-Bergomi volatility/price
simulator (:mod:`options_engine.models.rbergomi`); later phases add pricing, calibration,
the distributional surrogate, jumps, and the dynamic drift model.
"""

from __future__ import annotations
