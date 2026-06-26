"""Operational-mode services: the fail-closed broker factory and runners (SPEC §1.2, §7).

Public surface:

* :func:`create_broker` -- the single, fail-closed broker factory keyed off the operational
  mode (simulated for backtest/paper; live only through the full safety gauntlet).
"""

from __future__ import annotations

from .runner import create_broker

__all__ = ["create_broker"]
