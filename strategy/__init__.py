"""Strategy layer: condor construction, sizing, entry/exit, and risk supervision.

Brings together the distribution, regime gate, news gate, and quoted chain into concrete
trading decisions, all bounded by the deterministic risk supervisor (SPEC §4.5, §5, §6).

Public surface:

* State: :class:`Account`, :class:`OpenPosition`.
* Construction: :func:`select_iron_condor`, :class:`CondorCandidate`,
  :class:`CondorSelectionConfig`.
* Sizing: :func:`size_position`, :func:`kelly_fraction`, :class:`SizingInputs`,
  :class:`SizingResult`.
* Risk supervisor: :class:`RiskSupervisor`, :class:`RiskSupervisorConfig`,
  :class:`RiskCheckResult`.
* Entry: :class:`EntryEvaluator`, :class:`EntryConfig`, :class:`EntryDecision`.
* Exit: :func:`evaluate_exit`, :class:`ExitConfig`, :class:`ExitDecision`.
"""

from __future__ import annotations

from .account import Account, OpenPosition
from .condor_selection import (
    CondorCandidate,
    CondorSelectionConfig,
    select_iron_condor,
)
from .entry import EntryConfig, EntryDecision, EntryEvaluator
from .exit import ExitConfig, ExitDecision, evaluate_exit
from .risk_supervisor import RiskCheckResult, RiskSupervisor, RiskSupervisorConfig
from .sizing import SizingInputs, SizingResult, kelly_fraction, size_position

__all__ = [
    "Account",
    "CondorCandidate",
    "CondorSelectionConfig",
    "EntryConfig",
    "EntryDecision",
    "EntryEvaluator",
    "ExitConfig",
    "ExitDecision",
    "OpenPosition",
    "RiskCheckResult",
    "RiskSupervisor",
    "RiskSupervisorConfig",
    "SizingInputs",
    "SizingResult",
    "evaluate_exit",
    "kelly_fraction",
    "select_iron_condor",
    "size_position",
]
