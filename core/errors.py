"""Structured exception hierarchy for the options engine.

Design principles (institutional-grade requirement: no silent failures, actionable
errors):

* A single root exception, :class:`OptionsEngineError`, lets callers catch *anything*
  this system raises while still allowing fine-grained handling of specific subclasses.
* Every exception carries an optional machine-readable ``context`` mapping so that
  failures can be logged structurally and reconstructed during post-mortem analysis.
* Subclasses are organized by concern (configuration, validation, numerical, data,
  calibration, risk, execution). New modules should reuse these rather than raising bare
  ``ValueError`` / ``RuntimeError``, so that error handling stays consistent across the
  codebase.

The hierarchy intentionally mirrors the module layout described in ``SPEC.md`` §7.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CalibrationError",
    "ConfigurationError",
    "ConvergenceError",
    "DataError",
    "ExecutionError",
    "ModelStateError",
    "NumericalError",
    "OptionsEngineError",
    "RiskLimitError",
    "ValidationError",
]


class OptionsEngineError(Exception):
    """Root of all exceptions raised by the options engine.

    Parameters
    ----------
    message:
        Human-readable, actionable description of what went wrong.
    context:
        Optional structured key/value pairs that aid debugging and structured logging.
        Values should be small and serializable (numbers, strings, short sequences).
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        if not self.context:
            return self.message
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({rendered})"

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{type(self).__name__}(message={self.message!r}, context={self.context!r})"


class ConfigurationError(OptionsEngineError):
    """Raised when configuration is missing, malformed, or internally inconsistent."""


class ValidationError(OptionsEngineError):
    """Raised when an input fails type/range/format/domain validation."""


class NumericalError(OptionsEngineError):
    """Raised on numerical failures (non-finite values, singular matrices, etc.)."""


class ConvergenceError(NumericalError):
    """Raised when an iterative or Monte-Carlo procedure fails to meet its tolerance.

    Carrying the achieved vs. required tolerance in ``context`` is strongly encouraged so
    that operators can decide whether to widen tolerance or increase sample size.
    """


class DataError(OptionsEngineError):
    """Raised on missing, stale, malformed, or schema-violating market/news data."""


class CalibrationError(OptionsEngineError):
    """Raised when model calibration fails or produces out-of-domain parameters."""


class ModelStateError(OptionsEngineError):
    """Raised when a model/component is used in an invalid lifecycle state.

    Example: requesting predictions from an estimator that has not been fitted.
    """


class RiskLimitError(OptionsEngineError):
    """Raised when an action would breach a hard risk limit.

    This is used by the deterministic risk supervisor (SPEC §4.5). It is *not* a bug; it
    is the system correctly refusing to take an unsafe action. Callers must handle it
    explicitly rather than swallowing it.
    """


class ExecutionError(OptionsEngineError):
    """Raised on order-management / broker-interaction failures."""
