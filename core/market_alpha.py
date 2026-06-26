"""Market alpha: a multi-dimensional calmness signal in [0, 1]^K.

The engine's market model is parameterised by a :class:`MarketAlpha` -- a tuple of
unit-interval features that describe how "intense" each model dimension is:

* ``alpha[0]`` -- **overall calmness**. ``alpha[0] = 1`` means a smooth,
  near-Black-Scholes world (Hurst ~ 0.5, no jumps, no noise); ``alpha[0] = 0``
  means maximally rough volatility (Hurst ~ 0.05), frequent jumps and shocks.
* ``alpha[1]`` -- **Avellaneda-Stoikov noise suppression**. Higher values attenuate
  the additive quote noise the market maker injects; ``1`` is silent quotes,
  ``0`` is maximum noise.
* ``alpha[2]`` -- **drift-noise suppression**. Same convention for the multiplicative
  drift noise injected under the physical measure.
* ``alpha[3]`` -- **jump suppression**. Higher values turn jumps off and shrink their
  size; ``1`` is no-jump Merton dynamics.
* ``alpha[4]`` -- **shock suppression**. Higher values shrink the probability and
  magnitude of exogenous price shocks.

The **helper critic** agent (see :mod:`options_engine.agent.helper_critic`) learns to
find the alpha vector that drives every internal diagnostic of the main PPO-Transformer
agent toward ``1`` -- a deliberately calm, predictable, fully-profitable, stable market.
In the limit where every internal feature is exactly ``1``, the main agent sees the
easiest possible market and its reward is maximised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .validation import check_unit_interval

__all__ = [
    "ALPHA_DIM",
    "DEFAULT_ALPHA_DIM",
    "MarketAlpha",
    "alpha_components",
    "alpha_to_drift_noise",
    "alpha_to_eta",
    "alpha_to_hurst",
    "alpha_to_jump_intensity",
    "alpha_to_jump_size",
    "alpha_to_shock_intensity",
    "alpha_to_stoikov_noise",
]

#: Number of named alpha components (each in [0, 1]). The order is fixed:
#: 0=overall_calmness, 1=stoikov_noise_suppression, 2=drift_noise_suppression,
#: 3=jump_suppression, 4=shock_suppression.
DEFAULT_ALPHA_DIM: Final[int] = 5

#: Alias used elsewhere in the codebase for the alpha dimension.
ALPHA_DIM: Final[int] = DEFAULT_ALPHA_DIM

_ALPHA_NAMES: Final[tuple[str, ...]] = (
    "overall_calmness",
    "stoikov_noise_suppression",
    "drift_noise_suppression",
    "jump_suppression",
    "shock_suppression",
)


@dataclass(frozen=True, slots=True)
class MarketAlpha:
    """A multi-dimensional calmness vector with each feature in ``[0, 1]``."""

    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.features:
            raise ValidationError("alpha must have at least one feature", context={})
        if len(self.features) > DEFAULT_ALPHA_DIM:
            raise ValidationError(
                f"alpha cannot have more than {DEFAULT_ALPHA_DIM} features",
                context={"got": len(self.features), "max": DEFAULT_ALPHA_DIM},
            )
        for i, f in enumerate(self.features):
            check_unit_interval(float(f), name=f"alpha[{i}]", inclusive=True)

    # ---- factories --------------------------------------------------------------

    @classmethod
    def scalar(cls, value: float) -> MarketAlpha:
        """Build a length-1 alpha. ``value`` must be in ``[0, 1]``."""
        return cls(features=(float(value),))

    @classmethod
    def zeros(cls) -> MarketAlpha:
        """The maximally-rough, maximally-noisy alpha (alpha = (0, 0, 0, 0, 0))."""
        return cls(features=tuple([0.0] * DEFAULT_ALPHA_DIM))

    @classmethod
    def ones(cls) -> MarketAlpha:
        """The fully-calm, fully-stable alpha (alpha = (1, 1, 1, 1, 1))."""
        return cls(features=tuple([1.0] * DEFAULT_ALPHA_DIM))

    @classmethod
    def from_components(
        cls,
        *,
        overall_calmness: float = 1.0,
        stoikov_noise_suppression: float = 1.0,
        drift_noise_suppression: float = 1.0,
        jump_suppression: float = 1.0,
        shock_suppression: float = 1.0,
    ) -> MarketAlpha:
        return cls(
            features=(
                float(overall_calmness),
                float(stoikov_noise_suppression),
                float(drift_noise_suppression),
                float(jump_suppression),
                float(shock_suppression),
            )
        )

    # ---- accessors --------------------------------------------------------------

    @property
    def is_scalar(self) -> bool:
        """True if this alpha has a single component."""
        return len(self.features) == 1

    @property
    def scalar_value(self) -> float:
        """Return the scalar alpha value (only valid when ``is_scalar``)."""
        if not self.is_scalar:
            raise ValidationError(
                "alpha is not scalar", context={"dim": len(self.features)}
            )
        return float(self.features[0])

    def as_array(self) -> NDArray[np.float64]:
        """Return the alpha as a 1-D NumPy array of length ``len(self)``."""
        return np.asarray(self.features, dtype=np.float64)

    def padded(self) -> MarketAlpha:
        """Right-pad a short alpha to the default dimension with the value ``1`` (calm)."""
        if len(self.features) == DEFAULT_ALPHA_DIM:
            return self
        pad_value = 1.0
        padded_features = tuple(self.features) + (pad_value,) * (
            DEFAULT_ALPHA_DIM - len(self.features)
        )
        return MarketAlpha(features=padded_features)

    def clipped(self, *, lo: float = 0.0, hi: float = 1.0) -> MarketAlpha:
        """Return a copy with every component clipped to ``[lo, hi]``."""
        return MarketAlpha(
            features=tuple(float(np.clip(f, lo, hi)) for f in self.features)
        )

    def __getitem__(self, idx: int) -> float:
        """Return the ``idx``-th feature, padding with ``1.0`` if absent."""
        if 0 <= idx < len(self.features):
            return float(self.features[idx])
        if 0 <= idx < DEFAULT_ALPHA_DIM:
            return 1.0
        raise IndexError(f"alpha index {idx} out of range [0, {DEFAULT_ALPHA_DIM})")

    def __len__(self) -> int:
        return len(self.features)

    def __str__(self) -> str:
        names = _ALPHA_NAMES[: len(self.features)]
        parts = ", ".join(f"{name}={f:.3f}" for name, f in zip(names, self.features))
        return f"MarketAlpha({parts})"


# ----------------------------------------------------------------------------
# alpha -> model-parameter mappings
# ----------------------------------------------------------------------------

_HURST_MAX: float = 0.49  # stay strictly inside rBergomi's (0, 0.5) open interval
_HURST_MIN: float = 0.05
_ETA_MAX: float = 2.0
_ETA_MIN: float = 0.1
_STOIKOV_NOISE_MAX: float = 0.5
_DRIFT_NOISE_MAX: float = 0.1
_JUMP_LAMBDA_MAX: float = 2.0
_JUMP_SIGMA_MAX: float = 0.2
_SHOCK_PROB_MAX: float = 0.5
_SHOCK_SIZE_MAX: float = 0.3


def alpha_components(alpha: MarketAlpha) -> dict[str, float]:
    """Return the named alpha components as a dict (missing components -> ``1``)."""
    return {name: alpha[i] for i, name in enumerate(_ALPHA_NAMES)}


def alpha_to_hurst(alpha: MarketAlpha) -> float:
    """Map alpha to the rBergomi Hurst exponent.

    ``alpha[0] = 1`` -> ``H = 0.5`` (smooth, BS-like). ``alpha[0] = 0`` -> ``H = 0.05``.
    """
    return _HURST_MIN + alpha[0] * (_HURST_MAX - _HURST_MIN)


def alpha_to_eta(alpha: MarketAlpha) -> float:
    """Map alpha to vol-of-vol ``eta`` (higher when alpha is low)."""
    return _ETA_MIN + (1.0 - alpha[0]) * (_ETA_MAX - _ETA_MIN)


def alpha_to_stoikov_noise(alpha: MarketAlpha) -> float:
    """Map alpha to Avellaneda-Stoikov quote-noise factor (higher when less suppressed)."""
    return max(0.0, (1.0 - alpha[1])) * _STOIKOV_NOISE_MAX


def alpha_to_drift_noise(alpha: MarketAlpha) -> float:
    """Map alpha to multiplicative drift-noise factor (higher when less suppressed)."""
    return max(0.0, (1.0 - alpha[2])) * _DRIFT_NOISE_MAX


def alpha_to_jump_intensity(alpha: MarketAlpha) -> float:
    """Map alpha to Merton jump intensity ``lambda`` (per year)."""
    return max(0.0, (1.0 - alpha[3])) * _JUMP_LAMBDA_MAX


def alpha_to_jump_size(alpha: MarketAlpha) -> float:
    """Map alpha to Merton jump-size standard deviation ``sigma_J``."""
    return max(0.0, (1.0 - alpha[3])) * _JUMP_SIGMA_MAX


def alpha_to_shock_intensity(alpha: MarketAlpha) -> float:
    """Map alpha to exogenous-shock probability per step."""
    return max(0.0, (1.0 - alpha[4])) * _SHOCK_PROB_MAX


def alpha_shock_size_std(alpha: MarketAlpha) -> float:
    """Map alpha to exogenous-shock size standard deviation."""
    return max(0.0, (1.0 - alpha[4])) * _SHOCK_SIZE_MAX


def alpha_to_rbergomi_horizon(alpha: MarketAlpha) -> float:
    """Trade horizon the helper critic prefers: long when calm, short when rough."""
    return 7.0 + 7.0 * alpha[0]
