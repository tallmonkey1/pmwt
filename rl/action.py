r"""Parameterized action space for the RL trading agent (SPEC §4.1).

The agent's action is *hybrid*: a discrete strategic head and a continuous tactical head,
flattened into a single fixed-length real vector so it is compatible with continuous-control
algorithms (PPO, distributional critics). The decoder turns the raw vector into a structured,
validated :class:`DecodedAction` the environment can act on.

Layout of the raw action vector (length :data:`ACTION_DIM`):

==========  =============================================================================
 index      meaning
==========  =============================================================================
 0..2       strategic logits over {HARVEST_THETA, HARVEST_GAMMA, FLAT}; argmax selects
 3          tail-probability knob in [-1, 1] -> short-strike tail probability
 4          wing-width knob in [-1, 1] -> wing width fraction
 5          size-fraction knob in [-1, 1] -> fraction of the Kelly-capped size to deploy
==========  =============================================================================

Everything is deterministic given the raw vector, and every continuous knob is squashed from
the unbounded policy output into a bounded, economically-sensible range, so a wild network
output can never produce an out-of-domain trade -- it is merely clipped to the nearest
admissible action. Hard risk limits still live in the risk supervisor (SPEC §4.5); this layer
only bounds the *parameterization*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.enums import StrategicAction
from ..core.errors import ValidationError

__all__ = ["ACTION_DIM", "N_STRATEGIC", "ActionBounds", "DecodedAction", "decode_action"]

#: Number of discrete strategic choices (HARVEST_THETA, HARVEST_GAMMA, FLAT).
N_STRATEGIC: int = 3

#: Dimensionality of the raw action vector.
ACTION_DIM: int = N_STRATEGIC + 3

_STRATEGIC_ORDER: tuple[StrategicAction, ...] = (
    StrategicAction.HARVEST_THETA,
    StrategicAction.HARVEST_GAMMA,
    StrategicAction.FLAT,
)


@dataclass(frozen=True, slots=True)
class ActionBounds:
    """Admissible ranges for the continuous tactical knobs.

    Each ``[-1, 1]`` knob is affinely mapped into its ``(min, max)`` range. Defaults are
    chosen for short-premium index condors: tail probabilities of 5%-35% per side, wing
    widths of 1%-8% of spot, and a size fraction of 0-100% of the Kelly-capped size.
    """

    tail_probability: tuple[float, float] = (0.05, 0.35)
    wing_width_fraction: tuple[float, float] = (0.01, 0.08)
    size_fraction: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        for name, (lo, hi) in (
            ("tail_probability", self.tail_probability),
            ("wing_width_fraction", self.wing_width_fraction),
            ("size_fraction", self.size_fraction),
        ):
            if not (lo < hi):
                raise ValidationError(
                    f"{name} range must satisfy lo < hi", context={"lo": lo, "hi": hi}
                )
        if self.tail_probability[0] <= 0.0 or self.tail_probability[1] >= 0.5:
            raise ValidationError("tail_probability must lie within (0, 0.5)", context={})
        if self.wing_width_fraction[0] <= 0.0:
            raise ValidationError("wing_width_fraction must be positive", context={})
        if self.size_fraction[0] < 0.0:
            raise ValidationError("size_fraction must be non-negative", context={})


@dataclass(frozen=True, slots=True)
class DecodedAction:
    """A validated, structured action ready for the environment to execute."""

    strategic: StrategicAction
    tail_probability: float
    wing_width_fraction: float
    size_fraction: float

    @property
    def is_trade(self) -> bool:
        """True if the strategic head requests opening/harvesting (not FLAT)."""
        return self.strategic is not StrategicAction.FLAT


def _affine_unit_to_range(knob: float, lo: float, hi: float) -> float:
    """Map a knob in ``[-1, 1]`` to ``[lo, hi]`` (clamping out-of-range inputs)."""
    clamped = float(np.clip(knob, -1.0, 1.0))
    unit = 0.5 * (clamped + 1.0)  # [-1, 1] -> [0, 1]
    return lo + unit * (hi - lo)


def decode_action(
    raw: NDArray[np.float64], *, bounds: ActionBounds | None = None
) -> DecodedAction:
    """Decode a raw policy action vector into a structured, bounded :class:`DecodedAction`.

    Parameters
    ----------
    raw:
        The policy output, shape ``(ACTION_DIM,)``. Non-finite values are rejected; finite
        but out-of-range continuous knobs are clipped (never an error), so the environment is
        robust to an untrained or exploratory policy.
    bounds:
        The admissible tactical ranges; defaults to :class:`ActionBounds`.
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (ACTION_DIM,):
        raise ValidationError(
            "action must have shape (ACTION_DIM,)",
            context={"shape": arr.shape, "expected": ACTION_DIM},
        )
    if not np.all(np.isfinite(arr)):
        raise ValidationError("action contains non-finite values", context={})

    b = bounds or ActionBounds()
    strategic_logits = arr[:N_STRATEGIC]
    strategic = _STRATEGIC_ORDER[int(np.argmax(strategic_logits))]

    tail = _affine_unit_to_range(arr[N_STRATEGIC], *b.tail_probability)
    wing = _affine_unit_to_range(arr[N_STRATEGIC + 1], *b.wing_width_fraction)
    size = _affine_unit_to_range(arr[N_STRATEGIC + 2], *b.size_fraction)

    return DecodedAction(
        strategic=strategic,
        tail_probability=tail,
        wing_width_fraction=wing,
        size_fraction=size,
    )
