r"""Observation construction for the RL agent (SPEC §4.3).

The observation is the point-in-time, leakage-free feature vector the policy conditions on.
It is assembled from quantities the rest of the engine already computes causally:

* **Distribution features** -- model win-probability proxies and tail shape from the terminal
  distribution (the "edge" the agent is trading).
* **Regime features** -- calibrated low/mid/high probabilities now and next step.
* **Microstructure features** -- the current at-the-money relative spread (the friction the
  agent must respect).
* **Portfolio features** -- normalized margin usage, drawdown, open-position count.
* **Calendar features** -- normalized time-to-expiry and a news-cool-off flag.

Every feature is bounded and finite by construction; the observation space declares matching
bounds so a conformant RL library can validate it. The builder never reads any future
information -- it is the same data the strategy layer would see live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.enums import VolRegime
from ..core.errors import ValidationError
from ..regime.detector import RegimeNowcast

__all__ = ["OBSERVATION_DIM", "ObservationInputs", "build_observation", "observation_bounds"]

#: Dimensionality of the observation vector. Kept explicit so the env and tests stay in sync.
OBSERVATION_DIM: int = 15


@dataclass(frozen=True, slots=True)
class ObservationInputs:
    """The causal quantities required to build one observation."""

    # Distribution / edge.
    prob_in_one_sigma: float  # P(terminal in +/- 1 model-sigma)
    left_tail_prob: float  # P(terminal below the 5th percentile threshold scale)
    right_tail_prob: float  # symmetric upper tail mass proxy
    expected_move: float  # model expected absolute move (std of log-return)
    # Regime.
    regime: RegimeNowcast
    # Microstructure.
    atm_relative_spread: float
    # Portfolio.
    margin_utilization: float  # in [0, 1+]: total margin / equity
    drawdown: float  # in [0, 1]
    open_position_fraction: float  # open positions / max positions, in [0, 1]
    # Calendar / event.
    time_to_expiry_fraction: float  # remaining fraction of the trade horizon, in [0, 1]
    news_cooloff_active: bool


def _finite(value: float, *, name: str) -> float:
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite", context={"name": name, "value": value})
    return float(value)


def build_observation(inputs: ObservationInputs) -> NDArray[np.float32]:
    """Return the fixed-length observation vector (float32) for the policy.

    Bounds are enforced by clipping each feature into its declared range, guaranteeing the
    observation always lies inside :func:`observation_bounds` regardless of upstream noise.
    """
    regime = inputs.regime
    low_now = regime.current_prob(VolRegime.LOW)
    mid_now = regime.current_prob(VolRegime.MID)
    high_now = regime.current_prob(VolRegime.HIGH)
    low_next = regime.next_prob(VolRegime.LOW)
    high_next = regime.next_prob(VolRegime.HIGH)

    features = np.array(
        [
            np.clip(_finite(inputs.prob_in_one_sigma, name="prob_in_one_sigma"), 0.0, 1.0),
            np.clip(_finite(inputs.left_tail_prob, name="left_tail_prob"), 0.0, 1.0),
            np.clip(_finite(inputs.right_tail_prob, name="right_tail_prob"), 0.0, 1.0),
            np.clip(_finite(inputs.expected_move, name="expected_move"), 0.0, 1.0),
            np.clip(low_now, 0.0, 1.0),
            np.clip(mid_now, 0.0, 1.0),
            np.clip(high_now, 0.0, 1.0),
            np.clip(low_next, 0.0, 1.0),
            np.clip(high_next, 0.0, 1.0),
            np.clip(_finite(inputs.atm_relative_spread, name="atm_relative_spread"), 0.0, 1.0),
            np.clip(_finite(inputs.margin_utilization, name="margin_utilization"), 0.0, 2.0),
            np.clip(_finite(inputs.drawdown, name="drawdown"), 0.0, 1.0),
            np.clip(
                _finite(inputs.open_position_fraction, name="open_position_fraction"), 0.0, 1.0
            ),
            np.clip(
                _finite(inputs.time_to_expiry_fraction, name="time_to_expiry_fraction"), 0.0, 1.0
            ),
            np.float32(1.0 if inputs.news_cooloff_active else 0.0),
        ],
        dtype=np.float32,
    )
    if features.shape != (OBSERVATION_DIM,):  # pragma: no cover - guarded by construction
        raise ValidationError(
            "observation has unexpected dimension",
            context={"shape": features.shape, "expected": OBSERVATION_DIM},
        )
    return features


def observation_bounds() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return ``(low, high)`` bounds matching :func:`build_observation`.

    Most features are probabilities in ``[0, 1]``; margin utilization is allowed up to ``2``
    to represent (capped) leverage. These bounds are used to construct the Gymnasium
    observation space.
    """
    low = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    high = np.ones(OBSERVATION_DIM, dtype=np.float32)
    high[10] = 2.0  # margin_utilization
    return low, high
