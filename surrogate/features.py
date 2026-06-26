r"""Feature construction and scaling for the distribution surrogate.

The surrogate learns the map

    (rBergomi parameters, horizon)  ->  terminal log-return quantiles.

The raw inputs live on very different scales (``H`` ~ 0.1, ``eta`` ~ 1.5, ``rho`` ~ -0.7,
``xi0`` ~ 0.04, horizon ~ 0.02-0.1 years), so we (a) apply domain-aware transforms that
linearize the model's dependence and (b) standardize to zero mean / unit variance. Both are
essential for stable neural-network training (SPEC §13.9: performance; §2.5: accuracy).

Design choices, justified:

* ``sqrt(xi0 * horizon)`` is the dominant scale of the terminal distribution (it is the
  Black-Scholes total volatility), so we include it explicitly -- this gives the network an
  almost-linear handle on the spread of the distribution and dramatically improves accuracy
  for a given network size.
* The :class:`FeatureScaler` is *fit on training data only* and then frozen, so there is no
  train/serve skew and no data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ModelStateError, ValidationError
from ..core.validation import check_array_finite

__all__ = ["N_FEATURES", "FeatureScaler", "RawInputs", "build_feature_matrix"]

#: Number of engineered features produced by :func:`build_feature_matrix`.
N_FEATURES: int = 6


@dataclass(frozen=True, slots=True)
class RawInputs:
    """A batch of raw model inputs to be turned into surrogate features.

    Each attribute is a 1-D array of equal length (one entry per scenario). Using a typed
    container (rather than a bare matrix) keeps the column meaning unambiguous.
    """

    hurst: NDArray[np.float64]
    eta: NDArray[np.float64]
    rho: NDArray[np.float64]
    xi0: NDArray[np.float64]
    horizon: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = {
            "hurst": self.hurst,
            "eta": self.eta,
            "rho": self.rho,
            "xi0": self.xi0,
            "horizon": self.horizon,
        }
        sizes = set()
        for name, arr in arrays.items():
            a = np.asarray(arr, dtype=np.float64)
            check_array_finite(a, name=name)
            if a.ndim != 1:
                raise ValidationError(f"{name} must be 1-D", context={"ndim": a.ndim})
            sizes.add(a.size)
        if len(sizes) != 1 or 0 in sizes:
            raise ValidationError(
                "all raw input arrays must share a single non-zero length",
                context={"sizes": sorted(sizes)},
            )

    @property
    def n_samples(self) -> int:
        """Number of scenarios in the batch."""
        return int(np.asarray(self.hurst).size)


def build_feature_matrix(inputs: RawInputs) -> NDArray[np.float64]:
    r"""Return the engineered, *unscaled* feature matrix of shape ``(n_samples, N_FEATURES)``.

    Features (columns):

    0. ``hurst``
    1. ``eta``
    2. ``rho``
    3. ``log(xi0)``                  -- variance is positive and varies multiplicatively
    4. ``sqrt(horizon)``             -- diffusive time scaling
    5. ``sqrt(xi0 * horizon)``       -- total Black-Scholes volatility (dominant spread)
    """
    h = np.asarray(inputs.hurst, dtype=np.float64)
    eta = np.asarray(inputs.eta, dtype=np.float64)
    rho = np.asarray(inputs.rho, dtype=np.float64)
    xi0 = np.asarray(inputs.xi0, dtype=np.float64)
    horizon = np.asarray(inputs.horizon, dtype=np.float64)

    if np.any(xi0 <= 0.0):
        raise ValidationError("xi0 must be strictly positive", context={})
    if np.any(horizon <= 0.0):
        raise ValidationError("horizon must be strictly positive", context={})

    features = np.column_stack(
        [
            h,
            eta,
            rho,
            np.log(xi0),
            np.sqrt(horizon),
            np.sqrt(xi0 * horizon),
        ]
    )
    return np.ascontiguousarray(features, dtype=np.float64)


@dataclass
class FeatureScaler:
    """Standardizes features to zero mean / unit variance, fit on training data only.

    Stateful by design: it is ``fit`` once on the training matrix, then ``transform`` is
    applied identically at train and serve time. Attempting to ``transform`` before ``fit``
    raises, preventing silent train/serve skew.
    """

    mean_: NDArray[np.float64] | None = None
    scale_: NDArray[np.float64] | None = None

    def fit(self, features: NDArray[np.float64]) -> FeatureScaler:
        """Fit the scaler on a feature matrix and return ``self``."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValidationError(
                "features must be 2-D with >= 2 rows to fit a scaler", context={"shape": x.shape}
            )
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        # Guard against zero-variance columns (constant features) -> unit scale.
        scale = np.where(scale < 1e-12, 1.0, scale)
        self.mean_ = mean
        self.scale_ = scale
        return self

    @property
    def is_fitted(self) -> bool:
        """True if the scaler has been fit."""
        return self.mean_ is not None and self.scale_ is not None

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardize a feature matrix using the fitted statistics."""
        mean = self.mean_
        scale = self.scale_
        if mean is None or scale is None:
            raise ModelStateError("FeatureScaler must be fit before transform", context={})
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != mean.shape[0]:
            raise ValidationError(
                "feature width mismatch with fitted scaler",
                context={"got": x.shape, "expected_cols": int(mean.shape[0])},
            )
        return np.asarray((x - mean) / scale, dtype=np.float64)

    def fit_transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convenience: fit then transform the same matrix."""
        return self.fit(features).transform(features)

    def state_dict(self) -> dict[str, list[float]]:
        """Return a JSON-serializable snapshot of the fitted statistics."""
        mean = self.mean_
        scale = self.scale_
        if mean is None or scale is None:
            raise ModelStateError("cannot serialize an unfitted scaler", context={})
        return {"mean": mean.tolist(), "scale": scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> FeatureScaler:
        """Reconstruct a scaler from :meth:`state_dict` output."""
        scaler = cls()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
        return scaler
