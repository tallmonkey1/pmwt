r"""The distribution surrogate: trainable wrapper around the monotone quantile network.

This is the public, framework-aware object that ties the pieces together (SPEC §2.5):
feature scaling, the monotone quantile network, the pinball-loss training loop, prediction
into a :class:`~options_engine.surrogate.distribution.SurrogateDistribution`, and
serialization. It is fully reproducible (explicit seeding) and validates its lifecycle
(predicting before training raises).

The training loop is a standard, well-understood Adam loop with:

* an explicit train/validation split for early stopping (best-validation checkpointing),
* gradient clipping for stability,
* deterministic seeding of all RNGs (NumPy split + torch).

It is intentionally CPU-friendly and small; the surrogate exists to be *fast at inference*,
not to be a large model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ..core.errors import ModelStateError, ValidationError
from ..core.logging import get_logger
from .dataset import TrainingData
from .distribution import SurrogateDistribution
from .features import FeatureScaler, RawInputs, build_feature_matrix
from .losses import pinball_loss
from .quantile_network import MonotoneQuantileNetwork

__all__ = ["DistributionSurrogate", "TrainingConfig", "TrainingReport"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyper-parameters for surrogate training."""

    hidden_sizes: tuple[int, ...] = (128, 128)
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    validation_fraction: float = 0.2
    grad_clip_norm: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValidationError("learning_rate must be positive", context={})
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValidationError("validation_fraction must lie in (0, 1)", context={})
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValidationError("batch_size/max_epochs/patience must be >= 1", context={})


@dataclass
class TrainingReport:
    """Diagnostics returned by :meth:`DistributionSurrogate.fit`."""

    best_val_loss: float
    epochs_run: int
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class DistributionSurrogate:
    """Learns and serves the rBergomi terminal-distribution quantile map.

    Use :meth:`fit` with :class:`~options_engine.surrogate.dataset.TrainingData`, then
    :meth:`predict` for a single scenario or :meth:`predict_batch` for many.
    """

    def __init__(self) -> None:
        self._network: MonotoneQuantileNetwork | None = None
        self._scaler: FeatureScaler | None = None
        self._quantile_levels: NDArray[np.float64] | None = None
        self._config: TrainingConfig | None = None

    # -- lifecycle -------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True once :meth:`fit` (or :meth:`load`) has produced a usable network."""
        return self._network is not None and self._scaler is not None

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise ModelStateError("surrogate must be trained before use", context={})

    # -- training --------------------------------------------------------------------

    def fit(self, data: TrainingData, *, config: TrainingConfig | None = None) -> TrainingReport:
        """Train the surrogate on Monte-Carlo-labelled data and return a report."""
        cfg = config or TrainingConfig()
        if data.n_samples < 4:
            raise ValidationError(
                "need at least four training samples", context={"n": data.n_samples}
            )

        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        scaler = FeatureScaler()
        x_all = scaler.fit_transform(data.features)
        y_all = np.ascontiguousarray(data.quantiles, dtype=np.float64)
        levels = np.ascontiguousarray(data.quantile_levels, dtype=np.float64)

        # Deterministic train/validation split.
        n = x_all.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, round(cfg.validation_fraction * n))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        if train_idx.size == 0:
            raise ValidationError("training split is empty; add more samples", context={"n": n})

        x_train = torch.tensor(x_all[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_all[train_idx], dtype=torch.float32)
        x_val = torch.tensor(x_all[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_all[val_idx], dtype=torch.float32)
        levels_t = torch.tensor(levels, dtype=torch.float32)

        network = MonotoneQuantileNetwork(
            n_features=data.features.shape[1],
            n_quantiles=levels.size,
            hidden_sizes=cfg.hidden_sizes,
            dropout=cfg.dropout,
        )
        optimizer = torch.optim.Adam(
            network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        report = self._train_loop(
            network=network,
            optimizer=optimizer,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            levels_t=levels_t,
            cfg=cfg,
        )

        self._network = network
        self._scaler = scaler
        self._quantile_levels = levels
        self._config = cfg
        _logger.info(
            "surrogate_trained",
            extra={
                "best_val_loss": report.best_val_loss,
                "epochs": report.epochs_run,
                "n_train": int(x_train.shape[0]),
                "n_val": int(x_val.shape[0]),
            },
        )
        return report

    @staticmethod
    def _train_loop(
        *,
        network: MonotoneQuantileNetwork,
        optimizer: torch.optim.Optimizer,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        levels_t: torch.Tensor,
        cfg: TrainingConfig,
    ) -> TrainingReport:
        """Run the Adam training loop with early stopping; load the best checkpoint."""
        best_val = float("inf")
        best_state: dict[str, torch.Tensor] = copy.deepcopy(network.state_dict())
        epochs_without_improve = 0
        report = TrainingReport(best_val_loss=best_val, epochs_run=0)

        for epoch in range(cfg.max_epochs):
            train_loss = DistributionSurrogate._run_epoch(
                network=network,
                optimizer=optimizer,
                x_train=x_train,
                y_train=y_train,
                levels_t=levels_t,
                cfg=cfg,
                epoch=epoch,
            )
            network.eval()
            with torch.no_grad():
                val_loss = float(pinball_loss(network(x_val), y_val, levels_t).item())
            report.train_losses.append(train_loss)
            report.val_losses.append(val_loss)

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = copy.deepcopy(network.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= cfg.patience:
                    break

        network.load_state_dict(best_state)
        network.eval()
        report.best_val_loss = best_val
        report.epochs_run = len(report.train_losses)
        return report

    @staticmethod
    def _run_epoch(
        *,
        network: MonotoneQuantileNetwork,
        optimizer: torch.optim.Optimizer,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        levels_t: torch.Tensor,
        cfg: TrainingConfig,
        epoch: int,
    ) -> float:
        """Run one training epoch and return its mean batch loss."""
        network.train()
        n_train = x_train.shape[0]
        batch_perm = torch.randperm(
            n_train, generator=torch.Generator().manual_seed(cfg.seed + epoch)
        )
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = batch_perm[start : start + cfg.batch_size]
            optimizer.zero_grad()
            preds = network(x_train[idx])
            loss = pinball_loss(preds, y_train[idx], levels_t)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        return epoch_loss / max(n_batches, 1)

    # -- inference -------------------------------------------------------------------

    def predict_quantiles(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return predicted quantiles ``(batch, Q)`` for an unscaled feature matrix."""
        self._require_trained()
        assert self._network is not None and self._scaler is not None  # for type-checkers
        x = self._scaler.transform(np.asarray(features, dtype=np.float64))
        with torch.no_grad():
            preds = self._network(torch.tensor(x, dtype=torch.float32))
        return np.ascontiguousarray(preds.numpy(), dtype=np.float64)

    def predict(
        self,
        *,
        hurst: float,
        eta: float,
        rho: float,
        xi0: float,
        horizon: float,
        initial_spot: float = 100.0,
    ) -> SurrogateDistribution:
        """Predict the terminal distribution for a single scenario."""
        self._require_trained()
        assert self._quantile_levels is not None
        raw = RawInputs(
            hurst=np.array([hurst]),
            eta=np.array([eta]),
            rho=np.array([rho]),
            xi0=np.array([xi0]),
            horizon=np.array([horizon]),
        )
        features = build_feature_matrix(raw)
        quantiles = self.predict_quantiles(features)[0]
        return SurrogateDistribution(
            quantile_levels=self._quantile_levels,
            quantile_values=quantiles,
            horizon=horizon,
            initial_spot=initial_spot,
        )

    @property
    def quantile_levels(self) -> NDArray[np.float64]:
        """The probability grid the surrogate predicts on."""
        self._require_trained()
        assert self._quantile_levels is not None
        return self._quantile_levels

    # -- serialization ---------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the trained surrogate (network + scaler + levels) to ``path``."""
        self._require_trained()
        assert self._network is not None and self._scaler is not None
        assert self._quantile_levels is not None and self._config is not None
        payload = {
            "network_state": self._network.state_dict(),
            "n_features": self._network.n_features,
            "n_quantiles": self._network.n_quantiles,
            "hidden_sizes": list(self._config.hidden_sizes),
            "dropout": self._config.dropout,
            "scaler": self._scaler.state_dict(),
            "quantile_levels": self._quantile_levels.tolist(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> DistributionSurrogate:
        """Load a surrogate previously written by :meth:`save`. Returns ``self``."""
        payload = torch.load(path, weights_only=False)
        network = MonotoneQuantileNetwork(
            n_features=int(payload["n_features"]),
            n_quantiles=int(payload["n_quantiles"]),
            hidden_sizes=tuple(payload["hidden_sizes"]),
            dropout=float(payload["dropout"]),
        )
        network.load_state_dict(payload["network_state"])
        network.eval()
        self._network = network
        self._scaler = FeatureScaler.from_state_dict(payload["scaler"])
        self._quantile_levels = np.asarray(payload["quantile_levels"], dtype=np.float64)
        return self
