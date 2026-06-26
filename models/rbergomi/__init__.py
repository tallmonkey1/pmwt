"""Rough-Bergomi (rBergomi) volatility/price simulation (SPEC §2.1, §2.3).

Public surface:

* :class:`RBergomiParams`, :class:`ForwardVariance` -- the validated model parameters.
* :class:`HybridSimulator` -- the production hybrid-scheme simulator (ground truth).
* :class:`CholeskySimulator` -- the exact (validation-grade) simulator (ground truth).
* :class:`SimulationPaths`, :class:`TerminalDistribution` -- result containers.
* :func:`build_terminal_distribution`, :func:`mean_standard_error`,
  :class:`MonteCarloSummary` -- aggregation with Monte-Carlo error control.
* :class:`NeuralRBergomiSimulator`, :class:`TrainableSequenceDataset`,
  :func:`train_neural_rbergomi` -- fast inference surrogate trained on TRUE paths.
* :func:`build_rbergomi_params_from_alpha` -- alpha-driven parameter construction.
"""

from __future__ import annotations

from .alpha_calibration import alpha_diagnostics, build_rbergomi_params_from_alpha
from .diagnostics import (
    MonteCarloSummary,
    build_terminal_distribution,
    mean_standard_error,
)
from .neural_simulator import (
    NeuralRBergomiConfig,
    NeuralRBergomiSimulator,
    TrainableSequenceDataset,
    build_dataset_from_paths,
    train_neural_rbergomi,
)
from .params import ForwardVariance, RBergomiParams
from .results import SimulationPaths, TerminalDistribution
from .simulator import CholeskySimulator, HybridSimulator, RBergomiSimulator

__all__ = [
    "CholeskySimulator",
    "ForwardVariance",
    "HybridSimulator",
    "MonteCarloSummary",
    "NeuralRBergomiConfig",
    "NeuralRBergomiSimulator",
    "RBergomiParams",
    "RBergomiSimulator",
    "SimulationPaths",
    "TerminalDistribution",
    "TrainableSequenceDataset",
    "alpha_diagnostics",
    "build_dataset_from_paths",
    "build_rbergomi_params_from_alpha",
    "build_terminal_distribution",
    "mean_standard_error",
    "train_neural_rbergomi",
] 
