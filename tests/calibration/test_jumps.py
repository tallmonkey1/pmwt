"""Tests for the BNS jump-detection test."""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.calibration.jumps import bns_jump_test
from options_engine.core.errors import ValidationError


class TestBnsJumpTest:
    def test_no_jumps_in_diffusion(self) -> None:
        # A pure diffusion should rarely trigger the jump flag.
        rng = np.random.default_rng(0)
        triggers = 0
        for _ in range(50):
            r = rng.normal(0.0, 0.01, size=400)
            if bns_jump_test(r, significance=0.01).jumps_detected:
                triggers += 1
        # At 1% significance, false positives must be rare.
        assert triggers <= 3

    def test_detects_large_jump(self) -> None:
        rng = np.random.default_rng(1)
        detections = 0
        for _ in range(20):
            r = rng.normal(0.0, 0.01, size=400)
            r[200] = 0.15  # a clear jump
            if bns_jump_test(r, significance=0.05).jumps_detected:
                detections += 1
        # The test should detect the injected jump in the large majority of samples.
        assert detections >= 15

    def test_relative_jump_positive_with_jump(self) -> None:
        rng = np.random.default_rng(2)
        r = rng.normal(0.0, 0.01, size=400)
        r[100] = 0.2
        result = bns_jump_test(r)
        assert result.relative_jump > 0.0
        assert result.realized_variance > result.bipower_variation

    def test_flat_window_no_jump(self) -> None:
        result = bns_jump_test(np.zeros(10))
        assert not result.jumps_detected
        assert result.relative_jump == 0.0

    def test_rejects_short(self) -> None:
        with pytest.raises(ValidationError):
            bns_jump_test(np.array([0.01, 0.02, 0.03]))

    def test_rejects_bad_significance(self) -> None:
        with pytest.raises(ValidationError):
            bns_jump_test(np.ones(10) * 0.01, significance=0.0)
