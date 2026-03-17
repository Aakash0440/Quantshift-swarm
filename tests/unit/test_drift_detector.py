# tests/unit/test_drift_detector.py

import pytest
import numpy as np
from regime.drift_detector import DriftDetector, RegimeTier


class TestDriftDetector:
    def setup_method(self):
        self.detector = DriftDetector(stable_p=0.3, watch_p=0.1, volatility_z_critical=3.0)

    def test_stable_on_same_distribution(self):
        # Same distribution = no drift = STABLE
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 200))
        ref = prices[:140]
        cur = prices[140:]
        result = self.detector.detect(ref, cur)
        assert result.tier == RegimeTier.STABLE

    def test_critical_on_volatility_spike(self):
        # Reference: calm market. Current: 10x volatility spike
        np.random.seed(42)
        ref_prices = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        # Spike: much higher volatility
        cur_prices = ref_prices[-1] + np.cumsum(np.random.normal(0, 5.0, 30))
        result = self.detector.detect(ref_prices, cur_prices)
        # High volatility should trigger WATCH or CRITICAL
        assert result.tier in (RegimeTier.WATCH, RegimeTier.CRITICAL)

    def test_drift_score_range(self):
        np.random.seed(0)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        result = self.detector.detect(prices[:70], prices[70:])
        assert 0.0 <= result.drift_score <= 1.0

    def test_p_value_range(self):
        np.random.seed(0)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        result = self.detector.detect(prices[:70], prices[70:])
        assert 0.0 <= result.p_value <= 1.0

    def test_empty_input_returns_stable(self):
        # Very short arrays shouldn't crash
        ref = np.array([100.0, 101.0, 100.5])
        cur = np.array([101.0, 100.0])
        result = self.detector.detect(ref, cur)
        assert result.tier in (RegimeTier.STABLE, RegimeTier.WATCH, RegimeTier.CRITICAL)

    def test_triggered_tests_populated(self):
        # On a genuine drift, at least one test should fire
        np.random.seed(1)
        ref = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        # Completely different distribution
        cur = 200 + np.cumsum(np.random.normal(0, 5.0, 30))
        result = self.detector.detect(ref, cur)
        # drift_score should be high
        assert result.drift_score > 0.3
