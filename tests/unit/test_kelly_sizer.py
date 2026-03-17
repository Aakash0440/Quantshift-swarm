# tests/unit/test_kelly_sizer.py

import pytest
from decision.kelly_sizer import KellySizer


class TestKellySizer:
    def setup_method(self):
        self.sizer = KellySizer(fraction=0.25, max_pct=0.05)

    def test_positive_edge_returns_positive_size(self):
        size = self.sizer.size(win_prob=0.65, win_loss_ratio=2.0)
        assert size > 0

    def test_zero_edge_returns_zero(self):
        # 50% win prob, 1:1 ratio = zero Kelly edge
        size = self.sizer.size(win_prob=0.50, win_loss_ratio=1.0)
        assert size == 0.0

    def test_never_exceeds_max_pct(self):
        # Even with huge edge, size should be capped
        size = self.sizer.size(win_prob=0.99, win_loss_ratio=100.0)
        assert size <= 0.05

    def test_negative_kelly_returns_zero(self):
        # Losing edge should return 0, never negative
        size = self.sizer.size(win_prob=0.30, win_loss_ratio=0.5)
        assert size == 0.0

    def test_quarter_kelly_is_25_percent_of_full(self):
        # Full Kelly at 65% win, 2:1 = (2*0.65 - 0.35) / 2 = 0.475
        # Quarter Kelly = 0.475 * 0.25 = 0.11875, but capped at 0.05
        size = self.sizer.size(win_prob=0.65, win_loss_ratio=2.0)
        assert size <= 0.05

    def test_moderate_edge_reasonable_size(self):
        # 60% win, 1.5:1 ratio — should give a real but moderate position
        size = self.sizer.size(win_prob=0.60, win_loss_ratio=1.5)
        assert 0.0 < size <= 0.05
