# regime/volatility_monitor.py
# Tracks realized volatility and computes z-scores.
# Acts as an early warning system before drift_detector fires.
# Also computes a VIX proxy from SPY options-implied vol (yfinance free).

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timezone


class VolatilityMonitor:
    """
    Two-layer volatility monitoring:

    Layer 1 — Realized vol z-score:
      Track rolling 30-day realized vol.
      When current vol is 3+ standard deviations above baseline → CRITICAL signal.

    Layer 2 — VIX proxy:
      Pull ^VIX from yfinance (free).
      VIX > 30 = elevated fear → WATCH
      VIX > 40 = extreme fear → CRITICAL

    Feeds into BlackSwanHandler and DriftDetector.
    """

    VIX_WATCH_THRESHOLD    = 25.0
    VIX_CRITICAL_THRESHOLD = 35.0
    Z_SCORE_CRITICAL       = 3.0
    Z_SCORE_WATCH          = 2.0
    WINDOW_DAYS            = 30

    def __init__(self):
        self.returns_history: deque = deque(maxlen=self.WINDOW_DAYS * 24)  # hourly
        self.vix_history: deque = deque(maxlen=100)
        self._baseline_vol: float | None = None
        self._baseline_std: float | None = None

    def update_returns(self, new_returns: list[float]) -> None:
        """Add new price returns to rolling window."""
        for r in new_returns:
            self.returns_history.append(r)
        self._update_baseline()

    def _update_baseline(self) -> None:
        """Recompute baseline vol from rolling window."""
        if len(self.returns_history) < 48:  # need at least 2 days
            return
        returns = np.array(list(self.returns_history))
        # Rolling 7-day windows to get distribution of vols
        window = 7 * 24
        vols = []
        for i in range(window, len(returns)):
            segment = returns[i - window:i]
            vols.append(np.std(segment) * np.sqrt(252 * 24))  # annualized
        if vols:
            self._baseline_vol = float(np.mean(vols))
            self._baseline_std = float(np.std(vols)) + 1e-9

    def current_vol(self) -> float:
        """Current realized vol (annualized)."""
        recent = list(self.returns_history)[-168:]  # last 7 days hourly
        if len(recent) < 10:
            return 0.0
        return float(np.std(recent) * np.sqrt(252 * 24))

    def vol_z_score(self) -> float:
        """How many std deviations is current vol above baseline?"""
        if self._baseline_vol is None or self._baseline_std is None:
            return 0.0
        cur_vol = self.current_vol()
        return (cur_vol - self._baseline_vol) / self._baseline_std

    async def fetch_vix(self) -> float | None:
        """Fetch latest VIX from yfinance (free). Returns VIX level."""
        try:
            import yfinance as yf
            vix = yf.download("^VIX", period="5d", interval="1h", progress=False)
            if not vix.empty:
                latest_vix = float(vix["Close"].iloc[-1])
                self.vix_history.append(latest_vix)
                return latest_vix
        except Exception as e:
            print(f"VIX fetch failed: {e}")
        return None

    def volatility_signal(self, vix: float | None = None) -> dict:
        """
        Returns current volatility assessment.
        tier: "normal" | "elevated" | "extreme"
        """
        z = self.vol_z_score()
        cur_vol = self.current_vol()

        tier = "normal"
        reasons = []

        # Z-score check
        if z >= self.Z_SCORE_CRITICAL:
            tier = "extreme"
            reasons.append(f"Vol z-score {z:.1f}σ (>{self.Z_SCORE_CRITICAL}σ threshold)")
        elif z >= self.Z_SCORE_WATCH:
            tier = "elevated"
            reasons.append(f"Vol z-score {z:.1f}σ (>{self.Z_SCORE_WATCH}σ threshold)")

        # VIX check (overrides if worse)
        if vix is not None:
            if vix >= self.VIX_CRITICAL_THRESHOLD:
                tier = "extreme"
                reasons.append(f"VIX {vix:.1f} (>{self.VIX_CRITICAL_THRESHOLD} critical)")
            elif vix >= self.VIX_WATCH_THRESHOLD and tier == "normal":
                tier = "elevated"
                reasons.append(f"VIX {vix:.1f} (>{self.VIX_WATCH_THRESHOLD} watch)")

        return {
            "tier": tier,
            "vol_z_score": round(z, 2),
            "current_vol_annualized": round(cur_vol, 4),
            "vix": vix,
            "reasons": reasons,
            "is_critical": tier == "extreme",
        }
