# prediction/online_retrainer.py
# Automatically retrains models when regime drift is detected.
# Also runs weekly scheduled retrains via cron (see CI/CD yaml).
# "Deploy if Sharpe improves" logic — don't replace a working model with a worse one.

import asyncio
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass

from regime.drift_detector import RegimeTier
from prediction.walk_forward import WalkForwardValidator


@dataclass
class RetrainResult:
    triggered_at: datetime
    trigger_reason: str
    old_sharpe: float
    new_sharpe: float
    deployed: bool
    note: str


class OnlineRetrainer:
    """
    Monitors model performance and triggers retraining when:
      1. Drift detector reports CRITICAL regime
      2. Rolling Sharpe drops below threshold (0.3)
      3. Weekly scheduled retrain (via CI/CD cron)

    Deploy policy:
      Only replace existing model if new_sharpe > old_sharpe * 0.9
      (allow 10% regression — avoids thrashing on noise)
    """

    MIN_SHARPE_THRESHOLD   = 0.30
    DEPLOY_SHARPE_RATIO    = 0.90   # new must be >= 90% of old to deploy
    MIN_RETRAIN_HOURS      = 6      # don't retrain more often than every 6h
    MAX_RETRAIN_SAMPLES    = 2000   # use last N samples for retraining

    def __init__(self, tft_predictor, xgb_predictor, walk_forward: WalkForwardValidator):
        self.tft = tft_predictor
        self.xgb = xgb_predictor
        self.walk_forward = walk_forward

        self.last_retrain_at: datetime | None = None
        self.current_sharpe: float = 0.0
        self.retrain_history: list[RetrainResult] = []

        # Buffer of recent (features, target) for retraining
        self.data_buffer: list[dict] = []

    def add_data_point(self, features: dict, target: float) -> None:
        """Call after each completed trade to accumulate training data."""
        self.data_buffer.append({**features, "target": target})
        if len(self.data_buffer) > self.MAX_RETRAIN_SAMPLES:
            self.data_buffer.pop(0)

    def _hours_since_last_retrain(self) -> float:
        if self.last_retrain_at is None:
            return float("inf")
        delta = datetime.now(timezone.utc) - self.last_retrain_at
        return delta.total_seconds() / 3600

    def should_retrain(self, regime: RegimeTier, rolling_sharpe: float) -> tuple[bool, str]:
        """Returns (should_retrain, reason)."""
        if self._hours_since_last_retrain() < self.MIN_RETRAIN_HOURS:
            return False, "cooldown"

        if len(self.data_buffer) < 100:
            return False, "insufficient_data"

        if regime == RegimeTier.CRITICAL:
            return True, "critical_regime_drift"

        if rolling_sharpe < self.MIN_SHARPE_THRESHOLD:
            return True, f"sharpe_below_threshold ({rolling_sharpe:.2f} < {self.MIN_SHARPE_THRESHOLD})"

        return False, ""

    async def retrain(self, reason: str) -> RetrainResult:
        """
        Execute retraining.
        1. Run walk-forward validation on new data
        2. If new Sharpe > old Sharpe * 0.9, deploy new model
        """
        import pandas as pd
        print(f"[OnlineRetrainer] Starting retrain. Reason: {reason}")

        if len(self.data_buffer) < 100:
            return RetrainResult(
                triggered_at=datetime.now(timezone.utc),
                trigger_reason=reason,
                old_sharpe=self.current_sharpe,
                new_sharpe=0.0,
                deployed=False,
                note="insufficient_data",
            )

        df = pd.DataFrame(self.data_buffer)

        # Run walk-forward on new data
        wf_results = self.walk_forward.run(df, model_fn=lambda train: self._make_predictor(train))
        summary = self.walk_forward.summary(wf_results)
        new_sharpe = summary.get("mean_sharpe", 0.0)

        # Deploy decision
        min_acceptable = self.current_sharpe * self.DEPLOY_SHARPE_RATIO
        should_deploy = new_sharpe >= max(min_acceptable, 0.0)

        if should_deploy:
            # Actually retrain XGB (fast) and flag TFT for retrain (slow, do async)
            try:
                self.xgb.train(df)
                print(f"[OnlineRetrainer] XGB retrained. New Sharpe: {new_sharpe:.2f}")
            except Exception as e:
                print(f"[OnlineRetrainer] XGB retrain failed: {e}")
                should_deploy = False

        result = RetrainResult(
            triggered_at=datetime.now(timezone.utc),
            trigger_reason=reason,
            old_sharpe=round(self.current_sharpe, 3),
            new_sharpe=round(new_sharpe, 3),
            deployed=should_deploy,
            note=f"Deployed: {should_deploy}. Walk-forward windows: {summary.get('n_positive_windows', 0)}/{summary.get('n_windows', 0)} positive.",
        )

        if should_deploy:
            self.current_sharpe = new_sharpe
            self.last_retrain_at = datetime.now(timezone.utc)

        self.retrain_history.append(result)
        print(f"[OnlineRetrainer] {result.note}")
        return result

    def _make_predictor(self, train_df):
        """Create a simple predictor function for walk-forward validation."""
        import pandas as pd
        from prediction.xgb_model import XGBPredictor
        predictor = XGBPredictor()
        predictor.train(train_df)

        def predict_row(row):
            features = {col: row.get(col, 0.0) for col in XGBPredictor.FEATURE_COLS}
            return predictor.predict(features)

        return predict_row
