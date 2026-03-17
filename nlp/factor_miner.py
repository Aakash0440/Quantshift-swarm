# nlp/factor_miner.py
# LightGBM factor mining — finds which signal features actually predict price moves.
# Keeps only factors with Information Coefficient (IC) > 0.05.
# Auto-kills underperforming factors after 50 trades.
#
# IC = Spearman correlation between predicted factor and actual forward return
# IC > 0.05 = statistically meaningful signal
# IC < 0.02 = noise, discard

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Factor:
    name: str
    ic: float
    ic_std: float
    n_observations: int
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_ic: float = 0.0


class FactorMiner:
    """
    Mines predictive factors from signal features using LightGBM.

    Workflow:
      1. Collect (features, forward_return) pairs over time
      2. Train LightGBM to predict forward_return from features
      3. Extract feature importances → compute IC per feature
      4. Keep only IC > threshold (default 0.05)
      5. Re-evaluate every N trades, kill underperformers

    Features mined from signals:
      - sentiment_news_score
      - sentiment_twitter_score
      - sentiment_reddit_score
      - sec_insider_signal
      - funding_rate
      - volume_z_score
      - drift_score
      - n_sources_agreeing
      - source_consensus_strength
    """

    IC_THRESHOLD = 0.05
    MIN_OBSERVATIONS = 50
    RETRAIN_EVERY_N = 50

    def __init__(self):
        self.factors: dict[str, Factor] = {}
        self.observations: list[dict] = []   # {features, forward_return}
        self.model: lgb.Booster | None = None
        self.trade_count = 0

    def add_observation(self, features: dict[str, float], forward_return: float) -> None:
        """Call this after each completed trade with actual realized return."""
        self.observations.append({**features, "_forward_return": forward_return})
        self.trade_count += 1

        if self.trade_count % self.RETRAIN_EVERY_N == 0 and len(self.observations) >= self.MIN_OBSERVATIONS:
            self._retrain()

    def _compute_ic(self, predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
        """Spearman IC between predictions and forward returns."""
        if len(predictions) < 10:
            return 0.0, 1.0
        corr, _ = stats.spearmanr(predictions, actuals)
        ic = float(corr) if not np.isnan(corr) else 0.0
        ic_std = float(np.std([ic]))  # simplified — ideally rolling IC std
        return ic, ic_std

    def _retrain(self) -> None:
        """Retrain LightGBM and update factor ICs."""
        df = pd.DataFrame(self.observations[-500:])  # last 500 observations
        feature_cols = [c for c in df.columns if c != "_forward_return"]
        X = df[feature_cols].fillna(0)
        y = df["_forward_return"]

        if len(X) < self.MIN_OBSERVATIONS:
            return

        params = {
            "objective": "regression",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "verbosity": -1,
        }

        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self.model = model

            predictions = model.predict(X)

            # Compute IC per feature using permutation approach
            for feature in feature_cols:
                feat_series = X[feature].values
                ic, ic_std = self._compute_ic(feat_series, y.values)

                self.factors[feature] = Factor(
                    name=feature,
                    ic=round(ic, 4),
                    ic_std=round(ic_std, 4),
                    n_observations=len(df),
                    active=abs(ic) >= self.IC_THRESHOLD,
                    last_ic=round(ic, 4),
                )

        except Exception as e:
            print(f"FactorMiner retrain failed: {e}")

    def get_active_factors(self) -> list[Factor]:
        """Returns only factors with IC > threshold."""
        return [f for f in self.factors.values() if f.active]

    def score_features(self, features: dict[str, float]) -> float:
        """
        Predict signal quality score from features.
        Returns: 0.0 to 1.0 confidence multiplier.
        Uses LightGBM model if trained, else returns 1.0 (no adjustment).
        """
        if self.model is None or len(self.observations) < self.MIN_OBSERVATIONS:
            return 1.0  # no model yet, don't adjust

        try:
            active_factor_names = {f.name for f in self.get_active_factors()}
            if not active_factor_names:
                return 1.0

            # Use only active factors
            feat_values = {k: features.get(k, 0.0) for k in active_factor_names}
            X = pd.DataFrame([feat_values]).fillna(0)

            # Align columns with training
            for col in self.model.feature_name_:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.model.feature_name_]

            predicted_return = float(self.model.predict(X)[0])
            # Convert predicted return to confidence multiplier [0.5, 1.5]
            multiplier = 1.0 + float(np.clip(predicted_return * 10, -0.5, 0.5))
            return round(multiplier, 3)

        except Exception:
            return 1.0

    def summary(self) -> dict:
        return {
            "n_factors_total": len(self.factors),
            "n_factors_active": len(self.get_active_factors()),
            "n_observations": len(self.observations),
            "top_factors": sorted(
                [(f.name, f.ic) for f in self.get_active_factors()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5],
        }
