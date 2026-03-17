# prediction/xgb_model.py
# XGBoost baseline predictor.
# Faster to train than TFT, good for lower-data situations and sanity checks.
# Used in the ensemble alongside TFT — if both agree, confidence goes up.

import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class XGBPredictor:
    """
    XGBoost baseline for market direction prediction.

    Features (same as TFT inputs):
      - sentiment_news, sentiment_twitter, sentiment_reddit
      - sec_insider_signal
      - funding_rate
      - drift_score
      - volume_z
      - price momentum (1h, 4h, 24h returns)
      - volatility (rolling std)

    Target: forward return over prediction horizon (4h or 24h)

    Why XGBoost alongside TFT:
      - XGBoost trains in seconds, TFT takes minutes
      - XGBoost handles missing features gracefully (important early on)
      - When XGB and TFT agree → higher confidence in ensemble
      - When they disagree → ensemble discounts prediction
    """

    FEATURE_COLS = [
        "sentiment_news",
        "sentiment_twitter",
        "sentiment_reddit",
        "sec_insider_signal",
        "funding_rate",
        "drift_score",
        "volume_z",
        "momentum_1h",
        "momentum_4h",
        "momentum_24h",
        "volatility_24h",
        "n_sources",
    ]

    def __init__(self, horizon: str = "4h"):
        self.horizon = horizon
        self.model: xgb.XGBRegressor | None = None
        self.feature_importance: dict = {}
        self.train_rmse: float = 0.0

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required feature columns exist, fill missing with 0."""
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.FEATURE_COLS].fillna(0.0)

    def train(self, df: pd.DataFrame, target_col: str = "target") -> dict:
        """
        Train XGBoost on historical feature + target data.
        Uses TimeSeriesSplit for validation (no lookahead bias).
        Returns training metrics.
        """
        if len(df) < 50:
            print(f"XGBPredictor: not enough data ({len(df)} rows). Need 50+.")
            return {"status": "insufficient_data"}

        X = self._prepare_features(df.copy())
        y = df[target_col].fillna(0)

        # TimeSeriesSplit — never use future data for validation
        tscv = TimeSeriesSplit(n_splits=3)
        val_rmses = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            val_rmses.append(rmse)

        # Final model on all data
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
        )
        self.model.fit(X, y)

        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.FEATURE_COLS, importance.tolist()))

        self.train_rmse = float(np.mean(val_rmses))

        return {
            "status": "trained",
            "n_samples": len(df),
            "val_rmse_mean": round(self.train_rmse, 6),
            "top_features": sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    def predict(self, features: dict[str, float]) -> dict:
        """
        Predict direction + magnitude from a single feature vector.
        Returns dict compatible with BayesianEnsemble.
        """
        if self.model is None:
            # No model yet — return neutral with low confidence
            return {
                "direction": "NEUTRAL",
                "magnitude_pct": 0.0,
                "confidence": 0.40,
                "model": "xgb_untrained",
            }

        try:
            X = pd.DataFrame([features])
            X = self._prepare_features(X)
            predicted_return = float(self.model.predict(X)[0])

            # Convert raw return prediction to direction + confidence
            magnitude_pct = abs(predicted_return) * 100

            if predicted_return > 0.002:   # > 0.2% predicted move
                direction = "UP"
                confidence = min(0.5 + magnitude_pct * 5, 0.95)
            elif predicted_return < -0.002:
                direction = "DOWN"
                confidence = min(0.5 + magnitude_pct * 5, 0.95)
            else:
                direction = "NEUTRAL"
                confidence = 0.45

            return {
                "direction": direction,
                "magnitude_pct": round(magnitude_pct, 3),
                "confidence": round(confidence, 3),
                "raw_return": round(predicted_return, 6),
                "model": "xgb",
            }

        except Exception as e:
            print(f"XGBPredictor.predict failed: {e}")
            return {
                "direction": "NEUTRAL",
                "magnitude_pct": 0.0,
                "confidence": 0.40,
                "model": "xgb_error",
            }

    def predict_from_signal(self, aggregated_signal) -> dict:
        """Convenience wrapper: build feature dict from AggregatedSignal."""
        components = aggregated_signal.components or {}

        features = {
            "sentiment_news":     components.get("reuters", (0, 0))[0],
            "sentiment_twitter":  components.get("twitter", (0, 0))[0],
            "sentiment_reddit":   components.get("reddit", (0, 0))[0],
            "sec_insider_signal": components.get("sec_4", (0, 0))[0],
            "funding_rate":       components.get("funding_rate", (0, 0))[0],
            "drift_score":        0.0,  # will be injected from regime layer
            "volume_z":           0.0,
            "momentum_1h":        0.0,
            "momentum_4h":        0.0,
            "momentum_24h":       0.0,
            "volatility_24h":     0.0,
            "n_sources":          float(len(components)),
        }
        return self.predict(features)
