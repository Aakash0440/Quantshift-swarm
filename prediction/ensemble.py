import numpy as np
from dataclasses import dataclass

@dataclass
class EnsemblePrediction:
    ticker: str
    direction: str
    magnitude_pct: float
    ci_low: float
    ci_high: float
    confidence: float
    horizon: str
    model_votes: dict

class BayesianEnsemble:
    MODEL_WEIGHTS = {"tft":0.50,"xgb":0.30,"sentiment":0.20}

    def combine(self, tft_pred: dict, xgb_pred: dict, sentiment_signal: dict,
                ticker: str, horizon: str = "4h") -> EnsemblePrediction:
        scores = []; votes = {}
        for model, weight in self.MODEL_WEIGHTS.items():
            pred = tft_pred if model=="tft" else xgb_pred if model=="xgb" else sentiment_signal
            sign = 1.0 if pred.get("direction")=="UP" else -1.0
            scores.append(sign * pred.get("confidence",0.5) * weight)
            votes[model] = (pred.get("direction"), round(pred.get("confidence",0),3))

        combined = float(np.sum(scores))
        direction = "UP" if combined > 0.05 else "DOWN" if combined < -0.05 else "NEUTRAL"
        magnitude = sum([
            tft_pred.get("magnitude_pct",0)*self.MODEL_WEIGHTS["tft"],
            xgb_pred.get("magnitude_pct",0)*self.MODEL_WEIGHTS["xgb"],
            sentiment_signal.get("magnitude_pct",0)*self.MODEL_WEIGHTS["sentiment"],
        ])
        return EnsemblePrediction(
            ticker=ticker, direction=direction,
            magnitude_pct=round(magnitude,2),
            ci_low=tft_pred.get("ci_low",0), ci_high=tft_pred.get("ci_high",0),
            confidence=round(abs(combined),3), horizon=horizon, model_votes=votes,
        )
