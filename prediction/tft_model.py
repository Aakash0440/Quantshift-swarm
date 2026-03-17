import pandas as pd
import numpy as np

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    print("[TFTPredictor] pytorch_forecasting not installed — using XGBoost ensemble only.")

class TFTPredictor:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.max_encoder_length = 168
        self.max_prediction_length = 4

    def train(self, df: pd.DataFrame, max_epochs: int = 20) -> None:
        if not TFT_AVAILABLE:
            print("[TFTPredictor] TFT unavailable. Use XGBPredictor instead.")
            return
        dataset = self._prepare_dataset(df)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model = self._build_model(dataset)
        trainer = pl.Trainer(max_epochs=max_epochs, gradient_clip_val=0.1)
        trainer.fit(self.model, train_dataloaders=dataloader)

    def _prepare_dataset(self, df):
        return TimeSeriesDataSet(
            df, time_idx="time_idx", target="target", group_ids=["series_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["sentiment_news","sentiment_twitter",
                "sentiment_reddit","sec_insider_signal","funding_rate","drift_score","volume_z"],
            target_normalizer=None,
        )

    def _build_model(self, train_dataset):
        return TemporalFusionTransformer.from_dataset(
            train_dataset, learning_rate=1e-3, hidden_size=64,
            attention_head_size=4, dropout=0.1, hidden_continuous_size=16,
            output_size=7, loss=QuantileLoss(), reduce_on_plateau_patience=4,
        )

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            return {"direction":"NEUTRAL","magnitude_pct":0.0,"ci_low":0.0,"ci_high":0.0,"confidence":0.40}
        dataset = self._prepare_dataset(df)
        predictions = self.model.predict(dataset, mode="quantiles", return_x=True)
        p50 = float(predictions[0][-1][3])
        p10 = float(predictions[0][-1][1])
        p90 = float(predictions[0][-1][5])
        direction = "UP" if p50 > 0 else "DOWN"
        confidence = min(abs(p50)/(abs(p90-p10)+1e-9), 1.0)
        return {"direction":direction,"magnitude_pct":round(p50*100,2),
                "ci_low":round(p10*100,2),"ci_high":round(p90*100,2),"confidence":round(confidence,3)}
