# scripts/retrain.py
# Manual retrain trigger.
# Run this when you want to force a model update outside the automatic schedule.
# Also called by the GitHub Actions cron job every Sunday.
#
# Usage:
#   python scripts/retrain.py
#   python scripts/retrain.py --ticker NVDA --days 90
#   python scripts/retrain.py --all          (retrain all tickers)
#   python scripts/retrain.py --force        (skip Sharpe improvement check)

import asyncio
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


async def retrain(tickers: list[str], days: int, force: bool):
    with open("config/base.yaml") as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*50}")
    print(f"QUANTSHIFT v1 — Manual Retrain")
    print(f"Tickers: {tickers}")
    print(f"Days:    {days}")
    print(f"Force:   {force}")
    print(f"Time:    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}\n")

    from ingestion.market_data import MarketDataFetcher
    from prediction.xgb_model import XGBPredictor
    from prediction.walk_forward import WalkForwardValidator

    fetcher = MarketDataFetcher()
    wf = WalkForwardValidator(n_windows=8, train_months=4, test_months=1)

    results = {}

    for ticker in tickers:
        print(f"\n── Retraining {ticker} ──────────────────────────")
        is_crypto = ticker in ["BTC", "ETH", "SOL", "BNB", "AVAX"]

        # Fetch data
        try:
            if is_crypto:
                df = await fetcher.get_ohlcv_crypto(f"{ticker}/USDT", interval="1h", days=days)
            else:
                df = await fetcher.get_ohlcv_stocks(ticker, interval="1h", days=days)
        except Exception as e:
            print(f"  Data fetch failed: {e}")
            results[ticker] = {"status": "failed", "reason": str(e)}
            continue

        if len(df) < 100:
            print(f"  Not enough data ({len(df)} bars). Skipping.")
            results[ticker] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        # Build feature df
        feature_df = _build_feature_df(df)

        if len(feature_df) < 100:
            print(f"  Feature df too small after dropna. Skipping.")
            results[ticker] = {"status": "skipped", "reason": "insufficient_features"}
            continue

        # Run walk-forward
        def model_fn(train_df):
            predictor = XGBPredictor()
            if len(train_df) >= 50:
                predictor.train(train_df)
            def predict_row(row):
                features = {col: float(row.get(col, 0.0)) for col in XGBPredictor.FEATURE_COLS}
                return predictor.predict(features)
            return predict_row

        try:
            wf_results = wf.run(feature_df, model_fn)
            summary = wf.summary(wf_results)
            new_sharpe = summary.get("mean_sharpe", 0.0)
        except Exception as e:
            print(f"  Walk-forward failed: {e}")
            results[ticker] = {"status": "failed", "reason": str(e)}
            continue

        print(f"  Walk-forward Sharpe: {new_sharpe:.3f}")
        print(f"  Positive windows: {summary.get('n_positive_windows', 0)}/{summary.get('n_windows', 0)}")

        # Train final model
        if new_sharpe > 0.0 or force:
            try:
                final_model = XGBPredictor()
                train_result = final_model.train(feature_df)
                print(f"  XGB trained. Val RMSE: {train_result.get('val_rmse_mean', 0):.6f}")

                # Save model
                model_path = f"models/{ticker}_xgb.json"
                import os
                os.makedirs("models", exist_ok=True)
                if final_model.model:
                    final_model.model.save_model(model_path)
                    print(f"  Model saved to {model_path}")

                results[ticker] = {
                    "status": "deployed",
                    "sharpe": round(new_sharpe, 3),
                    "model_path": model_path,
                }
            except Exception as e:
                print(f"  Model save failed: {e}")
                results[ticker] = {"status": "failed", "reason": str(e)}
        else:
            print(f"  Sharpe {new_sharpe:.3f} <= 0.0 — skipping deploy (use --force to override)")
            results[ticker] = {
                "status": "skipped_low_sharpe",
                "sharpe": round(new_sharpe, 3),
            }

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("RETRAIN SUMMARY")
    print(f"{'='*50}")
    for ticker, r in results.items():
        status_icon = "✅" if r["status"] == "deployed" else "⚠️" if "skipped" in r["status"] else "❌"
        sharpe_str = f" | Sharpe: {r['sharpe']:.3f}" if "sharpe" in r else ""
        print(f"  {status_icon} {ticker:<10} {r['status']}{sharpe_str}")
    print(f"{'='*50}\n")


def _build_feature_df(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    close = df["close"]
    df["target"]         = close.pct_change(4).shift(-4)
    df["momentum_1h"]    = close.pct_change(1)
    df["momentum_4h"]    = close.pct_change(4)
    df["momentum_24h"]   = close.pct_change(24)
    df["volatility_24h"] = close.pct_change().rolling(24).std()
    df["volume_z"]       = (df["volume"] - df["volume"].rolling(24).mean()) / (df["volume"].rolling(24).std() + 1e-9)
    df["sentiment_news"]     = 0.0
    df["sentiment_twitter"]  = 0.0
    df["sentiment_reddit"]   = 0.0
    df["sec_insider_signal"] = 0.0
    df["funding_rate"]       = 0.0
    df["drift_score"]        = 0.0
    df["n_sources"]          = 0.0
    return df.dropna()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUANTSHIFT Model Retrain")
    parser.add_argument("--ticker", default=None, help="Single ticker to retrain (e.g. NVDA)")
    parser.add_argument("--all",    action="store_true", help="Retrain all tickers from config")
    parser.add_argument("--days",   default=180, type=int, help="Days of training data")
    parser.add_argument("--force",  action="store_true", help="Deploy even if Sharpe didn't improve")
    args = parser.parse_args()

    with open("config/base.yaml") as f:
        config = yaml.safe_load(f)

    if args.all:
        stock_tickers  = config["markets"]["stocks"]["universe"]
        crypto_tickers = [p.split("/")[0] for p in config["markets"]["crypto"]["pairs"]]
        tickers = stock_tickers + crypto_tickers
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        # Default: retrain first 5 stock tickers (quick run)
        tickers = config["markets"]["stocks"]["universe"][:5]

    asyncio.run(retrain(tickers, args.days, args.force))
