# scripts/run_backtest.py
# Runs a historical backtest using pre-fetched OHLCV data.
# Usage:
#   python scripts/run_backtest.py
#   python scripts/run_backtest.py --ticker NVDA --days 180
#   python scripts/run_backtest.py --ticker BTC --days 90 --capital 50000

import asyncio
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


async def run_backtest(ticker: str, days: int, capital: float, output_pdf: bool):
    with open("config/base.yaml") as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*50}")
    print(f"QUANTSHIFT v1 — Backtest")
    print(f"Ticker: {ticker} | Days: {days} | Capital: ${capital:,.0f}")
    print(f"{'='*50}\n")

    # ── Step 1: Fetch historical data ─────────────────────────────────────
    from ingestion.market_data import MarketDataFetcher
    fetcher = MarketDataFetcher()

    print(f"Fetching {days} days of hourly data for {ticker}...")
    is_crypto = ticker in ["BTC", "ETH", "SOL", "BNB", "AVAX"]

    try:
        if is_crypto:
            price_df = await fetcher.get_ohlcv_crypto(f"{ticker}/USDT", interval="1h", days=days)
        else:
            price_df = await fetcher.get_ohlcv_stocks(ticker, interval="1h", days=days)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    if price_df.empty:
        print("No data returned. Check ticker and date range.")
        return

    print(f"Loaded {len(price_df)} hourly bars.")

    # ── Step 2: Generate synthetic signals from price data ────────────────
    # In a full backtest, you'd replay real signals.
    # For now, generate signals from technical indicators as a baseline.
    print("Generating baseline signals from momentum + volatility...")

    signals_df = _generate_baseline_signals(price_df, ticker, config)

    # ── Step 3: Run backtest engine ───────────────────────────────────────
    from backtest.engine import BacktestEngine
    from backtest.metrics import print_metrics

    engine = BacktestEngine(
        starting_capital=capital,
        slippage_bps=config.get("execution", {}).get("slippage_model", {}).get(
            "crypto_bps" if is_crypto else "stocks_bps", 5
        ),
    )

    print("Running backtest...")
    result = engine.run(price_df, signals_df)

    # ── Step 4: Print results ─────────────────────────────────────────────
    print_metrics(result.metrics)

    # Walk-forward validation
    print("Running walk-forward validation (12 windows)...")
    from prediction.walk_forward import WalkForwardValidator
    from prediction.xgb_model import XGBPredictor

    # Build feature df from price_df
    feature_df = _build_feature_df(price_df)

    if len(feature_df) >= 200:
        wf = WalkForwardValidator(n_windows=min(12, len(feature_df) // 50))

        def model_fn(train_df):
            predictor = XGBPredictor()
            if len(train_df) >= 50:
                predictor.train(train_df)
            def predict_row(row):
                features = {col: row.get(col, 0.0) for col in XGBPredictor.FEATURE_COLS}
                return predictor.predict(features)
            return predict_row

        wf_results = wf.run(feature_df, model_fn)
        summary = wf.summary(wf_results)

        print("\nWalk-Forward Validation:")
        print(f"  Mean Sharpe:      {summary.get('mean_sharpe', 0):.3f}")
        print(f"  Std Sharpe:       {summary.get('std_sharpe', 0):.3f}")
        print(f"  Mean Win Rate:    {summary.get('mean_win_rate', 0):.1%}")
        print(f"  Positive Windows: {summary.get('n_positive_windows', 0)}/{summary.get('n_windows', 0)}")

        if summary.get("mean_sharpe", 0) > 0.8:
            print("\n  ✅ Sharpe > 0.8 — strategy looks robust. Consider paper trading next.")
        elif summary.get("mean_sharpe", 0) > 0.3:
            print("\n  ⚠️  Sharpe 0.3–0.8 — marginal. More data or signal tuning needed.")
        else:
            print("\n  ❌ Sharpe < 0.3 — not ready for live trading. Tune signals first.")
    else:
        print("Not enough data for walk-forward validation (need 200+ bars).")

    # ── Step 5: Generate PDF report ───────────────────────────────────────
    if output_pdf:
        try:
            from backtest.report_generator import generate_report
            pdf_path = f"quantshift_backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            generate_report(result, output_path=pdf_path)
            print(f"\nPDF report saved: {pdf_path}")
        except Exception as e:
            print(f"PDF generation failed (optional): {e}")

    print("\nBacktest complete.")


def _generate_baseline_signals(price_df: pd.DataFrame, ticker: str, config: dict) -> pd.DataFrame:
    """
    Generate simple momentum-based signals as a backtest baseline.
    In production, replay actual QUANTSHIFT signals stored in DB.
    """
    df = price_df.copy()
    close = df["close"]

    # Momentum signals
    df["momentum_4h"]  = close.pct_change(4)
    df["momentum_24h"] = close.pct_change(24)
    df["volatility"]   = close.pct_change().rolling(24).std()

    # Simple signal: positive momentum + low volatility = BUY
    df["direction"] = "NEUTRAL"
    df["confidence"] = 0.50
    df["position_pct"] = 0.0
    df["regime"] = "STABLE"
    df["ticker"] = ticker

    buy_mask = (df["momentum_4h"] > 0.005) & (df["volatility"] < df["volatility"].quantile(0.7))
    sell_mask = df["momentum_4h"] < -0.005

    df.loc[buy_mask,  "direction"] = "UP"
    df.loc[buy_mask,  "confidence"] = 0.65
    df.loc[buy_mask,  "position_pct"] = 0.03
    df.loc[sell_mask, "direction"] = "DOWN"
    df.loc[sell_mask, "confidence"] = 0.60

    return df[["ticker", "direction", "confidence", "position_pct", "regime"]]


def _build_feature_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature dataframe from OHLCV for walk-forward validation."""
    df = price_df.copy()
    close = df["close"]

    df["target"]         = close.pct_change(4).shift(-4)  # 4h forward return
    df["momentum_1h"]    = close.pct_change(1)
    df["momentum_4h"]    = close.pct_change(4)
    df["momentum_24h"]   = close.pct_change(24)
    df["volatility_24h"] = close.pct_change().rolling(24).std()
    df["volume_z"]       = (df["volume"] - df["volume"].rolling(24).mean()) / (df["volume"].rolling(24).std() + 1e-9)

    # Placeholder signal features (in live, these come from NLP pipeline)
    df["sentiment_news"]     = 0.0
    df["sentiment_twitter"]  = 0.0
    df["sentiment_reddit"]   = 0.0
    df["sec_insider_signal"] = 0.0
    df["funding_rate"]       = 0.0
    df["drift_score"]        = 0.0
    df["n_sources"]          = 0.0

    return df.dropna()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUANTSHIFT Backtester")
    parser.add_argument("--ticker",  default="NVDA", help="Ticker to backtest (e.g. NVDA, BTC)")
    parser.add_argument("--days",    default=180, type=int, help="Days of historical data")
    parser.add_argument("--capital", default=100_000.0, type=float, help="Starting capital in USD")
    parser.add_argument("--pdf",     action="store_true", help="Generate PDF report")
    args = parser.parse_args()

    asyncio.run(run_backtest(args.ticker, args.days, args.capital, args.pdf))
