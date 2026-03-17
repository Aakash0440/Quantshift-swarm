import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio, yaml, numpy as np
from dotenv import load_dotenv
load_dotenv()

from ingestion.pipeline import IngestionPipeline
from cleaning.bot_filter import BotFilter
from cleaning.entity_extractor import EntityExtractor
from nlp.finbert_engine import FinBERTEngine, SentimentResult
from nlp.signal_aggregator import BayesianAggregator
from regime.drift_detector import DriftDetector, RegimeTier
from ingestion.market_data import MarketDataFetcher
from prediction.ensemble import BayesianEnsemble, EnsemblePrediction
from decision.agent_loop import AgentLoop
from decision.explainer import SignalExplainer
from blackswan.handler import BlackSwanHandler
from execution.signal_only import SignalOnlyExecutor
from alerts.telegram_bot import TelegramAlerter

async def run():
    with open("config/base.yaml") as f:
        config = yaml.safe_load(f)

    tickers      = config["markets"]["stocks"]["universe"]
    crypto_pairs = config["markets"]["crypto"]["pairs"]

    alerter    = TelegramAlerter(config)
    pipeline   = IngestionPipeline(tickers, crypto_pairs, config)
    bot_filter = BotFilter()
    extractor  = EntityExtractor(known_tickers=set(tickers + [p.split("/")[0] for p in crypto_pairs]))
    finbert    = FinBERTEngine()
    aggregator = BayesianAggregator()
    drift      = DriftDetector()
    market     = MarketDataFetcher()
    agent      = AgentLoop(config)
    explainer  = SignalExplainer()
    blackswan  = BlackSwanHandler(config, alerter)
    executor   = SignalOnlyExecutor(alerter, explainer)

    print("QUANTSHIFT-SWARM v1 starting...")
    await alerter.send("QUANTSHIFT-SWARM v1 started — signal-only mode active")

    while True:
        try:
            # Step 1: Ingest all sources (news + SEC + MiroFish + optional reddit/twitter/onchain)
            raw_signals = await pipeline.fetch_all()

            # Step 2: Clean
            signals_with_entities = [extractor.extract(s) for s in raw_signals]
            clean_signals, n_filtered = bot_filter.filter(signals_with_entities)

            # Step 3: NLP scoring
            sentiment_results = []
            for signal in clean_signals:
                if not signal.ticker:
                    continue
                # Route to correct model: CryptoBERT for crypto, FinBERT for stocks
                score, conf, label = finbert.analyze(signal.text)
                sentiment_results.append(SentimentResult(
                    ticker=signal.ticker, source=signal.source,
                    score=score, confidence=conf, label=label,
                    trust_weight=signal.trust_weight,
                    metadata=signal.metadata,
                ))

            # Step 4: Group by ticker
            ticker_signals = {}
            for sr in sentiment_results:
                ticker_signals.setdefault(sr.ticker, []).append(sr)

            # Step 5: Regime detection using SPY as market proxy
            regime = RegimeTier.WATCH
            vol_z  = 0.0
            try:
                spy_result = await market.get_ohlcv_stocks("SPY", interval="1h", days=60)
                # Handle both DataFrame and tuple return types
                if isinstance(spy_result, tuple):
                    spy_df = spy_result[0]
                else:
                    spy_df = spy_result
                if hasattr(spy_df, 'values'):
                    spy_prices = spy_df["close"].values
                elif hasattr(spy_df, '__getitem__'):
                    spy_prices = np.array(spy_df["close"])
                else:
                    spy_prices = np.array(spy_df)
                ref = spy_prices[:int(len(spy_prices)*0.7)]
                cur = spy_prices[int(len(spy_prices)*0.7):]
                drift_result = drift.detect(ref, cur)
                regime = drift_result.tier
                vol_z  = drift_result.details.get("volatility_z", 0)
            except Exception as e:
                print(f"Regime detection failed: {e}")

            # Step 6: Black swan check
            paused = await blackswan.check(regime=regime, volatility_z=vol_z)
            if paused:
                print("Bot paused — black swan handler activated")
                await asyncio.sleep(3600)
                continue

            # Step 7: Predict and decide per ticker
            for ticker, signals in ticker_signals.items():
                agg = aggregator.aggregate(signals, n_filtered=n_filtered)
                if agg is None or agg.confidence < config["prediction"]["min_confidence_threshold"]:
                    continue

                prediction = EnsemblePrediction(
                    ticker=ticker, direction=agg.direction,
                    magnitude_pct=agg.magnitude_pct,
                    ci_low=agg.magnitude_pct*0.5, ci_high=agg.magnitude_pct*1.5,
                    confidence=agg.confidence, horizon="4h",
                    model_votes={"sentiment":(agg.direction, agg.confidence)},
                )
                decision    = agent.run(prediction, regime)
                explanation = explainer.explain(prediction, agg, n_filtered)
                await executor.execute(decision, explanation)

            print(f"Cycle complete. Regime: {regime.value}. Next run in 1h.")
            await asyncio.sleep(3600)

        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback; traceback.print_exc()
            await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(run())
