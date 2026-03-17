# tests/integration/test_full_cycle.py
# Full pipeline integration test: signals -> clean -> NLP -> regime -> decision
# Uses synthetic data — no API calls needed.

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone
from ingestion.base import RawSignal
from cleaning.bot_filter import BotFilter
from cleaning.entity_extractor import EntityExtractor
from nlp.finbert_engine import FinBERTEngine, SentimentResult
from nlp.signal_aggregator import BayesianAggregator
from regime.drift_detector import DriftDetector, RegimeTier
from prediction.ensemble import BayesianEnsemble, EnsemblePrediction
from decision.agent_loop import AgentLoop, TradeDecision


def make_sentiment_result(ticker: str, source: str, score: float, confidence: float) -> SentimentResult:
    return SentimentResult(
        ticker=ticker,
        source=source,
        score=score,
        confidence=confidence,
        label="positive" if score > 0 else "negative",
        trust_weight=0.8,
    )


MOCK_CONFIG = {
    "prediction": {"min_confidence_threshold": 0.60},
    "decision": {
        "kelly_fraction": 0.25,
        "max_position_pct": 0.05,
        "min_edge_pct": 0.015,
        "consecutive_loss_limit": 3,
        "regime_position_scale": {"stable": 1.0, "watch": 0.5, "critical": 0.0},
    },
}


class TestFullCycle:

    def test_bullish_signals_produce_buy(self):
        """Strong bullish signals across 3 sources should produce a BUY decision."""
        aggregator = BayesianAggregator()
        agent = AgentLoop(MOCK_CONFIG)

        signals = [
            make_sentiment_result("NVDA", "reuters", score=0.8, confidence=0.85),
            make_sentiment_result("NVDA", "sec_4",  score=0.9, confidence=0.95),
            make_sentiment_result("NVDA", "reddit", score=0.6, confidence=0.70),
        ]

        agg = aggregator.aggregate(signals)
        assert agg is not None
        assert agg.direction == "UP"
        assert agg.confidence > 0.60

        prediction = EnsemblePrediction(
            ticker="NVDA",
            direction=agg.direction,
            magnitude_pct=agg.magnitude_pct,
            ci_low=agg.magnitude_pct * 0.5,
            ci_high=agg.magnitude_pct * 1.5,
            confidence=agg.confidence,
            horizon="4h",
            model_votes={"sentiment": (agg.direction, agg.confidence)},
        )

        decision = agent.run(prediction, RegimeTier.STABLE)
        assert decision.action in ("UP", "BUY")
        assert decision.position_pct > 0

    def test_critical_regime_always_alerts_human(self):
        """CRITICAL regime must produce ALERT_HUMAN regardless of signal strength."""
        aggregator = BayesianAggregator()
        agent = AgentLoop(MOCK_CONFIG)

        signals = [
            make_sentiment_result("NVDA", "reuters", score=0.9, confidence=0.95),
            make_sentiment_result("NVDA", "sec_4",  score=0.95, confidence=0.99),
        ]
        agg = aggregator.aggregate(signals)

        prediction = EnsemblePrediction(
            ticker="NVDA",
            direction="UP",
            magnitude_pct=5.0,
            ci_low=2.0,
            ci_high=8.0,
            confidence=0.90,
            horizon="4h",
            model_votes={},
        )

        # CRITICAL regime should override even high-confidence signal
        decision = agent.run(prediction, RegimeTier.CRITICAL)
        assert decision.action == "ALERT_HUMAN"
        assert decision.position_pct == 0.0

    def test_low_confidence_produces_hold(self):
        """Below-threshold confidence should produce HOLD."""
        agent = AgentLoop(MOCK_CONFIG)
        prediction = EnsemblePrediction(
            ticker="AAPL",
            direction="UP",
            magnitude_pct=0.5,
            ci_low=0.1,
            ci_high=0.9,
            confidence=0.35,  # below 0.60 threshold
            horizon="4h",
            model_votes={},
        )
        decision = agent.run(prediction, RegimeTier.STABLE)
        assert decision.action == "HOLD"

    def test_watch_regime_reduces_position(self):
        """WATCH regime should produce 50% of normal position size."""
        agent = AgentLoop(MOCK_CONFIG)
        prediction = EnsemblePrediction(
            ticker="MSFT",
            direction="UP",
            magnitude_pct=3.0,
            ci_low=1.5,
            ci_high=4.5,
            confidence=0.75,
            horizon="4h",
            model_votes={},
        )
        decision_stable = agent.run(prediction, RegimeTier.STABLE)
        decision_watch  = agent.run(prediction, RegimeTier.WATCH)

        if decision_stable.action not in ("HOLD", "ALERT_HUMAN") and decision_watch.action not in ("HOLD", "ALERT_HUMAN"):
            assert decision_watch.position_pct <= decision_stable.position_pct * 0.6

    def test_entity_extraction_finds_tickers(self):
        """EntityExtractor should correctly identify $NVDA in text."""
        extractor = EntityExtractor(known_tickers={"NVDA", "AAPL", "MSFT", "BTC"})
        signal = RawSignal(
            source="twitter",
            ticker=None,
            text="$NVDA earnings beat by 15%, guidance raised for Q3",
            url=None,
            timestamp=datetime.now(timezone.utc),
        )
        result = extractor.extract(signal)
        assert result.ticker == "NVDA"

    def test_crypto_ticker_extraction(self):
        """EntityExtractor should find BTC in crypto text."""
        extractor = EntityExtractor(known_tickers={"NVDA", "BTC", "ETH"})
        signal = RawSignal(
            source="reddit",
            ticker=None,
            text="BTC looks ready to break 70k, funding rates normalizing",
            url=None,
            timestamp=datetime.now(timezone.utc),
        )
        result = extractor.extract(signal)
        assert result.ticker == "BTC"
