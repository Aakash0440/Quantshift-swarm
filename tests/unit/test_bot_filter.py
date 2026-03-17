# tests/unit/test_bot_filter.py

import pytest
from datetime import datetime, timezone
from ingestion.base import RawSignal
from cleaning.bot_filter import BotFilter


def make_signal(text: str, source: str = "twitter", metadata: dict = None) -> RawSignal:
    return RawSignal(
        source=source,
        ticker="NVDA",
        text=text,
        url=None,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata or {},
        trust_weight=0.55,
    )


class TestBotFilter:
    def setup_method(self):
        self.filter = BotFilter()

    def test_new_account_flagged(self):
        signal = make_signal("NVDA going to moon!", metadata={"account_age_days": 5})
        # The filter checks metadata — new accounts are suspicious
        result = self.filter._is_suspicious_account(signal.metadata)
        assert result is True

    def test_old_account_passes(self):
        signal = make_signal("NVDA earnings beat", metadata={"account_age_days": 365})
        result = self.filter._is_suspicious_account(signal.metadata)
        assert result is False

    def test_high_tweet_rate_flagged(self):
        metadata = {"account_age_days": 100, "tweets_per_day": 200}
        result = self.filter._is_suspicious_account(metadata)
        assert result is True

    def test_duplicate_texts_filtered(self):
        # Two nearly identical texts should trigger dedup
        text_a = "NVDA is the best AI stock, buy now before it's too late!"
        text_b = "NVDA is the best AI stock, buy now before it is too late!"  # slight variation
        signals = [make_signal(text_a), make_signal(text_b)]
        duplicates = self.filter._detect_duplicate_campaign(signals)
        # One of them should be flagged as duplicate
        assert len(duplicates) >= 1

    def test_diverse_signals_pass(self):
        signals = [
            make_signal("NVDA reports record earnings, AI chip demand surges", source="reuters"),
            make_signal("Insider at NVDA purchased 50000 shares last week", source="sec_4"),
            make_signal("WSB loves NVDA for next earnings play", source="reddit"),
        ]
        clean, n_filtered = self.filter.filter(signals)
        # All three are genuinely different — should mostly pass
        assert len(clean) >= 2
        assert n_filtered <= 1

    def test_empty_signals(self):
        clean, n_filtered = self.filter.filter([])
        assert clean == []
        assert n_filtered == 0
