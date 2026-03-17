# ingestion/mirofish_source.py
# MiroFish swarm intelligence integration for QUANTSHIFT
# MiroFish repo: github.com/666ghj/MiroFish
#
# HOW IT WORKS:
#   1. QUANTSHIFT detects a high-impact event (earnings, CPI, Fed, insider buy, etc.)
#   2. Seeds MiroFish with that event text as input
#   3. MiroFish spins up 1000 agents (bulls, bears, retail, institutional, media)
#   4. Agents interact and form consensus over simulated 6h horizon
#   5. ReportAgent outputs crowd sentiment score
#   6. QUANTSHIFT uses this as highest-weight signal (0.90) in Bayesian aggregator
#
# SETUP:
#   git clone https://github.com/666ghj/MiroFish
#   cd MiroFish && pip install -r requirements.txt
#   python app.py    # starts on localhost:8001 by default
#   Set MIROFISH_URL=http://localhost:8001 in .env

import httpx
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque
from ingestion.base import SignalSource, RawSignal


# ── Event classifier constants ─────────────────────────────────────────────────

HIGH_IMPACT_PATTERNS = {
    "earnings":     ["earnings", "eps", "revenue beat", "revenue miss", "guidance",
                     "quarterly results", "q1", "q2", "q3", "q4", "annual report"],
    "fed":          ["federal reserve", "fed rate", "fomc", "interest rate decision",
                     "powell", "rate hike", "rate cut", "basis points", "monetary policy"],
    "inflation":    ["cpi", "pce", "inflation", "consumer price", "core inflation",
                     "ppi", "producer price"],
    "insider":      ["form 4", "insider", "sec filing", "executive purchase",
                     "director bought", "officer sold", "10% owner"],
    "merger":       ["merger", "acquisition", "takeover", "buyout", "m&a",
                     "agreed to acquire", "deal valued"],
    "geopolitical": ["war", "conflict", "sanctions", "nuclear", "invasion",
                     "military", "attack", "escalation", "crisis"],
    "credit":       ["bankruptcy", "default", "downgrade", "chapter 11",
                     "liquidity crisis", "bank run", "insolvency"],
    "regulatory":   ["sec charges", "doj", "antitrust", "fine", "ban",
                     "investigation", "subpoena", "settlement"],
    "product":      ["fda approval", "clinical trial", "product launch",
                     "recall", "patent", "breakthrough"],
    "macro":        ["gdp", "unemployment", "jobs report", "nonfarm payroll",
                     "retail sales", "manufacturing", "ism"],
}

# Minimum confidence to trigger a simulation (save compute)
MIN_EVENT_CONFIDENCE = 0.40

# Cache: don't re-simulate the same event within 30 minutes
SIMULATION_CACHE_MINUTES = 30


@dataclass
class SimulationResult:
    ticker: str
    event_text: str
    event_type: str
    score: float              # -1.0 bearish to +1.0 bullish
    confidence: float         # 0.0 to 1.0
    bullish_pct: float        # % of agents that turned bullish
    bearish_pct: float        # % of agents that turned bearish
    consensus_strength: float # how strongly agents agreed
    n_agents: int
    horizon_hours: int
    simulated_at: datetime
    raw_report: dict = field(default_factory=dict)


@dataclass
class FeedbackRecord:
    """Tracks MiroFish accuracy for dynamic weight adjustment."""
    simulated_at: datetime
    ticker: str
    predicted_direction: str   # UP / DOWN
    predicted_magnitude: float
    actual_direction: str | None = None
    actual_magnitude: float | None = None
    correct: bool | None = None
    resolved_at: datetime | None = None


class EventClassifier:
    """
    Determines if a news event is high-impact enough to warrant
    a MiroFish simulation. Low-impact events are skipped to save compute.

    Returns: (event_type, confidence_score)
    confidence_score: 0.0 = definitely not high impact, 1.0 = definitely is
    """

    def classify(self, text: str) -> tuple[str | None, float]:
        text_lower = text.lower()
        best_type = None
        best_score = 0.0

        for event_type, keywords in HIGH_IMPACT_PATTERNS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                # Score = matches / total keywords, capped at 1.0
                score = min(matches / max(len(keywords) * 0.3, 1), 1.0)
                if score > best_score:
                    best_score = score
                    best_type = event_type

        return best_type, round(best_score, 3)

    def is_high_impact(self, text: str) -> tuple[bool, str, float]:
        """Returns (is_high_impact, event_type, confidence)"""
        event_type, confidence = self.classify(text)
        return confidence >= MIN_EVENT_CONFIDENCE, event_type or "general", confidence


class MiroFishFeedbackTracker:
    """
    Tracks MiroFish prediction accuracy over time.
    Adjusts signal weight dynamically:
      - If MiroFish has been >70% accurate recently → weight = 1.10
      - If MiroFish has been 50-70% accurate → weight = 0.90 (baseline)
      - If MiroFish has been <50% accurate → weight = 0.65
    """

    BASE_WEIGHT = 0.90
    WINDOW_SIZE = 50   # last 50 predictions for rolling accuracy

    def __init__(self):
        self.records: deque = deque(maxlen=200)
        self.pending: dict[str, FeedbackRecord] = {}  # ticker -> latest pending

    def record_prediction(self, ticker: str, direction: str, magnitude: float) -> str:
        """Log a new prediction. Returns a tracking ID."""
        record = FeedbackRecord(
            simulated_at=datetime.now(timezone.utc),
            ticker=ticker,
            predicted_direction=direction,
            predicted_magnitude=magnitude,
        )
        track_id = f"{ticker}_{int(datetime.now(timezone.utc).timestamp())}"
        self.pending[track_id] = record
        return track_id

    def resolve_prediction(self, track_id: str, actual_direction: str, actual_magnitude: float) -> None:
        """Called after trade resolves to update accuracy."""
        if track_id not in self.pending:
            return
        record = self.pending.pop(track_id)
        record.actual_direction = actual_direction
        record.actual_magnitude = actual_magnitude
        record.correct = (record.predicted_direction == actual_direction)
        record.resolved_at = datetime.now(timezone.utc)
        self.records.append(record)

    def rolling_accuracy(self) -> float:
        """Returns accuracy over last WINDOW_SIZE resolved predictions."""
        resolved = [r for r in self.records if r.correct is not None]
        if len(resolved) < 5:
            return 0.60  # assume moderate accuracy before enough data
        recent = resolved[-self.WINDOW_SIZE:]
        return sum(1 for r in recent if r.correct) / len(recent)

    def current_weight(self) -> float:
        """Dynamic weight based on rolling accuracy."""
        accuracy = self.rolling_accuracy()
        if accuracy >= 0.70:
            return 1.10   # MiroFish is hot — trust it more
        elif accuracy >= 0.55:
            return 0.90   # baseline
        elif accuracy >= 0.45:
            return 0.70   # underperforming — reduce trust
        else:
            return 0.50   # clearly not working — heavy discount

    def summary(self) -> dict:
        resolved = [r for r in self.records if r.correct is not None]
        return {
            "n_resolved": len(resolved),
            "n_pending": len(self.pending),
            "rolling_accuracy": round(self.rolling_accuracy(), 3),
            "current_weight": round(self.current_weight(), 3),
        }


class MiroFishSource(SignalSource):
    """
    MiroFish swarm intelligence as a QUANTSHIFT signal source.

    Integration modes:
      1. LIVE   — MiroFish server running locally at MIROFISH_URL
      2. MOCK   — synthetic simulation for testing (MIROFISH_MOCK=true in .env)

    Signal weight: dynamic (0.50 - 1.10) based on rolling accuracy
    Only triggered for high-impact events (saves compute)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.base_url = os.getenv("MIROFISH_URL", "http://localhost:8001")
        self.mock_mode = os.getenv("MIROFISH_MOCK", "false").lower() == "true"
        self.timeout = 120  # simulations can take up to 2 minutes

        self.classifier = EventClassifier()
        self.feedback = MiroFishFeedbackTracker()

        # Simple cache: event_hash -> SimulationResult
        self._cache: dict[str, tuple[datetime, SimulationResult]] = {}

        # Config
        self.n_agents = self.config.get("mirofish", {}).get("n_agents", 1000)
        self.horizon_hours = self.config.get("mirofish", {}).get("horizon_hours", 6)

    def _cache_key(self, text: str, ticker: str) -> str:
        """Simple hash for deduplication."""
        import hashlib
        combined = f"{ticker}:{text[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _is_cached(self, cache_key: str) -> SimulationResult | None:
        """Return cached result if within cooldown window."""
        if cache_key not in self._cache:
            return None
        cached_at, result = self._cache[cache_key]
        age_minutes = (datetime.now(timezone.utc) - cached_at).total_seconds() / 60
        if age_minutes < SIMULATION_CACHE_MINUTES:
            return result
        del self._cache[cache_key]
        return None

    async def _run_simulation_live(self, event_text: str, ticker: str, event_type: str) -> dict:
        """Call live MiroFish server."""
        payload = {
            "seed": event_text,
            "n_agents": self.n_agents,
            "horizon_hours": self.horizon_hours,
            "inject_variables": {
                "ticker": ticker,
                "event_type": event_type,
                "market_context": f"Financial market simulation for {ticker}",
            },
            "agent_personas": [
                {"type": "retail_bull",        "weight": 0.30},
                {"type": "retail_bear",        "weight": 0.20},
                {"type": "institutional_long", "weight": 0.20},
                {"type": "institutional_short","weight": 0.15},
                {"type": "media_analyst",      "weight": 0.10},
                {"type": "algo_trader",        "weight": 0.05},
            ]
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/simulate",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()

    def _run_simulation_mock(self, event_text: str, ticker: str, event_type: str) -> dict:
        """
        Mock simulation for testing without MiroFish server.
        Generates realistic-looking output based on event type and keywords.
        """
        import random
        import hashlib

        # Seed random with event text for reproducibility
        seed = int(hashlib.md5(event_text[:100].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Base sentiment by event type
        base_bullish = {
            "earnings":     0.60,  # earnings beats are more common
            "fed":          0.45,  # rate decisions are uncertain
            "inflation":    0.35,  # high inflation is bad for markets
            "insider":      0.70,  # insider buys are bullish
            "merger":       0.65,  # M&A usually lifts target
            "geopolitical": 0.30,  # war is bearish
            "credit":       0.25,  # bankruptcy is bearish
            "regulatory":   0.35,  # regulation is usually bad
            "product":      0.60,  # launches/approvals are bullish
            "macro":        0.50,  # mixed
            "general":      0.50,
        }.get(event_type, 0.50)

        # Add noise
        bullish_pct = max(0.10, min(0.90, base_bullish + rng.gauss(0, 0.12)))
        bearish_pct = max(0.10, min(0.90 - bullish_pct, (1 - bullish_pct) * 0.7))
        neutral_pct = 1 - bullish_pct - bearish_pct
        consensus = abs(bullish_pct - bearish_pct)

        return {
            "status": "completed",
            "n_agents": self.n_agents,
            "horizon_hours": self.horizon_hours,
            "report": {
                "bullish_agent_pct":  round(bullish_pct, 3),
                "bearish_agent_pct":  round(bearish_pct, 3),
                "neutral_agent_pct":  round(neutral_pct, 3),
                "consensus_strength": round(consensus, 3),
                "dominant_narrative": (
                    f"Agents primarily {'bullish' if bullish_pct > bearish_pct else 'bearish'} "
                    f"on {ticker} following {event_type} event. "
                    f"Consensus strength: {consensus:.0%}."
                ),
                "emergent_behaviors": [
                    "sentiment_cascade" if consensus > 0.4 else "mixed_signals",
                    "institutional_leading" if rng.random() > 0.5 else "retail_leading",
                ],
                "price_impact_estimate": round((bullish_pct - 0.5) * 10, 2),
            },
            "mock": True,
        }

    def _parse_result(
        self,
        raw: dict,
        ticker: str,
        event_text: str,
        event_type: str,
    ) -> SimulationResult:
        """Parse MiroFish output into a SimulationResult."""
        report = raw.get("report", {})

        bullish_pct = float(report.get("bullish_agent_pct", 0.5))
        bearish_pct = float(report.get("bearish_agent_pct", 0.5))
        consensus   = float(report.get("consensus_strength", 0.0))

        # Convert to -1 to +1 score
        raw_score = bullish_pct - bearish_pct
        # Weight by consensus — low consensus = unreliable signal
        weighted_score = raw_score * (0.5 + consensus * 0.5)
        confidence = min(abs(weighted_score) * (0.6 + consensus * 0.4), 0.95)

        return SimulationResult(
            ticker=ticker,
            event_text=event_text[:200],
            event_type=event_type,
            score=round(float(weighted_score), 4),
            confidence=round(float(confidence), 4),
            bullish_pct=round(bullish_pct, 3),
            bearish_pct=round(bearish_pct, 3),
            consensus_strength=round(consensus, 3),
            n_agents=int(raw.get("n_agents", self.n_agents)),
            horizon_hours=int(raw.get("horizon_hours", self.horizon_hours)),
            simulated_at=datetime.now(timezone.utc),
            raw_report=report,
        )

    async def simulate_event(
        self,
        event_text: str,
        ticker: str,
    ) -> SimulationResult | None:
        """
        Main entry point. Classifies event, runs simulation if high-impact.
        Returns SimulationResult or None if event not high-impact enough.
        """
        # Step 1: Classify event
        is_high_impact, event_type, event_confidence = self.classifier.is_high_impact(event_text)

        if not is_high_impact:
            return None

        # Step 2: Check cache
        cache_key = self._cache_key(event_text, ticker)
        cached = self._is_cached(cache_key)
        if cached:
            print(f"[MiroFish] Cache hit for {ticker} {event_type} event")
            return cached

        print(f"[MiroFish] Running {self.n_agents}-agent simulation for {ticker} "
              f"({event_type}, confidence={event_confidence:.0%})...")

        # Step 3: Run simulation
        try:
            if self.mock_mode:
                raw = self._run_simulation_mock(event_text, ticker, event_type)
                print(f"[MiroFish] Mock simulation complete (set MIROFISH_MOCK=false for live)")
            else:
                raw = await self._run_simulation_live(event_text, ticker, event_type)

            result = self._parse_result(raw, ticker, event_text, event_type)

            # Step 4: Cache result
            self._cache[cache_key] = (datetime.now(timezone.utc), result)

            direction = "UP" if result.score > 0 else "DOWN"
            print(f"[MiroFish] {ticker}: {direction} | score={result.score:+.3f} | "
                  f"confidence={result.confidence:.0%} | "
                  f"bulls={result.bullish_pct:.0%} bears={result.bearish_pct:.0%} | "
                  f"consensus={result.consensus_strength:.0%}")

            return result

        except httpx.ConnectError:
            print(f"[MiroFish] Server not reachable at {self.base_url}. "
                  f"Set MIROFISH_MOCK=true or start MiroFish: python app.py")
            return None
        except Exception as e:
            print(f"[MiroFish] Simulation failed for {ticker}: {e}")
            return None

    async def fetch(self, tickers: list[str], window_hours: int = 6) -> list[RawSignal]:
        """
        SignalSource interface implementation.
        Screens recent signals from other sources and runs simulations
        for high-impact events.
        Called by IngestionPipeline alongside other sources.
        """
        # In the full pipeline, this is called with recently-fetched news
        # For standalone use, pull recent headlines and simulate
        signals = []
        dynamic_weight = self.feedback.current_weight()

        # Pull recent RSS headlines to find high-impact events
        try:
            import feedparser
            test_feeds = [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
            ]
            ticker_set = set(t.upper() for t in tickers)
            recent_events = []

            for feed_url in test_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:
                        text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                        # Find matching ticker
                        for ticker in ticker_set:
                            if ticker in text.upper() or f"${ticker}" in text.upper():
                                recent_events.append((ticker, text))
                                break
                except Exception:
                    continue

            # Simulate high-impact events (max 3 per cycle to limit compute)
            simulated = 0
            for ticker, event_text in recent_events[:10]:
                if simulated >= 3:
                    break

                result = await self.simulate_event(event_text, ticker)
                if result is None:
                    continue

                simulated += 1
                direction = "UP" if result.score > 0 else "DOWN"
                sentiment_str = "bullish" if result.score > 0 else "bearish"

                narrative = result.raw_report.get("dominant_narrative", "")
                emergent  = result.raw_report.get("emergent_behaviors", [])

                signal_text = (
                    f"MiroFish swarm simulation ({result.n_agents} agents, "
                    f"{result.horizon_hours}h horizon): "
                    f"{ticker} {sentiment_str} signal from {result.event_type} event. "
                    f"Bullish agents: {result.bullish_pct:.0%}, "
                    f"Bearish agents: {result.bearish_pct:.0%}, "
                    f"Consensus: {result.consensus_strength:.0%}. "
                    f"{narrative} "
                    f"Emergent behaviors: {', '.join(emergent)}."
                )

                signals.append(RawSignal(
                    source="mirofish",
                    ticker=ticker,
                    text=signal_text,
                    url=None,
                    timestamp=datetime.now(timezone.utc),
                    trust_weight=dynamic_weight,
                    metadata={
                        "event_type":          result.event_type,
                        "score":               result.score,
                        "confidence":          result.confidence,
                        "bullish_pct":         result.bullish_pct,
                        "bearish_pct":         result.bearish_pct,
                        "consensus_strength":  result.consensus_strength,
                        "n_agents":            result.n_agents,
                        "mirofish_weight":     dynamic_weight,
                        "mock":                self.mock_mode,
                    }
                ))

        except Exception as e:
            print(f"[MiroFish] fetch() failed: {e}")

        return signals

    def get_feedback_summary(self) -> dict:
        return self.feedback.summary()
