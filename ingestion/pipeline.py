# ingestion/pipeline.py
# Unified ingestion runner — all signal sources including MiroFish
# MiroFish runs AFTER other sources so it can simulate events found by RSS/SEC

import asyncio
from ingestion.news_source import NewsRSSSource
from ingestion.base import RawSignal

# Optional sources — all loaded lazily, none crash startup if unavailable
try:
    from ingestion.reddit_source import RedditSource
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

try:
    from ingestion.sec_edgar_source import SECEdgarSource
    SEC_AVAILABLE = True
except ImportError:
    SEC_AVAILABLE = False

try:
    from ingestion.twitter_source import TwitterNitterSource
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    from ingestion.onchain_source import OnChainSource
    ONCHAIN_AVAILABLE = True
except ImportError:
    ONCHAIN_AVAILABLE = False

try:
    from ingestion.mirofish_source import MiroFishSource
    MIROFISH_AVAILABLE = True
except ImportError:
    MIROFISH_AVAILABLE = False


class IngestionPipeline:
    """
    Unified signal ingestion pipeline.

    Source priority order:
      1. Reuters/Yahoo RSS news       — high trust, always available, no credentials
      2. SEC EDGAR (Form 4, 8-K)     — highest trust, legal filings (lazy init)
      3. MiroFish swarm simulation   — high trust, crowd prediction
      4. Reddit (WSB, investing)     — medium trust, requires credentials
      5. Twitter/Nitter              — medium trust, real-time chatter
      6. On-chain data               — high trust for crypto

    All sources are fault-tolerant — if one fails the rest continue.
    Reddit is disabled by default until credentials are added to .env.
    """

    def __init__(
        self,
        tickers: list,
        crypto_pairs: list,
        config: dict = None,
        use_reddit: bool = False,    # disabled by default — enable once credentials added
        use_twitter: bool = True,
        use_onchain: bool = True,
        use_mirofish: bool = True,
    ):
        self.tickers      = tickers
        self.crypto_pairs = crypto_pairs
        self.config       = config or {}

        # Core sources — always attempt, but fault-tolerant
        self.core_sources = [NewsRSSSource()]

        # SEC Edgar — lazy init (only connects to SEC.gov during fetch, not startup)
        if SEC_AVAILABLE:
            try:
                self.core_sources.append(SECEdgarSource())
            except Exception as e:
                print(f"[Pipeline] SEC Edgar disabled at startup: {e}")

        # Reddit — only if credentials exist in .env
        if use_reddit and REDDIT_AVAILABLE:
            try:
                import os
                if os.getenv("REDDIT_CLIENT_ID"):
                    self.core_sources.append(RedditSource())
                else:
                    print("[Pipeline] Reddit disabled: add REDDIT_CLIENT_ID to .env to enable")
            except Exception as e:
                print(f"[Pipeline] Reddit disabled: {e}")

        # Optional sources
        self.optional_sources = []

        if use_twitter and TWITTER_AVAILABLE:
            try:
                self.optional_sources.append(TwitterNitterSource(self.config))
            except Exception as e:
                print(f"[Pipeline] Twitter/Nitter disabled: {e}")

        if use_onchain and ONCHAIN_AVAILABLE:
            try:
                self.optional_sources.append(OnChainSource(self.config))
            except Exception as e:
                print(f"[Pipeline] OnChain disabled: {e}")

        # MiroFish — runs after core sources to simulate detected events
        self.mirofish = None
        if use_mirofish and MIROFISH_AVAILABLE:
            try:
                self.mirofish = MiroFishSource(self.config)
                print("[Pipeline] MiroFish swarm intelligence: ENABLED")
            except Exception as e:
                print(f"[Pipeline] MiroFish disabled: {e}")
        else:
            print("[Pipeline] MiroFish: not available (set MIROFISH_MOCK=true to enable)")

        active = []
        active += [type(s).__name__.replace("Source","").replace("RSS","_rss").lower()
                   for s in self.core_sources]
        active += [type(s).__name__.replace("Source","").replace("Nitter","_nitter").lower()
                   for s in self.optional_sources]
        if self.mirofish:
            active.append("mirofish")

        print(f"[Pipeline] Active sources: {', '.join(active)}")

    async def fetch_all(self) -> list:
        """
        Fetch signals from all sources in parallel.
        MiroFish runs after core sources so it can react to events they found.
        """
        all_signals = []

        # Step 1: All core + optional sources in parallel
        all_regular = self.core_sources + self.optional_sources
        tasks   = [source.fetch(self.tickers) for source in all_regular]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, list):
                all_signals.extend(result)
            elif isinstance(result, Exception):
                source_name = type(all_regular[i]).__name__
                print(f"[Pipeline] {source_name} fetch error: {result}")

        # Step 2: MiroFish simulations on events found above
        if self.mirofish:
            try:
                mf_signals = await self.mirofish.fetch(self.tickers)
                all_signals.extend(mf_signals)
                if mf_signals:
                    print(f"[Pipeline] MiroFish: {len(mf_signals)} simulation signal(s) generated")
            except Exception as e:
                print(f"[Pipeline] MiroFish fetch error: {e}")

        # Summary
        source_counts = {}
        for s in all_signals:
            source_counts[s.source] = source_counts.get(s.source, 0) + 1

        if source_counts:
            print(f"[Pipeline] Signals: {len(all_signals)} total | "
                  + " | ".join(f"{k}:{v}" for k, v in source_counts.items()))
        else:
            print("[Pipeline] No signals fetched this cycle")

        return all_signals

    def get_mirofish_stats(self) -> dict:
        if self.mirofish:
            return self.mirofish.get_feedback_summary()
        return {"status": "mirofish_not_enabled"}