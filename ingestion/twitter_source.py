# ingestion/twitter_source.py
# Free Twitter data via Nitter RSS — no API key required
# Nitter is a privacy-respecting Twitter frontend that exposes RSS feeds
# If nitter.net goes down, swap instance URL in config — many mirrors exist

import feedparser
import httpx
from datetime import datetime, timezone
from ingestion.base import SignalSource, RawSignal

# Public Nitter instances — try in order if one is down
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
]

CRYPTO_KEYWORDS = ["BTC", "ETH", "SOL", "crypto", "bitcoin", "ethereum"]
STOCK_KEYWORDS  = ["earnings", "short squeeze", "insider buy", "FDA", "merger", "upgrade", "downgrade"]


class TwitterNitterSource(SignalSource):
    """
    Scrapes financial Twitter via Nitter RSS feeds.
    Covers:
      - $TICKER search feeds  (e.g. nitter.net/search?q=%24NVDA&f=tweets)
      - Keyword search feeds  (e.g. "earnings beat", "short squeeze")
      - Specific account feeds for known finance influencers
    Zero API cost.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.instance = self.config.get("nitter_instance", NITTER_INSTANCES[0])
        self.timeout = 10

    def _build_search_url(self, query: str) -> str:
        import urllib.parse
        encoded = urllib.parse.quote(query)
        return f"{self.instance}/search/rss?q={encoded}&f=tweets"

    def _build_account_url(self, username: str) -> str:
        return f"{self.instance}/{username}/rss"

    def _try_fetch_feed(self, url: str) -> list:
        """Try primary instance, fall back to mirrors on failure."""
        for instance in NITTER_INSTANCES:
            try:
                # Replace the instance prefix with the current mirror
                adjusted_url = url.replace(self.instance, instance)
                feed = feedparser.parse(adjusted_url)
                if feed.entries:
                    return feed.entries
            except Exception:
                continue
        return []

    async def fetch(self, tickers: list[str], window_hours: int = 6) -> list[RawSignal]:
        signals = []
        ticker_set = set(t.upper() for t in tickers)

        # Build search queries
        queries = []

        # $TICKER mentions
        for ticker in tickers[:10]:  # limit to avoid rate limits
            queries.append(f"${ticker}")

        # Finance keywords
        for kw in STOCK_KEYWORDS[:3]:
            queries.append(kw)

        # Crypto keywords if relevant
        crypto_tickers = [t for t in tickers if t in CRYPTO_KEYWORDS]
        if crypto_tickers:
            queries.append("bitcoin OR ethereum crypto")

        # Known finance accounts to follow
        finance_accounts = [
            "unusual_whales",   # options flow + dark pool data
            "DeItaone",         # breaking news
            "tier10k",          # earnings + catalyst
        ]

        # Fetch search feeds
        for query in queries:
            url = self._build_search_url(query)
            entries = self._try_fetch_feed(url)

            for entry in entries[:20]:
                text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                # match to known tickers
                matched_ticker = None
                for ticker in ticker_set:
                    if f"${ticker}" in text.upper() or f" {ticker} " in f" {text.upper()} ":
                        matched_ticker = ticker
                        break

                signals.append(RawSignal(
                    source="twitter",
                    ticker=matched_ticker,
                    text=text[:400],
                    url=entry.get("link"),
                    timestamp=datetime.now(timezone.utc),
                    trust_weight=0.55,
                    metadata={
                        "query": query,
                        "feed_type": "search",
                        # Note: Nitter RSS doesn't expose follower/age data
                        # bot_filter will apply text-based duplicate detection only
                    }
                ))

        # Fetch account feeds
        for account in finance_accounts:
            url = self._build_account_url(account)
            entries = self._try_fetch_feed(url)

            for entry in entries[:10]:
                text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                matched_ticker = None
                for ticker in ticker_set:
                    if f"${ticker}" in text.upper():
                        matched_ticker = ticker
                        break

                signals.append(RawSignal(
                    source="twitter",
                    ticker=matched_ticker,
                    text=text[:400],
                    url=entry.get("link"),
                    timestamp=datetime.now(timezone.utc),
                    trust_weight=0.65,   # slightly higher for known accounts
                    metadata={
                        "account": account,
                        "feed_type": "account",
                    }
                ))

        return signals
