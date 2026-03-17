import feedparser
from datetime import datetime, timezone
from ingestion.base import SignalSource, RawSignal

RSS_FEEDS = [
    ("reuters",       "https://feeds.reuters.com/reuters/businessNews"),
    ("yahoo",         "https://feeds.finance.yahoo.com/rss/2.0/headline"),
    ("coindesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
]
SOURCE_WEIGHTS = {"reuters":1.0,"yahoo":0.85,"coindesk":0.80,"cointelegraph":0.70}

class NewsRSSSource(SignalSource):
    async def fetch(self, tickers: list, window_hours: int = 24) -> list:
        signals = []
        ticker_set = set(t.upper() for t in tickers)
        for source_name, url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:50]:
                    text = f"{entry.get('title','')} {entry.get('summary','')}"
                    matched = next((t for t in ticker_set if t in text.upper()), None)
                    signals.append(RawSignal(
                        source=source_name, ticker=matched, text=text,
                        url=entry.get("link"),
                        timestamp=datetime.now(timezone.utc),
                        trust_weight=SOURCE_WEIGHTS.get(source_name, 0.7)
                    ))
            except Exception as e:
                print(f"RSS fetch failed for {source_name}: {e}")
        return signals
