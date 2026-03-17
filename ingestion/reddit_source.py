import praw, os
from datetime import datetime, timezone
from ingestion.base import SignalSource, RawSignal

class RedditSource(SignalSource):
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT","QuantShift/1.0"),
            read_only=True,
        )
        self.subreddits = ["wallstreetbets","investing","stocks","CryptoCurrency"]

    async def fetch(self, tickers: list, window_hours: int = 24) -> list:
        signals = []
        ticker_set = set(t.upper() for t in tickers)
        for sub_name in self.subreddits:
            try:
                sub = self.reddit.subreddit(sub_name)
                for post in sub.hot(limit=100):
                    text = f"{post.title} {post.selftext}"
                    matched = next((t for t in ticker_set if f"${t}" in text.upper() or f" {t} " in f" {text.upper()} "), None)
                    signals.append(RawSignal(
                        source="reddit", ticker=matched, text=text[:500],
                        url=f"https://reddit.com{post.permalink}",
                        timestamp=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        trust_weight=0.60,
                        metadata={"upvotes":post.score,"subreddit":sub_name}
                    ))
            except Exception as e:
                print(f"Reddit fetch failed for r/{sub_name}: {e}")
        return signals
