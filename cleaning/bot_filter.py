import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.base import RawSignal

class BotFilter:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.consensus_required = 3

    def _is_suspicious_account(self, metadata: dict) -> bool:
        age_days  = metadata.get("account_age_days", 999)
        tweet_rate = metadata.get("tweets_per_day", 0)
        ff_ratio   = metadata.get("follower_following_ratio", 1.0)
        score = 0
        if age_days < 30:   score += 1
        if tweet_rate > 50: score += 1
        if ff_ratio < 0.1:  score += 1
        return score >= 2

    def _detect_duplicate_campaign(self, signals: list) -> set:
        texts = [s.text for s in signals]
        if len(texts) < 2:
            return set()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        flagged = set()
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9)
                if sim > 0.9:
                    flagged.add(j)
        return flagged

    def filter(self, signals: list) -> tuple:
        after_account = [s for s in signals if not self._is_suspicious_account(s.metadata)]
        dup_indices   = self._detect_duplicate_campaign(after_account)
        clean = [s for i, s in enumerate(after_account) if i not in dup_indices]
        return clean, len(signals) - len(clean)
