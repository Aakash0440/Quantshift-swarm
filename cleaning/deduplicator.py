# cleaning/deduplicator.py
# Removes near-duplicate signals across sources using cosine similarity.
# Prevents the same news story reposted on 5 sites from being counted 5x.

import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.base import RawSignal


class Deduplicator:
    """
    Removes semantically duplicate signals.
    Example: Reuters article + Yahoo Finance repost = same event, keep only Reuters (higher trust).

    Uses all-MiniLM-L6-v2 — fast, small, good enough for dedup.
    Threshold: cosine sim > 0.85 = duplicate (slightly lower than bot filter's 0.9).
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Source priority — when two signals are duplicates, keep the higher-priority one
        self.source_priority = {
            "sec_4":    10,
            "sec_8k":   10,
            "reuters":   9,
            "bloomberg": 9,
            "yahoo":     7,
            "coindesk":  7,
            "cointelegraph": 6,
            "onchain":   8,
            "reddit":    4,
            "twitter":   3,
        }

    def _get_priority(self, source: str) -> int:
        for key, val in self.source_priority.items():
            if key in source.lower():
                return val
        return 5

    def deduplicate(self, signals: list[RawSignal]) -> tuple[list[RawSignal], int]:
        """
        Returns (deduplicated_signals, n_removed).
        Among duplicate pairs, keeps the higher-trust source.
        """
        if len(signals) <= 1:
            return signals, 0

        texts = [s.text[:256] for s in signals]

        try:
            embeddings = self.embedder.encode(texts, show_progress_bar=False, batch_size=32)
        except Exception:
            return signals, 0

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        # Find duplicate groups
        to_remove = set()
        n = len(embeddings)

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= self.threshold:
                    # Keep higher-priority source
                    pri_i = self._get_priority(signals[i].source)
                    pri_j = self._get_priority(signals[j].source)
                    if pri_j >= pri_i:
                        to_remove.add(i)
                        break  # i is removed, no need to compare further
                    else:
                        to_remove.add(j)

        deduped = [s for i, s in enumerate(signals) if i not in to_remove]
        return deduped, len(to_remove)
