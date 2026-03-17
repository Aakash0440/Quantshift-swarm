# cleaning/source_weighter.py
# Dynamic trust scoring per source.
# Base weights from config, adjusted by:
#   - Recent accuracy (if tracking enabled)
#   - Source freshness (older article = lower weight)
#   - Signal count balance (prevent one source from dominating)

from datetime import datetime, timezone, timedelta
from ingestion.base import RawSignal


class SourceWeighter:
    """
    Assigns and updates trust weights for each signal source.

    Base weights (from guide spec):
      sec_form4 / sec_8k : 1.00  (legal filing — highest trust)
      reuters             : 1.00
      bloomberg           : 1.00
      yahoo               : 0.85
      coindesk            : 0.80
      cointelegraph       : 0.70
      reddit              : 0.60
      twitter             : 0.55
      onchain             : 0.80

    Dynamic adjustments:
      - Freshness decay: signals > 4h old get 0.8x weight
      - Volume cap: if one source contributes > 60% of signals, cap at 0.7x
    """

    BASE_WEIGHTS: dict[str, float] = {
        "sec_4":         1.00,
        "sec_8k":        1.00,
        "reuters":       1.00,
        "bloomberg":     1.00,
        "yahoo":         0.85,
        "coindesk":      0.80,
        "cointelegraph": 0.70,
        "reddit":        0.60,
        "twitter":       0.55,
        "onchain":       0.80,
    }

    def _base_weight(self, source: str) -> float:
        source_lower = source.lower()
        for key, weight in self.BASE_WEIGHTS.items():
            if key in source_lower:
                return weight
        return 0.65  # default for unknown sources

    def _freshness_multiplier(self, timestamp: datetime) -> float:
        """Decay weight for stale signals."""
        now = datetime.now(timezone.utc)
        age_hours = (now - timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None
                     else now - timestamp).total_seconds() / 3600

        if age_hours <= 1:
            return 1.0
        elif age_hours <= 4:
            return 0.90
        elif age_hours <= 12:
            return 0.75
        elif age_hours <= 24:
            return 0.60
        else:
            return 0.40

    def _volume_multiplier(self, source: str, source_counts: dict[str, int], total: int) -> float:
        """Cap sources that dominate the signal pool."""
        if total == 0:
            return 1.0
        share = source_counts.get(source, 0) / total
        if share > 0.60:
            return 0.70
        elif share > 0.40:
            return 0.85
        return 1.0

    def apply_weights(self, signals: list[RawSignal]) -> list[RawSignal]:
        """
        Mutates trust_weight on each signal based on base + dynamic adjustments.
        Returns the updated signal list.
        """
        # Count signals per source for volume cap
        source_counts: dict[str, int] = {}
        for s in signals:
            source_counts[s.source] = source_counts.get(s.source, 0) + 1
        total = len(signals)

        for signal in signals:
            base = self._base_weight(signal.source)
            freshness = self._freshness_multiplier(signal.timestamp)
            volume = self._volume_multiplier(signal.source, source_counts, total)

            signal.trust_weight = round(base * freshness * volume, 3)

        return signals
