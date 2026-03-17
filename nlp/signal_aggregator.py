# nlp/signal_aggregator.py
# Bayesian multi-source fusion — updated with MiroFish as a signal source
# MiroFish weight is DYNAMIC — adjusted by feedback tracker accuracy

import numpy as np
from dataclasses import dataclass


@dataclass
class AggregatedSignal:
    ticker: str
    direction: str
    magnitude_pct: float
    confidence: float
    horizon: str
    components: dict
    n_bot_signals_filtered: int
    mirofish_triggered: bool = False


class BayesianAggregator:
    """
    Combines signals from multiple sources using Bayesian probability updating.
    NOT majority vote — actual posterior probability computation.

    Source weights (base):
      sec_4 / sec_8k : 1.00  — legal filing, highest trust
      reuters        : 1.00
      mirofish       : 0.90  — dynamic, adjusted by feedback tracker
      bloomberg      : 0.90
      yahoo          : 0.75
      coindesk       : 0.72
      onchain        : 0.80
      reddit         : 0.55
      twitter        : 0.50

    MiroFish special behavior:
      - If MiroFish and SEC both agree → confidence multiplied by 1.3
      - If MiroFish contradicts SEC → confidence reduced by 0.8
      - MiroFish weight is dynamic (0.50-1.10) based on rolling accuracy
    """

    BASE_WEIGHTS: dict[str, float] = {
        "sec_4":         1.00,
        "sec_8k":        1.00,
        "reuters":       1.00,
        "mirofish":      0.90,   # dynamic — overridden by metadata.mirofish_weight
        "bloomberg":     0.90,
        "yahoo":         0.75,
        "coindesk":      0.72,
        "cointelegraph": 0.68,
        "onchain":       0.80,
        "reddit":        0.55,
        "twitter":       0.50,
    }

    def _get_weight(self, source: str, trust_weight: float, metadata: dict) -> float:
        """Get effective weight for a signal, respecting dynamic MiroFish weight."""
        if source == "mirofish":
            # Use dynamic weight from feedback tracker if available
            return metadata.get("mirofish_weight", self.BASE_WEIGHTS["mirofish"])

        base = self.BASE_WEIGHTS.get(source, 0.60)
        return base * trust_weight

    def _mirofish_agreement_multiplier(
        self,
        mirofish_score: float | None,
        sec_score: float | None,
    ) -> float:
        """
        If MiroFish and SEC agree, boost confidence.
        If they disagree, reduce confidence.
        """
        if mirofish_score is None or sec_score is None:
            return 1.0

        mf_direction = 1 if mirofish_score > 0 else -1
        sec_direction = 1 if sec_score > 0 else -1

        if mf_direction == sec_direction:
            return 1.30   # agreement boost
        else:
            return 0.75   # disagreement discount

    def aggregate(
        self,
        sentiment_results: list,
        funding_rate: float = 0.0,
        onchain_flow: float = 0.0,
        n_filtered: int = 0,
    ) -> "AggregatedSignal | None":

        if not sentiment_results:
            return None

        ticker = sentiment_results[0].ticker
        prior = 0.5
        weighted_scores = []
        components = {}
        mirofish_triggered = False

        # Track MiroFish and SEC scores for agreement multiplier
        mirofish_score = None
        sec_score = None

        for sr in sentiment_results:
            metadata = getattr(sr, "metadata", {}) or {}
            w = self._get_weight(sr.source, sr.trust_weight, metadata)

            weighted_scores.append(sr.score * w)
            components[sr.source] = (round(sr.score, 3), round(w, 3))

            if sr.source == "mirofish":
                mirofish_triggered = True
                mirofish_score = sr.score
                # Add consensus strength as extra weight if available
                consensus = metadata.get("consensus_strength", 0.5)
                if consensus > 0.6:
                    # Strong consensus — double-count the signal
                    weighted_scores.append(sr.score * w * 0.5)

            if sr.source in ("sec_4", "sec_8k"):
                sec_score = sr.score

        # Add crypto-specific signals
        if abs(funding_rate) > 0.0005:
            fr_score = funding_rate * 100
            components["funding_rate"] = (round(fr_score, 3), 0.30)
            weighted_scores.append(fr_score * 0.30)

        if abs(onchain_flow) > 0:
            oc_score = float(np.sign(onchain_flow) * min(abs(onchain_flow) / 10000, 1.0))
            components["onchain"] = (round(oc_score, 3), 0.25)
            weighted_scores.append(oc_score * 0.25)

        # Bayesian posterior update
        raw_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0
        posterior = prior + (raw_score * (1 - prior) if raw_score > 0 else raw_score * prior)
        posterior = float(np.clip(posterior, 0.0, 1.0))

        # Apply MiroFish-SEC agreement multiplier
        agreement_mult = self._mirofish_agreement_multiplier(mirofish_score, sec_score)

        confidence = abs(posterior - 0.5) * 2 * agreement_mult
        confidence = min(float(confidence), 0.95)

        direction = "UP" if posterior > 0.55 else "DOWN" if posterior < 0.45 else "NEUTRAL"
        magnitude = abs(raw_score) * 10

        # Annotate if MiroFish boosted or discounted the signal
        if mirofish_triggered and agreement_mult != 1.0:
            boost_str = "boosted" if agreement_mult > 1.0 else "discounted"
            components["_mirofish_agreement"] = (
                f"{boost_str} ({agreement_mult:.2f}x)", 0
            )

        return AggregatedSignal(
            ticker=ticker,
            direction=direction,
            magnitude_pct=round(magnitude, 2),
            confidence=round(confidence, 3),
            horizon="4h",
            components=components,
            n_bot_signals_filtered=n_filtered,
            mirofish_triggered=mirofish_triggered,
        )
