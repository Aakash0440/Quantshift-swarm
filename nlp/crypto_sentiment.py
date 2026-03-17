# nlp/crypto_sentiment.py
# Crypto-specific sentiment model using ElKulako/cryptobert
# Trained on crypto Twitter/Reddit — understands "wen moon", "rekt", "ngmi" etc.
# Use this for crypto signals; use FinBERT for stock news.

from transformers import pipeline
import torch
from dataclasses import dataclass


@dataclass
class CryptoSentimentResult:
    ticker: str
    source: str
    score: float          # -1.0 (bearish) to +1.0 (bullish)
    confidence: float
    label: str
    trust_weight: float


class CryptoSentimentEngine:
    """
    CryptoBERT — fine-tuned on 3.2M crypto-specific tweets.
    Handles crypto slang that confuses FinBERT:
      "WAGMI" → bullish
      "rekt"  → bearish
      "wen moon" → bullish (retail FOMO)
      "FUD"   → bearish
    """

    def __init__(self):
        model_name = "ElKulako/cryptobert"
        try:
            self.pipe = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                truncation=True,
            )
            self.available = True
        except Exception as e:
            print(f"CryptoBERT load failed: {e}. Falling back to FinBERT for crypto.")
            self.available = False
            # Fallback to FinBERT
            from nlp.finbert_engine import FinBERTEngine
            self._fallback = FinBERTEngine()

        # CryptoBERT label mapping
        # Model outputs: Bullish / Bearish (binary)
        self.label_map = {
            "bullish": 1.0,
            "bearish": -1.0,
            "positive": 1.0,   # in case model uses FinBERT-style labels
            "negative": -1.0,
            "neutral":  0.0,
        }

    def analyze(self, text: str) -> tuple[float, float, str]:
        """Returns (score, confidence, label)"""
        if not self.available:
            return self._fallback.analyze(text)

        try:
            result = self.pipe(text[:512])[0]
            label = result["label"].lower()
            confidence = result["score"]
            score = self.label_map.get(label, 0.0) * confidence
            return score, confidence, label
        except Exception:
            return 0.0, 0.0, "neutral"

    def is_crypto_text(self, text: str) -> bool:
        """Quick check: should this text use CryptoBERT vs FinBERT?"""
        crypto_indicators = [
            "btc", "eth", "sol", "crypto", "bitcoin", "ethereum",
            "defi", "nft", "web3", "altcoin", "hodl", "wagmi", "ngmi",
            "wen moon", "rekt", "fud", "pump", "dump", "whale",
            "funding rate", "perpetual", "liquidation",
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in crypto_indicators)
