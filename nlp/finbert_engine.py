from transformers import pipeline
import torch
from dataclasses import dataclass

@dataclass
class SentimentResult:
    ticker: str
    source: str
    score: float
    confidence: float
    label: str
    trust_weight: float
    metadata: dict = None

class FinBERTEngine:
    def __init__(self):
        self.pipe = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1,
            max_length=512, truncation=True,
        )
        self.label_map = {"positive":1.0,"negative":-1.0,"neutral":0.0}

    def analyze(self, text: str) -> tuple:
        try:
            result = self.pipe(text[:512])[0]
            label = result["label"].lower()
            confidence = result["score"]
            score = self.label_map.get(label, 0.0) * confidence
            return score, confidence, label
        except Exception:
            return 0.0, 0.0, "neutral"
