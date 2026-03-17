from dataclasses import dataclass
from enum import Enum

class RegimeTier(Enum):
    STABLE   = "STABLE"
    WATCH    = "WATCH"
    CRITICAL = "CRITICAL"

@dataclass
class TradeDecision:
    ticker: str
    action: str
    position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    reason: str
    regime: object
    confidence: float

class KellySizer:
    def __init__(self, fraction: float = 0.25, max_pct: float = 0.05):
        self.fraction = fraction
        self.max_pct  = max_pct
    def size(self, win_prob: float, win_loss_ratio: float) -> float:
        b = win_loss_ratio; p = win_prob; q = 1.0 - p
        kelly = (b * p - q) / b
        return min(max(0.0, kelly * self.fraction), self.max_pct)

class AgentLoop:
    def __init__(self, config: dict):
        self.min_confidence = config.get("prediction",{}).get("min_confidence_threshold", 0.60)
        self.min_edge_pct   = config.get("decision",{}).get("min_edge_pct", 0.015)
        self.sizer = KellySizer(
            fraction=config.get("decision",{}).get("kelly_fraction", 0.25),
            max_pct=config.get("decision",{}).get("max_position_pct", 0.05),
        )

    def observe(self, prediction, regime) -> dict:
        return {"ticker": prediction.ticker, "direction": prediction.direction,
                "confidence": prediction.confidence, "magnitude": prediction.magnitude_pct,
                "regime": regime}

    def think(self, obs: dict) -> dict:
        regime     = obs["regime"]
        confidence = obs["confidence"]
        magnitude  = obs["magnitude"]
        regime_val = regime.value if hasattr(regime, "value") else str(regime)

        if regime_val == "CRITICAL":
            return {"action":"ALERT_HUMAN","reason":"CRITICAL regime — model stale, trading paused"}
        if confidence < self.min_confidence:
            return {"action":"HOLD","reason":f"Confidence {confidence:.1%} below {self.min_confidence:.1%}"}
        if magnitude < self.min_edge_pct * 100:
            return {"action":"HOLD","reason":f"Edge {magnitude:.2f}% too small"}
        if obs["direction"] == "NEUTRAL":
            return {"action":"HOLD","reason":"No directional signal"}

        scale = {"STABLE":1.0,"WATCH":0.5,"CRITICAL":0.0}.get(regime_val, 1.0)
        pos = self.sizer.size(win_prob=confidence,
                              win_loss_ratio=magnitude/max(magnitude*0.5,0.01)) * scale
        return {"action": obs["direction"], "position_pct": round(pos, 4),
                "reason": f"Edge {magnitude:.2f}% | Conf {confidence:.1%} | Regime {regime_val}"}

    def act(self, thought: dict, obs: dict) -> TradeDecision:
        mag = obs.get("magnitude", 1.0)
        return TradeDecision(
            ticker=obs["ticker"], action=thought.get("action","HOLD"),
            position_pct=thought.get("position_pct", 0.0),
            stop_loss_pct=mag*0.5, take_profit_pct=mag,
            reason=thought.get("reason",""), regime=obs["regime"],
            confidence=obs["confidence"],
        )

    def run(self, prediction, regime) -> TradeDecision:
        obs = self.observe(prediction, regime)
        return self.act(self.think(obs), obs)
