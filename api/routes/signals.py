from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone

router = APIRouter()

class SignalResponse(BaseModel):
    ticker: str
    direction: str
    magnitude_pct: float
    confidence: float
    ci_low: float
    ci_high: float
    horizon: str
    regime: str
    explanation: str
    timestamp: datetime

@router.get("/{ticker}", response_model=SignalResponse)
async def get_signal(ticker: str):
    # In production: reads from Redis cache populated by run_live.py
    return SignalResponse(
        ticker=ticker.upper(), direction="UP", magnitude_pct=2.4,
        confidence=0.71, ci_low=1.1, ci_high=3.7, horizon="4h",
        regime="STABLE",
        explanation="Start run_live.py to populate live signals",
        timestamp=datetime.now(timezone.utc),
    )
