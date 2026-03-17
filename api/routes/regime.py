from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone

router = APIRouter()

class RegimeResponse(BaseModel):
    tier: str
    drift_score: float
    p_value: float
    message: str
    timestamp: datetime

@router.get("/status", response_model=RegimeResponse)
async def get_regime():
    return RegimeResponse(
        tier="STABLE", drift_score=0.12, p_value=0.44,
        message="Market operating within normal regime parameters",
        timestamp=datetime.now(timezone.utc),
    )
