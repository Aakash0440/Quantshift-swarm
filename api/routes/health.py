# api/routes/health.py
# Health check endpoint — confirms the bot is alive and all systems are green.
# Used by Railway/Docker for container health checks.

from fastapi import APIRouter
from api.models import HealthResponse
from datetime import datetime, timezone

router = APIRouter()

# Module-level state — updated by run_live.py via shared state or Redis
_start_time = datetime.now(timezone.utc)
_last_signal_at: datetime | None = None
_n_signals_today: int = 0
_current_regime: str = "STABLE"
_sources_active: list[str] = []


def update_health_state(
    last_signal_at: datetime | None = None,
    n_signals_today: int | None = None,
    regime: str | None = None,
    sources_active: list[str] | None = None,
) -> None:
    """Called by run_live.py to update health state."""
    global _last_signal_at, _n_signals_today, _current_regime, _sources_active
    if last_signal_at is not None:
        _last_signal_at = last_signal_at
    if n_signals_today is not None:
        _n_signals_today = n_signals_today
    if regime is not None:
        _current_regime = regime
    if sources_active is not None:
        _sources_active = sources_active


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns system health status.
    Used by Docker HEALTHCHECK and Railway health probe.

    status: "ok"       — all systems nominal
            "degraded" — running but some sources offline
            "down"     — critical failure
    """
    now = datetime.now(timezone.utc)
    uptime_seconds = (now - _start_time).total_seconds()
    uptime_hours = uptime_seconds / 3600

    # Determine health status
    if _current_regime == "CRITICAL":
        status = "degraded"   # running but trading paused
    elif _last_signal_at is None:
        status = "degraded"   # never sent a signal yet
    else:
        hours_since_signal = (now - _last_signal_at).total_seconds() / 3600
        status = "degraded" if hours_since_signal > 3 else "ok"

    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_hours=round(uptime_hours, 2),
        last_signal_at=_last_signal_at,
        regime=_current_regime,
        n_signals_today=_n_signals_today,
        sources_active=_sources_active,
        timestamp=now,
    )


@router.get("/ping", tags=["System"])
async def ping():
    """Minimal liveness probe — returns 200 OK immediately."""
    return {"status": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}
