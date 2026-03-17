# api/models.py
# Pydantic schemas for all API request/response types.

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# ── Signal models ──────────────────────────────────────────────────────────────

class SignalComponent(BaseModel):
    source: str
    score: float
    weight: float
    sentiment: str   # "bullish" | "bearish" | "neutral"


class SignalResponse(BaseModel):
    ticker: str
    direction: str                  # UP | DOWN | NEUTRAL
    magnitude_pct: float
    confidence: float
    ci_low: float
    ci_high: float
    horizon: str                    # "4h" | "24h"
    regime: str                     # STABLE | WATCH | CRITICAL
    action: str                     # BUY | SELL | HOLD | ALERT_HUMAN
    position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    components: list[SignalComponent] = []
    n_bot_signals_filtered: int = 0
    explanation: str = ""
    model_votes: dict = {}
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "NVDA",
                "direction": "UP",
                "magnitude_pct": 3.2,
                "confidence": 0.741,
                "ci_low": 1.8,
                "ci_high": 4.6,
                "horizon": "4h",
                "regime": "STABLE",
                "action": "BUY",
                "position_pct": 0.007,
                "stop_loss_pct": 1.6,
                "take_profit_pct": 3.2,
                "n_bot_signals_filtered": 3,
                "explanation": "Reuters: bullish +0.42 (34%), SEC Form 4 insider buy (28%)",
                "timestamp": "2026-03-16T10:30:00Z",
            }
        }


class SignalListResponse(BaseModel):
    signals: list[SignalResponse]
    generated_at: datetime
    regime: str
    n_tickers_analyzed: int


# ── Regime models ──────────────────────────────────────────────────────────────

class RegimeResponse(BaseModel):
    tier: str               # STABLE | WATCH | CRITICAL
    drift_score: float
    p_value: float
    hours_in_tier: float
    position_scale: float   # 1.0 | 0.5 | 0.0
    should_trade: bool
    vol_z_score: float
    vix: Optional[float] = None
    triggered_tests: list[str] = []
    message: str
    timestamp: datetime


# ── Health models ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str             # "ok" | "degraded" | "down"
    version: str = "1.0.0"
    uptime_hours: float
    last_signal_at: Optional[datetime] = None
    regime: str
    n_signals_today: int
    sources_active: list[str] = []
    timestamp: datetime


# ── Portfolio models ──────────────────────────────────────────────────────────

class PositionResponse(BaseModel):
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime


class PortfolioResponse(BaseModel):
    total_value: float
    cash: float
    positions: list[PositionResponse] = []
    day_pnl: float
    day_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    drawdown_pct: float
    timestamp: datetime


# ── Backtest models ───────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    ticker: str = "NVDA"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    starting_capital: float = Field(default=100_000.0, gt=0)
    slippage_bps: float = Field(default=5.0, ge=0, le=50)


class BacktestMetrics(BaseModel):
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_holding_hours: float


class BacktestResponse(BaseModel):
    request: BacktestRequest
    metrics: BacktestMetrics
    equity_curve: list[float] = []
    report_url: Optional[str] = None
    completed_at: datetime
