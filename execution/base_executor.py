# execution/base_executor.py
# Abstract base class for all execution adapters.
# All executors (Alpaca, CCXT, SignalOnly) must implement this interface.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class OrderStatus(Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class OrderResult:
    order_id: str | None
    ticker: str
    action: str              # BUY / SELL / HOLD
    status: OrderStatus
    shares: float = 0.0
    fill_price: float = 0.0
    estimated_value: float = 0.0
    slippage_cost: float = 0.0
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: dict = field(default_factory=dict)


class BaseExecutor(ABC):
    """
    Abstract executor interface.
    All adapters must implement execute() and get_portfolio_value().
    """

    def __init__(self, config: dict):
        self.config = config
        self.execution_log: list[OrderResult] = []

    @abstractmethod
    async def execute(self, decision, portfolio_value: float) -> OrderResult:
        """
        Execute a trade decision.
        decision: TradeDecision from agent_loop.py
        portfolio_value: current total portfolio value in USD
        Returns: OrderResult
        """
        pass

    @abstractmethod
    async def get_portfolio_value(self) -> float:
        """Returns current total portfolio value in USD."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Returns list of current open positions."""
        pass

    def log_order(self, result: OrderResult) -> None:
        self.execution_log.append(result)

    def execution_stats(self) -> dict:
        """Summary stats on all executed orders."""
        if not self.execution_log:
            return {"n_orders": 0}

        filled = [r for r in self.execution_log if r.status == OrderStatus.FILLED]
        skipped = [r for r in self.execution_log if r.status == OrderStatus.SKIPPED]
        failed = [r for r in self.execution_log if r.status == OrderStatus.FAILED]

        return {
            "n_total": len(self.execution_log),
            "n_filled": len(filled),
            "n_skipped": len(skipped),
            "n_failed": len(failed),
            "total_slippage_cost": sum(r.slippage_cost for r in filled),
        }
