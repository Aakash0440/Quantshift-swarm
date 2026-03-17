# execution/alpaca_adapter.py
# Paper and live stock trading via Alpaca
# Install when ready: pip install alpaca-trade-api
# Keep BINANCE_TESTNET=true and SIGNAL_ONLY_MODE=true until backtesting is validated

import os

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from execution.base_executor import BaseExecutor, OrderResult, OrderStatus
from execution.slippage_model import SlippageModel


class AlpacaAdapter(BaseExecutor):
    """
    Stock execution via Alpaca Markets.
    Paper trading by default (ALPACA_BASE_URL points to paper API).
    Always uses LIMIT orders — never market orders.

    To install: pip install alpaca-trade-api
    To enable paper trading: set ALPACA_BASE_URL=https://paper-api.alpaca.markets in .env
    To go live: set ALPACA_BASE_URL=https://api.alpaca.markets (only after 60+ days validation)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.slippage = SlippageModel(
            bps=config.get("execution", {}).get("slippage_model", {}).get("stocks_bps", 5)
        )
        self.max_retries = config.get("execution", {}).get("retry_attempts", 3)
        self._api = None  # lazy init — only connect when needed

    def _get_api(self):
        """Lazy init Alpaca client — only called when actually executing."""
        if not ALPACA_AVAILABLE:
            raise RuntimeError(
                "alpaca_trade_api not installed. Run: pip install alpaca-trade-api"
            )
        if self._api is None:
            self._api = tradeapi.REST(
                key_id=os.getenv("ALPACA_API_KEY", ""),
                secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            )
        return self._api

    async def execute(self, decision, portfolio_value: float) -> OrderResult:
        """Execute a stock trade via Alpaca."""

        # Guard: alpaca not installed
        if not ALPACA_AVAILABLE:
            return OrderResult(
                order_id=None,
                ticker=getattr(decision, "ticker", "UNKNOWN"),
                action=getattr(decision, "action", "HOLD"),
                status=OrderStatus.SKIPPED,
                reason="alpaca_trade_api not installed — run: pip install alpaca-trade-api",
            )

        # Guard: non-trade actions
        if getattr(decision, "action", "HOLD") not in ("UP", "BUY"):
            return OrderResult(
                order_id=None,
                ticker=decision.ticker,
                action=decision.action,
                status=OrderStatus.SKIPPED,
                reason=f"Non-buy action: {decision.action}",
            )

        try:
            api = self._get_api()
            trade_value = portfolio_value * decision.position_pct

            # Get current price
            latest = api.get_latest_trade(decision.ticker)
            current_price = float(latest.price)

            # Apply slippage to get limit price
            limit_price = self.slippage.adjust(current_price, side="buy")
            shares = int(trade_value / limit_price)

            if shares < 1:
                return OrderResult(
                    order_id=None,
                    ticker=decision.ticker,
                    action=decision.action,
                    status=OrderStatus.SKIPPED,
                    reason=f"Position too small: ${trade_value:.2f} / ${limit_price:.2f} = {shares} shares",
                )

            # Submit limit order with retries
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    order = api.submit_order(
                        symbol=decision.ticker,
                        qty=shares,
                        side="buy",
                        type="limit",
                        limit_price=round(limit_price, 2),
                        time_in_force="day",
                    )
                    result = OrderResult(
                        order_id=str(order.id),
                        ticker=decision.ticker,
                        action=decision.action,
                        status=OrderStatus.SUBMITTED,
                        shares=float(shares),
                        fill_price=limit_price,
                        estimated_value=shares * limit_price,
                        slippage_cost=self.slippage.estimate_cost(trade_value),
                    )
                    self.log_order(result)
                    return result

                except Exception as e:
                    last_error = e
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))

            return OrderResult(
                order_id=None,
                ticker=decision.ticker,
                action=decision.action,
                status=OrderStatus.FAILED,
                reason=f"Failed after {self.max_retries} attempts: {last_error}",
            )

        except Exception as e:
            return OrderResult(
                order_id=None,
                ticker=getattr(decision, "ticker", "UNKNOWN"),
                action=getattr(decision, "action", "HOLD"),
                status=OrderStatus.FAILED,
                reason=str(e),
            )

    async def get_portfolio_value(self) -> float:
        """Returns total portfolio value in USD."""
        if not ALPACA_AVAILABLE:
            return 0.0
        try:
            account = self._get_api().get_account()
            return float(account.portfolio_value)
        except Exception as e:
            print(f"[AlpacaAdapter] Portfolio value fetch failed: {e}")
            return 0.0

    async def get_positions(self) -> list[dict]:
        """Returns current open positions."""
        if not ALPACA_AVAILABLE:
            return []
        try:
            positions = self._get_api().list_positions()
            return [
                {
                    "ticker": p.symbol,
                    "shares": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception as e:
            print(f"[AlpacaAdapter] Positions fetch failed: {e}")
            return []
