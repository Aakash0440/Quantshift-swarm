# execution/ccxt_adapter.py
# Crypto execution via CCXT — supports Binance, Bybit, Kraken, Coinbase, etc.
# Always uses LIMIT orders (never market orders).
# Testnet mode by default — set BINANCE_TESTNET=false for live.

import ccxt.async_support as ccxt_async
import os
from execution.base_executor import BaseExecutor, OrderResult, OrderStatus
from execution.slippage_model import SlippageModel


class CCXTAdapter(BaseExecutor):
    """
    Crypto execution adapter using CCXT library.
    Supports: Binance, Bybit, Kraken, OKX, Coinbase Advanced.

    Default: Binance testnet (paper trading).
    Switch to live: set BINANCE_TESTNET=false in .env.

    Order flow:
      1. Get current orderbook mid price
      2. Apply slippage model to get limit price
      3. Submit limit order with 3-retry logic
      4. Return OrderResult

    IMPORTANT: Never uses market orders — always limit.
    """

    EXCHANGE_CLASSES = {
        "binance": ccxt_async.binance,
        "bybit":   ccxt_async.bybit,
        "kraken":  ccxt_async.kraken,
        "okx":     ccxt_async.okx,
    }

    def __init__(self, config: dict):
        super().__init__(config)
        exchange_name = config.get("markets", {}).get("crypto", {}).get("exchange", "binance")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        exchange_cls = self.EXCHANGE_CLASSES.get(exchange_name, ccxt_async.binance)

        self.exchange = exchange_cls({
            "apiKey":    os.getenv("BINANCE_API_KEY", ""),
            "secret":    os.getenv("BINANCE_SECRET_KEY", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if testnet and exchange_name == "binance":
            self.exchange.set_sandbox_mode(True)

        self.slippage = SlippageModel(
            bps=config.get("execution", {}).get("slippage_model", {}).get("crypto_bps", 10)
        )
        self.max_retries = config.get("execution", {}).get("retry_attempts", 3)

    async def _get_mid_price(self, symbol: str) -> float:
        """Get mid price from orderbook."""
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=5)
            best_bid = ob["bids"][0][0] if ob["bids"] else 0
            best_ask = ob["asks"][0][0] if ob["asks"] else 0
            return (best_bid + best_ask) / 2
        except Exception:
            # Fallback to ticker
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])

    async def execute(self, decision, portfolio_value: float) -> OrderResult:
        """Execute a crypto trade decision."""
        if decision.action not in ("UP", "BUY", "DOWN", "SELL"):
            return OrderResult(
                order_id=None, ticker=decision.ticker,
                action=decision.action, status=OrderStatus.SKIPPED,
                reason=f"Non-trade action: {decision.action}",
            )

        # Map ticker to CCXT symbol
        symbol = f"{decision.ticker}/USDT"
        side = "buy" if decision.action in ("UP", "BUY") else "sell"

        try:
            mid_price = await self._get_mid_price(symbol)
        except Exception as e:
            return OrderResult(
                order_id=None, ticker=decision.ticker,
                action=decision.action, status=OrderStatus.FAILED,
                reason=f"Price fetch failed: {e}",
            )

        # Apply slippage
        limit_price = self.slippage.adjust(mid_price, side=side)
        trade_value = portfolio_value * decision.position_pct
        amount = trade_value / limit_price

        if amount < 0.001:
            return OrderResult(
                order_id=None, ticker=decision.ticker,
                action=decision.action, status=OrderStatus.SKIPPED,
                reason="Amount too small (< 0.001)",
            )

        # Submit with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                order = await self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=round(amount, 6),
                    price=round(limit_price, 6),
                )

                result = OrderResult(
                    order_id=str(order.get("id")),
                    ticker=decision.ticker,
                    action=decision.action,
                    status=OrderStatus.SUBMITTED,
                    shares=amount,
                    fill_price=limit_price,
                    estimated_value=trade_value,
                    slippage_cost=self.slippage.estimate_cost(trade_value),
                    raw_response=order,
                )
                self.log_order(result)
                return result

            except ccxt_async.InsufficientFunds:
                return OrderResult(
                    order_id=None, ticker=decision.ticker,
                    action=decision.action, status=OrderStatus.FAILED,
                    reason="Insufficient funds",
                )
            except Exception as e:
                last_error = e
                import asyncio
                await asyncio.sleep(1 * (attempt + 1))

        return OrderResult(
            order_id=None, ticker=decision.ticker,
            action=decision.action, status=OrderStatus.FAILED,
            reason=f"Failed after {self.max_retries} attempts: {last_error}",
        )

    async def get_portfolio_value(self) -> float:
        """Returns total USDT value of portfolio."""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_free = float(balance.get("USDT", {}).get("total", 0))
            return usdt_free
        except Exception as e:
            print(f"Portfolio value fetch failed: {e}")
            return 0.0

    async def get_positions(self) -> list[dict]:
        """Returns current open positions."""
        try:
            balance = await self.exchange.fetch_balance()
            positions = []
            for asset, values in balance.get("total", {}).items():
                if asset != "USDT" and float(values) > 0:
                    positions.append({"asset": asset, "amount": float(values)})
            return positions
        except Exception as e:
            print(f"Positions fetch failed: {e}")
            return []

    async def close(self) -> None:
        await self.exchange.close()
