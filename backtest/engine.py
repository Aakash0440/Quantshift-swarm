# backtest/engine.py
# Backtest engine wrapping the signal pipeline.
# Replays historical signals through the full decision stack.
# Applies slippage model and realistic fill simulation.

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime | None
    ticker: str
    direction: str
    entry_price: float
    exit_price: float | None
    shares: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    signal_confidence: float = 0.0
    regime: str = "STABLE"
    slippage_cost: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    starting_capital: float = 100_000.0
    final_capital: float = 0.0
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Simple vectorized backtester for QUANTSHIFT signals.

    How it works:
      1. Load historical OHLCV + pre-computed signals
      2. For each bar: run decision logic → generate trade
      3. Apply slippage model to entry/exit prices
      4. Track P&L, equity curve, drawdown

    Slippage applied:
      - Entry: price * (1 + slippage_bps/10000) for buys
      - Exit:  price * (1 - slippage_bps/10000) for sells

    NOT a full event-driven backtester (use Backtrader for that).
    This is a fast signal-replay engine for quick validation.
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
        commission_pct: float = 0.001,   # 0.1% per trade
        max_position_pct: float = 0.05,
    ):
        self.starting_capital = starting_capital
        self.slippage_bps = slippage_bps
        self.commission_pct = commission_pct
        self.max_position_pct = max_position_pct

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * (self.slippage_bps / 10_000)
        return price + slip if side == "buy" else price - slip

    def run(
        self,
        price_df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest.

        price_df:   DatetimeIndex, columns: open, high, low, close, volume
        signals_df: DatetimeIndex, columns:
                      ticker, direction, confidence, magnitude_pct,
                      horizon, regime, position_pct

        Both DataFrames must have the same DatetimeIndex.
        """
        result = BacktestResult(starting_capital=self.starting_capital)
        capital = self.starting_capital
        equity = [capital]
        positions: dict[str, dict] = {}   # ticker -> {shares, entry_price, entry_time}

        # Align indices
        common_index = price_df.index.intersection(signals_df.index)
        price_df = price_df.loc[common_index]
        signals_df = signals_df.loc[common_index]

        for ts in common_index:
            price_row = price_df.loc[ts]
            signal_row = signals_df.loc[ts]

            current_price = float(price_row.get("close", 0))
            if current_price <= 0:
                equity.append(capital)
                result.timestamps.append(ts)
                continue

            ticker = str(signal_row.get("ticker", "UNKNOWN"))
            direction = str(signal_row.get("direction", "NEUTRAL"))
            confidence = float(signal_row.get("confidence", 0))
            position_pct = float(signal_row.get("position_pct", 0))
            regime = str(signal_row.get("regime", "STABLE"))

            # Close existing position if direction reversed or regime critical
            if ticker in positions and (direction in ("DOWN", "NEUTRAL") or regime == "CRITICAL"):
                pos = positions.pop(ticker)
                exit_price = self._apply_slippage(current_price, "sell")
                commission = pos["shares"] * exit_price * self.commission_pct
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - commission
                pnl_pct = (exit_price / pos["entry_price"] - 1) * 100

                result.trades.append(Trade(
                    entry_time=pos["entry_time"],
                    exit_time=ts,
                    ticker=ticker,
                    direction="BUY",
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    shares=pos["shares"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    signal_confidence=confidence,
                    regime=regime,
                ))
                capital += pnl

            # Open new long position
            if (direction == "UP" and confidence >= 0.60 and
                    regime != "CRITICAL" and ticker not in positions):

                trade_value = capital * min(position_pct, self.max_position_pct)
                entry_price = self._apply_slippage(current_price, "buy")
                commission = trade_value * self.commission_pct
                shares = (trade_value - commission) / entry_price

                if shares > 0 and trade_value > 10:
                    positions[ticker] = {
                        "shares": shares,
                        "entry_price": entry_price,
                        "entry_time": ts,
                    }

            equity.append(capital)
            result.timestamps.append(ts)

        # Close remaining positions at end
        if price_df.shape[0] > 0:
            final_price = float(price_df.iloc[-1].get("close", 0))
            for ticker, pos in positions.items():
                exit_price = self._apply_slippage(final_price, "sell")
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                capital += pnl

        result.equity_curve = equity
        result.final_capital = capital

        from backtest.metrics import compute_metrics
        result.metrics = compute_metrics(
            equity_curve=np.array(equity),
            trades=result.trades,
            starting_capital=self.starting_capital,
        )

        return result
