# backtest/metrics.py
# Computes standard trading performance metrics from backtest results.

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.engine import Trade


def compute_metrics(
    equity_curve: np.ndarray,
    trades: list,
    starting_capital: float = 100_000.0,
) -> dict:
    """
    Compute full performance metrics from equity curve + trade list.

    Returns:
        total_return_pct
        annualized_return_pct
        sharpe_ratio             (annualized, assumes hourly bars)
        sortino_ratio
        max_drawdown_pct
        calmar_ratio             (annualized_return / max_drawdown)
        win_rate
        avg_win_pct
        avg_loss_pct
        profit_factor
        n_trades
        avg_holding_hours
    """
    metrics = {}

    if len(equity_curve) < 2:
        return {"error": "insufficient_data"}

    # ── Returns ────────────────────────────────────────────────────────────
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)

    total_return = (equity_curve[-1] / starting_capital - 1) * 100
    n_bars = len(equity_curve)
    # Assume hourly bars: annualize by multiplying by sqrt(252 * 24)
    annualized_return = total_return * (8760 / n_bars)

    metrics["total_return_pct"]      = round(float(total_return), 2)
    metrics["annualized_return_pct"] = round(float(annualized_return), 2)

    # ── Sharpe ────────────────────────────────────────────────────────────
    if returns.std() > 1e-9:
        sharpe = float(returns.mean() / returns.std()) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = round(sharpe, 3)

    # ── Sortino (downside only) ───────────────────────────────────────────
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 1e-9:
        sortino = float(returns.mean() / downside.std()) * np.sqrt(252 * 24)
    else:
        sortino = 0.0
    metrics["sortino_ratio"] = round(sortino, 3)

    # ── Max Drawdown ─────────────────────────────────────────────────────
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / (peak + 1e-9)
    max_dd = float(drawdowns.min()) * 100
    metrics["max_drawdown_pct"] = round(max_dd, 2)

    # ── Calmar ───────────────────────────────────────────────────────────
    if abs(max_dd) > 0.01:
        calmar = annualized_return / abs(max_dd)
    else:
        calmar = 0.0
    metrics["calmar_ratio"] = round(calmar, 3)

    # ── Trade-level stats ─────────────────────────────────────────────────
    if trades:
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]

        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics["n_trades"]   = len(trades)
        metrics["win_rate"]   = round(len(wins) / len(trades) if trades else 0, 3)
        metrics["avg_win_pct"]  = round(float(np.mean([t.pnl_pct for t in trades if t.pnl > 0])) if wins else 0, 3)
        metrics["avg_loss_pct"] = round(float(np.mean([t.pnl_pct for t in trades if t.pnl <= 0])) if losses else 0, 3)

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        metrics["profit_factor"] = round(gross_profit / max(gross_loss, 0.01), 3)

        # Average holding time
        holding_hours = []
        for t in trades:
            if t.exit_time and t.entry_time:
                h = (t.exit_time - t.entry_time).total_seconds() / 3600
                holding_hours.append(h)
        metrics["avg_holding_hours"] = round(float(np.mean(holding_hours)) if holding_hours else 0, 1)

    else:
        metrics.update({
            "n_trades": 0, "win_rate": 0, "avg_win_pct": 0,
            "avg_loss_pct": 0, "profit_factor": 0, "avg_holding_hours": 0,
        })

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty print metrics to console."""
    print("\n" + "=" * 45)
    print("QUANTSHIFT BACKTEST RESULTS")
    print("=" * 45)
    print(f"  Total Return:        {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Annualized Return:   {metrics.get('annualized_return_pct', 0):+.2f}%")
    print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}")
    print(f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}")
    print("-" * 45)
    print(f"  Trades:              {metrics.get('n_trades', 0)}")
    print(f"  Win Rate:            {metrics.get('win_rate', 0):.1%}")
    print(f"  Avg Win:             {metrics.get('avg_win_pct', 0):+.2f}%")
    print(f"  Avg Loss:            {metrics.get('avg_loss_pct', 0):+.2f}%")
    print(f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
    print(f"  Avg Hold Time:       {metrics.get('avg_holding_hours', 0):.1f}h")
    print("=" * 45 + "\n")
