# decision/signal_formatter.py
# Formats trade signals into clean, human-readable strings for Telegram/API.
# Two formats: SHORT (one-liner for quick scan) and FULL (detailed breakdown).

from datetime import datetime, timezone


def format_signal(
    ticker: str,
    direction: str,
    magnitude_pct: float,
    confidence: float,
    ci_low: float,
    ci_high: float,
    horizon: str,
    regime: str,
    components: dict,
    action: str,
    position_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    n_filtered: int = 0,
    model_votes: dict | None = None,
    short: bool = False,
) -> str:
    """
    Generate formatted signal string.

    Short format (for quick Telegram previews):
      🟢 NVDA UP +3.2% | 74% confidence | 4h | STABLE

    Full format:
      Full breakdown with explanation, components, action details.
    """

    direction_icon = {
        "UP":      "🟢",
        "DOWN":    "🔴",
        "NEUTRAL": "🟡",
    }.get(direction, "⚪")

    regime_icon = {
        "STABLE":   "✅",
        "WATCH":    "⚠️",
        "CRITICAL": "🚨",
    }.get(regime, "❓")

    if short:
        return (
            f"{direction_icon} {ticker} {direction} {magnitude_pct:+.1f}% "
            f"| {confidence:.0%} conf | {horizon} | {regime} {regime_icon}"
        )

    # Full format
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{direction_icon} SIGNAL: {ticker}  |  {direction}  |  {magnitude_pct:+.1f}%",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Confidence:   {confidence:.1%}",
        f"CI Range:     [{ci_low:+.1f}%, {ci_high:+.1f}%]",
        f"Horizon:      {horizon}",
        f"Regime:       {regime} {regime_icon}",
        "",
        "Why this prediction:",
    ]

    # Sort components by weight
    sorted_components = sorted(
        components.items(),
        key=lambda x: abs(x[1][1]) if isinstance(x[1], tuple) else abs(x[1]),
        reverse=True,
    )

    for source, value in sorted_components:
        if isinstance(value, tuple):
            score, weight = value
        else:
            score, weight = float(value), 0.5

        sentiment_str = "bullish ↑" if score > 0.05 else "bearish ↓" if score < -0.05 else "neutral →"
        lines.append(f"  {source:<20s}  {score:+.3f}  (wt {weight:.0%})  {sentiment_str}")

    if n_filtered > 0:
        lines.append(f"\n  ⚠️  {n_filtered} bot/duplicate signal(s) filtered")

    if model_votes:
        lines.append("")
        lines.append("Model votes:")
        for model, (direction_vote, conf) in model_votes.items():
            vote_icon = "🟢" if direction_vote == "UP" else "🔴" if direction_vote == "DOWN" else "🟡"
            lines.append(f"  {model:<12s}  {vote_icon} {direction_vote:<8s}  {conf:.0%}")

    lines.append("")
    lines.append("━━━ TRADE ACTION ━━━")

    action_icon = {
        "UP":          "📈 BUY",
        "BUY":         "📈 BUY",
        "DOWN":        "📉 SELL / SHORT",
        "HOLD":        "⏸ HOLD",
        "ALERT_HUMAN": "🚨 MANUAL REVIEW",
        "NEUTRAL":     "⏸ HOLD",
    }.get(action, f"• {action}")

    lines.append(f"Action:       {action_icon}")

    if position_pct > 0:
        lines.append(f"Position:     {position_pct:.1%} of portfolio")
        lines.append(f"Stop Loss:    -{stop_loss_pct:.2f}%")
        lines.append(f"Take Profit:  +{take_profit_pct:.2f}%")

    lines.append(f"Generated:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_regime_change(old_regime: str, new_regime: str, drift_score: float, p_value: float) -> str:
    """Format a regime change alert."""
    icon = "🚨" if new_regime == "CRITICAL" else "⚠️" if new_regime == "WATCH" else "✅"
    return (
        f"{icon} REGIME CHANGE DETECTED\n\n"
        f"  {old_regime} → {new_regime}\n"
        f"  Drift score: {drift_score:.3f}\n"
        f"  P-value: {p_value:.4f}\n\n"
        f"  {'Trading PAUSED — manual review required.' if new_regime == 'CRITICAL' else 'Position sizes reduced to 50%.' if new_regime == 'WATCH' else 'Trading resumed — regime normalized.'}"
    )


def format_blackswan(triggers: list[str], portfolio_value: float, drawdown: float) -> str:
    """Format a black swan alert."""
    trigger_lines = "\n".join(f"  • {t}" for t in triggers)
    return (
        f"🚨🚨 BLACK SWAN DETECTED — BOT PAUSED 🚨🚨\n\n"
        f"Triggers:\n{trigger_lines}\n\n"
        f"Portfolio: ${portfolio_value:,.2f}\n"
        f"Drawdown:  {drawdown:.1%}\n\n"
        f"Action: All trading halted. Capital preserved.\n"
        f"Manual review required before resuming.\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
