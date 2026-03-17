from datetime import datetime, timezone

class SignalExplainer:
    def explain(self, prediction, aggregated_signal, n_filtered: int = 0) -> str:
        icon = "UP" if prediction.direction == "UP" else "DOWN"
        lines = [
            f"[{icon}] SIGNAL: {prediction.ticker}",
            f"Direction: {prediction.direction}  |  {prediction.magnitude_pct:+.1f}%",
            f"Confidence: {prediction.confidence:.1%}",
            f"CI Range: [{prediction.ci_low:.1f}%, {prediction.ci_high:.1f}%]",
            f"Horizon: {prediction.horizon}",
            f"Regime: {getattr(prediction, 'regime', 'STABLE')}",
            "",
            "Why this prediction:",
        ]
        components = getattr(aggregated_signal, "components", {}) or {}
        sorted_components = sorted(
            components.items(),
            key=lambda x: abs(x[1][1]) if isinstance(x[1], tuple) else abs(x[1]),
            reverse=True,
        )
        for source, value in sorted_components:
            if source.startswith("_"): continue
            score, weight = value if isinstance(value, tuple) else (float(value), 0.5)
            sentiment_str = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
            lines.append(f"  {source:<22s}  {score:+.3f}  (wt {weight:.0%})  {sentiment_str}")

        mf_triggered = getattr(aggregated_signal, "mirofish_triggered", False)
        if mf_triggered:
            lines.append("  MiroFish swarm simulation contributed to this signal")

        if n_filtered > 0:
            lines.append(f"\n  {n_filtered} bot/duplicate signal(s) filtered from analysis")

        model_votes = getattr(prediction, "model_votes", {})
        if model_votes:
            lines.append("\nModel votes:")
            for model, v in model_votes.items():
                direction_vote, conf = v if isinstance(v, tuple) else (str(v), 0.5)
                icon2 = "UP" if direction_vote == "UP" else "DOWN" if direction_vote == "DOWN" else "~"
                lines.append(f"  {model:<14s}  {icon2}  {conf:.0%}")

        lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        return "\n".join(lines)
