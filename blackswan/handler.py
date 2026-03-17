from datetime import datetime, timezone

class BlackSwanHandler:
    def __init__(self, config: dict, alerter):
        self.config   = config
        self.alerter  = alerter
        self.paused   = False
        self.consecutive_losses    = 0
        self.peak_portfolio_value  = None
        self.current_portfolio_value = 1.0

    def update_portfolio(self, value: float) -> None:
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = value
        self.peak_portfolio_value = max(self.peak_portfolio_value, value)
        self.current_portfolio_value = value

    def record_trade_result(self, pnl: float) -> None:
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0

    def _drawdown(self) -> float:
        if not self.peak_portfolio_value: return 0.0
        return (self.peak_portfolio_value - self.current_portfolio_value) / self.peak_portfolio_value

    async def check(self, regime, volatility_z: float = 0.0) -> bool:
        from regime.drift_detector import RegimeTier
        triggers = []
        regime_val = regime.value if hasattr(regime, "value") else str(regime)

        if regime_val == "CRITICAL":
            triggers.append("CRITICAL regime detected — model predictions unreliable")

        vol_limit = self.config.get("blackswan",{}).get("volatility_z_limit", 3.0)
        if volatility_z >= vol_limit:
            triggers.append(f"Volatility spike: {volatility_z:.1f}σ (limit: {vol_limit}σ)")

        loss_limit = self.config.get("decision",{}).get("consecutive_loss_limit", 3)
        if self.consecutive_losses >= loss_limit:
            triggers.append(f"{self.consecutive_losses} consecutive losses (limit: {loss_limit})")

        dd = self._drawdown()
        dd_limit = self.config.get("blackswan",{}).get("drawdown_pct_limit", 0.15)
        if dd >= dd_limit:
            triggers.append(f"Portfolio drawdown: {dd:.1%} (limit: {dd_limit:.1%})")

        if triggers:
            self.paused = True
            msg = "BLACK SWAN DETECTED — BOT PAUSED\n\nTriggers:\n"
            msg += "\n".join(f"  - {t}" for t in triggers)
            msg += "\n\nAction: All trading halted. Manual review required."
            await self.alerter.send(msg)
            return True
        return False

    def is_paused(self) -> bool:
        return self.paused

    def resume(self) -> None:
        self.paused = False
        self.consecutive_losses = 0
