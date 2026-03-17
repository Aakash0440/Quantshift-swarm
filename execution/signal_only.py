class SignalOnlyExecutor:
    """
    Default executor — emits signals to Telegram/console only.
    Never touches real money. Use this for first 60+ days.
    """
    def __init__(self, alerter, explainer):
        self.alerter  = alerter
        self.explainer = explainer

    async def execute(self, decision, explanation: str) -> None:
        if decision.action in ("HOLD", "NEUTRAL"):
            return
        message = f"""
{explanation}

--- Trade Signal (Signal-Only Mode) ---
  Action:      {decision.action}
  Position:    {decision.position_pct:.1%} of portfolio
  Stop Loss:   -{decision.stop_loss_pct:.2f}%
  Take Profit: +{decision.take_profit_pct:.2f}%
  Regime:      {getattr(decision.regime,'value',str(decision.regime))}
  Confidence:  {decision.confidence:.1%}
"""
        await self.alerter.send(message)
