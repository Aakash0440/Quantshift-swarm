# blackswan/capital_lock.py
# Capital protection during CRITICAL regime / black swan events.
# Two modes:
#   "cash"  — do nothing, just stop trading, keep capital in cash
#   "tbill" — conceptually park in T-bills / stablecoin yield
#             (in paper mode, just track notional; in live, you'd manually move)

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class LockMode(Enum):
    UNLOCKED = "unlocked"
    CASH     = "cash"      # stop trading, hold cash
    TBILL    = "tbill"     # stop trading, move to safe yield instrument


@dataclass
class LockEvent:
    timestamp: datetime
    mode: LockMode
    reason: str
    portfolio_value: float
    unlocked_at: datetime | None = None


class CapitalLock:
    """
    When triggered, halts all execution and optionally moves to safe assets.

    In signal-only mode: just stops emitting trade signals.
    In paper mode: marks portfolio as "locked" in the ledger.
    In live mode: you need to manually close positions. The bot will alert you via Telegram.

    The lock automatically re-evaluates every 24h.
    Re-entry requires:
      - Regime back to STABLE or WATCH for 2+ consecutive checks
      - Volatility z-score below 2.0
      - Manual confirmation (optional, configurable)
    """

    def __init__(self, mode: str = "cash", require_manual_confirm: bool = False):
        self.lock_mode_str = mode
        self.require_manual_confirm = require_manual_confirm
        self.current_mode = LockMode.UNLOCKED
        self.locked_at: datetime | None = None
        self.portfolio_value_at_lock: float = 0.0
        self.history: list[LockEvent] = []
        self.manual_confirmed: bool = False

        # Notional yield while locked (T-bill proxy)
        # ~5% APY for cash, ~4.5% for T-bill
        self.annual_yield = {
            LockMode.CASH:  0.050,
            LockMode.TBILL: 0.045,
        }

    def lock(self, reason: str, portfolio_value: float) -> LockEvent:
        """Engage capital lock."""
        mode = LockMode.TBILL if self.lock_mode_str == "tbill" else LockMode.CASH

        event = LockEvent(
            timestamp=datetime.now(timezone.utc),
            mode=mode,
            reason=reason,
            portfolio_value=portfolio_value,
        )
        self.history.append(event)
        self.current_mode = mode
        self.locked_at = event.timestamp
        self.portfolio_value_at_lock = portfolio_value
        self.manual_confirmed = False

        print(f"[CapitalLock] LOCKED in {mode.value} mode. Reason: {reason}")
        return event

    def unlock(self) -> bool:
        """
        Attempt to unlock capital.
        Returns True if unlocked, False if blocked (e.g. awaiting manual confirm).
        """
        if self.require_manual_confirm and not self.manual_confirmed:
            print("[CapitalLock] Unlock blocked — awaiting manual confirmation.")
            return False

        if self.current_mode == LockMode.UNLOCKED:
            return True

        # Record unlock in last event
        if self.history:
            self.history[-1].unlocked_at = datetime.now(timezone.utc)

        self.current_mode = LockMode.UNLOCKED
        print("[CapitalLock] UNLOCKED. Trading can resume.")
        return True

    def confirm_manual(self) -> None:
        """Human confirms it's safe to re-enter. Call via API or Telegram command."""
        self.manual_confirmed = True
        print("[CapitalLock] Manual confirmation received.")

    def is_locked(self) -> bool:
        return self.current_mode != LockMode.UNLOCKED

    def notional_yield_since_lock(self) -> float:
        """Estimated yield earned while locked (notional, for reporting)."""
        if not self.locked_at or self.current_mode == LockMode.UNLOCKED:
            return 0.0
        hours_locked = (datetime.now(timezone.utc) - self.locked_at).total_seconds() / 3600
        annual_rate = self.annual_yield.get(self.current_mode, 0.0)
        return self.portfolio_value_at_lock * annual_rate * (hours_locked / 8760)

    def status(self) -> dict:
        return {
            "locked": self.is_locked(),
            "mode": self.current_mode.value,
            "locked_at": self.locked_at.isoformat() if self.locked_at else None,
            "notional_yield": round(self.notional_yield_since_lock(), 2),
            "awaiting_manual_confirm": self.require_manual_confirm and not self.manual_confirmed,
        }
