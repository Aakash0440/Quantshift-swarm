# regime/regime_state.py
# Tracks current regime and maintains history.
# Provides regime-aware position scaling and re-entry logic.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque

from regime.drift_detector import RegimeTier


@dataclass
class RegimeEvent:
    timestamp: datetime
    tier: RegimeTier
    drift_score: float
    p_value: float
    note: str = ""


class RegimeState:
    """
    Stateful regime tracker.
    Maintains:
      - Current tier (STABLE / WATCH / CRITICAL)
      - How long we've been in current tier
      - Full history for analysis
      - Cooldown logic: don't flip back to STABLE too fast after CRITICAL

    Cooldown rules:
      CRITICAL → WATCH : requires 2 consecutive STABLE detections
      WATCH → STABLE   : requires 1 consecutive STABLE detection
      Any → CRITICAL   : immediate (safety first)
    """

    COOLDOWN_REQUIRED = {
        RegimeTier.CRITICAL: 2,  # need 2 clean checks before stepping down
        RegimeTier.WATCH: 1,
        RegimeTier.STABLE: 0,
    }

    def __init__(self):
        self.current_tier: RegimeTier = RegimeTier.STABLE
        self.tier_since: datetime = datetime.now(timezone.utc)
        self.consecutive_clean: int = 0
        self.history: deque = deque(maxlen=200)
        self._prev_tier: RegimeTier = RegimeTier.STABLE

    def update(self, new_tier: RegimeTier, drift_score: float, p_value: float) -> bool:
        """
        Update regime state with new detection result.
        Returns True if regime actually changed.
        """
        now = datetime.now(timezone.utc)
        changed = False

        # Always escalate immediately (safety)
        tier_severity = {RegimeTier.STABLE: 0, RegimeTier.WATCH: 1, RegimeTier.CRITICAL: 2}
        current_severity = tier_severity[self.current_tier]
        new_severity = tier_severity[new_tier]

        if new_severity > current_severity:
            # Escalation: immediate
            self._prev_tier = self.current_tier
            self.current_tier = new_tier
            self.tier_since = now
            self.consecutive_clean = 0
            changed = True
            note = f"ESCALATED from {self._prev_tier.value} to {new_tier.value}"

        elif new_severity < current_severity:
            # De-escalation: require cooldown
            self.consecutive_clean += 1
            required = self.COOLDOWN_REQUIRED.get(self.current_tier, 1)

            if self.consecutive_clean >= required:
                # Step down one level (not directly to STABLE from CRITICAL)
                target = RegimeTier.WATCH if self.current_tier == RegimeTier.CRITICAL else RegimeTier.STABLE
                self._prev_tier = self.current_tier
                self.current_tier = target
                self.tier_since = now
                self.consecutive_clean = 0
                changed = True
                note = f"DE-ESCALATED from {self._prev_tier.value} to {target.value} after {required} clean checks"
            else:
                note = f"Cooldown: {self.consecutive_clean}/{required} clean checks before de-escalating"

        else:
            # Same severity
            if new_severity == 0:
                self.consecutive_clean += 1
            note = f"Maintained {self.current_tier.value}"

        self.history.append(RegimeEvent(
            timestamp=now,
            tier=self.current_tier,
            drift_score=drift_score,
            p_value=p_value,
            note=note if 'note' in dir() else "",
        ))

        return changed

    def hours_in_current_tier(self) -> float:
        delta = datetime.now(timezone.utc) - self.tier_since
        return delta.total_seconds() / 3600

    def position_scale(self) -> float:
        """Returns position sizing multiplier for current regime."""
        return {
            RegimeTier.STABLE:   1.0,
            RegimeTier.WATCH:    0.5,
            RegimeTier.CRITICAL: 0.0,
        }[self.current_tier]

    def should_trade(self) -> bool:
        return self.current_tier != RegimeTier.CRITICAL

    def status_dict(self) -> dict:
        return {
            "tier": self.current_tier.value,
            "hours_in_tier": round(self.hours_in_current_tier(), 1),
            "position_scale": self.position_scale(),
            "should_trade": self.should_trade(),
            "consecutive_clean": self.consecutive_clean,
        }
