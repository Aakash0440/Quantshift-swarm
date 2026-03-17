class KellySizer:
    """Quarter Kelly position sizer. Never risks more than max_pct of portfolio."""
    def __init__(self, fraction: float = 0.25, max_pct: float = 0.05):
        self.fraction = fraction
        self.max_pct  = max_pct

    def size(self, win_prob: float, win_loss_ratio: float) -> float:
        b = win_loss_ratio; p = win_prob; q = 1.0 - p
        kelly = (b * p - q) / b
        fractional = max(0.0, kelly * self.fraction)
        return min(fractional, self.max_pct)
