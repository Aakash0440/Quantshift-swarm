class SlippageModel:
    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def adjust(self, price: float, side: str) -> float:
        slip = price * (self.bps / 10_000)
        return price + slip if side == "buy" else price - slip

    def estimate_cost(self, trade_value: float) -> float:
        return trade_value * (self.bps / 10_000)
