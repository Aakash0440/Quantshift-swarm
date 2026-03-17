# Lazy imports — adapters loaded only when used, not on startup
# Prevents missing optional deps (alpaca, ccxt) from blocking startup

from execution.signal_only import SignalOnlyExecutor
from execution.slippage_model import SlippageModel

def get_alpaca():
    from execution.alpaca_adapter import AlpacaAdapter
    return AlpacaAdapter

def get_ccxt():
    from execution.ccxt_adapter import CCXTAdapter
    return CCXTAdapter

__all__ = ["SignalOnlyExecutor", "SlippageModel", "get_alpaca", "get_ccxt"]
