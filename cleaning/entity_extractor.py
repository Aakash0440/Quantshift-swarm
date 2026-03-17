import re
from ingestion.base import RawSignal

TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
CRYPTO_PATTERN = re.compile(r'\b(BTC|ETH|SOL|BNB|AVAX|XRP)\b')

class EntityExtractor:
    def __init__(self, known_tickers: set):
        self.known_tickers = known_tickers

    def extract(self, signal: RawSignal) -> RawSignal:
        if signal.ticker:
            return signal
        text_upper = signal.text.upper()
        matches = TICKER_PATTERN.findall(signal.text)
        for m in matches:
            if m in self.known_tickers:
                signal.ticker = m
                return signal
        crypto_matches = CRYPTO_PATTERN.findall(text_upper)
        if crypto_matches:
            signal.ticker = crypto_matches[0]
            return signal
        for ticker in self.known_tickers:
            if f" {ticker} " in f" {text_upper} ":
                signal.ticker = ticker
                return signal
        return signal
