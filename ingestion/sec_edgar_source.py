from sec_edgar_downloader import Downloader
import os
from datetime import datetime, timezone
from ingestion.base import SignalSource, RawSignal

class SECEdgarSource(SignalSource):
    def __init__(self):
        self.dl = Downloader("QuantShift","quantshift@example.com","/tmp/sec_filings")

    async def fetch(self, tickers: list, window_hours: int = 168) -> list:
        signals = []
        for ticker in tickers:
            for form_type in ["4","8-K"]:
                try:
                    self.dl.get(form_type, ticker, limit=5)
                    filing_dir = f"/tmp/sec_filings/{ticker}/{form_type.replace('-','')}"
                    if os.path.exists(filing_dir):
                        for fname in os.listdir(filing_dir)[:3]:
                            fpath = os.path.join(filing_dir, fname, "primary-document.html")
                            if os.path.exists(fpath):
                                with open(fpath) as f:
                                    content = f.read()[:1000]
                                signals.append(RawSignal(
                                    source=f"sec_{form_type.lower().replace('-','')}",
                                    ticker=ticker,
                                    text=f"SEC {form_type} filing for {ticker}: {content}",
                                    url=None,
                                    timestamp=datetime.now(timezone.utc),
                                    trust_weight=1.0,
                                    metadata={"form_type":form_type}
                                ))
                except Exception as e:
                    print(f"SEC EDGAR failed for {ticker} {form_type}: {e}")
        return signals
