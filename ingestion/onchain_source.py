# ingestion/onchain_source.py
# On-chain data: Glassnode free tier + Binance funding rates
# Free tier gives: exchange netflow, SOPR, MVRV, active addresses
# No API key required for some endpoints; optional key for higher limits

import httpx
import os
from datetime import datetime, timezone, timedelta
from ingestion.base import SignalSource, RawSignal

GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"
BINANCE_BASE   = "https://fapi.binance.com/fapi/v1"


class OnChainSource(SignalSource):
    """
    Pulls on-chain and derivatives data for crypto signals:
      - BTC/ETH exchange netflow  (are coins moving TO or FROM exchanges?)
        Inflow  → likely selling pressure → bearish signal
        Outflow → accumulation/withdrawal → bullish signal
      - Funding rates on perpetual futures
        Positive → longs paying shorts → market is bullish/overleveraged
        Negative → shorts paying longs → market is bearish/overleveraged
      - Whale transaction count (Glassnode free)

    Glassnode free tier: ~10 metrics, daily resolution, no key needed for some
    Binance funding rates: completely free, no key required
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.glassnode_key = os.getenv("GLASSNODE_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=15)

    async def _get_funding_rates(self, symbols: list[str]) -> list[dict]:
        """Binance perpetual futures funding rates — completely free."""
        results = []
        for symbol in symbols:
            try:
                url = f"{BINANCE_BASE}/fundingRate"
                params = {"symbol": symbol.replace("/", ""), "limit": 1}
                resp = await self.client.get(url, params=params)
                data = resp.json()
                if data and isinstance(data, list):
                    rate = float(data[-1].get("fundingRate", 0))
                    results.append({
                        "symbol": symbol,
                        "funding_rate": rate,
                        "annualized_pct": rate * 3 * 365 * 100,  # 3x/day * 365
                        "signal": "bullish" if rate > 0.001 else "bearish" if rate < -0.001 else "neutral",
                    })
            except Exception as e:
                print(f"Funding rate fetch failed for {symbol}: {e}")
        return results

    async def _get_exchange_netflow(self, asset: str = "BTC") -> dict | None:
        """
        Glassnode exchange netflow.
        Positive = net inflow to exchanges (bearish)
        Negative = net outflow from exchanges (bullish)
        """
        try:
            params = {"a": asset, "i": "24h"}
            if self.glassnode_key:
                params["api_key"] = self.glassnode_key

            url = f"{GLASSNODE_BASE}/transactions/transfers_volume_exchanges_net"
            resp = await self.client.get(url, params=params)
            data = resp.json()

            if data and isinstance(data, list):
                latest = data[-1]
                value = latest.get("v", 0)
                return {
                    "asset": asset,
                    "netflow_usd": value,
                    "signal": "bearish" if value > 0 else "bullish",
                    "timestamp": latest.get("t"),
                }
        except Exception as e:
            print(f"Glassnode netflow fetch failed for {asset}: {e}")
        return None

    async def _get_whale_transactions(self, asset: str = "BTC") -> dict | None:
        """Count of transactions > $100K — proxy for institutional activity."""
        try:
            params = {"a": asset, "i": "24h"}
            if self.glassnode_key:
                params["api_key"] = self.glassnode_key

            url = f"{GLASSNODE_BASE}/transactions/count_large"
            resp = await self.client.get(url, params=params)
            data = resp.json()

            if data and isinstance(data, list):
                current = data[-1].get("v", 0)
                prev = data[-2].get("v", 0) if len(data) > 1 else current
                change_pct = ((current - prev) / max(prev, 1)) * 100
                return {
                    "asset": asset,
                    "whale_tx_count": current,
                    "change_pct_24h": round(change_pct, 2),
                    "signal": "bullish" if change_pct > 10 else "bearish" if change_pct < -10 else "neutral",
                }
        except Exception as e:
            print(f"Glassnode whale tx fetch failed for {asset}: {e}")
        return None

    async def fetch(self, tickers: list[str], window_hours: int = 24) -> list[RawSignal]:
        signals = []

        # Only process crypto tickers
        crypto_map = {
            "BTC": "BTCUSDT", "ETH": "ETHUSDT",
            "SOL": "SOLUSDT", "BNB": "BNBUSDT", "AVAX": "AVAXUSDT",
        }
        crypto_tickers = [t for t in tickers if t in crypto_map]

        if not crypto_tickers:
            return signals

        # Funding rates (Binance, free)
        symbols = [crypto_map[t] for t in crypto_tickers if t in crypto_map]
        funding_rates = await self._get_funding_rates(symbols)

        for fr in funding_rates:
            ticker = fr["symbol"].replace("USDT", "")
            text = (
                f"On-chain: {ticker} funding rate {fr['funding_rate']:+.4f} "
                f"({fr['annualized_pct']:+.1f}% annualized). "
                f"Signal: {fr['signal']}. "
                f"{'Longs paying shorts = market overleveraged bullish.' if fr['funding_rate'] > 0 else 'Shorts paying longs = market overleveraged bearish.'}"
            )
            signals.append(RawSignal(
                source="onchain",
                ticker=ticker,
                text=text,
                url=None,
                timestamp=datetime.now(timezone.utc),
                trust_weight=0.80,
                metadata={
                    "metric": "funding_rate",
                    "value": fr["funding_rate"],
                    "signal": fr["signal"],
                }
            ))

        # Exchange netflow for BTC and ETH (Glassnode)
        for asset in ["BTC", "ETH"]:
            if asset in crypto_tickers:
                netflow = await self._get_exchange_netflow(asset)
                if netflow:
                    direction = "INTO" if netflow["netflow_usd"] > 0 else "OUT OF"
                    text = (
                        f"On-chain: {asset} exchange netflow {netflow['netflow_usd']:+,.0f} USD. "
                        f"Coins moving {direction} exchanges. "
                        f"Signal: {netflow['signal']}."
                    )
                    signals.append(RawSignal(
                        source="onchain",
                        ticker=asset,
                        text=text,
                        url=None,
                        timestamp=datetime.now(timezone.utc),
                        trust_weight=0.85,
                        metadata={
                            "metric": "exchange_netflow",
                            "value": netflow["netflow_usd"],
                            "signal": netflow["signal"],
                        }
                    ))

        # Whale transactions for BTC
        if "BTC" in crypto_tickers:
            whale = await self._get_whale_transactions("BTC")
            if whale:
                text = (
                    f"On-chain: BTC whale transactions (>$100K) = {whale['whale_tx_count']}. "
                    f"Change 24h: {whale['change_pct_24h']:+.1f}%. "
                    f"Signal: {whale['signal']}."
                )
                signals.append(RawSignal(
                    source="onchain",
                    ticker="BTC",
                    text=text,
                    url=None,
                    timestamp=datetime.now(timezone.utc),
                    trust_weight=0.80,
                    metadata={
                        "metric": "whale_transactions",
                        "value": whale["whale_tx_count"],
                        "signal": whale["signal"],
                    }
                ))

        await self.client.aclose()
        return signals
