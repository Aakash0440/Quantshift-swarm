import yfinance as yf
import pandas as pd
from binance import AsyncClient
from datetime import datetime, timedelta, timezone

class MarketDataFetcher:
    async def get_ohlcv_stocks(self, ticker: str, interval: str = "1h", days: int = 90) -> pd.DataFrame:
        df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        return df

    async def get_ohlcv_crypto(self, symbol: str, interval: str = "1h", days: int = 90) -> pd.DataFrame:
        client = await AsyncClient.create()
        klines = await client.get_historical_klines(
            symbol.replace("/",""), interval,
            str(int((datetime.now(timezone.utc)-timedelta(days=days)).timestamp()*1000))
        )
        await client.close_connection()
        df = pd.DataFrame(klines, columns=["timestamp","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")[["open","high","low","close","volume"]].astype(float)
        return df

    async def get_funding_rates(self, symbol: str = "BTCUSDT") -> list:
        client = await AsyncClient.create()
        rates = await client.get_funding_rate(symbol=symbol, limit=100)
        await client.close_connection()
        return rates
