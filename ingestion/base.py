from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator
import asyncio

@dataclass
class RawSignal:
    source: str
    ticker: str | None
    text: str
    url: str | None
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    trust_weight: float = 1.0

class SignalSource(ABC):
    @abstractmethod
    async def fetch(self, tickers: list, window_hours: int = 24) -> list:
        pass
    async def stream(self, tickers: list) -> AsyncGenerator:
        while True:
            signals = await self.fetch(tickers)
            for s in signals:
                yield s
            await asyncio.sleep(60)
