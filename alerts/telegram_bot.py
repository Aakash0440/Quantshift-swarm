import os

class TelegramAlerter:
    def __init__(self, config: dict = None):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.bot     = None
        if self.token and self.chat_id:
            try:
                from telegram import Bot
                self.bot = Bot(token=self.token)
            except ImportError:
                print("[Telegram] python-telegram-bot not installed")
        else:
            print("Telegram not configured — alerts print to console only")

    async def send(self, message: str) -> None:
        if self.bot:
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
            except Exception as e:
                print(f"[Telegram] Send failed: {e}\n{message}")
        else:
            print(f"[ALERT]\n{message}")
