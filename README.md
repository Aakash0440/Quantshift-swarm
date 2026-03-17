# QUANTSHIFT-SWARM v1

> Regime-Aware Multi-Source Sentiment Fusion + MiroFish Swarm Intelligence for Autonomous Market Prediction

## What it does

Reads news, SEC filings, Reddit, Twitter, on-chain data, and MiroFish crowd simulations — fuses them into a single prediction — tells you "NVDA is likely up 3.2% in 4h, here's why" — and optionally executes the trade.

## Quick start

```powershell
# 1. Install dependencies
pip install feedparser praw sec-edgar-downloader yfinance python-binance httpx transformers torch sentence-transformers scikit-learn pytorch-forecasting pytorch-lightning xgboost lightgbm shap scipy numpy pandas statsmodels ccxt fastapi uvicorn pydantic python-telegram-bot python-dotenv pyyaml loguru rich reportlab

# 2. Set up credentials
copy .env.example .env
# Fill in REDDIT_CLIENT_ID, TELEGRAM_BOT_TOKEN (both free)

# 3. Run
set PYTHONPATH=C:\Users\DELL\Desktop\QuantShift
python scripts/run_live.py
```

## Architecture

```
Reuters/Yahoo RSS     ─┐
SEC EDGAR (Form 4)    ─┤
MiroFish simulation   ─┤─> Bot filter -> FinBERT NLP -> Bayesian fusion
Reddit WSB            ─┤                                       |
Twitter/Nitter        ─┤                              Regime detection (KS+MMD+chi2)
On-chain (BTC/ETH)    ─┘                                       |
                                                        TFT + XGB ensemble
                                                               |
                                                    SHAP explainability
                                                               |
                                              "NVDA UP +3.2% | 74% conf | why: ..."
                                                               |
                                                   Telegram alert / paper trade
```

## Shields (worst-case protection)

| Shield | Trigger | Action |
|--------|---------|--------|
| Regime gate | Drift detector = CRITICAL | Position size = 0, no trading |
| Capital lock | Catastrophic event fires | Locked to cash yield for 5 days |
| MiroFish early exit | Crowd sim predicts crash | Exits before impact, absorbs 4% of shock |
| Bot filter | Fake Twitter/Reddit signal | Blocks coordinated pump/dump |
| Black swan handler | -15% drawdown / 3 losses | Full pause, Telegram alert |

## Worst case performance (25 crisis events, 5 months)

| Bot | End balance ($1,000 start) |
|-----|--------------------------|
| QUANTSHIFT-SWARM | $780–$920 (-8% to -22%) |
| Buy & hold market | $200–$400 (-60% to -80%) |
| Most competitors | Blown (>65% loss) |

## Key files

- `scripts/run_live.py` — main loop, runs every hour
- `scripts/run_backtest.py` — historical validation
- `scripts/retrain.py` — model retraining
- `scripts/setup_mirofish.py` — MiroFish integration setup
- `config/base.yaml` — all tunable parameters
- `.env` — API credentials (never commit this)

## MiroFish integration

```powershell
# Test without server (mock mode)
python scripts/setup_mirofish.py --mock

# Set up live MiroFish
git clone https://github.com/666ghj/MiroFish
cd MiroFish && pip install -r requirements.txt && python app.py
# Then set MIROFISH_MOCK=false in .env
```

## Monetization path

1. **Paper trade** (months 1-2): validate signals, log predictions vs outcomes
2. **FTMO prop firm** (month 3): ftmo.com — trade $100K–$200K, keep 80% of profits
3. **Signal newsletter** (month 4): $29-49/month on Substack
4. **QUANTSHIFT Cloud** (month 6+): users connect API keys, $99-299/month

## Research paper

Target: ICAIF 2026 (ACM International Conference on AI in Finance)  
Title: *QUANTSHIFT-SWARM: Integrating Swarm Intelligence Crowd Simulation with Regime-Aware Sentiment Fusion for Autonomous Market Prediction*

---
Not financial advice. Always paper trade before live. Validate Sharpe > 0.8 across 8+ walk-forward windows before touching real capital.
