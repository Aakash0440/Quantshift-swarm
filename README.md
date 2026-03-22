# QUANTSHIFT-SWARM

> Regime-aware sentiment fusion + MiroFish swarm intelligence for autonomous market prediction.

[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/status-paper%20trading-yellow?style=flat-square)]()
[![Target](https://img.shields.io/badge/paper-ICAIF%202026-red?style=flat-square)]()

Reads news, SEC filings, Reddit, Twitter, and on-chain data — runs them through a MiroFish crowd simulation — and outputs **"NVDA is likely up 3.2% in 4h, here's why"**. Optionally executes the trade.

---

## How it works

```
Reuters / Yahoo RSS  ┐
SEC EDGAR Form 4     ├──► Bot Filter ──► FinBERT NLP ──► Bayesian Fusion
Reddit WSB           │                                          │
Twitter / Nitter     │                               Regime Detection (KS+MMD+Chi²)
MiroFish Swarm       │                                          │
On-chain BTC/ETH     ┘                               TFT + XGB Ensemble + SHAP
                                                                │
                                        "NVDA UP +3.2% | 74% conf | why: ..."
                                                                │
                                               Telegram alert / paper trade
```

---

## Quickstart

```bash
# 1. Install
pip install feedparser praw sec-edgar-downloader yfinance python-binance httpx \
  transformers torch sentence-transformers scikit-learn pytorch-forecasting \
  pytorch-lightning xgboost lightgbm shap scipy numpy pandas statsmodels ccxt \
  fastapi uvicorn pydantic python-telegram-bot python-dotenv pyyaml loguru

# 2. Configure (REDDIT_CLIENT_ID + TELEGRAM_BOT_TOKEN — both free)
cp .env.example .env

# 3. Run
set PYTHONPATH=C:\Users\DELL\Desktop\QuantShift
python scripts/run_live.py
```

**MiroFish (mock mode — no server needed):**
```bash
python scripts/setup_mirofish.py --mock
```

---

## Worst-case results

25 crisis events, 5 months, $1,000 starting capital.

| System | End balance | Max drawdown |
|---|---|---|
| **QUANTSHIFT-SWARM** | **$780 – $920** | −8% to −22% |
| Buy & hold | $200 – $400 | −60% to −80% |
| Most competitors | Blown | >65% |

---

## Circuit breakers

| Shield | Trigger | Action |
|---|---|---|
| Regime gate | Drift = CRITICAL | Position size → 0 |
| Capital lock | Catastrophic event | Cash-only for 5 days |
| MiroFish exit | Swarm predicts crash | Exit before impact |
| Bot filter | Coordinated pump/dump | Signal blocked |
| Black swan | −15% drawdown / 3 losses | Full pause + Telegram alert |
| Drawdown ceiling | Breach threshold | Auto-reduce exposure |

---

## Key files

| File | Purpose |
|---|---|
| `scripts/run_live.py` | Main loop (hourly) |
| `scripts/run_backtest.py` | Walk-forward validation |
| `scripts/retrain.py` | Model retraining |
| `scripts/setup_mirofish.py` | MiroFish setup |
| `config/base.yaml` | Tunable parameters |
| `.env` | API credentials — never commit |

---

## Roadmap

| Phase | Timeline | |
|---|---|---|
| Paper trade | Months 1–2 | Validate signals, Sharpe > 0.8 across 8+ windows |
| FTMO prop firm | Month 3 | Trade $100K–200K, keep 80% of profits |
| Signal newsletter | Month 4 | $29–49/mo on Substack |
| QUANTSHIFT Cloud | Month 6+ | $99–299/mo SaaS |

---

## Research

**QUANTSHIFT-SWARM: Integrating Swarm Intelligence Crowd Simulation with Regime-Aware Sentiment Fusion for Autonomous Market Prediction**  
Target: ICAIF 2026 — ACM International Conference on AI in Finance

---

> Not financial advice. Paper trade first. Validate Sharpe > 0.8 across 8+ walk-forward windows before touching real capital.  
> MIT © 2026 Aakash Ali
