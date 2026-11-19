# AlphaFilter (FinanceIQ v6): AI Market-Intelligence Dashboard

A full-stack financial analytics platform built for the Warwick Fintech Hackathon (February–March 2026), in **Category 1: Data in Finance**. The brief was to build a tool that uses financial data to inform decisions or identify opportunities.

*Team project with [Vansh Shah](https://github.com/vanshshah10002-prog). Originally developed in [ihstepura/hackathon_final_draft](https://github.com/ihstepura/hackathon_final_draft).*

## The problem

Retail investors are drowning in data: news, Reddit threads, SEC filings and price charts. Most tools only report what happened, such as "AAPL fell 3%". They rarely explain why, or whether the news still matters. AlphaFilter aims to turn that noise into an actionable signal for the advanced retail investor who can't afford a Bloomberg terminal.

## What makes it different

- **Signal decay:** news loses value as the market prices it in. Each article's sentiment score is discounted exponentially with its age, so a two-day-old headline counts for much less than one from ten minutes ago.
- **Divergence alerts:** compares 3-day price action with 3-day sentiment and flags two cases:
  - *Teflon stocks:* the price rises despite bad news.
  - *Value traps:* the price falls despite good news.
- **Supply-chain contagion:** pulls the Risk Factors section from a company's latest 10-K filing, uses an LLM to sort the risks into categories, and combines them with sentiment about peer companies into a contagion score.
- **Multi-agent analysis:** fundamental, technical and sentiment analyst agents each score the stock, then a director agent weighs their views into a consensus verdict. Their reasoning streams to the user live.
- **Explainable forecasts:** an LSTM model predicts the next 10 days of prices. It shows which input features drove the forecast, by measuring how the prediction changes when each feature is masked out (occlusion sensitivity).

## Features

The dashboard has four pages, plus an AI assistant in a sidebar available on every page:

| Page | What it shows |
|---|---|
| **Dashboard** | Company brief, key fundamentals, latest news and the 10-day price forecast |
| **Charts** | TradingView candlestick charts, technical indicators and detailed fundamentals |
| **News** | Sentiment for each article using FinBERT (a finance-tuned language model), with social media analytics |
| **Explain** | Which features drove each AI forecast, with a console for asking custom questions |

The AI assistant is fed the stock's actual data in its system prompt, so it answers from real numbers rather than generic text.

The backend also provides:
- Black-Scholes option pricing, the Greeks and implied volatility
- Monte Carlo price simulation using geometric Brownian motion
- A long/short strategy backtester
- Paper trading
- A global market overview

## Architecture

```
frontend/  Next.js 16 · React 19 · Tailwind · TradingView Lightweight Charts · Jotai
   │  REST + server-sent events (SSE) for live agent output
backend/   FastAPI · SQLAlchemy · Redis cache
   ├── agents/      multi-agent analysis (fundamental, technical, sentiment)
   ├── routers/     analysis, market, AI, agents, portfolio endpoints
   └── services/
        ├── alphamath.py           signal decay and divergence engine
        ├── contagion_service.py   SEC 10-K risk extraction and contagion score
        ├── ai_service.py          PyTorch LSTM forecast and feature importance
        ├── news_service.py        Google News RSS and FinBERT sentiment
        ├── social_service.py      Reddit and Twitter data
        ├── llm_service.py         Groq (Llama 3.3 70B), with Ollama as a fallback
        ├── options_service.py     Black-Scholes and Greeks
        └── backtest_service.py    long/short backtester
```

The Python files in the root folder (`app.py`, `portfolio_engine.py`, `options_engine.py` and others) are the earlier Flask version (FinanceIQ v5), described in [`README_v5_original.md`](README_v5_original.md). Parts of it were ported into the v6 backend.

## My contribution

I led the final build sprint (28 Feb–1 Mar 2026), with 21 of the repo's 28 commits:

- **Dashboard and real data:** redesigned the dashboard into the four-page layout and connected it to live API data.
- **AI assistant:** built the sidebar assistant and fed each stock's real financial data into its prompt.
- **Reliability fixes:** got the FinBERT per-article sentiment and the LSTM forecasts working reliably, and sourced social analytics from Finnhub news data.
- **Forecast chart:** built the predicted-trajectory chart with time and price labels.

## Running it locally

**Backend** (Python 3.10+, with PostgreSQL and Redis running):
```bash
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env     # add your API keys
uvicorn main:app --reload      # http://localhost:8000/docs
```

**Frontend** (Node 20+):
```bash
cd frontend
npm install
npm run dev                    # http://localhost:3000
```

API keys used: Groq, Finnhub, Financial Modeling Prep, FRED, Alpha Vantage and sec-api. Most features still work without some of them: the contagion engine has a demo mode, and the LLM falls back to a local Ollama model.

## Tech stack

Python · FastAPI · PyTorch · Hugging Face Transformers (FinBERT) · Groq / Llama 3.3 · yfinance · SQLAlchemy · Redis · Next.js · React · TypeScript · Tailwind CSS
