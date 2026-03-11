# Axiom Core Bot — Recall Network Competition Agent

**AEF v1.0 · BTC · SOL · CHZ · ETH**

A systematic trading bot implementing the Unified Axiom Execution Framework
for the Recall Network Crypto Paper Trading and Perpetual Futures competitions.

---

## Quick Start

```bash
# 1. Clone / copy this directory
cd axiom_core_bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
echo "RECALL_API_KEY=pk_live_your_key_here" > .env

# 4. Test in sandbox (dry-run — no trades execute)
python axiom_core_bot.py --mode sandbox --dry-run

# 5. Run with HITL approval prompts
python axiom_core_bot.py --mode sandbox --hitl

# 6. Fully autonomous (live competition)
python axiom_core_bot.py --mode production
```

---

## Architecture — Four-Signal Stack

```
Market Data Collection
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Regime Classifier                              │
│  BTC-Nasdaq Corr + VIX + ETF Flow Direction     │
│  → macro_driven | neutral | narrative_driven    │
└─────────────────┬───────────────────────────────┘
                  │  Adaptive Weights
                  ▼
┌──────────────────────────────────────────────────────────┐
│  L1 Macro Baseline     (30–45%)  │  L2 On-Chain (35%)   │
│  L3 Sentiment NLP       (8–35%)  │  L4 Technical (12–20%)│
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              Quorum Engine
         Score = Σ(weight × layer_score)
         Long   if Score ≥ +0.65
         Short  if Score ≤ -0.65
         No trade otherwise
                       │
                       ▼
          Fractional Kelly Sizer
          Position = f_half × conviction_scaler × portfolio
          Hard caps: BTC 20% | ETH 18% | SOL 15% | CHZ 8%
                       │
                       ▼
           Triple-Barrier Signal
           Stop-Loss | Take-Profit×2 | Time Barrier
                       │
                       ▼
            HITL Interrupt (--hitl flag)
            Human approval before execution
                       │
                       ▼
            Recall API → Trade Execute
```

---

## Layer Details

| Layer | Weight (Neutral) | Primary Driver |
|-------|-----------------|----------------|
| L1 Macro | 30% | VIX, BTC-Nasdaq correlation, ETF flows, F&G |
| L2 On-Chain | 35% | Funding rates, exchange flows, structural levels |
| L3 Sentiment | 20% | Fear & Greed, social velocity |
| L4 Technical | 15% | Support/resistance zones — no mid-range trades |

Weights shift dynamically based on detected regime:
- **Macro-Driven** (VIX>28, Corr>70%, ETF outflows): L1→45%, Sentiment→8%
- **Narrative-Driven** (VIX<18, Corr<40%, ETF inflows): Sentiment→35%, L1→15%

---

## Triple-Barrier Parameters

| Asset | Stop-Loss | TP1 | TP2 | Time Limit |
|-------|-----------|-----|-----|------------|
| BTC | -5% | +8% | +15% | 21 days |
| ETH | -5% | +8% | +15% | 18 days |
| SOL | -4% | +7% | +14% | 10 days |
| CHZ | -6% | +12% | — | 5 days |

---

## Kill Switch

If portfolio drawdown exceeds **37.5%** (1.5× the 25% historical max),
the bot:
1. Flattens all positions immediately via market orders
2. Sets state to HALTED
3. Requires manual restart

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RECALL_API_KEY` | Your Recall Network API key (from agent registration) |

---

## Competition URLs

| Mode | URL |
|------|-----|
| Sandbox | `https://api.sandbox.competitions.recall.network` |
| Production | `https://api.competitions.recall.network` |

---

## Extending the Bot

The bot is modular — each layer is a standalone function:

- **`score_macro()`** — add live VIX API, Fed NLP, ETF flow tracker
- **`score_onchain()`** — add Glassnode / Dune Analytics webhooks
- **`score_sentiment()`** — add LunarCrush social velocity API
- **`score_technical()`** — update `STRUCTURE_LEVELS` dict with fresh levels
- **`KELLY_PARAMS`** — update win-rate/payoff as backtest data accumulates
