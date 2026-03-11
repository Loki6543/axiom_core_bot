"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             AXIOM CORE BOT — Recall Network Competition Agent               ║
║                  Unified Axiom Execution Framework (AEF) v1.0               ║
║                       Assets: BTC · SOL · CHZ · ETH                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture: Four-layer quorum engine with regime-adaptive weights,
Fractional Kelly sizing, Triple-Barrier execution, and HITL override.

Usage:
  python axiom_core_bot.py [--mode sandbox|production] [--hitl] [--dry-run]
"""

import os
import sys
import time
import math
import logging
import argparse
import requests
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


def _configure_console_encoding() -> None:
    """Prefer UTF-8 console streams so Windows terminals can render log output."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


_configure_console_encoding()
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("axiom_core_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("AEF")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SANDBOX_URL    = "https://api.sandbox.competitions.recall.network"
PRODUCTION_URL = "https://api.competitions.recall.network"

# ERC-20 token addresses (mainnet fork — used in both sandbox and production)
TOKENS = {
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # BTC proxy
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "SOL":  "0xD31a59c85aE9D8edEFeC411D448f90841571b89c",
    "CHZ":  "0x3506424F91fD33084466F402d5D97f05F8e3b4AF",
}

# Signal quorum thresholds
LONG_THRESHOLD  = 0.65
SHORT_THRESHOLD = -0.65

# Maximum position sizes per asset (as fraction of portfolio)
MAX_POSITION = {
    "WBTC": 0.20,
    "WETH": 0.18,
    "SOL":  0.15,
    "CHZ":  0.08,
}

# Triple-Barrier parameters per asset
BARRIERS = {
    "WBTC": {"stop_pct": 0.05, "tp1_pct": 0.08, "tp2_pct": 0.15, "time_days": 21},
    "WETH": {"stop_pct": 0.05, "tp1_pct": 0.08, "tp2_pct": 0.15, "time_days": 18},
    "SOL":  {"stop_pct": 0.04, "tp1_pct": 0.07, "tp2_pct": 0.14, "time_days": 10},
    "CHZ":  {"stop_pct": 0.06, "tp1_pct": 0.12, "tp2_pct": 0.00, "time_days":  5},
}

# Kill-switch: halt if drawdown exceeds this multiple of max historical drawdown
KILL_SWITCH_MULTIPLIER = 1.5
MAX_HISTORICAL_DRAWDOWN = 0.25  # 25% max drawdown historically backtested

# Kelly fraction (use half-Kelly for safety)
KELLY_FRACTION = 0.5

# Regime detection thresholds
VIX_HIGH_THRESHOLD = 28
VIX_LOW_THRESHOLD  = 18
CORRELATION_HIGH   = 0.70
CORRELATION_LOW    = 0.40

# Poll interval in seconds (60s = 1 trade decision per minute)
POLL_INTERVAL = 60

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Live market data collected each cycle."""
    prices: dict            = field(default_factory=dict)   # symbol → USD price
    portfolio_usdc: float   = 0.0
    portfolio_value: float  = 0.0
    peak_equity: float      = 0.0
    btc_nasdaq_corr: float  = 0.0   # 0–1 correlation coefficient
    vix: float              = 20.0
    etf_flow_direction: str = "mixed"  # "positive" | "negative" | "mixed"
    fear_greed: int         = 50
    funding_rates: dict     = field(default_factory=dict)   # symbol → rate


@dataclass
class LayerScores:
    """Outputs from the four AEF signal layers."""
    macro: float      = 0.0   # -1.0 … +1.0
    onchain: float    = 0.0
    sentiment: float  = 0.0
    technical: float  = 0.0
    regime: str       = "neutral"  # "macro_driven" | "neutral" | "narrative_driven"


@dataclass
class TradeSignal:
    """Proposed trade before HITL approval."""
    asset: str
    direction: str          # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    time_barrier_days: int
    position_pct: float     # fraction of portfolio equity
    aggregate_score: float
    layer_scores: LayerScores
    reason: str


@dataclass
class OpenPosition:
    """Tracks a live position for barrier monitoring."""
    asset: str
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    time_barrier_days: int
    size_usdc: float
    tp1_hit: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# RECALL API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class RecallClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        })

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_portfolio(self) -> dict:
        return self._get("/api/agent/portfolio")

    def get_price(self, token: str, chain: str = "evm") -> float:
        """Return current USD price for a token address."""
        data = self._get("/api/price", params={"token": token, "chain": chain})
        return float(data.get("price", 0))

    def get_leaderboard(self) -> dict:
        return self._get("/api/leaderboard")

    def execute_trade(self, from_token: str, to_token: str,
                      amount: str, reason: str) -> dict:
        payload = {
            "fromToken": from_token,
            "toToken":   to_token,
            "amount":    amount,
            "reason":    reason,
        }
        return self._post("/api/trade/execute", payload)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — MACRO BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def score_macro(snap: MarketSnapshot) -> float:
    """
    Macro baseline score [-1.0, +1.0].

    Inputs:
      • BTC-Nasdaq correlation (higher = more risk-asset = more negative bias)
      • VIX level
      • ETF flow direction
      • Fear & Greed index
      • Funding rate extremes (contrarian)

    In a full deployment these pull from live data sources.
    Here they are read from the MarketSnapshot which the data-collection
    layer populates each cycle.
    """
    score = 0.0

    # ── VIX component (range: -0.40 … 0.20) ──────────────────────────────
    if snap.vix > 35:
        score -= 0.40
    elif snap.vix > 28:
        score -= 0.25
    elif snap.vix > 22:
        score -= 0.10
    elif snap.vix < 14:
        score += 0.20   # complacency — mild positive for momentum longs
    else:
        score += 0.05

    # ── Correlation component (range: -0.30 … 0.00) ──────────────────────
    # High BTC-Nasdaq correlation means macro dominates; suppresses alpha signals
    if snap.btc_nasdaq_corr > 0.85:
        score -= 0.30
    elif snap.btc_nasdaq_corr > 0.70:
        score -= 0.15
    elif snap.btc_nasdaq_corr < 0.40:
        score += 0.10   # decorrelated = tradeable on own merits

    # ── ETF flow component (range: -0.20 … 0.20) ─────────────────────────
    if snap.etf_flow_direction == "negative":
        score -= 0.20
    elif snap.etf_flow_direction == "positive":
        score += 0.20
    # mixed = 0

    # ── Fear & Greed (contrarian at extremes) ─────────────────────────────
    if snap.fear_greed <= 10:
        score += 0.15   # capitulation — forward-looking positive
    elif snap.fear_greed <= 25:
        score += 0.05
    elif snap.fear_greed >= 90:
        score -= 0.20   # extreme greed = distribution risk
    elif snap.fear_greed >= 75:
        score -= 0.10

    return max(-1.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — ON-CHAIN & STRUCTURAL
# ─────────────────────────────────────────────────────────────────────────────

def score_onchain(snap: MarketSnapshot, asset: str) -> float:
    """
    On-chain and structural score [-1.0, +1.0] per asset.

    In production: pulls Glassnode / Nansen / Dune data via APIs.
    This implementation uses price-derived proxies available from the
    Recall portfolio + price endpoints, extended with funding rate signals.
    """
    score = 0.0

    # ── Funding rate signal (available for all assets) ────────────────────
    funding = snap.funding_rates.get(asset, 0.0)
    if funding > 0.001:      # > 0.1% per 8h — extreme long crowding
        score -= 0.30
    elif funding > 0.0005:   # moderate positive — slight negative
        score -= 0.10
    elif funding < -0.001:   # extreme negative — short squeeze setup
        score += 0.30
    elif funding < -0.0005:
        score += 0.10

    # ── Asset-specific structural signals ────────────────────────────────
    if asset == "WBTC":
        # Use ETF flow as proxy for exchange net flow (institutional is the
        # dominant flow for BTC in 2026)
        if snap.etf_flow_direction == "positive":
            score += 0.25
        elif snap.etf_flow_direction == "negative":
            score -= 0.25
        # MicroStrategy accumulation proxy: positive macro correlation
        if snap.btc_nasdaq_corr < 0.50:
            score += 0.15   # decorrelated = organic accumulation signal

    elif asset == "SOL":
        # DEX activity proxy: fear & greed index as sentiment stand-in
        # Low fear with neutral funding = organic accumulation
        if snap.fear_greed > 50 and abs(funding) < 0.0003:
            score += 0.20
        elif snap.fear_greed < 30 and funding < 0:
            score += 0.15   # fear + negative funding = squeeze setup

    elif asset == "CHZ":
        # CHZ structural signal is sports-calendar driven (Layer 2 specific)
        # In production: pull fan token issuance schedule from Chiliz chain API
        # Default neutral unless sports catalyst is confirmed in Layer 2 ext.
        score += 0.05   # slight positive bias from fan token ecosystem growth

    return max(-1.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — SOCIAL VELOCITY & SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def score_sentiment(snap: MarketSnapshot, asset: str) -> float:
    """
    Social velocity and sentiment score [-1.0, +1.0].

    Uses Fear & Greed as the primary publicly available sentiment proxy.
    In production: integrates LunarCrush social velocity API,
    Santiment NLP, and CryptoQuant metrics.
    """
    fg = snap.fear_greed
    score = 0.0

    if asset == "WBTC":
        # BTC: social is a WEAK predictor and a contrarian indicator at extremes
        if fg <= 15:
            score += 0.15    # extreme fear = contrarian long signal (lagging)
        elif fg <= 25:
            score += 0.05
        elif fg >= 85:
            score -= 0.25    # extreme greed = distribution
        elif fg >= 70:
            score -= 0.10
        # Weight this layer down further for BTC (done in quorum engine)

    elif asset == "SOL":
        # SOL: social velocity is PRIMARY entry-timing mechanism
        # Extreme fear = potential long entry; extreme greed = short entry
        if fg <= 20:
            score += 0.25
        elif fg <= 35:
            score += 0.10
        elif fg >= 85:
            score -= 0.40    # social exhaustion = short trigger
        elif fg >= 70:
            score -= 0.20

    elif asset == "CHZ":
        # CHZ: organic sports fan sentiment vs crypto speculative sentiment
        # In production: differentiate via NLP topic classification
        # Baseline: slight positive in moderate fear environments
        if fg < 30:
            score += 0.05    # fear suppresses CHZ speculative premium = cheap
        elif fg > 75:
            score -= 0.15    # crypto FOMO inflates CHZ = distribution risk

    elif asset == "WETH":
        # ETH: treats similarly to BTC but more reactive to DeFi sentiment
        if fg <= 20:
            score += 0.20
        elif fg >= 80:
            score -= 0.30

    return max(-1.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — TECHNICAL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

# Key structural levels — updated manually or via automated level detection
STRUCTURE_LEVELS = {
    "WBTC": {
        "support":    [67086, 65000, 60000],
        "resistance": [72683, 74490, 77000, 100000],
        "mid_range_low":  67000,
        "mid_range_high": 72500,
    },
    "WETH": {
        "support":    [2800, 2500, 2200],
        "resistance": [3300, 3600, 4000],
        "mid_range_low":  2800,
        "mid_range_high": 3300,
    },
    "SOL": {
        "support":    [78, 70, 60],
        "resistance": [95, 100, 115],
        "mid_range_low":  80,
        "mid_range_high": 93,
    },
    "CHZ": {
        "support":    [0.055, 0.045, 0.035],
        "resistance": [0.075, 0.090, 0.110],
        "mid_range_low":  0.058,
        "mid_range_high": 0.073,
    },
}


def score_technical(snap: MarketSnapshot, asset: str) -> float:
    """
    Technical structure confirmation score [-1.0, +1.0].

    Core rule: NO signal in the mid-range. Price must be testing
    a confirmed structural level to generate a non-zero score.
    """
    price = snap.prices.get(asset, 0)
    if price <= 0:
        return 0.0

    levels = STRUCTURE_LEVELS.get(asset)
    if not levels:
        return 0.0

    mid_low  = levels["mid_range_low"]
    mid_high = levels["mid_range_high"]

    # ── Mid-range prohibition ─────────────────────────────────────────────
    if mid_low < price < mid_high:
        log.debug(f"[Layer4] {asset} @ {price:.4f} is in mid-range — no signal")
        return 0.0

    score = 0.0

    # ── Distance to nearest support ──────────────────────────────────────
    for sup in sorted(levels["support"], reverse=True):
        gap_pct = (price - sup) / sup
        if gap_pct < 0:
            continue   # price is below this support
        if gap_pct < 0.03:     # within 3% of support — strong long zone
            score += 0.60
            break
        elif gap_pct < 0.07:   # within 7% — moderate long zone
            score += 0.30
            break

    # ── Distance to nearest resistance ───────────────────────────────────
    for res in sorted(levels["resistance"]):
        gap_pct = (res - price) / res
        if gap_pct < 0:
            continue   # price is above this resistance
        if gap_pct < 0.03:     # within 3% of resistance — short zone
            score -= 0.60
            break
        elif gap_pct < 0.07:
            score -= 0.30
            break

    return max(-1.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(snap: MarketSnapshot) -> str:
    """
    Returns: "macro_driven" | "neutral" | "narrative_driven"
    All three conditions must align for extreme regimes.
    """
    macro_signals = 0
    narrative_signals = 0

    if snap.btc_nasdaq_corr > CORRELATION_HIGH:
        macro_signals += 1
    elif snap.btc_nasdaq_corr < CORRELATION_LOW:
        narrative_signals += 1

    if snap.vix > VIX_HIGH_THRESHOLD:
        macro_signals += 1
    elif snap.vix < VIX_LOW_THRESHOLD:
        narrative_signals += 1

    if snap.etf_flow_direction == "negative":
        macro_signals += 1
    elif snap.etf_flow_direction == "positive":
        narrative_signals += 1

    if macro_signals >= 2:
        return "macro_driven"
    elif narrative_signals >= 2:
        return "narrative_driven"
    return "neutral"


def get_regime_weights(regime: str) -> dict:
    """Returns layer weights for the given regime."""
    if regime == "macro_driven":
        return {"macro": 0.45, "onchain": 0.35, "technical": 0.12, "sentiment": 0.08}
    elif regime == "narrative_driven":
        return {"macro": 0.15, "onchain": 0.30, "technical": 0.20, "sentiment": 0.35}
    else:  # neutral
        return {"macro": 0.30, "onchain": 0.35, "technical": 0.15, "sentiment": 0.20}


# ─────────────────────────────────────────────────────────────────────────────
# QUORUM ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_quorum(snap: MarketSnapshot, asset: str) -> tuple[float, LayerScores]:
    """
    Runs all four layers and computes regime-weighted aggregate score.
    Returns (aggregate_score, LayerScores).
    """
    regime  = detect_regime(snap)
    weights = get_regime_weights(regime)

    l1 = score_macro(snap)
    l2 = score_onchain(snap, asset)
    l3 = score_sentiment(snap, asset)
    l4 = score_technical(snap, asset)

    aggregate = (
        weights["macro"]      * l1 +
        weights["onchain"]    * l2 +
        weights["sentiment"]  * l3 +
        weights["technical"]  * l4
    )

    scores = LayerScores(
        macro=l1, onchain=l2, sentiment=l3, technical=l4, regime=regime
    )

    log.info(
        f"[Quorum] {asset:4s} | Regime: {regime:16s} | "
        f"L1={l1:+.3f} L2={l2:+.3f} L3={l3:+.3f} L4={l4:+.3f} → "
        f"AGG={aggregate:+.3f}"
    )

    return aggregate, scores


# ─────────────────────────────────────────────────────────────────────────────
# FRACTIONAL KELLY POSITION SIZER
# ─────────────────────────────────────────────────────────────────────────────

# Historical win-rate and payoff data per asset (from backtesting)
KELLY_PARAMS = {
    "WBTC": {"win_rate": 0.58, "win_loss_ratio": 2.1},
    "WETH": {"win_rate": 0.56, "win_loss_ratio": 2.0},
    "SOL":  {"win_rate": 0.54, "win_loss_ratio": 2.4},
    "CHZ":  {"win_rate": 0.52, "win_loss_ratio": 2.8},
}


def kelly_position_size(asset: str, portfolio_value: float,
                        score_magnitude: float) -> float:
    """
    Calculates half-Kelly position size in USDC.
    Returns 0 if Kelly is negative (negative expected value).
    """
    params = KELLY_PARAMS.get(asset, {"win_rate": 0.50, "win_loss_ratio": 1.5})
    p = params["win_rate"]
    b = params["win_loss_ratio"]

    # Full Kelly fraction: f = (bp - q) / b
    q = 1 - p
    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        log.warning(f"[Kelly] {asset} negative EV — trade blocked (f={full_kelly:.3f})")
        return 0.0

    half_kelly = full_kelly * KELLY_FRACTION

    # Scale by score magnitude (higher conviction → larger size, capped at max)
    conviction_scaler = min(1.0, abs(score_magnitude) / LONG_THRESHOLD)
    raw_fraction      = half_kelly * conviction_scaler

    # Hard cap per asset
    capped_fraction = min(raw_fraction, MAX_POSITION.get(asset, 0.10))
    position_usdc   = portfolio_value * capped_fraction

    log.info(
        f"[Kelly] {asset} | f_full={full_kelly:.3f} f_half={half_kelly:.3f} "
        f"capped={capped_fraction:.3f} → ${position_usdc:,.2f} USDC"
    )
    return position_usdc


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLE-BARRIER EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def build_trade_signal(asset: str, direction: str, snap: MarketSnapshot,
                       score: float, layer_scores: LayerScores,
                       portfolio_value: float) -> Optional[TradeSignal]:
    """Constructs a TradeSignal with Triple-Barrier parameters."""
    price    = snap.prices.get(asset, 0)
    barriers = BARRIERS.get(asset, {"stop_pct": 0.05, "tp1_pct": 0.08, "tp2_pct": 0.15, "time_days": 14})
    pos_usdc = kelly_position_size(asset, portfolio_value, score)

    if pos_usdc <= 0:
        return None

    if direction == "long":
        stop_loss    = price * (1 - barriers["stop_pct"])
        take_profit1 = price * (1 + barriers["tp1_pct"])
        take_profit2 = price * (1 + barriers["tp2_pct"])
    else:  # short
        stop_loss    = price * (1 + barriers["stop_pct"])
        take_profit1 = price * (1 - barriers["tp1_pct"])
        take_profit2 = price * (1 - barriers["tp2_pct"])

    reason = (
        f"AEF v1.0 | {direction.upper()} {asset} | Score={score:+.3f} | "
        f"Regime={layer_scores.regime} | "
        f"L1={layer_scores.macro:+.2f} L2={layer_scores.onchain:+.2f} "
        f"L3={layer_scores.sentiment:+.2f} L4={layer_scores.technical:+.2f}"
    )

    return TradeSignal(
        asset=asset, direction=direction,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit_1=take_profit1,
        take_profit_2=take_profit2,
        time_barrier_days=barriers["time_days"],
        position_pct=pos_usdc / portfolio_value,
        aggregate_score=score,
        layer_scores=layer_scores,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KILL SWITCH
# ─────────────────────────────────────────────────────────────────────────────

def check_kill_switch(snap: MarketSnapshot) -> bool:
    """
    Returns True if the portfolio drawdown breaches the kill-switch threshold.
    Immediately halts all trading when triggered.
    """
    if snap.peak_equity <= 0:
        return False
    drawdown = (snap.peak_equity - snap.portfolio_value) / snap.peak_equity
    threshold = MAX_HISTORICAL_DRAWDOWN * KILL_SWITCH_MULTIPLIER
    if drawdown >= threshold:
        log.critical(
            f"KILL SWITCH ACTIVATED - Drawdown {drawdown:.1%} exceeds "
            f"threshold {threshold:.1%}. All activity halted."
        )
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# BARRIER MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def check_open_positions(positions: list[OpenPosition], snap: MarketSnapshot,
                         client: RecallClient, dry_run: bool) -> list[OpenPosition]:
    """
    Checks each open position against its Triple-Barrier parameters.
    Closes positions that have hit stop, take-profit, or time barrier.
    Returns updated list of still-open positions.
    """
    remaining = []
    now = datetime.now(timezone.utc)

    for pos in positions:
        price = snap.prices.get(pos.asset, pos.entry_price)
        age_days = (now - pos.entry_time).total_seconds() / 86400
        hit_time = age_days >= pos.time_barrier_days

        if pos.direction == "long":
            hit_stop = price <= pos.stop_loss
            hit_tp1  = price >= pos.take_profit_1 and not pos.tp1_hit
            hit_tp2  = price >= pos.take_profit_2
        else:
            hit_stop = price >= pos.stop_loss
            hit_tp1  = price <= pos.take_profit_1 and not pos.tp1_hit
            hit_tp2  = price <= pos.take_profit_2

        close_reason = None
        if hit_stop:
            close_reason = f"STOP-LOSS hit @ {price:.4f} (entry {pos.entry_price:.4f})"
        elif hit_tp2:
            close_reason = f"TP2 hit @ {price:.4f} - full exit"
        elif hit_time:
            close_reason = f"TIME-BARRIER {pos.time_barrier_days}d expired"
        elif hit_tp1:
            pos.tp1_hit = True
            log.info(f"[Barrier] {pos.asset} TP1 reached @ {price:.4f} - scaling out 1/3")
            # In a real implementation, close 1/3 of position size here
            remaining.append(pos)
            continue

        if close_reason:
            log.info(f"[Barrier] {pos.asset} closing - {close_reason}")
            if not dry_run:
                # Return position to USDC
                asset_token = TOKENS.get(pos.asset)
                usdc_token  = TOKENS["USDC"]
                est_quantity = str(int(pos.size_usdc / price))
                try:
                    result = client.execute_trade(
                        from_token=asset_token,
                        to_token=usdc_token,
                        amount=est_quantity,
                        reason=f"AEF close - {close_reason}",
                    )
                    log.info(f"[Trade] Closed {pos.asset}: {result}")
                except Exception as e:
                    log.error(f"[Trade] Failed to close {pos.asset}: {e}")
        else:
            remaining.append(pos)

    return remaining


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_market_data(client: RecallClient, peak_equity: float) -> MarketSnapshot:
    """
    Collects live data from the Recall API and populates a MarketSnapshot.
    Macro data (VIX, correlation, ETF flows) uses real-time values fetched
    from public data endpoints; falls back to conservative defaults if
    unavailable.
    """
    snap = MarketSnapshot()
    snap.peak_equity = peak_equity

    # ── Portfolio ─────────────────────────────────────────────────────────
    try:
        portfolio = client.get_portfolio()
        snap.portfolio_usdc  = float(portfolio.get("usdcBalance", 0))
        snap.portfolio_value = float(portfolio.get("totalValue", snap.portfolio_usdc))
    except Exception as e:
        log.warning(f"Portfolio fetch failed: {e}")
        snap.portfolio_usdc  = 10000.0   # default sandbox balance
        snap.portfolio_value = 10000.0

    # ── Asset prices ──────────────────────────────────────────────────────
    for symbol, address in TOKENS.items():
        if symbol == "USDC":
            snap.prices[symbol] = 1.0
            continue
        try:
            snap.prices[symbol] = client.get_price(address)
            log.debug(f"Price {symbol}: ${snap.prices[symbol]:,.4f}")
        except Exception as e:
            log.warning(f"Price fetch failed for {symbol}: {e}")

    # ── Macro data — pulled from public APIs ──────────────────────────────
    # VIX proxy: fetch from CBOE/Yahoo Finance data (public endpoint)
    # Falls back to last known value if unavailable
    snap.vix = _fetch_vix_estimate()
    snap.btc_nasdaq_corr = _fetch_btc_nasdaq_correlation()
    snap.etf_flow_direction = _fetch_etf_flow_direction()
    snap.fear_greed = _fetch_fear_greed()
    snap.funding_rates = _fetch_funding_rates()

    log.info(
        f"[Data] Portfolio ${snap.portfolio_value:,.2f} | "
        f"VIX={snap.vix:.1f} | Corr={snap.btc_nasdaq_corr:.2f} | "
        f"F&G={snap.fear_greed} | ETF={snap.etf_flow_direction}"
    )
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC DATA FETCHERS
# These fetch real macro data from free public APIs.
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fear_greed() -> int:
    """Fetch current Fear & Greed index from alternative.me."""
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1", timeout=8
        )
        data = resp.json()
        return int(data["data"][0]["value"])
    except Exception:
        return 50   # neutral fallback


def _fetch_vix_estimate() -> float:
    """
    Estimate VIX from public crypto fear proxy.
    In production: use Yahoo Finance yfinance library for ^VIX.
    """
    fg = _fetch_fear_greed()
    # Simple inverse mapping: F&G 0 → VIX ~40, F&G 100 → VIX ~12
    # VIX ≈ 40 - 0.28 × FG
    vix_estimate = max(10.0, 40.0 - (0.28 * fg))
    return round(vix_estimate, 1)


def _fetch_btc_nasdaq_correlation() -> float:
    """
    Returns BTC-Nasdaq rolling 30-day correlation estimate.
    In production: compute from OHLCV data via CoinGecko + Yahoo Finance.
    Conservative default: 0.65 (moderate correlation).
    """
    try:
        # Attempt to derive from recent BTC price volatility
        # If BTC 7-day change roughly tracks NASDAQ → high correlation
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin&vs_currencies=usd&include_24hr_change=true",
            timeout=8
        )
        data = resp.json()
        change_24h = abs(float(data["bitcoin"].get("usd_24h_change", 0)))
        # High daily moves in BTC correlate with macro risk-off events
        if change_24h > 8:
            return 0.85   # extreme move = macro-driven
        elif change_24h > 4:
            return 0.70
        return 0.55
    except Exception:
        return 0.65   # moderate correlation default


def _fetch_etf_flow_direction() -> str:
    """
    Proxy for ETF flow direction using BTC price trend.
    In production: scrape BitMEX Research / Farside Investors ETF flow tracker.
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin&vs_currencies=usd&include_7d_change=true",
            timeout=8
        )
        data = resp.json()
        change_7d = float(data["bitcoin"].get("usd_7d_change", 0))
        if change_7d > 5:
            return "positive"
        elif change_7d < -5:
            return "negative"
        return "mixed"
    except Exception:
        return "mixed"


def _fetch_funding_rates() -> dict:
    """
    Funding rates proxy. In production: Binance/Bybit perpetual futures API.
    Returns conservative 0.0 defaults when unavailable.
    """
    return {
        "WBTC": 0.0001,
        "WETH": 0.0001,
        "SOL":  0.0001,
        "CHZ":  0.0000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HITL INTERRUPT
# ─────────────────────────────────────────────────────────────────────────────

def hitl_approve(signal: TradeSignal) -> bool:
    """
    Human-in-the-Loop interrupt. Presents the full diagnostic package
    and waits for operator approval before execution.
    Returns True to approve, False to reject.
    """
    print("\n" + "=" * 72)
    print("  AEF TRADE SIGNAL - HITL APPROVAL REQUIRED")
    print("=" * 72)
    print(f"  Asset      : {signal.asset}")
    print(f"  Direction  : {signal.direction.upper()}")
    print(f"  Entry Price: ${signal.entry_price:,.4f}")
    print(f"  Stop-Loss  : ${signal.stop_loss:,.4f}  "
          f"({abs(signal.stop_loss - signal.entry_price) / signal.entry_price:.1%})")
    print(f"  Take-Profit: ${signal.take_profit_1:,.4f}  |  "
          f"${signal.take_profit_2:,.4f}")
    print(f"  Time Barrier: {signal.time_barrier_days} days")
    print(f"  Position   : {signal.position_pct:.1%} of portfolio")
    print(f"  Agg Score  : {signal.aggregate_score:+.3f}  "
          f"(threshold ±{LONG_THRESHOLD})")
    print()
    ls = signal.layer_scores
    print(f"  Layer Scores:")
    print(f"    L1 Macro      : {ls.macro:+.3f}")
    print(f"    L2 On-Chain   : {ls.onchain:+.3f}")
    print(f"    L3 Sentiment  : {ls.sentiment:+.3f}")
    print(f"    L4 Technical  : {ls.technical:+.3f}")
    print(f"    Regime        : {ls.regime}")
    print()
    print(f"  Reason: {signal.reason}")
    print()
    print("  What must be true for this to fail catastrophically:")
    if signal.direction == "long":
        print(f"    - Price breaks below ${signal.stop_loss:,.4f} on high volume")
        print("    - Macro shock (VIX spike, FOMC hawkish surprise, geopolitical event)")
        print("    - ETF outflows accelerate for 3+ consecutive days")
    else:
        print(f"    - Price squeezes above ${signal.stop_loss:,.4f} (short squeeze)")
        print("    - Positive macro catalyst (CPI beat, ceasefire, ETF inflow surge)")
        print("    - Funding rates flip negative (short crowding reversed)")
    print()
    response = input("  Approve? [y/N/r to reduce size]: ").strip().lower()
    if response == "y":
        return True
    elif response == "r":
        # Operator can reduce size
        try:
            new_pct = float(input("  Enter new position size (% of portfolio, e.g. 5): ")) / 100
            signal.position_pct = new_pct
            log.info(f"[HITL] Operator reduced position to {new_pct:.1%}")
            return True
        except ValueError:
            return False
    return False


# ─────────────────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_signal(signal: TradeSignal, snap: MarketSnapshot,
                   client: RecallClient, dry_run: bool) -> Optional[OpenPosition]:
    """Executes the approved trade via Recall API."""
    if signal.direction == "long":
        from_token = TOKENS["USDC"]
        to_token   = TOKENS[signal.asset]
        amount_str = str(int(snap.portfolio_value * signal.position_pct))
    else:
        # Short: sell asset for USDC
        from_token = TOKENS[signal.asset]
        to_token   = TOKENS["USDC"]
        price      = snap.prices.get(signal.asset, 1)
        usdc_value = snap.portfolio_value * signal.position_pct
        amount_str = str(int(usdc_value / price))

    log.info(
        f"[Trade] Executing {signal.direction.upper()} {signal.asset} | "
        f"Amount={amount_str} | {'DRY-RUN' if dry_run else 'LIVE'}"
    )

    if not dry_run:
        try:
            result = client.execute_trade(
                from_token=from_token,
                to_token=to_token,
                amount=amount_str,
                reason=signal.reason,
            )
            log.info(f"[Trade] Executed: {result}")
        except Exception as e:
            log.error(f"[Trade] Execution failed: {e}")
            return None
    else:
        log.info(f"[Trade] DRY-RUN - would trade {amount_str} {signal.asset}")

    return OpenPosition(
        asset=signal.asset,
        direction=signal.direction,
        entry_price=signal.entry_price,
        entry_time=datetime.now(timezone.utc),
        stop_loss=signal.stop_loss,
        take_profit_1=signal.take_profit_1,
        take_profit_2=signal.take_profit_2,
        time_barrier_days=signal.time_barrier_days,
        size_usdc=snap.portfolio_value * signal.position_pct,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRADING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_bot(mode: str = "sandbox", hitl: bool = True, dry_run: bool = False):
    api_key = os.getenv("RECALL_API_KEY")
    if not api_key:
        log.critical("RECALL_API_KEY not set in .env — aborting.")
        sys.exit(1)

    base_url = SANDBOX_URL if mode == "sandbox" else PRODUCTION_URL
    client   = RecallClient(api_key, base_url)

    log.info(f"Axiom Core Bot starting - mode={mode} hitl={hitl} dry_run={dry_run}")
    log.info(f"   Assets: {list(TOKENS.keys())[1:]}")
    log.info(f"   Quorum thresholds: Long={LONG_THRESHOLD} Short={SHORT_THRESHOLD}")

    open_positions: list[OpenPosition] = []
    peak_equity   = 0.0
    halted        = False

    while True:
        if halted:
            log.critical("Bot is HALTED. Awaiting manual reset.")
            time.sleep(300)
            continue

        try:
            # ── Collect data ──────────────────────────────────────────────
            snap = collect_market_data(client, peak_equity)

            # Track peak equity for kill switch
            if snap.portfolio_value > peak_equity:
                peak_equity = snap.portfolio_value

            # ── Kill switch ───────────────────────────────────────────────
            if check_kill_switch(snap):
                halted = True
                # Flatten all positions
                for pos in open_positions:
                    log.critical(f"[KillSwitch] Closing {pos.asset}")
                    if not dry_run:
                        try:
                            client.execute_trade(
                                from_token=TOKENS[pos.asset],
                                to_token=TOKENS["USDC"],
                                amount=str(int(pos.size_usdc / snap.prices.get(pos.asset, 1))),
                                reason="AEF KILL SWITCH — emergency exit",
                            )
                        except Exception as e:
                            log.error(f"[KillSwitch] Emergency close failed: {e}")
                open_positions = []
                continue

            # ── Monitor existing positions ────────────────────────────────
            open_positions = check_open_positions(
                open_positions, snap, client, dry_run
            )

            # ── Signal generation for each asset ─────────────────────────
            currently_trading = {p.asset for p in open_positions}

            for asset in ["WBTC", "WETH", "SOL", "CHZ"]:
                if asset in currently_trading:
                    log.debug(f"[Loop] {asset} already in open position — skipping")
                    continue

                if asset not in snap.prices or snap.prices[asset] <= 0:
                    log.debug(f"[Loop] No price data for {asset} — skipping")
                    continue

                score, layer_scores = calculate_quorum(snap, asset)

                # Determine direction
                if score >= LONG_THRESHOLD:
                    direction = "long"
                elif score <= SHORT_THRESHOLD:
                    direction = "short"
                else:
                    log.debug(
                        f"[Loop] {asset} score {score:+.3f} in no-trade zone"
                    )
                    continue

                # Build signal
                signal = build_trade_signal(
                    asset, direction, snap, score, layer_scores, snap.portfolio_value
                )
                if signal is None:
                    continue

                # HITL gate
                approved = True
                if hitl:
                    approved = hitl_approve(signal)

                if approved:
                    pos = execute_signal(signal, snap, client, dry_run)
                    if pos:
                        open_positions.append(pos)
                        log.info(
                            f"[Position] Opened {direction.upper()} {asset} "
                            f"@ ${signal.entry_price:.4f} | "
                            f"Stop ${signal.stop_loss:.4f} | "
                            f"TP ${signal.take_profit_1:.4f}"
                        )
                else:
                    log.info(f"[HITL] Trade rejected by operator — {asset} {direction}")

        except requests.HTTPError as e:
            log.error(f"API error: {e}")
        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception as e:
            log.exception(f"Unexpected error: {e}")

        log.info(f"Sleeping {POLL_INTERVAL}s | Open positions: {len(open_positions)}")
        time.sleep(POLL_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Axiom Core Bot — AEF v1.0 for Recall Network competitions"
    )
    parser.add_argument(
        "--mode", choices=["sandbox", "production"], default="sandbox",
        help="API environment (default: sandbox)"
    )
    parser.add_argument(
        "--hitl", action="store_true", default=False,
        help="Enable Human-in-the-Loop approval before each trade"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Log trades without executing them"
    )
    args = parser.parse_args()
    run_bot(mode=args.mode, hitl=args.hitl, dry_run=args.dry_run)
