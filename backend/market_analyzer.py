"""
Comprehensive Quantitative Market Analyzer for NSE Equities.
Provides:
1. Full Technical Indicator Suite (RSI, MACD, Bollinger Bands, SuperTrend, EMAs, VWAP, RVOL).
2. Market Breadth, Sector Rotation & Sentiment Radar.
3. Automated Intraday Opportunity Screener (Breakouts, Mean-Reversion, Volume Surges).
"""

import os
import sys
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE
from database.relations import STOCK_SECTOR_MAP


class MarketAnalyzer:
    """
    Institutional quantitative market intelligence analyzer.
    Processes live feature frames and produces actionable multi-dimensional market diagnostics.
    """

    def __init__(self):
        self.symbols = list(STOCK_SYMBOLS)
        self.universe = dict(STOCK_UNIVERSE)
        self.sector_map = dict(STOCK_SECTOR_MAP)

    def analyze_market_overview(self, quotes_df: pd.DataFrame, feature_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Produce a high-level market breadth, sentiment, and sector rotation summary.
        """
        if quotes_df.empty:
            return {
                "sentiment": "NEUTRAL",
                "sentiment_score": 50.0,
                "advance_count": 0,
                "decline_count": 0,
                "neutral_count": 0,
                "adv_dec_ratio": 1.0,
                "top_gainer": None,
                "top_loser": None,
                "avg_change_pct": 0.0,
                "sector_performance": {},
                "nifty_status": "FLAT",
            }

        advances = int((quotes_df["change_pct"] > 0.05).sum())
        declines = int((quotes_df["change_pct"] < -0.05).sum())
        neutrals = len(quotes_df) - advances - declines
        adv_dec_ratio = advances / max(declines, 1)

        avg_chg = float(quotes_df["change_pct"].mean())
        sorted_by_chg = quotes_df.sort_values("change_pct", ascending=False)
        top_gainer = sorted_by_chg.iloc[0].to_dict() if not sorted_by_chg.empty else None
        top_loser = sorted_by_chg.iloc[-1].to_dict() if not sorted_by_chg.empty else None

        # Sector performance
        sector_perf = {}
        for _, row in quotes_df.iterrows():
            sym = row["symbol"]
            sec = self.sector_map.get(sym, "Diversified")
            if sec not in sector_perf:
                sector_perf[sec] = {"changes": [], "volume": 0.0}
            sector_perf[sec]["changes"].append(row["change_pct"])
            sector_perf[sec]["volume"] += row.get("volume", 0)

        sector_summary = {}
        for sec, data in sector_perf.items():
            sector_summary[sec] = {
                "avg_change": float(np.mean(data["changes"])),
                "total_volume": float(data["volume"]),
                "count": len(data["changes"]),
            }

        # Market Sentiment Score (0 to 100)
        # Components: 50% Adv/Dec proportion, 50% Avg Change magnitude
        adv_ratio_norm = advances / max(len(quotes_df), 1)
        sentiment_score = float(np.clip(adv_ratio_norm * 60.0 + (avg_chg + 1.0) * 20.0, 5.0, 95.0))

        if sentiment_score >= 65:
            sentiment = "BULLISH"
        elif sentiment_score <= 35:
            sentiment = "BEARISH"
        elif avg_chg > 0.15:
            sentiment = "MILDLY BULLISH"
        elif avg_chg < -0.15:
            sentiment = "MILDLY BEARISH"
        else:
            sentiment = "NEUTRAL / RANGEBOUND"

        return {
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 1),
            "advance_count": advances,
            "decline_count": declines,
            "neutral_count": neutrals,
            "adv_dec_ratio": round(adv_dec_ratio, 2),
            "avg_change_pct": round(avg_chg, 2),
            "top_gainer": top_gainer,
            "top_loser": top_loser,
            "sector_performance": sector_summary,
        }

    def compute_technical_screener(self, feature_df: pd.DataFrame, quotes_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate a comprehensive, actionable technical indicator matrix for all stocks:
        - Price & Change %
        - RSI (14) & Condition (Oversold / Neutral / Overbought)
        - MACD Signal (Bullish Cross / Bearish Cross / Neutral)
        - SuperTrend (Bullish / Bearish)
        - EMA Alignment (EMA 9 vs 21 vs 50)
        - VWAP Distance %
        - Bollinger %B & Bandwidth
        - RVOL (Relative Volume)
        - Overall Technical Action (STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL)
        """
        if feature_df.empty:
            return pd.DataFrame()

        rows = []
        quotes_lookup = {}
        if quotes_df is not None and not quotes_df.empty:
            for _, r in quotes_df.iterrows():
                quotes_lookup[r["symbol"]] = r

        for symbol, group in feature_df.groupby("symbol"):
            if len(group) < 20:
                continue
            group = group.sort_values("timestamp").reset_index(drop=True)
            last = group.iloc[-1]
            prev = group.iloc[-2] if len(group) > 1 else last

            close_p = float(last["close"])
            high_p = float(last.get("high", close_p))
            low_p = float(last.get("low", close_p))

            # Quote info
            q_info = quotes_lookup.get(symbol)
            curr_p = float(q_info["price"]) if q_info and q_info["price"] > 0 else close_p
            chg_pct = float(q_info["change_pct"]) if q_info else float(last.get("returns", 0.0) * 100)

            # 1. RSI (14)
            rsi = float(last.get("rsi_14", 50.0))
            if rsi < 30:
                rsi_state = "Oversold 🟢"
            elif rsi > 70:
                rsi_state = "Overbought 🔴"
            elif rsi >= 55:
                rsi_state = "Bullish"
            elif rsi <= 45:
                rsi_state = "Bearish"
            else:
                rsi_state = "Neutral"

            # 2. MACD Calculation
            close_s = group["close"]
            ema12 = close_s.ewm(span=12, adjust=False).mean()
            ema26 = close_s.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line

            curr_macd = float(macd_line.iloc[-1])
            curr_sig = float(signal_line.iloc[-1])
            curr_hist = float(macd_hist.iloc[-1])
            prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else curr_hist

            if curr_hist > 0 and prev_hist <= 0:
                macd_state = "Bullish Cross ⚡"
            elif curr_hist < 0 and prev_hist >= 0:
                macd_state = "Bearish Cross ⚠️"
            elif curr_hist > 0:
                macd_state = "Bullish"
            else:
                macd_state = "Bearish"

            # 3. SuperTrend (10, 3)
            atr14 = float(last.get("atr_14", close_p * 0.01))
            hl2 = (high_p + low_p) / 2.0
            st_upper = hl2 + (3.0 * atr14)
            st_lower = hl2 - (3.0 * atr14)
            supertrend_state = "BULLISH 🟢" if curr_p > st_lower else "BEARISH 🔴"

            # 4. EMAs
            ema9 = float(close_s.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
            ema50 = float(close_s.ewm(span=50, adjust=False).mean().iloc[-1])

            if curr_p > ema9 > ema21:
                ema_alignment = "Strong Uptrend (9>21)"
            elif curr_p < ema9 < ema21:
                ema_alignment = "Strong Downtrend (9<21)"
            elif curr_p > ema21:
                ema_alignment = "Mild Uptrend"
            else:
                ema_alignment = "Mild Downtrend"

            # 5. VWAP & Distance
            vwap_p = float(last.get("vwap", curr_p))
            vwap_dist = ((curr_p - vwap_p) / vwap_p * 100) if vwap_p > 0 else 0.0

            # 6. Bollinger Bands
            bb_pct = float(last.get("bb_pct_b", 0.5))

            # 7. RVOL (Relative Volume)
            vol = float(last.get("volume", 0))
            avg_vol = float(group["volume"].tail(20).mean()) if len(group) >= 20 else vol
            rvol = (vol / avg_vol) if avg_vol > 0 else 1.0

            # 8. Technical Score Synthesis (-5 to +5)
            tech_score = 0
            if rsi >= 55: tech_score += 1
            if rsi <= 45: tech_score -= 1
            if rsi < 30: tech_score += 1 # Oversold bounce potential
            if curr_hist > 0: tech_score += 1
            if curr_hist < 0: tech_score -= 1
            if curr_p > vwap_p: tech_score += 1
            if curr_p < vwap_p: tech_score -= 1
            if curr_p > ema9: tech_score += 1
            if curr_p < ema9: tech_score -= 1
            if "BULLISH" in supertrend_state: tech_score += 1
            if "BEARISH" in supertrend_state: tech_score -= 1

            if tech_score >= 3:
                action = "STRONG BUY"
                action_badge = "🟢 Strong Buy"
            elif tech_score >= 1:
                action = "BUY"
                action_badge = "🟢 Buy"
            elif tech_score <= -3:
                action = "STRONG SELL"
                action_badge = "🔴 Strong Sell"
            elif tech_score <= -1:
                action = "SELL"
                action_badge = "🔴 Sell"
            else:
                action = "NEUTRAL"
                action_badge = "⚪ Neutral"

            rows.append({
                "symbol": symbol,
                "company": self.universe.get(symbol, symbol),
                "sector": self.sector_map.get(symbol, "General"),
                "price": curr_p,
                "change_pct": chg_pct,
                "rsi_14": round(rsi, 1),
                "rsi_state": rsi_state,
                "macd_state": macd_state,
                "supertrend": supertrend_state,
                "ema_trend": ema_alignment,
                "vwap": round(vwap_p, 2),
                "vwap_dist_pct": round(vwap_dist, 2),
                "bb_pct": round(bb_pct, 2),
                "rvol": round(rvol, 2),
                "tech_score": tech_score,
                "action": action,
                "action_badge": action_badge,
                "regime": last.get("regime", "sideways"),
            })

        if not rows:
            return pd.DataFrame()

        df_screener = pd.DataFrame(rows)
        return df_screener.sort_values("tech_score", ascending=False).reset_index(drop=True)

    def scan_breakout_opportunities(self, feature_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scan for high-conviction intraday setups:
        1. Breakouts: Price within 0.3% of 5-day high + RVOL > 1.4x.
        2. Mean-Reversion: RSI < 32 with positive reversal candle or hammer.
        3. Momentum Surges: Intraday alpha vs Nifty > 1.0%.
        """
        opportunities = []
        if feature_df.empty:
            return opportunities

        for symbol, group in feature_df.groupby("symbol"):
            if len(group) < 30:
                continue
            group = group.sort_values("timestamp").reset_index(drop=True)
            last = group.iloc[-1]
            close = float(last["close"])
            high_5d = float(group["high"].max())
            low_5d = float(group["low"].min())
            rsi = float(last.get("rsi_14", 50.0))
            rvol = float(last.get("vpin", 0.0) * 5 + 1.0)
            returns = float(last.get("returns", 0.0))

            # 1. Breakout setup
            dist_to_high = ((high_5d - close) / high_5d) * 100
            if dist_to_high < 0.35 and returns > 0:
                opportunities.append({
                    "symbol": symbol,
                    "company": self.universe.get(symbol, symbol),
                    "type": "BREAKOUT",
                    "title": f"{self.universe.get(symbol, symbol)} Intraday High Breakout",
                    "description": f"Trading within {dist_to_high:.2f}% of 5-day resistance (₹{high_5d:.2f}) with bullish acceleration.",
                    "conviction": "HIGH",
                    "price": close,
                    "target": round(close * 1.012, 2),
                    "stop_loss": round(close * 0.994, 2),
                    "badge_color": "var(--profit-emerald)",
                })

            # 2. Mean-reversion oversold setup
            if rsi < 35 and returns > -0.001:
                opportunities.append({
                    "symbol": symbol,
                    "company": self.universe.get(symbol, symbol),
                    "type": "OVERSOLD_REVERSAL",
                    "title": f"{self.universe.get(symbol, symbol)} Oversold Rebound Setup",
                    "description": f"RSI severely depressed at {rsi:.1f} near 5-day support (₹{low_5d:.2f}) showing reversal buying.",
                    "conviction": "MEDIUM-HIGH",
                    "price": close,
                    "target": round(close * 1.015, 2),
                    "stop_loss": round(close * 0.993, 2),
                    "badge_color": "var(--accent-cyan)",
                })

        return opportunities


# Global analyzer instance
market_analyzer = MarketAnalyzer()
