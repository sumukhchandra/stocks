import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE
from ml_models.labels.triple_barrier import apply_triple_barrier_labels


OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "processed", "master_labeled_dataset.parquet")
)


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            (col[0] or "timestamp").lower() if isinstance(col, tuple) else str(col).lower()
            for col in df.columns
        ]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df


def fetch_stock_history(symbol, period="60d", interval="5m"):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()

    df = _flatten_columns(df.reset_index())
    time_col = "datetime" if "datetime" in df.columns else "date" if "date" in df.columns else "index"
    df = df.rename(columns={time_col: "timestamp", "adj close": "adj_close"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["symbol"] = symbol
    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_nifty_index_features(period="60d", interval="5m"):
    """Fetch NIFTY 50 benchmark index data for cross-market lead-lag signals."""
    try:
        nifty = fetch_stock_history("^NSEI", period=period, interval=interval)
        if nifty.empty:
            return pd.DataFrame()

        close = nifty["close"]
        vol = nifty["volume"].fillna(0)

        nifty["nifty_ret_1"] = close.pct_change(1).fillna(0)
        nifty["nifty_ret_3"] = close.pct_change(3).fillna(0)
        nifty["nifty_ret_5"] = close.pct_change(5).fillna(0)

        # NIFTY RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        nifty["nifty_rsi"] = 100 - (100 / (1 + rs)).fillna(50)

        # NIFTY VWAP
        nifty["nifty_vwap"] = (close * vol).cumsum() / vol.cumsum().replace(0, np.nan)
        nifty["nifty_vwap_dist"] = ((close - nifty["nifty_vwap"]) / nifty["nifty_vwap"]).fillna(0)

        cols = ["timestamp", "nifty_ret_1", "nifty_ret_3", "nifty_ret_5", "nifty_rsi", "nifty_vwap_dist"]
        return nifty[cols].sort_values("timestamp")
    except Exception as e:
        print(f"Warning: Could not fetch NIFTY 50 features: {e}")
        return pd.DataFrame()


def add_stock_features(df, nifty_df=None):
    # If NIFTY features are provided, merge on timestamp
    if nifty_df is not None and not nifty_df.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            nifty_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    else:
        for c in ["nifty_ret_1", "nifty_ret_3", "nifty_ret_5", "nifty_rsi", "nifty_vwap_dist"]:
            df[c] = 0.0

    frames = []
    for symbol, group in df.sort_values(["symbol", "timestamp"]).groupby("symbol"):
        group = group.copy().reset_index(drop=True)
        close = group["close"]
        high = group["high"]
        low = group["low"]
        op = group["open"]
        volume = group["volume"].replace(0, np.nan)

        returns = close.pct_change()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)

        true_range = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)

        # Core Technicals
        group["rsi_14"] = 100 - (100 / (1 + rs))
        group["vwap"] = (close * volume.fillna(0)).cumsum() / volume.fillna(0).cumsum().replace(0, np.nan)
        group["vwap_dist"] = (close - group["vwap"]) / group["vwap"]
        group["atr_14"] = true_range.rolling(14).mean()
        group["returns"] = returns
        group["volatility"] = returns.rolling(20).std()
        group["trend_strength"] = close.pct_change(12)
        group["ema_20"] = close.ewm(span=20, adjust=False).mean()
        group["ema_50"] = close.ewm(span=50, adjust=False).mean()
        group["ema_ratio"] = group["ema_20"] / group["ema_50"] - 1
        group["volume_zscore"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        group["volume_delta"] = volume.diff().fillna(0)
        group["vpin"] = (group["volume_delta"].abs() / volume.rolling(20).sum()).replace([np.inf, -np.inf], 0)

        spread_proxy = (high - low) / close.replace(0, np.nan)
        group["spread"] = spread_proxy
        group["mean_spread"] = spread_proxy
        group["imbalance"] = np.sign(group["close"] - group["open"]) * volume.fillna(0)
        group["mean_imbalance"] = group["imbalance"].rolling(5).mean()
        group["ofi"] = group["volume_delta"].rolling(5).mean().fillna(0)

        # Multi-timeframe Momentum & Acceleration
        group["momentum_5"] = close.pct_change(5)
        group["momentum_10"] = close.pct_change(10)
        group["momentum_20"] = close.pct_change(20)
        group["acceleration"] = returns.diff()

        # Volatility Ratio
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(50).std().replace(0, np.nan)
        group["vol_ratio"] = (vol_short / vol_long).fillna(1.0)
        group["volume_price_confirm"] = close.rolling(10).corr(volume.fillna(0)).fillna(0)
        group["bar_range_pct"] = (high - low) / close.replace(0, np.nan)
        group["gap_open"] = (op - close.shift(1)) / close.shift(1).replace(0, np.nan)

        # Institutional Reversal Indicators: Bollinger %B & Stochastic Oscillator
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std().replace(0, np.nan)
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        group["bb_pct_b"] = (close - lower_bb) / (upper_bb - lower_bb).replace(0, np.nan)

        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        group["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
        group["stoch_d"] = group["stoch_k"].rolling(3).mean()

        # Alpha vs NIFTY 50 Benchmark
        group["alpha_vs_nifty_5"] = group["momentum_5"] - group["nifty_ret_5"]
        group["alpha_vs_nifty_10"] = group["momentum_10"] - group["nifty_ret_5"]

        # Candlestick Physics & Geometry
        body = (close - op).abs()
        candle_range = (high - low).replace(0, np.nan)
        group["body_ratio"] = (body / candle_range).fillna(0)
        group["upper_wick"] = (high - pd.concat([op, close], axis=1).max(axis=1)) / candle_range.fillna(1)
        group["lower_wick"] = (pd.concat([op, close], axis=1).min(axis=1) - low) / candle_range.fillna(1)
        group["is_hammer"] = ((group["lower_wick"] > 0.6) & (group["body_ratio"] < 0.3)).astype(float)
        group["oversold_bounce"] = ((group["rsi_14"] < 35) & (returns > 0)).astype(float)

        # Time-of-Day Cyclical Encodings (Opening Rush vs Midday vs Closing)
        minute_of_day = group["timestamp"].dt.hour * 60 + group["timestamp"].dt.minute - (9 * 60 + 15)
        group["minute_of_day"] = minute_of_day
        group["sin_time"] = np.sin(2 * np.pi * minute_of_day / 375)
        group["cos_time"] = np.cos(2 * np.pi * minute_of_day / 375)
        group["is_opening_rush"] = (minute_of_day <= 45).astype(float)
        group["is_closing_rush"] = (minute_of_day >= 330).astype(float)

        group["last_funding_rate"] = 0.0
        group["last_macro_severity"] = 0.0
        group["macro_risk_factor"] = 0.0
        group["days_until_macro"] = 0.0
        group["days_since_macro"] = 0.0
        group["regime"] = np.select(
            [
                group["volatility"] > group["volatility"].rolling(100).quantile(0.80),
                group["ema_ratio"] > 0,
                group["ema_ratio"] < 0,
            ],
            ["high_volatility", "trending_bull", "trending_bear"],
            default="sideways",
        )
        group["company_name"] = STOCK_UNIVERSE.get(symbol, symbol)
        frames.append(group)

    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)


def build_stock_dataset(symbols=None, period="60d", interval="5m", output_path=OUTPUT_PATH):
    symbols = symbols or STOCK_SYMBOLS
    raw_frames = []
    skipped = []

    print(f"Fetching NIFTY 50 Benchmark features ({period}, {interval})...")
    nifty_df = fetch_nifty_index_features(period=period, interval=interval)

    print(f"Fetching NSE stock data: {', '.join(symbols)}")
    for symbol in symbols:
        try:
            df = fetch_stock_history(symbol, period=period, interval=interval)
            if df.empty:
                skipped.append(symbol)
                print(f"  - {symbol}: no data returned")
                continue
            raw_frames.append(df)
            print(f"  - {symbol}: {len(df)} bars")
        except Exception as exc:
            skipped.append(symbol)
            print(f"  - {symbol}: {exc}")

    if not raw_frames:
        raise RuntimeError("No stock data was fetched.")

    features = add_stock_features(pd.concat(raw_frames, ignore_index=True), nifty_df=nifty_df)
    labeled = []
    for symbol, group in features.groupby("symbol"):
        labeled_group = apply_triple_barrier_labels(
            group,
            pt_sl=[1.5, 1.0],
            t1=5,
            min_ret=0.0015,
        )
        labeled.append(labeled_group)

    final_df = pd.concat(labeled, ignore_index=True)
    final_df = final_df.dropna(subset=["target", "rsi_14", "atr_14", "vwap_dist", "volatility"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, engine="pyarrow")

    print(f"Saved {len(final_df)} labeled stock rows to {output_path}")
    print(final_df.groupby("symbol")["target"].agg(["count", "mean"]).to_string())
    if skipped:
        print(f"Skipped unavailable symbols: {', '.join(skipped)}")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Fetch NSE stock data and build labeled model dataset.")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    args = parser.parse_args()
    build_stock_dataset(period=args.period, interval=args.interval)


if __name__ == "__main__":
    main()
