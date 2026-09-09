import requests
import pandas as pd
import numpy as np

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
LIMIT = 100
INTERVAL = "1d"

def fetch_data(symbol):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT
    }
    try:
        response = requests.get(BINANCE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Use exponential moving average for smoother RSI calculation
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def main():
    results = []
    for symbol in SYMBOLS:
        df = fetch_data(symbol)
        if df is not None and not df.empty:
            df["rsi_14"] = calculate_rsi(df["close"], 14)
            df["ema_20"] = calculate_ema(df["close"], 20)
            
            # Trend strength = (Close - EMA20) / EMA20 * 100
            current_close = df["close"].iloc[-1]
            current_ema = df["ema_20"].iloc[-1]
            current_rsi = df["rsi_14"].iloc[-1]
            
            trend_strength = ((current_close - current_ema) / current_ema) * 100
            
            # Simple momentum score: normalized sum of RSI (rescaled around 50) and Trend Strength
            momentum_score = trend_strength + (current_rsi - 50) * 0.5
            
            results.append({
                "symbol": symbol,
                "close": current_close,
                "rsi_14": current_rsi,
                "ema_20_diff_pct": trend_strength,
                "momentum_score": momentum_score
            })
            
    if not results:
        print("Failed to fetch data.")
        return

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="momentum_score", ascending=False).reset_index(drop=True)
    
    print("--- Market Scan Results ---")
    print(df_results.to_string(index=False))
    print("\n")
    
    top_coin = df_results.iloc[0]
    print(f"Top Recommendation based on technicals: {top_coin['symbol']}")
    print(f"Momentum Score: {top_coin['momentum_score']:.2f}")

if __name__ == "__main__":
    main()
