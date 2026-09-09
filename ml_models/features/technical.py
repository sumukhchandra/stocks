import pandas as pd
import numpy as np

def calculate_rsi(df, period=14, price_col='price'):
    """Calculates Relative Strength Index."""
    if len(df) < period:
        return pd.Series(index=df.index, dtype=float)
        
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(df, volume_col='qty', price_col='price'):
    """Calculates Volume Weighted Average Price."""
    v = df[volume_col]
    p = df[price_col]
    vwap = (p * v).cumsum() / v.cumsum()
    return vwap

def calculate_atr(df, high_col='high', low_col='low', close_col='close', period=14):
    """Calculates Average True Range."""
    if high_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
        
    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - df[close_col].shift()).abs()
    tr3 = (df[low_col] - df[close_col].shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr
