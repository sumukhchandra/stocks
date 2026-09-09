import pandas as pd
import numpy as np

def calculate_order_imbalance(df, bid_qty_col='best_bid_qty', ask_qty_col='best_ask_qty'):
    """Calculates normalized order imbalance."""
    if bid_qty_col not in df.columns or ask_qty_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    imbalance = (df[bid_qty_col] - df[ask_qty_col]) / (df[bid_qty_col] + df[ask_qty_col] + 1e-8)
    return imbalance

def calculate_spread(df, bid_col='best_bid', ask_col='best_ask'):
    """Calculates bid-ask spread."""
    if bid_col not in df.columns or ask_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return df[ask_col] - df[bid_col]

def calculate_volume_delta(trade_df):
    """
    Calculates the difference between buy volume and sell volume.
    Requires trade data with 'is_buyer_maker' flag.
    If 'is_buyer_maker' is True, it means the trade was a sell (maker was buyer).
    """
    if 'is_buyer_maker' not in trade_df.columns or 'qty' not in trade_df.columns:
        return pd.Series(index=trade_df.index, dtype=float)
    
    # +1 for buy, -1 for sell
    trade_sign = np.where(trade_df['is_buyer_maker'], -1, 1)
    volume_delta = trade_sign * trade_df['qty']
    return volume_delta

def calculate_aggressive_bursts(group, window='1min'):
    """
    Identifies short-term rapid sequences of taker buy volume.
    Returns the rolling sum of buy volume over a short window.
    """
    qty_col = 'qty' if 'qty' in group.columns else 'quantity'
    if 'is_buyer_maker' not in group.columns or qty_col not in group.columns:
        return pd.Series(0.0, index=group.index, name='aggressive_bursts')
        
    buy_mask = ~group['is_buyer_maker']
    buy_vol = group[qty_col].where(buy_mask, 0)
    
    # Calculate rolling sum of buy volume over the short window
    rolling_buy = buy_vol.rolling(window).sum().fillna(0)
    return rolling_buy

def calculate_trade_clusters(group, percentile=0.90):
    """
    Identifies the volume of unusually large trades within a group.
    """
    qty_col = 'qty' if 'qty' in group.columns else 'quantity'
    if qty_col not in group.columns or len(group) == 0:
        return pd.Series(0.0, index=group.index, name='large_trade_volume')
        
    threshold = group[qty_col].quantile(percentile)
    if pd.isna(threshold):
        threshold = 0
    large_trades = group[qty_col].where(group[qty_col] > threshold, 0)
    return large_trades

def calculate_vpin(trade_df, n_buckets=50):
    """
    Calculates VPIN (Volume-synchronized Probability of Informed Trading).
    Approximated by volume-weighted imbalance across trade buckets.
    """
    if 'qty' not in trade_df.columns or 'is_buyer_maker' not in trade_df.columns:
        return 0.0
        
    total_vol = trade_df['qty'].sum()
    if total_vol == 0:
        return 0.0
        
    bucket_vol = total_vol / n_buckets
    trade_df = trade_df.copy()
    trade_df['cum_vol'] = trade_df['qty'].cumsum()
    trade_df['bucket'] = (trade_df['cum_vol'] / bucket_vol).astype(int)
    
    # Calculate buy/sell volume per bucket
    trade_df['buy_vol'] = np.where(~trade_df['is_buyer_maker'], trade_df['qty'], 0)
    trade_df['sell_vol'] = np.where(trade_df['is_buyer_maker'], trade_df['qty'], 0)
    
    bucket_stats = trade_df.groupby('bucket').agg({
        'buy_vol': 'sum',
        'sell_vol': 'sum'
    })
    
    # VPIN = sum|Sell - Buy| / Total Volume
    vpin = (bucket_stats['sell_vol'] - bucket_stats['buy_vol']).abs().sum() / total_vol
    return vpin

def calculate_order_flow_imbalance(df):
    """
    Calculates Order Flow Imbalance (OFI) based on Cont et al. (2014).
    OFI = Delta(BidVol) - Delta(AskVol)
    """
    if 'best_bid' not in df.columns or 'best_ask' not in df.columns:
        return pd.Series(0.0, index=df.index)
        
    # Delta Bid Vol
    bid_price_change = df['best_bid'].diff()
    bid_vol_ofi = np.where(bid_price_change > 0, df['best_bid_qty'],
                  np.where(bid_price_change < 0, -df['best_bid_qty'],
                  df['best_bid_qty'].diff()))
                  
    # Delta Ask Vol
    ask_price_change = df['best_ask'].diff()
    ask_vol_ofi = np.where(ask_price_change < 0, df['best_ask_qty'],
                  np.where(ask_price_change > 0, -df['best_ask_qty'],
                  -df['best_ask_qty'].diff()))
                  
    ofi = pd.Series(bid_vol_ofi - ask_vol_ofi, index=df.index).fillna(0)
    return ofi
