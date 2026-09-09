import pandas as pd
import os
import glob
from technical import calculate_rsi, calculate_vwap, calculate_atr
from microstructure import calculate_aggressive_bursts, calculate_trade_clusters, calculate_vpin

class HistoricalFeaturePipeline:
    def __init__(self, raw_data_dir="data/raw/klines", output_dir="data/processed"):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run(self):
        # Load klines
        files = glob.glob(os.path.join(self.raw_data_dir, "*.parquet"))
        if not files:
            print(f"No kline parquet files found in {self.raw_data_dir}")
            return
            
        dfs = []
        for f in files:
            dfs.append(pd.read_parquet(f))
            
        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None) # Normalize to naive for merging
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        print(f"Loaded {len(df)} candles.")
        
        # Load trades and compute microstructure features
        trades_dir = self.raw_data_dir.replace("klines", "trades")
        trade_files = glob.glob(os.path.join(trades_dir, "*.parquet"))
        trades_agg = None
        
        if trade_files:
            print(f"Found {len(trade_files)} trade archives. Aggregating microstructure features memory-efficiently...")
            
            all_agg_list = []
            for f in trade_files:
                print(f"Processing {os.path.basename(f)}...")
                # Only read necessary columns
                needed_cols = ['symbol', 'price', 'quantity', 'timestamp', 'is_buyer_maker']
                try:
                    trades_chunk = pd.read_parquet(f, columns=needed_cols)
                except ValueError:
                    # Try with 'qty' instead of 'quantity'
                    needed_cols = ['symbol', 'price', 'qty', 'timestamp', 'is_buyer_maker']
                    trades_chunk = pd.read_parquet(f, columns=needed_cols)
                    trades_chunk = trades_chunk.rename(columns={'qty': 'quantity'})
                    
                trades_chunk['timestamp'] = pd.to_datetime(trades_chunk['timestamp'], utc=True).dt.tz_localize(None)
                
                for symbol, group in trades_chunk.groupby('symbol'):
                    group = group.set_index('timestamp').sort_index()
                    
                    buy_mask = ~group['is_buyer_maker']
                    group['buy_vol'] = group['quantity'].where(buy_mask, 0)
                    group['sell_vol'] = group['quantity'].where(~buy_mask, 0)
                    
                    group['aggressive_bursts'] = calculate_aggressive_bursts(group)
                    group['large_trade_volume'] = calculate_trade_clusters(group)
                    
                    resampled = group.resample('5min').agg(
                        total_trades=('price', 'count'),
                        buy_vol=('buy_vol', 'sum'),
                        sell_vol=('sell_vol', 'sum'),
                        aggressive_bursts=('aggressive_bursts', 'max'),
                        large_trade_volume=('large_trade_volume', 'sum')
                    )
                    
                    # Add VPIN per bucket
                    # Ensure 'qty' is present for VPIN
                    if 'quantity' in group.columns and 'qty' not in group.columns:
                        group['qty'] = group['quantity']
                        
                    vpin_series = group.resample('5min').apply(lambda x: calculate_vpin(x) if not x.empty else 0.0)
                    resampled['vpin'] = vpin_series
                    
                    resampled['symbol'] = symbol
                    all_agg_list.append(resampled.reset_index())
                
                # Free memory
                del trades_chunk
                
            trades_agg_raw = pd.concat(all_agg_list, ignore_index=True)
            
            # Final aggregation in case symbols/timestamps overlap across files
            trades_agg = trades_agg_raw.groupby(['symbol', 'timestamp']).agg({
                'total_trades': 'sum',
                'buy_vol': 'sum',
                'sell_vol': 'sum',
                'aggressive_bursts': 'max',
                'large_trade_volume': 'sum',
                'vpin': 'mean' # VPIN is already a ratio, mean is a reasonable proxy if overlapping
            }).reset_index()
            
            trades_agg['volume_delta'] = trades_agg['buy_vol'] - trades_agg['sell_vol']
            trades_agg['taker_buy_ratio'] = trades_agg['buy_vol'] / (trades_agg['buy_vol'] + trades_agg['sell_vol'] + 1e-8)
            trades_agg['trade_intensity'] = trades_agg['total_trades']
            
            # Merge trades into klines
            df = pd.merge(df, trades_agg, on=['symbol', 'timestamp'], how='left')
            df['volume_delta'] = df['volume_delta'].fillna(0)
            df['taker_buy_ratio'] = df['taker_buy_ratio'].fillna(0.5)
            df['trade_intensity'] = df['trade_intensity'].fillna(0)
            df['aggressive_bursts'] = df['aggressive_bursts'].fillna(0)
            df['large_trade_volume'] = df['large_trade_volume'].fillna(0)
            df['vpin'] = df['vpin'].fillna(0)
            print("Microstructure features merged successfully.")
        else:
            print("No trades data found. Using placeholders.")
            df['volume_delta'] = 0.0
            df['taker_buy_ratio'] = 0.5
            df['trade_intensity'] = 0.0
            df['aggressive_bursts'] = 0.0
            df['large_trade_volume'] = 0.0
        
        feature_dfs = []
        for symbol, group in df.groupby('symbol'):
            group = group.copy()
            group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
            group['vwap'] = calculate_vwap(group, volume_col='volume', price_col='close')
            group['vwap_dist'] = (group['close'] - (group['vwap'] + 1e-8)) / (group['vwap'] + 1e-8)
            group['atr_14'] = calculate_atr(group, period=14)
            
            # True microstructure features mapped
            group['imbalance'] = group['taker_buy_ratio'] # Using true ratio instead of 0
            group['spread'] = 0.0 # Spread requires orderbook depth snaps, we don't have them yet, so keep 0
            group['type'] = 'klines'
            
            feature_dfs.append(group)
            
        features_df = pd.concat(feature_dfs, ignore_index=True)
        
        output_file = os.path.join(self.output_dir, "features_latest.parquet")
        features_df.to_parquet(output_file, engine='pyarrow')
        print(f"Generated historical features. Saved to {output_file}")

if __name__ == "__main__":
    pipeline = HistoricalFeaturePipeline()
    pipeline.run()
