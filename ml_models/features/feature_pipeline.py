import pandas as pd
import os
import glob
from technical import calculate_rsi, calculate_vwap, calculate_atr
from microstructure import calculate_order_imbalance, calculate_spread, calculate_volume_delta, calculate_vpin, calculate_order_flow_imbalance

class FeaturePipeline:
    def __init__(self, raw_data_dir="../data/orderbook", output_dir="../data/processed"):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_raw_data(self):
        """Loads and combines all raw parquet files."""
        files = glob.glob(f"{self.raw_data_dir}/*.parquet")
        if not files:
            print("No raw data found.")
            return None
            
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
        
    def resample_and_merge(self, df, timeframe='5T'):
        """
        Resamples raw tick/depth data into fixed timeframes (e.g., 5 min).
        """
        if df is None or df.empty:
            return None
            
        # Separate trades and depth
        trades = df[df['type'] == 'trade'].copy()
        depths = df[df['type'] == 'depth'].copy()
        
        # Resample trades to OHLCV
        if not trades.empty:
            trades.set_index('timestamp', inplace=True)
            # Group by symbol
            ohlcv_list = []
            for symbol, group in trades.groupby('symbol'):
                # Resample price to OHLC
                ohlc = group['price'].resample(timeframe).ohlc()
                # Resample volume
                volume = group['qty'].resample(timeframe).sum().rename('volume')
                
                # Calculate volume delta
                vol_delta = calculate_volume_delta(group).resample(timeframe).sum().rename('volume_delta')
                
                # Calculate VPIN
                vpin_val = calculate_vpin(group)
                
                merged = pd.concat([ohlc, volume, vol_delta], axis=1)
                merged['vpin'] = vpin_val
                merged['symbol'] = symbol
                ohlcv_list.append(merged)
            
            ohlcv_df = pd.concat(ohlcv_list)
        else:
            ohlcv_df = pd.DataFrame()
            
        # Resample depths (take mean for imbalance and spread)
        if not depths.empty:
            depths.set_index('timestamp', inplace=True)
            depth_list = []
            for symbol, group in depths.groupby('symbol'):
                mean_imbalance = group['imbalance'].resample(timeframe).mean().rename('mean_imbalance')
                mean_spread = group['spread'].resample(timeframe).mean().rename('mean_spread')
                
                # Calculate Order Flow Imbalance (OFI)
                ofi = calculate_order_flow_imbalance(group).resample(timeframe).sum().rename('ofi')
                
                merged = pd.concat([mean_imbalance, mean_spread, ofi], axis=1)
                merged['symbol'] = symbol
                depth_list.append(merged)
            
            depth_df = pd.concat(depth_list)
        else:
            depth_df = pd.DataFrame()
            
        # Merge trades and depths
        if not ohlcv_df.empty and not depth_df.empty:
            # We need to merge on both index (timestamp) and symbol
            ohlcv_df = ohlcv_df.reset_index().set_index(['timestamp', 'symbol'])
            depth_df = depth_df.reset_index().set_index(['timestamp', 'symbol'])
            final_df = ohlcv_df.join(depth_df, how='outer').reset_index()
        elif not ohlcv_df.empty:
            final_df = ohlcv_df.reset_index()
        else:
            final_df = depth_df.reset_index()
            
        return final_df

    def engineer_features(self, df):
        """Applies technical indicators to the resampled dataframe."""
        if df is None or df.empty:
            return df
            
        # Ensure it's sorted by time
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        feature_dfs = []
        for symbol, group in df.groupby('symbol'):
            group = group.copy()
            group['rsi_14'] = calculate_rsi(group, period=14, price_col='close')
            group['vwap'] = calculate_vwap(group, volume_col='volume', price_col='close')
            
            # Distance from VWAP
            group['vwap_dist'] = (group['close'] - group['vwap']) / group['vwap']
            
            group['atr_14'] = calculate_atr(group, period=14)
            
            feature_dfs.append(group)
            
        return pd.concat(feature_dfs, ignore_index=True)
        
    def run_pipeline(self):
        print("Loading raw data...")
        raw_df = self.load_raw_data()
        
        print("Resampling and merging (5m timeframe)...")
        resampled_df = self.resample_and_merge(raw_df, timeframe='5T')
        
        print("Engineering final features...")
        features_df = self.engineer_features(resampled_df)
        
        if features_df is not None and not features_df.empty:
            # Save to Parquet
            output_file = f"{self.output_dir}/features_latest.parquet"
            features_df.to_parquet(output_file, engine='pyarrow')
            print(f"Feature pipeline complete. Saved to {output_file}")
        else:
            print("No features generated.")

if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run_pipeline()
