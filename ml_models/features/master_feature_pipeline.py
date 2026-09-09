import pandas as pd
import os
import sys
import glob

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.historical_pipeline import HistoricalFeaturePipeline
from features.macro_features import MacroFeatureEngineer
from regimes.regime_classifier import RegimeClassifier
from labels.triple_barrier import apply_triple_barrier_labels

class MasterFeaturePipeline:
    def __init__(self, output_dir='data/processed'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def add_funding_rates(self, df):
        """Loads and merges funding rates into the main dataframe."""
        funding_dir = 'data/raw/futures/um/fundingRate'
        files = glob.glob(os.path.join(funding_dir, '*.parquet'))
        if not files:
            print("No funding rate data found.")
            return df
            
        dfs = []
        for f in files:
            dfs.append(pd.read_parquet(f))
        
        if not dfs:
            return df
            
        funding_df = pd.concat(dfs, ignore_index=True)
        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], utc=True).dt.tz_localize(None)
        
        # Sort and merge_asof since funding rates are every 8 hours
        df = df.sort_values('timestamp')
        funding_df = funding_df.sort_values('timestamp')
        
        merged_dfs = []
        for symbol, group in df.groupby('symbol'):
            sym_funding = funding_df[funding_df['symbol'] == symbol]
            if sym_funding.empty:
                merged_dfs.append(group)
                continue
                
            # Forward fill the funding rate
            merged = pd.merge_asof(
                group, 
                sym_funding[['timestamp', 'last_funding_rate']], 
                on='timestamp', 
                direction='backward'
            )
            merged_dfs.append(merged)
            
        return pd.concat(merged_dfs, ignore_index=True)

    def run(self):
        print("1. Running Historical Pipeline (OHLCV + Microstructure)...")
        # In a real environment, you'd run this. For now, we load what's already generated.
        # hfp = HistoricalFeaturePipeline()
        # hfp.run()
        
        base_features_path = os.path.join(self.output_dir, 'features_latest.parquet')
        if not os.path.exists(base_features_path):
            print("Base features not found. Please run historical_pipeline.py first.")
            return
            
        df = pd.read_parquet(base_features_path)
        
        print("2. Integrating Funding Rates...")
        df = self.add_funding_rates(df)
        
        print("3. Adding Macro Features...")
        macro_engineer = MacroFeatureEngineer()
        df = macro_engineer.add_macro_features(df)
        
        print("4. Applying Regime Classification (Leakage Protected)...")
        regime_classifier = RegimeClassifier()
        # FIX: Fit only on the training portion (first 80%) to prevent look-ahead bias
        split_idx = int(len(df) * 0.8)
        regime_classifier.fit(df.iloc[:split_idx])
        df['regime'] = regime_classifier.predict(df)
        
        print("5. Applying Elite Swing Labels (3% Move / 24H Horizon)...")
        # pt_sl=[3.0, 1.5] -> 3% target, 1.5% stop loss relative to min_ret=0.01 (1%)
        # Actually easier to use process_trend_data logic for Elite Swing:
        dfs = []
        for symbol, group in df.groupby('symbol'):
            group = group.copy().sort_values('timestamp')
            # 24H horizon
            group['future_max_high'] = group['high'].shift(-24).rolling(24).max().shift(-24)
            group['future_return_3%'] = (group['future_max_high'] - group['close']) / group['close']
            group['target'] = (group['future_return_3%'] > 0.03).astype(int)
            group['future_return_t1'] = group['close'].shift(-24).pct_change(24).shift(-24)
            dfs.append(group)
        df = pd.concat(dfs, ignore_index=True)
        
        output_path = os.path.join(self.output_dir, 'master_labeled_dataset.parquet')
        df.dropna(subset=['target']).to_parquet(output_path, engine='pyarrow')
        print(f"Master Pipeline Complete (Elite Swing Mode). Saved to {output_path}")
        print("Data Summary:")
        print(f"Total Rows: {len(df)}")
        print("Regime Distribution:")
        print(df['regime'].value_counts())
        
        return df

if __name__ == "__main__":
    pipeline = MasterFeaturePipeline()
    pipeline.run()
