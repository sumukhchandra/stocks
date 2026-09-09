import pandas as pd

import numpy as np

import glob

import os

import logging

from typing import Dict, List, Optional, Tuple



# Setup logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



class MarketDataValidator:

    """

    Institutional-grade market data validation pipeline.

    Ensures data integrity across historical and real-time sources.

    """

    

    def __init__(self, raw_data_dir: str = "data/raw"):

        self.raw_data_dir = raw_data_dir

        self.klines_dir = os.path.join(raw_data_dir, "klines")

        self.trades_dir = os.path.join(raw_data_dir, "trades")

        self.report = []



    def log_check(self, name: str, passed: bool, message: str):

        status = "PASS" if passed else "FAIL"

        self.report.append({"check": name, "status": status, "message": message})

        if not passed:

            logger.error(f"[{status}] {name}: {message}")

        else:

            logger.info(f"[{status}] {name}: {message}")



    def validate_ohlcv(self, df: pd.DataFrame, timeframe: str = "5min") -> bool:

        """Validates OHLCV integrity, continuity, and quality."""

        if df.empty:

            self.log_check("OHLCV Empty", False, "DataFrame is empty.")

            return False



        all_passed = True

        

        # 1. Timezone Consistency

        if df['timestamp'].dt.tz is None:

            self.log_check("Timezone", False, "Timestamps are naive. Expected UTC.")

            all_passed = False

        else:

            self.log_check("Timezone", True, "Timestamps are UTC.")



        # 2. Duplicates

        dupes = df['timestamp'].duplicated().sum()

        self.log_check("Duplicates", dupes == 0, f"Found {dupes} duplicate timestamps.")

        if dupes > 0: all_passed = False



        # 3. Monotonicity

        monotonic = df['timestamp'].is_monotonic_increasing

        self.log_check("Monotonicity", monotonic, "Timestamps are not strictly increasing.")

        if not monotonic: all_passed = False



        # 4. NaNs & Infs

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']

        nans = df[numeric_cols].isna().sum().sum()

        infs = np.isinf(df[numeric_cols]).sum().sum()

        self.log_check("NaNs/Infs", (nans == 0 and infs == 0), f"Found {nans} NaNs and {infs} Infs.")

        if nans > 0 or infs > 0: all_passed = False



        # 5. OHLC Integrity

        integrity_violations = (

            (df['low'] > df['open']) | 

            (df['low'] > df['close']) | 

            (df['high'] < df['open']) | 

            (df['high'] < df['close'])

        ).sum()

        self.log_check("OHLC Integrity", integrity_violations == 0, f"Found {integrity_violations} integrity violations (H<O, H<C, L>O, or L>C).")

        if integrity_violations > 0: all_passed = False



        # 6. Negative Values

        negatives = (df[numeric_cols] < 0).sum().sum()

        self.log_check("Negative Prices", negatives == 0, f"Found {negatives} negative values.")

        if negatives > 0: all_passed = False



        # 7. Continuity (Gaps)

        expected_diff = pd.Timedelta(timeframe)

        actual_diffs = df['timestamp'].diff().dropna()

        gaps = (actual_diffs > expected_diff).sum()

        self.log_check("Continuity", gaps == 0, f"Found {gaps} missing candles (> {timeframe} gap).")

        if gaps > 0: all_passed = False



        return all_passed



    def validate_trades(self, trades_dir: str):

        """Validates trade data for sequence gaps and abnormal price jumps."""

        files = sorted(glob.glob(os.path.join(trades_dir, "*.parquet")))

        if not files:

            self.log_check("Trades Source", False, "No trade files found.")

            return



        total_rows = 0

        missing_ids = 0

        abnormal_jumps = 0

        prev_price = None

        prev_id = None



        for f in files:

            df = pd.read_parquet(f)

            total_rows += len(df)

            

            # Check ID sequence

            if 'agg_trade_id' in df.columns:

                id_diffs = df['agg_trade_id'].diff().dropna()

                missing_ids += (id_diffs > 1).sum()

                

                if prev_id is not None:

                    if df['agg_trade_id'].iloc[0] - prev_id > 1:

                        missing_ids += 1

                prev_id = df['agg_trade_id'].iloc[-1]



            # Check Price Jumps (e.g. > 5% in a single trade event - very rare for liquid assets)

            if 'price' in df.columns:

                price_pct_change = df['price'].pct_change().abs()

                abnormal_jumps += (price_pct_change > 0.05).sum()

                

                if prev_price is not None:

                    if abs(df['price'].iloc[0] - prev_price) / prev_price > 0.05:

                        abnormal_jumps += 1

                prev_price = df['price'].iloc[-1]



        self.log_check("Trade Sequence", missing_ids == 0, f"Found {missing_ids} missing trade IDs.")

        self.log_check("Price Jumps", abnormal_jumps == 0, f"Found {abnormal_jumps} abnormal price jumps (>5%).")



    def run_full_validation(self):

        logger.info("Starting Full Market Data Validation Pipeline...")

        

        # Validate Klines

        kline_files = sorted(glob.glob(os.path.join(self.klines_dir, "*.parquet")))

        if kline_files:

            dfs = [pd.read_parquet(f) for f in kline_files]

            df_klines = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)

            df_klines['timestamp'] = pd.to_datetime(df_klines['timestamp'], utc=True)

            self.validate_ohlcv(df_klines)

        else:

            self.log_check("Klines Source", False, "No kline files found.")



        # Validate Trades

        self.validate_trades(self.trades_dir)



        self.print_summary()



    def print_summary(self):

        print("\n" + "="*50)

        print("VALIDATION SUMMARY REPORT")

        print("="*50)

        passes = sum(1 for r in self.report if r['status'] == 'PASS')

        fails = sum(1 for r in self.report if r['status'] == 'FAIL')

        

        for r in self.report:

            icon = "[OK]" if r['status'] == 'PASS' else "[FAIL]"

            print(f"{icon} {r['check']:<20} | {r['status']:<5} | {r['message']}")

            

        print("-" * 50)

        print(f"TOTAL: {len(self.report)} | PASSED: {passes} | FAILED: {fails}")

        print("="*50 + "\n")



if __name__ == "__main__":

    validator = MarketDataValidator()

    validator.run_full_validation()

