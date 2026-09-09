"""
Download Market Data CLI: Fetches and stores 5-minute OHLCV candles into Parquet storage.
"""

import argparse
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trading.data.collectors.market_data import MarketDataCollector
from trading.data.processors.cleaner import DataCleaner
from trading.data.storage.parquet_store import ParquetStore


def main():
    parser = argparse.ArgumentParser(description="Download NSE Market Data")
    parser.add_argument("--period", default="60d", help="Data period (e.g. 5d, 60d)")
    parser.add_argument("--interval", default="5m", help="Candle interval (e.g. 1m, 5m)")
    parser.add_argument("--output", default="master_labeled_dataset.parquet", help="Output filename")
    args = parser.parse_args()

    print(f"Ingesting market data (Period: {args.period}, Interval: {args.interval})...")
    collector = MarketDataCollector()
    raw_df = collector.fetch_all(period=args.period, interval=args.interval)
    if raw_df.empty:
        print("No market data fetched.")
        return

    cleaned_df = DataCleaner.clean(raw_df)
    store = ParquetStore()
    out_path = store.save_dataset(cleaned_df, filename=args.output)
    print(f"Successfully saved {len(cleaned_df):,} candles across {cleaned_df['symbol'].nunique()} stocks to: {out_path}")


if __name__ == "__main__":
    main()
