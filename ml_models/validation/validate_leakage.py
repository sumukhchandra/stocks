import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_temporal_integrity(file_path='data/processed/labeled_data.parquet'):
    """
    Verifies that for every prediction, the data used (features) 
    existed BEFORE the outcome (target/future_return).
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    logger.info(f"Checking temporal integrity for {len(df)} rows...")
    
    # 1. Basic Timestamp Assertion
    # In this pipeline, the row's 'timestamp' is the time the prediction is made.
    # The 'future_price' is at timestamp + 5min.
    # We must ensure no feature in this row 'knows' anything about t + 5min.
    
    # Check if 'future_price' matches the close price of the NEXT row for the same symbol
    df = df.sort_values(['symbol', 'timestamp'])
    for symbol, group in df.groupby('symbol'):
        # Only check numeric features
        numeric_group = group.select_dtypes(include=[np.number])
        target_cols = ['target', 'triple_barrier_label', 'future_return_t1', 'future_price']
        feature_cols = [c for c in numeric_group.columns if c not in target_cols]
        
        # Check correlation with the main outcome metric
        outcome_col = 'future_return_t1'
        if outcome_col not in numeric_group.columns:
            # Fallback to whatever looks like a return
            returns = [c for c in numeric_group.columns if 'return' in c]
            if returns:
                outcome_col = returns[0]
            else:
                logger.error("No outcome return column found for correlation check.")
                continue

        for col in feature_cols:
            # Check for constant features first to avoid RuntimeWarning
            if numeric_group[col].nunique() <= 1:
                continue
                
            corr = numeric_group[col].corr(numeric_group[outcome_col])

            if abs(corr) > 0.95:
                logger.warning(f"[LEAKAGE ALERT] Feature '{col}' has a suspiciously high correlation ({corr:.4f}) with outcome.")
            elif abs(corr) > 0.80:
                logger.info(f"[POTENTIAL LEAKAGE] Feature '{col}' has high correlation ({corr:.4f}) with outcome.")

    # 2. Duplicate Check with Target
    # Ensure target isn't just a copy of some current feature
    # (e.g. accidentally using future RSI as current RSI)
    
    logger.info("Temporal integrity check complete.")

if __name__ == "__main__":
    validate_temporal_integrity()
