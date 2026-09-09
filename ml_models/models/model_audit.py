import pandas as pd
import numpy as np

def audit_leakage(df):
    """
    Audits the dataframe for potential data leakage.
    Returns True if safe, False if leakage detected.
    """
    print("--- Running Data Leakage Audit ---")
    leakage_found = False
    
    exclude_cols = ['timestamp', 'symbol', 'target', 'future_price', 'future_return', 'type']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # 1. Check for suspicious feature names
    for feature in features:
        if 'future' in feature.lower() or 'next' in feature.lower() or 'target' in feature.lower():
            print(f"[!] WARNING: Potential leakage feature detected by name: {feature}")
            leakage_found = True
            
        # 2. Check correlation with future return (only for numeric)
        if 'future_return' in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            corr = df[feature].corr(df['future_return'])
            if abs(corr) > 0.95:
                print(f"[!] DANGER: Feature '{feature}' has extremely high correlation ({corr:.4f}) with future returns. Possible leakage.")
                leakage_found = True

    if not leakage_found:
        print("[+] PASS: No obvious data leakage detected.")
    else:
        print("[-] FAIL: Potential data leakage found. Review features immediately.")
        
    return not leakage_found

def validate_class_balance(df, target_col='target'):
    """
    Validates class balance and returns recommended scale_pos_weight for XGBoost.
    """
    print(f"\n--- Running Class Balance Validation ---")
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return 1.0
        
    counts = df[target_col].value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    total = len(df)
    
    print(f"Total samples: {total}")
    print(f"Class 0 (Negative/Skip): {neg_count} ({(neg_count/total)*100:.2f}%)")
    print(f"Class 1 (Positive/Trade): {pos_count} ({(pos_count/total)*100:.2f}%)")
    
    if pos_count == 0:
        print("[!] ERROR: No positive labels found. Cannot train model.")
        return 1.0
        
    scale_pos_weight = neg_count / pos_count
    
    if scale_pos_weight > 3.0 or scale_pos_weight < 0.33:
        print(f"[!] WARNING: High class imbalance detected. Recommended scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        print(f"[+] PASS: Classes are reasonably balanced.")
        
    return scale_pos_weight

if __name__ == "__main__":
    data_path = 'data/processed/labeled_data.parquet'
    import os
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        audit_leakage(df)
        validate_class_balance(df)
    else:
        print(f"Data file not found: {data_path}")
