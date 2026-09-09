import argparse
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.stocks_config import STOCK_SYMBOLS, STOCK_UNIVERSE
from data.fetch_stock_data import _flatten_columns, add_stock_features
from ml_models.labels.triple_barrier import apply_triple_barrier_labels
from ml_models.models.train_ensemble import ModelEnsemble
from ml_models.models.final_ensemble_engine import FinalEnsembleEngine

def fetch_stock_range(symbol, start_date, end_date, interval="5m"):
    df = yf.download(
        symbol,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()

    df = _flatten_columns(df.reset_index())
    time_col = "datetime" if "datetime" in df.columns else "date" if "date" in df.columns else "index"
    if time_col not in df.columns:
        return pd.DataFrame()
        
    df = df.rename(columns={time_col: "timestamp", "adj close": "adj_close"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["symbol"] = symbol
    return df.dropna(subset=["open", "high", "low", "close"])

def build_training_dataset(symbols, start_date, end_date):
    raw_frames = []
    print(f"Fetching historical training data from {start_date} to {end_date}...")
    for symbol in symbols:
        df = fetch_stock_range(symbol, start_date, end_date)
        if not df.empty:
            raw_frames.append(df)
            print(f"  - {symbol}: {len(df)} bars")
        else:
            print(f"  - {symbol}: no data")

    if not raw_frames:
        raise RuntimeError("No historical data fetched.")

    combined = pd.concat(raw_frames, ignore_index=True)
    features = add_stock_features(combined)
    
    labeled = []
    for symbol, group in features.groupby("symbol"):
        labeled_group = apply_triple_barrier_labels(
            group,
            pt_sl=[1.5, 1.0],
            t1=5,
            min_ret=0.0015,
        )
        labeled_group["target_ret"] = labeled_group["future_return_t1"]
        labeled.append(labeled_group)

    final_df = pd.concat(labeled, ignore_index=True)
    final_df = final_df.dropna(subset=["target", "rsi_14", "atr_14", "vwap_dist", "volatility"])
    return final_df

def run_simulation(target_date_str, starting_capital):
    if target_date_str.lower() == 'today':
        target_date = datetime.now().date()
    else:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        
    print(f"\n{'='*50}")
    print(f"INITIALIZING SIMULATOR FOR DATE: {target_date}")
    print(f"{'='*50}\n")
    
    # 1. Historical Train Date Range (60 days max for yfinance 5m)
    train_end = target_date
    train_start = target_date - timedelta(days=45)
    
    # 2. Fetch and Build Training Data
    try:
        train_df = build_training_dataset(STOCK_SYMBOLS, train_start, train_end)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return
        
    # 3. Train Models
    print("\nTraining Models on Historical Data (No Look-Ahead)...")
    model_base_dir = os.path.join(os.path.dirname(__file__), 'models', 'saved_models_sim')
    ensemble_dir = os.path.join(model_base_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)
    
    exclude = [
        'timestamp', 'symbol', 'target', 'future_price', 'future_return', 
        'future_return_t1', 'triple_barrier_label', 'is_synthetic',
        'ignore', 'close_time', 'open_time', 'future_max_high', 'future_return_3%',
        'actual_return', 'target_ret', 'regime', 'ot', 'ct', 'qv', 'nt', 'tbb', 'tbq', 'i', 'downloaded_at', 'company_name'
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
    feature_cols = [c for c in feature_cols if train_df[c].std() > 0]
    
    train_df = train_df.dropna(subset=['target'])
    X = train_df[feature_cols]
    y = train_df['target']
    
    if len(X) < 100:
        print("Not enough training data available.")
        return
        
    ensemble = ModelEnsemble(model_dir=ensemble_dir)
    ensemble.train_all(X, y, feature_cols)
    print("Model Training Complete.")
    
    # 4. Fetch Simulation Day Data (with 10-day warmup buffer for EMA/Volatility)
    print("\nFetching Intraday Simulation Data...")
    sim_start = target_date - timedelta(days=10)
    sim_end = target_date + timedelta(days=2) # To ensure we capture the full day
    
    sim_raw = []
    for symbol in STOCK_SYMBOLS:
        df = fetch_stock_range(symbol, sim_start, sim_end)
        if not df.empty:
            sim_raw.append(df)
            
    if not sim_raw:
        print("No data available for the simulation date.")
        return
        
    sim_features = add_stock_features(pd.concat(sim_raw, ignore_index=True))
    
    # Filter strictly to the target date
    sim_day_df = sim_features[sim_features['timestamp'].dt.date == target_date].sort_values('timestamp')
    if sim_day_df.empty:
        print(f"No trading data found exactly on {target_date}. Was the market closed?")
        return
        
    print(f"Loaded {len(sim_day_df)} simulation bars. Starting chronological replay...")
    
    # 5. Load Engine
    engine = FinalEnsembleEngine(model_dir=model_base_dir)
    
    # 6. Simulation Loop
    capital = starting_capital# Starting capital 1 Lakh INR
    trades = []
    active_positions = {}
    
    timestamps = sorted(sim_day_df['timestamp'].unique())
    
    for current_time in timestamps:
        current_bars = sim_day_df[sim_day_df['timestamp'] == current_time]
        
        # 6a. Manage Active Positions (Check SL/TP)
        closed_this_tick = []
        for symbol, pos in active_positions.items():
            bar = current_bars[current_bars['symbol'] == symbol]
            if bar.empty: continue
            
            high = bar['high'].iloc[0]
            low = bar['low'].iloc[0]
            close = bar['close'].iloc[0]
            
            # Check TP
            if high >= pos['tp_price']:
                pnl = (pos['tp_price'] - pos['entry_price']) * pos['qty']
                trades.append({'symbol': symbol, 'type': 'TP', 'pnl': pnl, 'close_time': current_time})
                capital += pnl
                closed_this_tick.append(symbol)
                print(f"[{current_time.time()}] TAKE PROFIT HIT: {symbol} | PnL: +INR {pnl:.2f} | Cap: INR {capital:.2f}")
                continue
                
            # Check SL
            if low <= pos['sl_price']:
                pnl = (pos['sl_price'] - pos['entry_price']) * pos['qty']
                trades.append({'symbol': symbol, 'type': 'SL', 'pnl': pnl, 'close_time': current_time})
                capital += pnl
                closed_this_tick.append(symbol)
                print(f"[{current_time.time()}] STOP LOSS HIT: {symbol} | PnL: -INR {abs(pnl):.2f} | Cap: INR {capital:.2f}")
                continue
                
            # Expiration
            if current_time >= pos['expiration']:
                pnl = (close - pos['entry_price']) * pos['qty']
                trades.append({'symbol': symbol, 'type': 'TIME', 'pnl': pnl, 'close_time': current_time})
                capital += pnl
                closed_this_tick.append(symbol)
                print(f"[{current_time.time()}] TIME EXIT: {symbol} | PnL: INR {pnl:.2f} | Cap: INR {capital:.2f}")
                
        for s in closed_this_tick:
            del active_positions[s]
            
        # 6b. Evaluate New Setups
        for symbol in STOCK_SYMBOLS:
            if symbol in active_positions: continue # Don't pyramid for now
            
            # Get historical window up to current time for features
            symbol_hist = sim_features[(sim_features['symbol'] == symbol) & (sim_features['timestamp'] <= current_time)]
            if len(symbol_hist) < 50: continue # Need warmup
            
            window = symbol_hist.tail(50)
            current_close = window['close'].iloc[-1]
            
            try:
                decision = engine.evaluate_state(window)
            except Exception as e:
                continue # Model might fail if features are missing
                
            prob = decision['final_probability']

            print(
                f"{current_time} | {symbol} | "
                f"Prob={prob:.4f} | "
                f"ExpRet={decision.get('expected_return',0):.4f}"
            )
            
            # Entry Logic: High Probability + Positive Expected Return
            if prob > 0.40 and decision.get('expected_return', 0) > 0.0005:
                # Calculate size based on capital (max 10% risk)
                trade_alloc = capital * 0.10
                qty = trade_alloc / current_close
                
                # Dynamic SL/TP based on ATR
                atr = window['atr_14'].iloc[-1]
                if pd.isna(atr) or atr == 0: atr = current_close * 0.005
                
                tp_price = current_close + (atr * 2)
                sl_price = current_close - atr
                
                active_positions[symbol] = {
                    'entry_price': current_close,
                    'qty': qty,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'entry_time': current_time,
                    'expiration': current_time + timedelta(minutes=int(decision['expected_holding_mins']))
                }
                print(f"[{current_time.time()}] NEW ENTRY: {symbol} @ INR {current_close:.2f} | Prob: {prob:.2f} | TP: INR {tp_price:.2f} | SL: INR {sl_price:.2f}")

    # 7. Close remaining positions at end of day
    if active_positions:
        print("\nMarket Closing - Liquidating Remaining Positions...")
        last_time = timestamps[-1]
        last_bars = sim_day_df[sim_day_df['timestamp'] == last_time]
        for symbol, pos in active_positions.items():
            bar = last_bars[last_bars['symbol'] == symbol]
            close = bar['close'].iloc[0] if not bar.empty else pos['entry_price']
            pnl = (close - pos['entry_price']) * pos['qty']
            trades.append({'symbol': symbol, 'type': 'EOD', 'pnl': pnl, 'close_time': last_time})
            capital += pnl
            print(f"[{last_time.time()}] EOD EXIT: {symbol} | PnL: INR {pnl:.2f} | Cap: INR {capital:.2f}")

    # 8. Summary Report
    print(f"\n{'='*50}")
    print("SIMULATION SUMMARY REPORT")
    print(f"{'='*50}")
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    losing_trades = len([t for t in trades if t['pnl'] <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    net_pnl = capital - starting_capital
    
    print(f"Target Date      : {target_date}")
    print(f"Starting Capital : INR {starting_capital:,.2f}")
    print(f"Ending Capital   : INR {capital:,.2f}")
    print(
    f"Net PnL          : INR {net_pnl:+,.2f} "
    f"({(net_pnl/starting_capital)*100:+.2f}%)"
)
    print(f"Total Trades     : {total_trades}")
    print(f"Winning Trades   : {winning_trades}")
    print(f"Losing Trades    : {losing_trades}")
    print(f"Win Rate         : {win_rate:.2f}%")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronological Stock Simulation")
    parser.add_argument(
        "--date",
        default="today",
        help="Simulation Date (YYYY-MM-DD or 'today')"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Starting Capital"
    )
    args = parser.parse_args()
    run_simulation(args.date, args.capital)
