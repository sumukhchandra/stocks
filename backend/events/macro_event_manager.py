import pandas as pd
import os

def create_macro_event_database():
    """
    Creates a database of major crypto and macro events for regime analysis.
    """
    events = [
        # Systemic Crashes / Liquidation Events
        {'timestamp': '2020-03-12', 'event_name': 'COVID_Crash', 'category': 'systemic_crash', 'severity': 1.0},
        {'timestamp': '2021-05-19', 'event_name': 'China_Mining_Ban', 'category': 'liquidation_event', 'severity': 0.8},
        {'timestamp': '2022-05-09', 'event_name': 'Luna_Collapse', 'category': 'systemic_crash', 'severity': 0.9},
        {'timestamp': '2022-11-08', 'event_name': 'FTX_Collapse', 'category': 'exchange_collapse', 'severity': 1.0},
        {'timestamp': '2023-03-10', 'event_name': 'SVB_Bank_Run', 'category': 'liquidity_stress', 'severity': 0.7},
        {'timestamp': '2024-01-10', 'event_name': 'BTC_ETF_Approval', 'category': 'institutional_inflow', 'severity': 0.5},
        {'timestamp': '2024-04-13', 'event_name': 'Iran_Israel_Conflict', 'category': 'geopolitical_risk', 'severity': 0.6},
        
        # Recurring Macro (High Volatility)
        {'timestamp': '2024-05-15', 'event_name': 'CPI_Release', 'category': 'macro_volatility', 'severity': 0.3},
        {'timestamp': '2024-06-12', 'event_name': 'FOMC_Meeting', 'category': 'macro_volatility', 'severity': 0.4},
    ]
    
    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    output_path = 'events/macro_event_calendar.parquet'
    df.to_parquet(output_path, engine='pyarrow')
    print(f"Macro Event Database created at {output_path}")
    return df

if __name__ == "__main__":
    create_macro_event_database()
