import pandas as pd
import numpy as np
import os

class MacroFeatureEngineer:
    """
    Generates features based on proximity and severity of macro events.
    """
    def __init__(self, event_path='events/macro_event_calendar.parquet'):
        if os.path.exists(event_path):
            self.events_df = pd.read_parquet(event_path)
        else:
            self.events_df = pd.DataFrame()

    def add_macro_features(self, df):
        if self.events_df.empty:
            return df
            
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Calculate days since last event and days until next event
        def get_proximity(row_ts):
            diffs = (row_ts - self.events_df['timestamp']).dt.days
            past_events = diffs[diffs >= 0]
            future_events = diffs[diffs < 0]
            
            days_since = past_events.min() if not past_events.empty else 365
            days_until = abs(future_events.max()) if not future_events.empty else 365
            
            # Severity of the closest event
            if not past_events.empty:
                last_idx = past_events.idxmin()
                last_severity = self.events_df.loc[last_idx, 'severity']
            else:
                last_severity = 0
                
            return pd.Series([days_since, days_until, last_severity])

        # This can be slow on large datasets, optimize with merge if possible
        # For research, we'll use a simpler approach: distance to any event
        df[['days_since_macro', 'days_until_macro', 'last_macro_severity']] = df['timestamp'].apply(get_proximity)
        
        # Risk metric: higher if close to an event (past or future)
        df['macro_risk_factor'] = (1.0 / (df['days_since_macro'] + 1)) + (1.0 / (df['days_until_macro'] + 1))
        df['macro_risk_factor'] *= df['last_macro_severity'].replace(0, 0.1)
        
        return df

if __name__ == "__main__":
    engineer = MacroFeatureEngineer()
    # Dummy test
    test_df = pd.DataFrame({'timestamp': pd.date_range('2022-11-01', '2022-11-15', freq='D')})
    res = engineer.add_macro_features(test_df)
    print(res[['timestamp', 'days_since_macro', 'days_until_macro', 'macro_risk_factor']])
