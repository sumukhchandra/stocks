import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

try:
    from pymongo import MongoClient
except ImportError:  # Optional dependency for local/offline use.
    MongoClient = None


class MongoStore:
    def __init__(self, database="stock_dashboard"):
        load_dotenv()
        self.uri = os.getenv("MONGODB_URI")
        self.client = None
        self.db = None
        if self.uri and MongoClient is not None:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[database]

    @property
    def enabled(self):
        return self.db is not None

    def ping(self):
        if not self.enabled:
            return False
        self.client.admin.command("ping")
        return True

    def insert_prices(self, session_id, rows):
        if not self.enabled or rows.empty:
            return 0

        records = rows.copy()
        records["session_id"] = session_id
        records["created_at"] = datetime.utcnow()
        docs = records.to_dict("records")
        self.db.live_prices.insert_many(docs, ordered=False)
        return len(docs)

    def load_session_prices(self, session_id, limit=500):
        if not self.enabled:
            return pd.DataFrame()

        cursor = (
            self.db.live_prices.find(
                {"session_id": session_id},
                {"_id": 0},
            )
            .sort("timestamp", -1)
            .limit(limit)
        )
        docs = list(cursor)
        if not docs:
            return pd.DataFrame()

        df = pd.DataFrame(docs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

