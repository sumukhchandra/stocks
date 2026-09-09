import asyncio
import pandas as pd
import os
from datetime import datetime

class OrderBookCollector:
    def __init__(self, output_dir="data/orderbook", buffer_limit=1000):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.buffer = []
        self.buffer_limit = buffer_limit
        
    async def process_message(self, data):
        # Determine message type
        if 'e' in data:
            event_type = data['e']
            if event_type == 'depthUpdate':
                await self.handle_depth(data)
            elif event_type == 'aggTrade':
                await self.handle_trade(data)
                
    async def handle_depth(self, data):
        # Extract best bid/ask
        # Binance format: 'b' for bids, 'a' for asks
        bids = data.get('b', [])
        asks = data.get('a', [])
        
        best_bid = float(bids[0][0]) if bids else None
        best_bid_qty = float(bids[0][1]) if bids else None
        best_ask = float(asks[0][0]) if asks else None
        best_ask_qty = float(asks[0][1]) if asks else None
        
        # Calculate imbalance
        imbalance = 0
        if best_bid_qty and best_ask_qty:
            imbalance = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)
            
        record = {
            'timestamp': pd.to_datetime(data['E'], unit='ms'),
            'symbol': data['s'],
            'type': 'depth',
            'best_bid': best_bid,
            'best_bid_qty': best_bid_qty,
            'best_ask': best_ask,
            'best_ask_qty': best_ask_qty,
            'imbalance': imbalance,
            'spread': best_ask - best_bid if best_ask and best_bid else None
        }
        
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_limit:
            self.save_buffer()
            
    async def handle_trade(self, data):
        record = {
            'timestamp': pd.to_datetime(data['E'], unit='ms'),
            'symbol': data['s'],
            'type': 'trade',
            'price': float(data['p']),
            'qty': float(data['q']),
            'is_buyer_maker': data['m'] # True if maker was buyer (sell order hit the book)
        }
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_limit:
            self.save_buffer()
            
    def save_buffer(self):
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/raw_data_{timestamp_str}.parquet"
        
        # Save to parquet using pyarrow engine
        df.to_parquet(filename, engine='pyarrow')
        print(f"Saved {len(df)} records to {filename}")
        
        # Clear buffer
        self.buffer = []

if __name__ == "__main__":
    from binance_stream import BinanceStream
    
    collector = OrderBookCollector()
    stream = BinanceStream(['btcusdt', 'ethusdt'])
    
    # Register collector callback
    stream.register_callback(collector.process_message)
    
    # Run the event loop
    try:
        asyncio.run(stream.connect())
    except KeyboardInterrupt:
        # Save any remaining data on exit
        collector.save_buffer()
        print("Shutdown complete.")
