import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceStream:
    def __init__(self, symbols, stream_types=['aggTrade', 'depth']):
        self.symbols = [s.lower() for s in symbols]
        self.stream_types = stream_types
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.callbacks = []

    def register_callback(self, callback):
        self.callbacks.append(callback)

    async def connect(self):
        streams = []
        for symbol in self.symbols:
            for stype in self.stream_types:
                if stype == 'depth':
                    streams.append(f"{symbol}@depth@100ms") # 100ms updates
                else:
                    streams.append(f"{symbol}@{stype}")
        
        url = f"{self.base_url}/" + "/".join(streams)
        
        while True:
            try:
                logger.info(f"Connecting to {url}")
                async with websockets.connect(url) as websocket:
                    logger.info("Connected successfully")
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        for callback in self.callbacks:
                            await callback(data)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket disconnected. Reconnecting in 5 seconds... {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in WebSocket: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    async def sample_callback(data):
        print(data)
    
    stream = BinanceStream(['btcusdt', 'ethusdt'])
    stream.register_callback(sample_callback)
    asyncio.run(stream.connect())
