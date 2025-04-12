# core/streaming/realtime_ws_client.py
import asyncio
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from collections import deque

class BinanceLiveStreamAggTrade:
    def __init__(self, symbol='btcusdt', buffer_size=500):
        self.symbol = symbol.lower()
        self.buffer = deque(maxlen=buffer_size)
        self.client = None
        self.bsm = None
        self.socket = None

    async def start_stream(self):
        try:
            self.client = await AsyncClient.create()
            self.bsm = BinanceSocketManager(self.client)
            self.socket = self.bsm.aggtrade_socket(symbol=self.symbol)

            async with self.socket as stream:
                print(f"✅ WebSocket streaming started for {self.symbol} @ aggTrade")
                while True:
                    res = await stream.recv()
                    row = {
                        'timestamp': int(res['T']),
                        'price': float(res['p']),
                        'volume': float(res['q']),
                    }
                    self.buffer.append(row)
                    print(f"[STREAM] Tick: {row}")
        except Exception as e:
            print(f"❌ WebSocket stream failed: {e}")

    def get_latest_dataframe(self):
        if self.buffer:
            df = pd.DataFrame(self.buffer)
            df.columns = [col.lower() for col in df.columns]
            print(f"[INFO] Retrieved dataframe: {df.shape}")
            return df
        return pd.DataFrame()
