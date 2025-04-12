# scripts/test_stream.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from core.streaming.realtime_ws_client import BinanceLiveStream

async def main():
    stream = BinanceLiveStream(symbol='BTCUSDT')
    await asyncio.gather(stream.start_stream())

if __name__ == '__main__':
    asyncio.run(main())