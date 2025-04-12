import asyncio
from core.streaming.realtime_ws_client import BinanceLiveStream
from core.agents.rsi_agent import RSIAgent
from core.agents.macd_agent import MACDAgent
from core.agents.bollinger_agent import BollingerAgent
from core.agents.aggregator_agent import AggregatorAgent
from core.environments.multi_agent_trading_environment import TradingEnvironment

from datetime import datetime

async def live_trading():
    stream = BinanceLiveStream(symbol='BTCUSDT', interval='1m')
    await asyncio.sleep(1)

    rsi_agent = RSIAgent()
    macd_agent = MACDAgent()
    boll_agent = BollingerAgent()
    aggregator = AggregatorAgent([rsi_agent, macd_agent, boll_agent])
    env = TradingEnvironment(initial_capital=1000)

    async def handle_data():
        while True:
            df = stream.get_latest_dataframe()

            if len(df) >= 30:
                current_row = df.iloc[-1:]
                indicators = {
                    'rsi': rsi_agent.compute(df),
                    'macd': macd_agent.compute(df),
                    'bollinger': boll_agent.compute(df)
                }

                action = aggregator.vote(df)
                price = current_row['close'].values[0]
                env.step(action, price)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Action: {action}, Price: {price:.2f}, Capital: {env.capital}, Position: {env.position}")

            await asyncio.sleep(1)

    await asyncio.gather(stream.start_stream(), handle_data())

if __name__ == "__main__":
    asyncio.run(live_trading())
