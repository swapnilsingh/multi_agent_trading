import streamlit as st
import pandas as pd
import asyncio
import time
import sys
import os
import queue
from datetime import datetime
import plotly.graph_objs as go
from collections import deque

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.streaming.realtime_ws_client import BinanceLiveStreamAggTrade
from core.agents import RSIAgent, MACDAgent, BollingerAgent, AggregatorAgent, ATRAgent, SMAAgent
from core.environments import RSITRadingEnv, MACDTradingEnv, BollingerTradingEnv, ATRTradingEnv, SMATradingEnv, GenericTradingEnv as TradingEnvironment
from core.model_management.model_manager import ModelManager

# Queues and global state
buffer_queue = queue.Queue()
trade_queue = queue.Queue()
buffer_df = pd.DataFrame()
trade_log = []
capital = 1000
position = 0
last_action = 0

st.set_page_config(page_title="Live Trading", layout="wide")

st.title("📈 Live Crypto Trading Dashboard")
price_plot = st.empty()
equity_plot = st.empty()
info_placeholder = st.empty()

async def main():
    stream = BinanceLiveStreamAggTrade(symbol='BTCUSDT')
    strategy_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'strategies')

    import yaml
    load_yaml = lambda name: yaml.safe_load(open(os.path.join(strategy_dir, f'{name}.yaml')))

    model_manager = ModelManager()
    agents = {
        'rsi': RSIAgent(load_yaml('rsi'), model_manager),
        'macd': MACDAgent(load_yaml('macd'), model_manager),
        'bollinger': BollingerAgent(load_yaml('bollinger'), model_manager),
        'atr': ATRAgent(load_yaml('atr'), model_manager),
        'sma': SMAAgent(load_yaml('sma_10_30'), model_manager),
    }
    aggregator = AggregatorAgent(list(agents.values()), model_manager)
    env = TradingEnvironment(initial_capital=1000)

    await asyncio.gather(stream.start_stream(), evaluate(stream, agents, aggregator, env))

async def evaluate(stream, agents, aggregator, env):
    global buffer_df, trade_log, last_action, capital, position

    while True:
        df = stream.get_latest_dataframe()
        if df is None or df.empty:
            await asyncio.sleep(0.5)
            continue

        df.columns = [c.lower() for c in df.columns]
        latest = df.iloc[-1:]
        buffer_queue.put(latest)

        price = latest['price'].values[0]
        timestamp = latest['timestamp'].values[0]

        actions = {}
        for name, agent in agents.items():
            try:
                env_cls = globals()[f'{name.upper()}TradingEnv']
                actions[name] = agent.act(df, env_cls(df))
            except:
                actions[name] = 0

        action = aggregator.vote_from_actions(actions)
        env.step(action, price)

        trade_queue.put({
            'timestamp': int(timestamp),
            'price': price,
            'action': 'BUY' if action == 1 else 'SELL' if action == -1 else 'HOLD'
        })

        capital, position, last_action = env.capital, env.position, action
        await asyncio.sleep(0.5)

def update_chart():
    global buffer_df
    updated = False

    while not buffer_queue.empty():
        new_data = buffer_queue.get()
        if buffer_df.empty:
            buffer_df = new_data
        else:
            buffer_df = pd.concat([buffer_df, new_data], ignore_index=True)
        updated = True

    while not trade_queue.empty():
        trade_log.append(trade_queue.get())

    if not updated:
        return

    df = buffer_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    trace_price = go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name='Price', line=dict(color='black'))
    trace_rsi = go.Scatter(x=df['timestamp'], y=df.get('rsi', pd.Series([None]*len(df))), name='RSI', yaxis='y2')
    trace_macd = go.Scatter(x=df['timestamp'], y=df.get('macd', pd.Series([None]*len(df))), name='MACD', yaxis='y2')
    trace_bb = go.Scatter(x=df['timestamp'], y=df.get('bb_upper', pd.Series([None]*len(df))), name='BB Upper', yaxis='y2')

    markers = []
    seen_labels = set()
    for t in trade_log[-100:]:
        t_time = pd.to_datetime(t['timestamp'], unit='ms')
        color = 'green' if t['action'] == 'BUY' else 'red' if t['action'] == 'SELL' else 'blue'
        symbol = 'triangle-up' if t['action'] == 'BUY' else 'triangle-down' if t['action'] == 'SELL' else 'circle'
        label = t['action']
        showlegend = label not in seen_labels
        markers.append(go.Scatter(
            x=[t_time],
            y=[t['price']],
            mode='markers',
            marker=dict(color=color, symbol=symbol, size=10),
            name=label,
            showlegend=showlegend
        ))
        seen_labels.add(label)

    layout_price = go.Layout(
        title='Live Price and Indicators',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Indicators', overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(traceorder="normal")
    )

    fig_price = go.Figure(data=[trace_price, trace_rsi, trace_macd, trace_bb] + markers, layout=layout_price)
    price_plot.plotly_chart(fig_price, use_container_width=True)

    equity = [1000]
    for trade in trade_log:
        if trade['action'] == 'BUY':
            equity.append(equity[-1] - trade['price'])
        elif trade['action'] == 'SELL':
            equity.append(equity[-1] + trade['price'])
        else:
            equity.append(equity[-1])

    times = [pd.to_datetime(t['timestamp'], unit='ms') for t in trade_log]
    equity_fig = go.Figure(data=[go.Scatter(x=times, y=equity[1:], mode='lines', name='Equity Curve')])
    equity_fig.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Capital', height=300)
    equity_plot.plotly_chart(equity_fig, use_container_width=True)

    info_placeholder.info(f"💰 Capital: {capital:.2f} | 📦 Position: {position} | 🔄 Last Action: {last_action}")

if __name__ == "__main__":
    import threading
    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()
    while True:
        update_chart()
        time.sleep(1)
