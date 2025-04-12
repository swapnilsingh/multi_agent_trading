import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
from core.utils.config_loader import load_config
from core.environments.base_trading_environment import BaseTradingEnv
from core.agents.rsi_agent import RSIAgent
from core.agents.macd_agent import MACDAgent
from core.agents.bollinger_agent import BollingerAgent
from core.agents.atr_agent import ATRAgent
from core.agents.sma_agent import SMAAgent

st.set_page_config(layout="wide")

# Load data
data = pd.read_csv('data/live_binance_ohlcv.csv', parse_dates=['timestamp'])
data.rename(columns={'timestamp': 'Date'}, inplace=True)
data = data.sort_values('Date')
dates = data['Date'].values
df = data.drop(columns=['Date']).reset_index(drop=True)

# Setup
def load_agents():
    configs = [
        load_config('configs/strategies/rsi.yaml'),
        load_config('configs/strategies/macd.yaml'),
        load_config('configs/strategies/bollinger.yaml'),
        load_config('configs/strategies/atr.yaml'),
        load_config('configs/strategies/sma_10_30.yaml'),
        load_config('configs/strategies/sma_20_50.yaml')
    ]
    agents = [
        RSIAgent(configs[0], None),
        MACDAgent(configs[1], None),
        BollingerAgent(configs[2], None),
        ATRAgent(configs[3], None),
        SMAAgent(configs[4], None),
        SMAAgent(configs[5], None)
    ]
    model_paths = [
        'models/RSIAgent.pkl',
        'models/MACDAgent.pkl',
        'models/BollingerAgent.pkl',
        'models/ATRAgent.pkl',
        'models/SMA_10_30Agent.pkl',
        'models/SMA_20_50Agent.pkl'
    ]
    for agent, path in zip(agents, model_paths):
        agent.load_model(path)
    return agents

window_size = 50
initial_balance = 1000
agents = load_agents()
env_main = BaseTradingEnv(df, window_size, initial_balance)
envs = [BaseTradingEnv(df, window_size, initial_balance) for _ in range(len(agents))]

# Backtest
price_series = []
equity_series = []
buy_signals = []
sell_signals = []
done = False

env_main.reset()
for env in envs:
    env.reset()

while not done:
    step = env_main.current_step
    current_price = df.loc[step, 'Close']
    price_series.append(current_price)

    actions = [agent.act(env.get_current_state()) for agent, env in zip(agents, envs)]
    final_action = 1 if actions.count(1) > max(actions.count(0), actions.count(-1)) else -1 if actions.count(-1) > actions.count(0) else 0
    reward, done = env_main.execute_action(final_action)
    if final_action == 1:
        buy_signals.append(step)
    elif final_action == -1:
        sell_signals.append(step)

    balance = env_main.balance
    equity = balance
    if env_main.position != 0 and env_main.entry_price is not None:
        equity += current_price - env_main.entry_price if env_main.position == 1 else env_main.entry_price - current_price
    equity_series.append(equity)

    for env in envs:
        env.skip()

if env_main.position != 0:
    final_price = df.loc[min(env_main.current_step, len(df)-1), 'Close']
    env_main.balance += final_price - env_main.entry_price if env_main.position == 1 else env_main.entry_price - final_price
    env_main.position = 0
    env_main.entry_price = None

final_balance = env_main.balance
st.success(f"Final Balance: ${final_balance:.2f}")

# Indicators
close = df['Close']
high = df['High']
low = df['Low']
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
rsi_series = 100 - 100 / (1 + rs)
ema_fast = close.ewm(span=12, adjust=False).mean()
ema_slow = close.ewm(span=26, adjust=False).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=9, adjust=False).mean()
rolling_mean = close.rolling(window=20).mean()
rolling_std = close.rolling(window=20).std()
bb_upper = rolling_mean + 2 * rolling_std
bb_lower = rolling_mean - 2 * rolling_std
bb_mid = rolling_mean
prev_close = close.shift(1)
tr = pd.DataFrame({
    'hl': high - low,
    'hc': (high - prev_close).abs(),
    'lc': (low - prev_close).abs()
})
atr_series = tr.max(axis=1).rolling(window=14).mean()
sma10 = close.rolling(window=10).mean()
sma20 = close.rolling(window=20).mean()
sma30 = close.rolling(window=30).mean()
sma50 = close.rolling(window=50).mean()

# 1. Price only chart
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=dates, y=close, name='Price', line=dict(color='black')))
price_fig.add_trace(go.Scatter(x=dates, y=bb_upper, name='BB Upper', line=dict(dash='dot', color='gray')))
price_fig.add_trace(go.Scatter(x=dates, y=bb_mid, name='BB Mid', line=dict(dash='dot', color='blue')))
price_fig.add_trace(go.Scatter(x=dates, y=bb_lower, name='BB Lower', line=dict(dash='dot', color='gray')))
price_fig.add_trace(go.Scatter(x=dates, y=sma10, name='SMA 10', line=dict(color='magenta')))
price_fig.add_trace(go.Scatter(x=dates, y=sma20, name='SMA 20', line=dict(color='cyan')))
price_fig.add_trace(go.Scatter(x=dates, y=sma30, name='SMA 30', line=dict(dash='dot', color='magenta')))
price_fig.add_trace(go.Scatter(x=dates, y=sma50, name='SMA 50', line=dict(dash='dot', color='cyan')))
price_fig.add_trace(go.Scatter(x=dates[buy_signals], y=close.iloc[buy_signals], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=8)))
price_fig.add_trace(go.Scatter(x=dates[sell_signals], y=close.iloc[sell_signals], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=8)))
price_fig.update_layout(height=600, title="Price & Signals", xaxis_rangeslider_visible=True)

# 2. Indicator chart
indicator_fig = go.Figure()
indicator_fig.add_trace(go.Scatter(x=dates, y=rsi_series, name='RSI', line=dict(color='purple', dash='dot')))
indicator_fig.add_trace(go.Scatter(x=dates, y=macd_line, name='MACD', line=dict(color='brown', dash='dot')))
indicator_fig.add_trace(go.Scatter(x=dates, y=signal_line, name='Signal', line=dict(color='olive', dash='dot')))
indicator_fig.add_trace(go.Scatter(x=dates, y=atr_series, name='ATR', line=dict(color='orange')))
indicator_fig.update_layout(height=300, title="Indicators (RSI, MACD, ATR)", xaxis_rangeslider_visible=False)

# 3. Equity only
equity_fig = go.Figure()
equity_fig.add_trace(go.Scatter(x=dates[window_size:window_size + len(equity_series)], y=equity_series, name='Equity', line=dict(color='orange')))
equity_fig.update_layout(height=300, title="Equity Curve", xaxis_rangeslider_visible=True)

st.plotly_chart(price_fig, use_container_width=True)
st.plotly_chart(indicator_fig, use_container_width=True)
st.plotly_chart(equity_fig, use_container_width=True)
