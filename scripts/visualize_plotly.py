# visualize_plotly.py
import plotly.graph_objs as go
import pandas_ta as ta
from plotly.subplots import make_subplots

def plot_combined_plotly(data, trades, capital_start):
    macd_df = ta.macd(data['close'])
    rsi = ta.rsi(data['close'], length=14)
    bb = ta.bbands(data['close'], length=20)
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)

    # Capital tracking
    capital = capital_start
    capital_history = [capital]
    for action, price in trades:
        if action == 'buy' or action == 1:
            capital -= price
        elif action == 'sell' or action == -1:
            capital += price
        capital_history.append(capital)

    times = data.index
    price = data['close']

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.3, 0.15, 0.2, 0.15, 0.2],
                        subplot_titles=("Price + Trades + Bollinger Bands",
                                        "RSI", "MACD", "ATR", "Capital Evolution"))

    # --- Price + Trades + BB ---
    fig.add_trace(go.Scatter(x=times, y=price, mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
    if bb is not None:
        fig.add_trace(go.Scatter(x=times, y=bb['BBL_20_2.0'], name='BB Lower', line=dict(color='green', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bb['BBU_20_2.0'], name='BB Upper', line=dict(color='red', dash='dot')), row=1, col=1)

    # Action label mapping
    action_labels = {-1: "Sell", 0: "Hold", 1: "Buy", "sell": "Sell", "buy": "Buy", "hold": "Hold"}
    for i, (action, price_val) in enumerate(trades):
        if i >= len(times):
            continue
        label = action_labels.get(action, "Unknown")
        color = 'green' if label == "Buy" else 'red' if label == "Sell" else 'gray'
        symbol = 'triangle-up' if label == "Buy" else 'triangle-down' if label == "Sell" else 'circle'
        fig.add_trace(go.Scatter(
            x=[times[i]], y=[price_val], mode='markers',
            marker=dict(symbol=symbol, color=color, size=10),
            name=f'{label} Signal', showlegend=False
        ), row=1, col=1)

    # --- RSI ---
    fig.add_trace(go.Scatter(x=times, y=rsi, name='RSI (14)', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=[70]*len(times), name='Overbought (70)', line=dict(color='red', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=[30]*len(times), name='Oversold (30)', line=dict(color='green', dash='dot')), row=2, col=1)

    # --- MACD ---
    if macd_df is not None:
        fig.add_trace(go.Scatter(x=times, y=macd_df['MACD_12_26_9'], name='MACD Line', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=times, y=macd_df['MACDs_12_26_9'], name='Signal Line', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Bar(x=times, y=macd_df['MACDh_12_26_9'], name='Histogram', marker_color='gray', opacity=0.3), row=3, col=1)

    # --- ATR ---
    fig.add_trace(go.Scatter(x=times, y=atr, name='ATR (14)', line=dict(color='brown')), row=4, col=1)

    # --- Capital Evolution ---
    fig.add_trace(go.Scatter(x=list(range(len(capital_history))), y=capital_history, name='Capital', line=dict(color='orange')), row=5, col=1)

    fig.update_layout(height=1000, title="Multi-Agent Trading Strategy Visualization (Plotly)", showlegend=True)
    fig.show()
