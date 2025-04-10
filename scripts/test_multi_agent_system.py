# scripts/test_multi_agent_system.py
import sys
import os

# Add the parent directory of 'core' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from core.model_management.model_manager import ModelManager
from core.agents.rsi_agent import RSIAgent
from core.agents.macd_agent import MACDAgent
from core.agents.bollinger_agent import BollingerAgent
from core.agents.atr_agent import ATRAgent
from core.agents.aggregator_agent import AggregatorAgent
from core.environments.multi_agent_trading_environment import MultiAgentTradingEnvironment

def test_multi_agent_system():
    # Load market data (example: BTC/USD)
    data = pd.read_csv("data/processed/btc_ohlcv.csv")

    # Create the model manager
    model_manager = ModelManager()

    # Create agent configurations
    rsi_config = {'rsi_period': 14}
    macd_config = {'short_period': 12, 'long_period': 26, 'signal_period': 9}
    bollinger_config = {'window': 20, 'num_std_dev': 2}
    atr_config = {'window': 14, 'atr_threshold': 1.5}
    
    # Initialize agents with model manager
    rsi_agent = RSIAgent(rsi_config, model_manager=model_manager)
    macd_agent = MACDAgent(macd_config, model_manager=model_manager)
    bollinger_agent = BollingerAgent(bollinger_config, model_manager=model_manager)
    atr_agent = ATRAgent(atr_config, model_manager=model_manager)

    # Initialize the aggregator agent
    agents = [rsi_agent, macd_agent, bollinger_agent, atr_agent]
    aggregator_agent = AggregatorAgent(agents)

    # Create the Multi-Agent Trading Environment
    env = MultiAgentTradingEnvironment(market_data=data, agents=agents)

    # Run the simulation
    state = env.reset()
    done = False
    actions = []
    
    while not done:
        # Each agent decides on an action
        agent_actions = [agent.act(state) for agent in agents]
        
        # Aggregator makes the final decision
        final_action = aggregator_agent.make_final_decision(agent_actions)
        
        # Log the action taken
        actions.append(final_action)
        
        # Take a step in the environment
        state, reward, done = env.step(agent_actions)
    
    print("Final Balance:", env.balance)
    print("Trades Executed:", env.trades)
    print("Actions Taken:", actions)

if __name__ == "__main__":
    test_multi_agent_system()
