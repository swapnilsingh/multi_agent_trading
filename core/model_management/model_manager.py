# core/model_management/model_manager.py
import os
import pickle

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(self, model, agent_name, version="v1"):
        model_path = os.path.join(self.models_dir, f"{agent_name}_{version}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

    def load_model(self, agent_name, version="v1"):
        model_path = os.path.join(self.models_dir, f"{agent_name}_{version}.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return model
        else:
            print(f"Model for {agent_name} version {version} not found.")
            return None
