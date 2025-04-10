import yaml
import os

def load_config(filepath: str) -> dict:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
