import yaml
import numpy as np

def load_config():
    """
    Load the config from the config.yaml file
    """
    with open('models/FineTunedDistilBERT/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

