
import torch
from torchviz import make_dot
import sys
import os

# Add path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GCCBC_LSTM, Critic
from config import Config

def visualize_model():
    config = Config()
    model = GCCBC_LSTM(config)
    
    # Create dummy input
    batch_size = 1
    seq_len = config.WINDOW_SIZE
    feature_dim = config.TOTAL_FEATURE_DIM
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # Forward pass
    y, _ = model(x)
    
    # Generate graph
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render("neural_gcc_architecture")
    print("Graph saved to neural_gcc_architecture.png")

if __name__ == "__main__":
    visualize_model()
