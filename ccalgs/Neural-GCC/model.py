"""
LSTM-based model for GCC Behavior Cloning
"""
import torch
import torch.nn as nn
from typing import Tuple

from config import Config


class GCCBC_LSTM(nn.Module):
    """
    LSTM model for GCC Behavior Cloning
    
    Architecture:
        Input: [batch, seq_len, feature_dim] where feature_dim includes reserved features
        LSTM: Multi-layer LSTM for temporal modeling
        FC: Fully connected layers for bandwidth prediction
        Output: [batch, 1] bandwidth prediction
    """
    
    def __init__(self, config: Config):
        super(GCCBC_LSTM, self).__init__()
        
        self.config = config
        self.input_dim = config.TOTAL_FEATURE_DIM
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.num_layers = config.LSTM_NUM_LAYERS
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.DROPOUT if self.num_layers > 1 else 0,
        )
        
        # Fully connected layers
        fc_layers = []
        prev_size = self.hidden_size
        
        for fc_size in config.FC_HIDDEN_SIZES:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
            ])
            prev_size = fc_size
        
        # Output layer (bandwidth prediction)
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.ReLU())  # Bandwidth must be non-negative
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            x: [batch, seq_len, feature_dim]
            hidden: Optional hidden state for LSTM
        
        Returns:
            output: [batch, 1] bandwidth prediction
            hidden: LSTM hidden state
        """
        batch_size = x.size(0)
        
        # LSTM forward
        if hidden is None:
            # Use last time step output
            lstm_out, hidden = self.lstm(x)  # lstm_out: [batch, seq, hidden]
            last_output = lstm_out[:, -1, :]  # [batch, hidden]
        else:
            # Use provided hidden state (for online inference)
            lstm_out, hidden = self.lstm(x, hidden)
            last_output = lstm_out[:, -1, :]
        
        # FC forward
        output = self.fc(last_output)  # [batch, 1]
        
        return output, hidden
    
    def predict(self, x: torch.Tensor, hidden=None):
        """
        Prediction mode (no gradient)
        
        Args:
            x: [batch, seq_len, feature_dim] or [seq_len, feature_dim]
        
        Returns:
            bandwidth prediction in bps
        """
        self.eval()
        with torch.no_grad():
            # Handle single sample
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, seq_len, feature_dim]
            
            output, hidden = self.forward(x, hidden)
            
            # Denormalize if needed
            # (Assuming output is already in bps scale)
            
            return output.squeeze(), hidden
    
    def get_core_feature_mask(self):
        """
        Get mask for core features (used for gradient analysis)
        
        Returns:
            mask: [feature_dim] where 1 = core feature, 0 = reserved
        """
        mask = torch.zeros(self.input_dim)
        mask[:len(self.config.CORE_FEATURES)] = 1
        return mask
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WeightedMSELoss(nn.Module):
    """MSE Loss with sample weighting"""
    
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, pred, target, weight):
        """
        Args:
            pred: [batch, 1]
            target: [batch, 1]
            weight: [batch, 1]
        """
        mse = (pred - target) ** 2
        weighted_mse = mse * weight
        return weighted_mse.mean()


class WeightedMAPELoss(nn.Module):
    """Mean Absolute Percentage Error with weighting"""
    
    def __init__(self, epsilon=1.0):
        super(WeightedMAPELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target, weight):
        """
        Args:
            pred: [batch, 1]
            target: [batch, 1]
            weight: [batch, 1]
        """
        # Avoid division by zero
        mape = torch.abs((target - pred) / (target + self.epsilon))
        weighted_mape = mape * weight
        return weighted_mape.mean()


class CombinedLoss(nn.Module):
    """Combined MSE + MAPE loss"""
    
    def __init__(self, mse_weight=0.7, mape_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mape_weight = mape_weight
        self.mse_loss = WeightedMSELoss()
        self.mape_loss = WeightedMAPELoss()
    
    def forward(self, pred, target, weight):
        mse = self.mse_loss(pred, target, weight)
        mape = self.mape_loss(pred, target, weight)
        return self.mse_weight * mse + self.mape_weight * mape


if __name__ == '__main__':
    # Test model
    config = Config()
    config.print_config()
    
    model = GCCBC_LSTM(config)
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = config.WINDOW_SIZE
    feature_dim = config.TOTAL_FEATURE_DIM
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    print(f"\nInput shape: {x.shape}")
    
    output, hidden = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shapes: h={hidden[0].shape}, c={hidden[1].shape}")
    
    # Test loss
    target = torch.randn(batch_size, 1).abs() * 1e6  # Bandwidth in bps
    weight = torch.ones(batch_size, 1)
    
    criterion = CombinedLoss()
    loss = criterion(output, target, weight)
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test prediction
    single_x = torch.randn(seq_len, feature_dim)
    pred, _ = model.predict(single_x)
    print(f"\nSingle prediction: {pred.item():.2f} bps")
