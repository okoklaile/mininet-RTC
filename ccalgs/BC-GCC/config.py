"""
Configuration for BC-GCC Training
"""

class Config:
    # Dataset paths
    DATA_DIR = '../data'
    DATASETS = ['ghent', 'norway', 'NY', 'opennetlab']
    
    # Data split
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Feature configuration
    WINDOW_SIZE = 10  # Use past 10 time steps (2 seconds at 200ms intervals)
    
    # Core GCC features (enhanced with GCC-inspired features)
    CORE_FEATURES = [
        # Basic features (original 6)
        'delay',              # Current delay (ms)
        'loss_ratio',         # Current packet loss ratio
        'receiving_rate',     # Current receiving rate (bps)
        'prev_bandwidth',     # Previous bandwidth prediction (bps)
        'delay_gradient',     # Change in delay (1st order)
        'throughput_effective', # Effective throughput (recv_rate * (1-loss))
        
        # Delay statistics (GCC core signals)
        'delay_mean',         # Mean delay in window
        'delay_std',          # Delay standard deviation (network stability)
        'delay_min',          # Minimum delay (baseline RTT)
        'queue_delay',        # Queuing delay (delay - delay_min)
        'delay_accel',        # Delay acceleration (2nd order gradient)
        'delay_trend',        # Delay trend (linear regression slope)
        
        # Loss features
        'loss_change',        # Loss ratio change
        
        # Bandwidth utilization
        'bw_utilization',     # recv_rate / prev_bandwidth
        'recv_rate_mean',     # Mean receiving rate
        'recv_rate_std',      # Receiving rate stability
    ]
    
    # Reserved features for future RL (filled with zeros during BC training)
    RESERVED_FEATURES = [
        'reward',             # Reserved for RL reward signal
        'value_estimate',     # Reserved for RL value function
        'action_prob',        # Reserved for RL action probability
        'advantage',          # Reserved for RL advantage
        'custom_1',           # Reserved for future use
        'custom_2',           # Reserved for future use
        'custom_3',           # Reserved for future use
        'custom_4',           # Reserved for future use
    ]
    
    TOTAL_FEATURE_DIM = len(CORE_FEATURES) + len(RESERVED_FEATURES)  # 16 + 8 = 24
    
    # Target
    TARGET = 'bandwidth_prediction'
    
    # Normalization bounds (with clipping for extreme values)
    # Designed to cover 95-99% of normal network conditions
    NORM_STATS = {
        # Basic features
        'delay': {'min': 0, 'max': 10000},           # 10 seconds (covers normal + severe congestion)
        'loss_ratio': {'min': 0, 'max': 1},          # 0-100%
        'receiving_rate': {'min': 0, 'max': 10e6},   # 10 Mbps
        'prev_bandwidth': {'min': 0, 'max': 10e6},   # 10 Mbps
        'delay_gradient': {'min': -2000, 'max': 2000},  # ±2 seconds change
        'throughput_effective': {'min': 0, 'max': 10e6},  # 10 Mbps
        
        # Delay statistics
        'delay_mean': {'min': 0, 'max': 10000},      # Same as delay
        'delay_std': {'min': 0, 'max': 3000},        # Delay variation
        'delay_min': {'min': 0, 'max': 10000},       # Minimum delay
        'queue_delay': {'min': 0, 'max': 10000},     # Queuing delay
        'delay_accel': {'min': -1000, 'max': 1000},  # Acceleration
        'delay_trend': {'min': -500, 'max': 500},    # Trend slope
        
        # Loss features
        'loss_change': {'min': -0.5, 'max': 0.5},    # Loss change
        
        # Bandwidth features
        'bw_utilization': {'min': 0, 'max': 2},      # Can be >1 (overuse)
        'recv_rate_mean': {'min': 0, 'max': 10e6},   # Mean rate
        'recv_rate_std': {'min': 0, 'max': 5e6},     # Rate variation
        
        # Target
        'bandwidth_prediction': {'min': 0, 'max': 10e6},  # 10 Mbps
    }
    
    # Enable clipping for extreme values outside normalization range
    USE_CLIPPING = True
    
    # Model architecture (optimized for RTX 3090)
    LSTM_HIDDEN_SIZE = 256  # Increased from 128 (more capacity)
    LSTM_NUM_LAYERS = 2
    DROPOUT = 0.2
    FC_HIDDEN_SIZES = [128, 64]  # Increased from [64, 32]
    
    # Training hyperparameters (optimized for stability + RTX 3090)
    BATCH_SIZE = 2048  # Reduced from 512 for more stable gradients
    LEARNING_RATE = 2e-4  # Reduced from 1e-3 to prevent NaN (10x safer)
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    USE_AMP = True  # Mixed precision training for RTX 3090
    
    # Sample weighting (to handle data imbalance)
    # Reduced from [1.0, 50.0, 10.0] to prevent gradient explosion
    LOSS_WEIGHT_NO_LOSS = 1.0
    LOSS_WEIGHT_HAS_LOSS = 10.0      # Reduced from 50 (still 10x emphasis)
    LOSS_WEIGHT_HIGH_DELAY = 5.0     # Reduced from 10 (still 5x emphasis)
    
    # Loss thresholds for weighting
    LOSS_THRESHOLD = 0.01   # 1% packet loss
    HIGH_DELAY_THRESHOLD = 300  # ms
    
    # Oversampling strategy (reduced for faster training and better GPU utilization)
    OVERSAMPLE_FILES = [
        'opennetlab/rates_delay_loss_gcc_4G_3mbps.pickle',  # 10x (reduced from 50x)
        'NY/rates_delay_loss_gcc_BusBrooklyn_bus57New.pickle',  # 5x (reduced from 30x)
        'NY/rates_delay_loss_gcc_Ferry_Ferry4.pickle',  # 5x (reduced from 30x)
        'NY/rates_delay_loss_gcc_7Train_7trainNew.pickle',  # 5x (reduced from 20x)
    ]
    OVERSAMPLE_MULTIPLIERS = [10, 5, 5, 5]
    
    # Optimizer
    OPTIMIZER = 'adam'
    SCHEDULER = 'plateau'
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Checkpointing
    CHECKPOINT_DIR = 'checkpoints'
    SAVE_BEST_ONLY = True
    
    # Logging
    LOG_DIR = 'logs'
    LOG_INTERVAL = 10  # Log every N batches
    VAL_INTERVAL = 1   # Validate every N epochs
    
    # Device
    DEVICE = 'cuda'  # or 'cpu'
    
    # Random seed
    SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=" * 80)
        print("BC-GCC Training Configuration")
        print("=" * 80)
        print(f"Core Features ({len(cls.CORE_FEATURES)}): {cls.CORE_FEATURES}")
        print(f"Reserved Features ({len(cls.RESERVED_FEATURES)}): {cls.RESERVED_FEATURES}")
        print(f"Total Feature Dimension: {cls.TOTAL_FEATURE_DIM}")
        print(f"Window Size: {cls.WINDOW_SIZE}")
        print(f"LSTM: {cls.LSTM_NUM_LAYERS} layers x {cls.LSTM_HIDDEN_SIZE} hidden")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Sample Weights: No loss={cls.LOSS_WEIGHT_NO_LOSS}, Has loss={cls.LOSS_WEIGHT_HAS_LOSS}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 80)
