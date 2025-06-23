"""
Constants used throughout the federated learning framework.
"""

# Network defaults
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT = 300
MAX_CLIENTS = 100

# Training defaults
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0001

# Privacy defaults
DEFAULT_EPSILON = 0.5
DEFAULT_DELTA = 1e-5
DEFAULT_NOISE_MULTIPLIER = 1.1
DEFAULT_MAX_GRAD_NORM = 1.0

# Model defaults
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_CLASSES = 10
DEFAULT_DROPOUT = 0.5

# Data processing
SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.json', '.csv'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
SUPPORTED_TABULAR_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet', '.json'}
DEFAULT_IMAGE_SIZE = (224, 224)

# Error messages
ERROR_MESSAGES = {
    'config_not_found': "Configuration file not found: {}",
    'invalid_yaml': "Invalid YAML configuration: {}",
    'connection_failed': "Failed to connect to server",
    'invalid_data': "Invalid data at {}",
} 