"""
Constants used throughout the federated learning framework.

This module contains all configuration constants for the federated learning system,
including network settings, training parameters, privacy controls, and error handling.
"""

import os
from typing import Dict, Set, Tuple

# Network defaults
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT = 300
MAX_CLIENTS = 100
WEBSOCKET_PING_INTERVAL = 30
WEBSOCKET_PING_TIMEOUT = 10
MAX_MESSAGE_SIZE = 1024 * 1024 * 100  # 100MB
CONNECTION_RETRY_ATTEMPTS = 3
CONNECTION_RETRY_DELAY = 5

# Security and authentication
DEFAULT_SECRET_KEY = os.getenv("FL_SECRET_KEY", "your-secret-key-change-in-production")
JWT_EXPIRATION_HOURS = 24
PASSWORD_MIN_LENGTH = 8
MAX_LOGIN_ATTEMPTS = 5
SESSION_TIMEOUT_MINUTES = 60

# Training defaults
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.001

# Federated learning specific
DEFAULT_FEDERATION_ROUNDS = 100
DEFAULT_CLIENT_FRACTION = 0.1
DEFAULT_MIN_CLIENTS = 2
DEFAULT_MAX_CLIENTS_PER_ROUND = 10
DEFAULT_AGGREGATION_METHOD = "fedavg"  # fedavg, fedprox, fednova
DEFAULT_FEDPROX_MU = 0.001
DEFAULT_FEDNOVA_MOMENTUM = 0.9

# Privacy defaults
DEFAULT_EPSILON = 0.5
DEFAULT_DELTA = 1e-5
DEFAULT_NOISE_MULTIPLIER = 1.1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_CLIP_NORM = 1.0
DEFAULT_DP_ALGORITHM = "dp-sgd"  # dp-sgd, dp-adam, dp-adagrad

# Model defaults
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_CLASSES = 10
DEFAULT_DROPOUT = 0.5
DEFAULT_ACTIVATION = "relu"  # relu, tanh, sigmoid, leaky_relu
DEFAULT_OPTIMIZER = "adam"  # sgd, adam, adagrad, rmsprop
DEFAULT_LOSS_FUNCTION = "cross_entropy"  # cross_entropy, mse, mae

# Model architectures
SUPPORTED_MODEL_ARCHITECTURES = {
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network", 
    "lstm": "Long Short-Term Memory",
    "gru": "Gated Recurrent Unit",
    "transformer": "Transformer",
    "mlp": "Multi-Layer Perceptron",
    "resnet": "Residual Network",
    "vgg": "VGG Network",
    "alexnet": "AlexNet",
    "custom": "Custom Architecture"
}

# Data processing
SUPPORTED_TEXT_EXTENSIONS: Set[str] = {'.txt', '.md', '.json', '.csv', '.xml', '.html', '.pdf'}
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.svg'}
SUPPORTED_TABULAR_EXTENSIONS: Set[str] = {'.csv', '.xlsx', '.xls', '.parquet', '.json', '.h5', '.hdf5'}
SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_TEXT_MAX_LENGTH = 512
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_BATCH_PREFETCH = 2

# Data validation
MIN_SAMPLES_PER_CLIENT = 10
MAX_SAMPLES_PER_CLIENT = 100000
REQUIRED_COLUMNS = ["id", "label"]
OPTIONAL_COLUMNS = ["features", "metadata", "timestamp"]

# Performance and optimization
DEFAULT_NUM_WORKERS = 4
DEFAULT_PIN_MEMORY = True
DEFAULT_MIXED_PRECISION = True
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_GRADIENT_CLIPPING = True
DEFAULT_WEIGHT_INITIALIZATION = "xavier"  # xavier, he, normal, uniform

# Monitoring and logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = "federated_learning.log"
DEFAULT_METRICS_INTERVAL = 10  # seconds
DEFAULT_CHECKPOINT_INTERVAL = 5  # rounds
DEFAULT_EVALUATION_INTERVAL = 1  # rounds

# Storage and caching
DEFAULT_CACHE_SIZE = 1000
DEFAULT_MODEL_CACHE_TTL = 3600  # seconds
DEFAULT_DATA_CACHE_TTL = 1800  # seconds
DEFAULT_TEMP_DIR = "/tmp/federated_learning"
DEFAULT_MODEL_DIR = "./models"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    'config_not_found': "Configuration file not found: {}",
    'invalid_yaml': "Invalid YAML configuration: {}",
    'connection_failed': "Failed to connect to server: {}",
    'invalid_data': "Invalid data at {}: {}",
    'model_not_found': "Model not found: {}",
    'insufficient_data': "Insufficient data for training: minimum {} samples required",
    'privacy_violation': "Privacy budget exceeded: epsilon={}, delta={}",
    'aggregation_failed': "Model aggregation failed: {}",
    'client_timeout': "Client timeout after {} seconds",
    'server_unavailable': "Server unavailable: {}",
    'invalid_model_architecture': "Unsupported model architecture: {}",
    'data_validation_failed': "Data validation failed: {}",
    'authentication_failed': "Authentication failed: {}",
    'authorization_denied': "Authorization denied for resource: {}",
    'resource_not_found': "Resource not found: {}",
    'rate_limit_exceeded': "Rate limit exceeded: {}",
    'internal_server_error': "Internal server error: {}",
    'network_error': "Network error: {}",
    'serialization_error': "Serialization error: {}",
    'deserialization_error': "Deserialization error: {}",
}

# Warning messages
WARNING_MESSAGES: Dict[str, str] = {
    'low_data_quality': "Low data quality detected: {}",
    'privacy_budget_low': "Privacy budget running low: {}% remaining",
    'model_convergence_slow': "Model convergence is slow: consider adjusting learning rate",
    'client_dropout': "Client dropout detected: {} clients disconnected",
    'memory_usage_high': "High memory usage detected: {}%",
    'gpu_memory_high': "High GPU memory usage: {}%",
    'training_stalled': "Training appears to be stalled: loss not improving",
    'data_imbalance': "Data imbalance detected: class distribution uneven",
    'gradient_explosion': "Gradient explosion detected: consider gradient clipping",
    'gradient_vanishing': "Gradient vanishing detected: consider different activation function",
}

# Success messages
SUCCESS_MESSAGES: Dict[str, str] = {
    'model_saved': "Model successfully saved to: {}",
    'training_completed': "Training completed successfully in {} rounds",
    'aggregation_successful': "Model aggregation completed successfully",
    'client_connected': "Client {} connected successfully",
    'client_disconnected': "Client {} disconnected gracefully",
    'checkpoint_saved': "Checkpoint saved successfully: {}",
    'evaluation_completed': "Evaluation completed: accuracy={:.4f}, loss={:.4f}",
    'privacy_guarantee': "Privacy guarantee maintained: epsilon={}, delta={}",
    'data_processed': "Data processing completed: {} samples processed",
    'model_deployed': "Model deployed successfully to production",
}

# HTTP status codes for API responses
HTTP_STATUS_CODES = {
    'OK': 200,
    'CREATED': 201,
    'ACCEPTED': 202,
    'NO_CONTENT': 204,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'NOT_FOUND': 404,
    'METHOD_NOT_ALLOWED': 405,
    'CONFLICT': 409,
    'RATE_LIMITED': 429,
    'INTERNAL_SERVER_ERROR': 500,
    'SERVICE_UNAVAILABLE': 503,
}

# Environment variables
ENV_VARS = {
    'FL_SECRET_KEY': 'Secret key for JWT authentication',
    'FL_DATABASE_URL': 'Database connection URL',
    'FL_REDIS_URL': 'Redis connection URL for caching',
    'FL_LOG_LEVEL': 'Logging level (DEBUG, INFO, WARNING, ERROR)',
    'FL_ENVIRONMENT': 'Environment (development, staging, production)',
    'FL_MAX_CLIENTS': 'Maximum number of clients',
    'FL_PRIVACY_EPSILON': 'Privacy budget epsilon',
    'FL_PRIVACY_DELTA': 'Privacy budget delta',
    'FL_MODEL_DIR': 'Directory for storing models',
    'FL_DATA_DIR': 'Directory for storing data',
    'FL_TEMP_DIR': 'Directory for temporary files',
}

# Feature flags
FEATURE_FLAGS = {
    'ENABLE_DIFFERENTIAL_PRIVACY': True,
    'ENABLE_SECURE_AGGREGATION': False,
    'ENABLE_MODEL_COMPRESSION': True,
    'ENABLE_ADAPTIVE_LEARNING_RATE': True,
    'ENABLE_EARLY_STOPPING': True,
    'ENABLE_MODEL_CHECKPOINTING': True,
    'ENABLE_REAL_TIME_MONITORING': True,
    'ENABLE_AUTOMATIC_SCALING': False,
    'ENABLE_MULTI_GPU_TRAINING': False,
    'ENABLE_DISTRIBUTED_TRAINING': False,
}

# Version information
FRAMEWORK_VERSION = "1.0.0"
MIN_PYTHON_VERSION = "3.8"
SUPPORTED_PYTORCH_VERSION = ">=1.9.0"
SUPPORTED_TENSORFLOW_VERSION = ">=2.6.0" 