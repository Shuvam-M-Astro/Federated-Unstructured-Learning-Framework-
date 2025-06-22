"""
Model factory for creating different types of neural networks for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10, 
                 hidden_size: int = 128, dropout: float = 0.5):
        """Initialize CNN model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            hidden_size: Size of hidden layers
            dropout: Dropout rate
        """
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate the size after convolutions and pooling
        # Assuming input size of 224x224
        conv_output_size = 128 * 28 * 28  # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class RNNModel(nn.Module):
    """Recurrent Neural Network for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_size: int = 128, num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.5):
        """Initialize RNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of hidden layers
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(RNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding layer
        embedded = self.embedding(x)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        output = self.dropout(hidden[-1])
        output = self.fc(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer model for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_heads: int = 8, num_layers: int = 6,
                 hidden_size: int = 512, num_classes: int = 10,
                 max_length: int = 512, dropout: float = 0.1):
        """Initialize Transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_size: Size of hidden layers
            num_classes: Number of output classes
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        self.embedding_dim = embedding_dim
        self.max_length = max_length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len = x.size()
        
        # Embedding layer
        embedded = self.embedding(x)
        
        # Add positional encoding
        if seq_len <= self.max_length:
            embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        # Create attention mask for padding
        mask = (x == 0).to(x.device)
        
        # Transformer layers
        transformer_out = self.transformer(embedded, src_key_padding_mask=mask)
        
        # Global average pooling
        # Mask out padding tokens
        mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
        transformer_out = transformer_out.masked_fill(mask_expanded, 0)
        
        # Average over sequence length (excluding padding)
        lengths = (~mask).sum(dim=1, keepdim=True).float()
        pooled = transformer_out.sum(dim=1) / lengths
        
        # Classification head
        output = self.dropout(pooled)
        output = self.fc(output)
        
        return output


class TabularModel(nn.Module):
    """Neural network for tabular data classification."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64],
                 num_classes: int = 10, dropout: float = 0.3):
        """Initialize tabular model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(TabularModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.network(x)


class MultiModalModel(nn.Module):
    """Multi-modal model for handling mixed data types."""
    
    def __init__(self, text_config: Dict[str, Any], image_config: Dict[str, Any],
                 tabular_config: Dict[str, Any], num_classes: int = 10,
                 fusion_method: str = 'concat'):
        """Initialize multi-modal model.
        
        Args:
            text_config: Configuration for text processing
            image_config: Configuration for image processing
            tabular_config: Configuration for tabular processing
            num_classes: Number of output classes
            fusion_method: Method to fuse modalities ('concat', 'attention', 'weighted')
        """
        super(MultiModalModel, self).__init__()
        
        # Initialize sub-models
        self.text_model = None
        self.image_model = None
        self.tabular_model = None
        
        if text_config.get('enabled', False):
            model_type = text_config.get('model_type', 'rnn')
            if model_type == 'rnn':
                self.text_model = RNNModel(
                    vocab_size=text_config.get('vocab_size', 10000),
                    embedding_dim=text_config.get('embedding_dim', 128),
                    hidden_size=text_config.get('hidden_size', 128),
                    num_classes=text_config.get('text_features', 64)
                )
            elif model_type == 'transformer':
                self.text_model = TransformerModel(
                    vocab_size=text_config.get('vocab_size', 10000),
                    embedding_dim=text_config.get('embedding_dim', 128),
                    num_classes=text_config.get('text_features', 64)
                )
        
        if image_config.get('enabled', False):
            self.image_model = CNNModel(
                input_channels=image_config.get('input_channels', 3),
                num_classes=image_config.get('image_features', 64)
            )
        
        if tabular_config.get('enabled', False):
            self.tabular_model = TabularModel(
                input_size=tabular_config.get('input_size', 10),
                hidden_sizes=tabular_config.get('hidden_sizes', [64, 32]),
                num_classes=tabular_config.get('tabular_features', 32)
            )
        
        # Fusion layer
        self.fusion_method = fusion_method
        total_features = 0
        
        if self.text_model:
            total_features += text_config.get('text_features', 64)
        if self.image_model:
            total_features += image_config.get('image_features', 64)
        if self.tabular_model:
            total_features += tabular_config.get('tabular_features', 32)
        
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(total_features, num_heads=4, batch_first=True)
            self.fusion_layer = nn.Linear(total_features, 128)
        elif fusion_method == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(3))  # One weight per modality
            self.fusion_layer = nn.Linear(total_features, 128)
        else:  # concat
            self.fusion_layer = nn.Linear(total_features, 128)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, text_input=None, image_input=None, tabular_input=None):
        """Forward pass.
        
        Args:
            text_input: Text input tensor
            image_input: Image input tensor
            tabular_input: Tabular input tensor
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        features = []
        
        if self.text_model and text_input is not None:
            text_features = self.text_model(text_input)
            features.append(text_features)
        
        if self.image_model and image_input is not None:
            image_features = self.image_model(image_input)
            features.append(image_features)
        
        if self.tabular_model and tabular_input is not None:
            tabular_features = self.tabular_model(tabular_input)
            features.append(tabular_features)
        
        if not features:
            raise ValueError("No input provided to multi-modal model")
        
        # Fuse features
        if self.fusion_method == 'attention':
            # Stack features and apply attention
            stacked_features = torch.stack(features, dim=1)  # (batch, modalities, features)
            attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)
            fused_features = attended_features.mean(dim=1)  # Average over modalities
        elif self.fusion_method == 'weighted':
            # Weighted combination
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_features = sum(w * f for w, f in zip(weights, features))
        else:  # concat
            # Concatenate features
            fused_features = torch.cat(features, dim=1)
        
        # Final classification
        fused_features = self.fusion_layer(fused_features)
        output = self.classifier(fused_features)
        
        return output


class ModelFactory:
    """Factory for creating different types of models."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create a model based on type and configuration.
        
        Args:
            model_type: Type of model ('cnn', 'rnn', 'transformer', 'tabular', 'multimodal')
            config: Model configuration
            
        Returns:
            PyTorch model
        """
        if model_type == 'cnn':
            return CNNModel(
                input_channels=config.get('input_channels', 3),
                num_classes=config.get('num_classes', 10),
                hidden_size=config.get('hidden_size', 128),
                dropout=config.get('dropout', 0.5)
            )
        
        elif model_type == 'rnn':
            return RNNModel(
                vocab_size=config.get('vocab_size', 10000),
                embedding_dim=config.get('embedding_dim', 128),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                num_classes=config.get('num_classes', 10),
                dropout=config.get('dropout', 0.5)
            )
        
        elif model_type == 'transformer':
            return TransformerModel(
                vocab_size=config.get('vocab_size', 10000),
                embedding_dim=config.get('embedding_dim', 128),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 6),
                hidden_size=config.get('hidden_size', 512),
                num_classes=config.get('num_classes', 10),
                max_length=config.get('max_length', 512),
                dropout=config.get('dropout', 0.1)
            )
        
        elif model_type == 'tabular':
            return TabularModel(
                input_size=config.get('input_size', 10),
                hidden_sizes=config.get('hidden_sizes', [128, 64]),
                num_classes=config.get('num_classes', 10),
                dropout=config.get('dropout', 0.3)
            )
        
        elif model_type == 'multimodal':
            return MultiModalModel(
                text_config=config.get('text', {}),
                image_config=config.get('image', {}),
                tabular_config=config.get('tabular', {}),
                num_classes=config.get('num_classes', 10),
                fusion_method=config.get('fusion_method', 'concat')
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """Get information about a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': str(model)
        } 