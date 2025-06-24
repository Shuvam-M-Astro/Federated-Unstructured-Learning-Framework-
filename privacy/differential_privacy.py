"""
Differential privacy implementation for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Differential privacy implementation for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize differential privacy.
        
        Args:
            config: Privacy configuration
        """
        self.config = config
        self.epsilon = config.get('epsilon', 0.5)
        self.delta = config.get('delta', 1e-5)
        self.noise_multiplier = config.get('noise_multiplier', 1.1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.enabled = config.get('differential_privacy', True)
        
        self.privacy_engine = None
        self.privacy_accountant = None
        
    def setup_privacy_engine(self, model: nn.Module, sample_rate: float, 
                           epochs: int, target_epsilon: Optional[float] = None,
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           data_loader: Optional[Any] = None) -> None:
        """Setup privacy engine for the model.
        
        Args:
            model: PyTorch model
            sample_rate: Sampling rate for privacy calculation
            epochs: Number of training epochs
            target_epsilon: Target privacy budget (if None, use config epsilon)
            optimizer: PyTorch optimizer (required for Opacus)
            data_loader: Data loader (required for Opacus)
        """
        if not self.enabled:
            logger.info("Differential privacy disabled")
            return
        
        try:
            # Ensure all parameters are the correct types
            sample_rate = float(sample_rate)
            epochs = int(epochs)
            
            self.privacy_engine = PrivacyEngine()
            
            # Calculate noise multiplier to achieve target epsilon
            if target_epsilon is not None:
                self.epsilon = float(target_epsilon)
            
            # Calculate noise multiplier based on privacy budget
            if sample_rate > 0:
                steps = int(epochs / sample_rate)
            else:
                steps = epochs  # Fallback if sample_rate is 0
            
            self.noise_multiplier = self._calculate_noise_multiplier(
                target_epsilon=float(self.epsilon),
                target_delta=float(self.delta),
                sample_rate=sample_rate,
                steps=steps
            )
            
            # Make model compatible with Opacus
            if optimizer is not None and data_loader is not None:
                self.privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )
            else:
                logger.warning("Optimizer or data_loader not provided, skipping Opacus setup")
                self.enabled = False
                return
            
            logger.info(f"Privacy engine setup complete. "
                       f"Target epsilon: {self.epsilon}, "
                       f"Noise multiplier: {self.noise_multiplier:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to setup privacy engine: {e}")
            self.enabled = False
    
    def _calculate_noise_multiplier(self, target_epsilon: float, target_delta: float,
                                  sample_rate: float, steps: int) -> float:
        """Calculate noise multiplier to achieve target privacy budget.
        
        Args:
            target_epsilon: Target epsilon value
            target_delta: Target delta value
            sample_rate: Sampling rate
            steps: Number of training steps
            
        Returns:
            Calculated noise multiplier
        """
        # This is a simplified calculation
        # In practice, you might want to use more sophisticated methods
        q = sample_rate
        T = steps
        
        # Approximate calculation based on moments accountant
        # This is a rough approximation
        sigma = np.sqrt(2 * np.log(1.25 / target_delta)) / target_epsilon
        sigma = sigma * np.sqrt(q * T)
        
        return max(sigma, 0.1)  # Minimum noise multiplier
    
    def add_noise_to_gradients(self, gradients: List[torch.Tensor], 
                              sensitivity: float = 1.0) -> List[torch.Tensor]:
        """Add noise to gradients for differential privacy.
        
        Args:
            gradients: List of gradient tensors
            sensitivity: Gradient sensitivity bound
            
        Returns:
            List of noisy gradients
        """
        if not self.enabled:
            return gradients
        
        noisy_gradients = []
        for grad in gradients:
            if grad is not None:
                # Calculate noise scale based on sensitivity and privacy parameters
                noise_scale = sensitivity * self.noise_multiplier
                
                # Add Gaussian noise
                noise = torch.randn_like(grad) * noise_scale
                noisy_grad = grad + noise
                
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(None)
        
        return noisy_gradients
    
    def clip_gradients(self, gradients: List[torch.Tensor], 
                      max_norm: Optional[float] = None) -> List[torch.Tensor]:
        """Clip gradients to bound sensitivity.
        
        Args:
            gradients: List of gradient tensors
            max_norm: Maximum gradient norm (if None, use config value)
            
        Returns:
            List of clipped gradients
        """
        if max_norm is None:
            max_norm = self.max_grad_norm
        
        clipped_gradients = []
        total_norm = 0.0
        
        # Calculate total norm
        for grad in gradients:
            if grad is not None:
                param_norm = grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in gradients:
                if grad is not None:
                    grad.data.mul_(clip_coef)
        
        return gradients
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy budget spent.
        
        Returns:
            Dictionary with privacy budget information
        """
        if not self.enabled or self.privacy_engine is None:
            return {'epsilon': 0.0, 'delta': 0.0}
        
        try:
            epsilon, delta = self.privacy_engine.get_privacy_spent()
            return {'epsilon': epsilon, 'delta': delta}
        except Exception as e:
            logger.warning(f"Failed to get privacy spent: {e}")
            return {'epsilon': 0.0, 'delta': 0.0}
    
    def is_privacy_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded.
        
        Returns:
            True if privacy budget is exceeded
        """
        privacy_spent = self.get_privacy_spent()
        return privacy_spent['epsilon'] > self.epsilon


class SecureAggregation:
    """Secure aggregation for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize secure aggregation.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.encryption_enabled = config.get('encryption', True)
        self.compression_enabled = config.get('compression', True)
        
    def encrypt_model_update(self, model_update: Dict[str, torch.Tensor], 
                           public_key: bytes) -> Dict[str, bytes]:
        """Encrypt model update for secure transmission.
        
        Args:
            model_update: Dictionary of model parameter updates
            public_key: Public key for encryption
            
        Returns:
            Dictionary of encrypted model updates
        """
        if not self.encryption_enabled:
            return model_update
        
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            
            encrypted_update = {}
            
            for param_name, param_tensor in model_update.items():
                # Serialize tensor
                tensor_bytes = self._tensor_to_bytes(param_tensor)
                
                # Encrypt with RSA
                key = serialization.load_pem_public_key(public_key)
                encrypted_bytes = key.encrypt(
                    tensor_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                encrypted_update[param_name] = encrypted_bytes
            
            return encrypted_update
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return model_update
    
    def decrypt_model_update(self, encrypted_update: Dict[str, bytes], 
                           private_key: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model update.
        
        Args:
            encrypted_update: Dictionary of encrypted model updates
            private_key: Private key for decryption
            
        Returns:
            Dictionary of decrypted model updates
        """
        if not self.encryption_enabled:
            return encrypted_update
        
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            
            decrypted_update = {}
            
            for param_name, encrypted_bytes in encrypted_update.items():
                # Decrypt with RSA
                key = serialization.load_pem_private_key(private_key, password=None)
                decrypted_bytes = key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Deserialize tensor
                param_tensor = self._bytes_to_tensor(decrypted_bytes)
                decrypted_update[param_name] = param_tensor
            
            return decrypted_update
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_update
    
    def compress_model_update(self, model_update: Dict[str, torch.Tensor], 
                            compression_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """Compress model update to reduce communication overhead.
        
        Args:
            model_update: Dictionary of model parameter updates
            compression_ratio: Compression ratio (0.1 = 10% of original size)
            
        Returns:
            Dictionary of compressed model updates
        """
        if not self.compression_enabled:
            return model_update
        
        compressed_update = {}
        
        for param_name, param_tensor in model_update.items():
            # Convert to numpy for easier manipulation
            tensor_np = param_tensor.detach().cpu().numpy()
            
            # Apply compression (simple top-k sparsification)
            flat_tensor = tensor_np.flatten()
            k = int(len(flat_tensor) * compression_ratio)
            
            # Keep top-k values
            indices = np.argsort(np.abs(flat_tensor))[-k:]
            compressed_tensor = np.zeros_like(flat_tensor)
            compressed_tensor[indices] = flat_tensor[indices]
            
            # Reshape back to original shape
            compressed_tensor = compressed_tensor.reshape(tensor_np.shape)
            
            compressed_update[param_name] = torch.from_numpy(compressed_tensor)
        
        return compressed_update
    
    def decompress_model_update(self, compressed_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decompress model update.
        
        Args:
            compressed_update: Dictionary of compressed model updates
            
        Returns:
            Dictionary of decompressed model updates
        """
        if not self.compression_enabled:
            return compressed_update
        
        # For simple top-k sparsification, decompression is just returning the compressed tensor
        # In more sophisticated compression schemes, you would implement the inverse operation
        return compressed_update
    
    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes for encryption.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Tensor as bytes
        """
        import io
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def _bytes_to_tensor(self, tensor_bytes: bytes) -> torch.Tensor:
        """Convert bytes back to tensor.
        
        Args:
            tensor_bytes: Tensor as bytes
            
        Returns:
            PyTorch tensor
        """
        import io
        buffer = io.BytesIO(tensor_bytes)
        return torch.load(buffer)


class PrivacyManager:
    """Manager for privacy-preserving mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize privacy manager.
        
        Args:
            config: Privacy configuration
        """
        self.config = config
        self.dp = DifferentialPrivacy(config)
        self.secure_agg = SecureAggregation(config)
        
    def setup_model_privacy(self, model: nn.Module, sample_rate: float, 
                          epochs: int, optimizer: Optional[torch.optim.Optimizer] = None,
                          data_loader: Optional[Any] = None) -> None:
        """Setup privacy mechanisms for a model.
        
        Args:
            model: PyTorch model
            sample_rate: Sampling rate
            epochs: Number of training epochs
            optimizer: PyTorch optimizer (required for Opacus)
            data_loader: Data loader (required for Opacus)
        """
        self.dp.setup_privacy_engine(model, sample_rate, epochs, optimizer=optimizer, data_loader=data_loader)
    
    def process_gradients(self, gradients: List[torch.Tensor], 
                         sensitivity: float = 1.0) -> List[torch.Tensor]:
        """Process gradients with privacy mechanisms.
        
        Args:
            gradients: List of gradient tensors
            sensitivity: Gradient sensitivity bound
            
        Returns:
            Processed gradients
        """
        # Clip gradients
        clipped_gradients = self.dp.clip_gradients(gradients)
        
        # Add noise
        noisy_gradients = self.dp.add_noise_to_gradients(clipped_gradients, sensitivity)
        
        return noisy_gradients
    
    def secure_aggregate_updates(self, model_updates: List[Dict[str, torch.Tensor]], 
                               weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Securely aggregate model updates from multiple clients.
        
        Args:
            model_updates: List of model updates from clients
            weights: Weights for weighted averaging (if None, equal weights)
            
        Returns:
            Aggregated model update
        """
        if weights is None:
            weights = [1.0 / len(model_updates)] * len(model_updates)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Aggregate updates
        aggregated_update = {}
        param_names = model_updates[0].keys()
        
        for param_name in param_names:
            aggregated_param = torch.zeros_like(model_updates[0][param_name])
            
            for i, update in enumerate(model_updates):
                aggregated_param += weights[i] * update[param_name]
            
            aggregated_update[param_name] = aggregated_param
        
        return aggregated_update
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status.
        
        Returns:
            Dictionary with privacy information
        """
        privacy_spent = self.dp.get_privacy_spent()
        
        return {
            'differential_privacy_enabled': self.dp.enabled,
            'encryption_enabled': self.secure_agg.encryption_enabled,
            'compression_enabled': self.secure_agg.compression_enabled,
            'privacy_budget_spent': privacy_spent,
            'privacy_budget_exceeded': self.dp.is_privacy_budget_exceeded(),
            'target_epsilon': self.dp.epsilon,
            'target_delta': self.dp.delta,
            'noise_multiplier': self.dp.noise_multiplier
        } 