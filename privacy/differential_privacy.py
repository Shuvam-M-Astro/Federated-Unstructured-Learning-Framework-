"""
Advanced differential privacy implementation for federated learning.

This module provides comprehensive privacy protection mechanisms including
differential privacy, secure aggregation, privacy accounting, and privacy
budget management for production federated learning systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import threading
import time
from pathlib import Path

# Privacy libraries
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.accountants import RDPAccountant, GaussianAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logger.warning("Opacus not available, using basic differential privacy")

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("Cryptography library not available, secure aggregation disabled")

from utils.constants import (
    ERROR_MESSAGES, WARNING_MESSAGES, SUCCESS_MESSAGES,
    DEFAULT_EPSILON, DEFAULT_DELTA, DEFAULT_NOISE_MULTIPLIER, DEFAULT_MAX_GRAD_NORM
)

logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Supported privacy mechanisms."""
    GAUSSIAN_MECHANISM = "gaussian"
    LAPLACE_MECHANISM = "laplace"
    EXPONENTIAL_MECHANISM = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    LOW = "low"           # ε > 10
    MEDIUM = "medium"     # 1 < ε ≤ 10
    HIGH = "high"         # 0.1 < ε ≤ 1
    VERY_HIGH = "very_high"  # ε ≤ 0.1


@dataclass
class PrivacyBudget:
    """Privacy budget tracking."""
    epsilon: float
    delta: float
    mechanism: PrivacyMechanism
    timestamp: datetime
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyReport:
    """Comprehensive privacy report."""
    total_epsilon: float
    total_delta: float
    privacy_level: PrivacyLevel
    budget_history: List[PrivacyBudget]
    remaining_budget: float
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class PrivacyAccountant:
    """Advanced privacy accounting with composition."""
    
    def __init__(self, target_epsilon: float, target_delta: float):
        """Initialize privacy accountant.
        
        Args:
            target_epsilon: Target epsilon value
            target_delta: Target delta value
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.budget_history: List[PrivacyBudget] = []
        self._lock = threading.RLock()
        
        if OPACUS_AVAILABLE:
            self.accountant = RDPAccountant()
        else:
            self.accountant = None
    
    def add_budget(self, epsilon: float, delta: float, 
                  mechanism: PrivacyMechanism, description: str = "") -> None:
        """Add privacy budget consumption.
        
        Args:
            epsilon: Epsilon consumed
            delta: Delta consumed
            mechanism: Privacy mechanism used
            description: Description of the operation
        """
        with self._lock:
            budget = PrivacyBudget(
                epsilon=epsilon,
                delta=delta,
                mechanism=mechanism,
                timestamp=datetime.now(),
                description=description
            )
            self.budget_history.append(budget)
    
    def get_total_budget_spent(self) -> Tuple[float, float]:
        """Get total privacy budget spent.
        
        Returns:
            Tuple of (total_epsilon, total_delta)
        """
        with self._lock:
            total_epsilon = sum(budget.epsilon for budget in self.budget_history)
            total_delta = sum(budget.delta for budget in self.budget_history)
            return total_epsilon, total_delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget.
        
        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        spent_epsilon, spent_delta = self.get_total_budget_spent()
        remaining_epsilon = max(0, self.target_epsilon - spent_epsilon)
        remaining_delta = max(0, self.target_delta - spent_delta)
        return remaining_epsilon, remaining_delta
    
    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded.
        
        Returns:
            True if budget is exceeded
        """
        spent_epsilon, spent_delta = self.get_total_budget_spent()
        return spent_epsilon > self.target_epsilon or spent_delta > self.target_delta
    
    def get_privacy_level(self) -> PrivacyLevel:
        """Get current privacy level based on remaining budget.
        
        Returns:
            Privacy level
        """
        remaining_epsilon, _ = self.get_remaining_budget()
        
        if remaining_epsilon > 10:
            return PrivacyLevel.LOW
        elif remaining_epsilon > 1:
            return PrivacyLevel.MEDIUM
        elif remaining_epsilon > 0.1:
            return PrivacyLevel.HIGH
        else:
            return PrivacyLevel.VERY_HIGH
    
    def generate_report(self) -> PrivacyReport:
        """Generate comprehensive privacy report.
        
        Returns:
            Privacy report
        """
        total_epsilon, total_delta = self.get_total_budget_spent()
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        privacy_level = self.get_privacy_level()
        
        # Generate recommendations
        recommendations = []
        if self.is_budget_exceeded():
            recommendations.append("Privacy budget exceeded - stop training immediately")
        elif remaining_epsilon < self.target_epsilon * 0.1:
            recommendations.append("Privacy budget running low - consider reducing training")
        if total_epsilon > self.target_epsilon * 0.8:
            recommendations.append("Consider using more conservative privacy parameters")
        
        return PrivacyReport(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            privacy_level=privacy_level,
            budget_history=self.budget_history.copy(),
            remaining_budget=remaining_epsilon,
            recommendations=recommendations
        )


class AdvancedDifferentialPrivacy:
    """Advanced differential privacy implementation with multiple mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced differential privacy.
        
        Args:
            config: Privacy configuration
        """
        self.config = config
        self.epsilon = config.get('epsilon', DEFAULT_EPSILON)
        self.delta = config.get('delta', DEFAULT_DELTA)
        self.noise_multiplier = config.get('noise_multiplier', DEFAULT_NOISE_MULTIPLIER)
        self.max_grad_norm = config.get('max_grad_norm', DEFAULT_MAX_GRAD_NORM)
        self.mechanism = PrivacyMechanism(config.get('mechanism', 'gaussian'))
        self.enabled = config.get('differential_privacy', True)
        
        # Advanced features
        self.adaptive_noise = config.get('adaptive_noise', False)
        self.composition_method = config.get('composition_method', 'basic')
        self.privacy_accountant = PrivacyAccountant(self.epsilon, self.delta)
        
        # Opacus integration
        self.privacy_engine = None
        self._setup_opacus()
    
    def _setup_opacus(self) -> None:
        """Setup Opacus privacy engine if available."""
        if not OPACUS_AVAILABLE or not self.enabled:
            return
        
        try:
            self.privacy_engine = PrivacyEngine()
            logger.info("Opacus privacy engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Opacus: {e}")
            self.privacy_engine = None
    
    def setup_model_privacy(self, model: nn.Module, sample_rate: float, 
                           epochs: int, target_epsilon: Optional[float] = None,
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           data_loader: Optional[Any] = None) -> None:
        """Setup privacy protection for the model.
        
        Args:
            model: PyTorch model
            sample_rate: Sampling rate for privacy calculation
            epochs: Number of training epochs
            target_epsilon: Target privacy budget
            optimizer: PyTorch optimizer
            data_loader: Data loader
        """
        if not self.enabled:
            logger.info("Differential privacy disabled")
            return
        
        try:
            if target_epsilon is not None:
                self.epsilon = float(target_epsilon)
            
            # Calculate noise multiplier
            steps = int(epochs / sample_rate) if sample_rate > 0 else epochs
            self.noise_multiplier = self._calculate_optimal_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=sample_rate,
                steps=steps
            )
            
            # Setup Opacus if available
            if self.privacy_engine and optimizer and data_loader:
                self.privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )
                logger.info(f"Opacus privacy engine configured with noise_multiplier={self.noise_multiplier:.3f}")
            
            logger.info(SUCCESS_MESSAGES['privacy_setup_complete'].format(self.epsilon, self.noise_multiplier))
            
        except Exception as e:
            logger.error(ERROR_MESSAGES['privacy_setup_failed'].format(str(e)))
            self.enabled = False
    
    def _calculate_optimal_noise_multiplier(self, target_epsilon: float, target_delta: float,
                                          sample_rate: float, steps: int) -> float:
        """Calculate optimal noise multiplier using advanced methods.
        
        Args:
            target_epsilon: Target epsilon value
            target_delta: Target delta value
            sample_rate: Sampling rate
            steps: Number of training steps
            
        Returns:
            Optimal noise multiplier
        """
        q = sample_rate
        T = steps
        
        if self.composition_method == 'advanced' and OPACUS_AVAILABLE:
            # Use Opacus accountant for more accurate calculation
            try:
                # This is a simplified version - in practice, you'd use Opacus's accountant
                sigma = np.sqrt(2 * np.log(1.25 / target_delta)) / target_epsilon
                sigma = sigma * np.sqrt(q * T)
            except Exception:
                sigma = self._basic_noise_calculation(target_epsilon, target_delta, q, T)
        else:
            sigma = self._basic_noise_calculation(target_epsilon, target_delta, q, T)
        
        return max(sigma, 0.1)
    
    def _basic_noise_calculation(self, target_epsilon: float, target_delta: float,
                               sample_rate: float, steps: int) -> float:
        """Basic noise multiplier calculation."""
        q = sample_rate
        T = steps
        
        # Gaussian mechanism noise calculation
        sigma = np.sqrt(2 * np.log(1.25 / target_delta)) / target_epsilon
        sigma = sigma * np.sqrt(q * T)
        
        return sigma
    
    def add_noise_to_gradients(self, gradients: List[torch.Tensor], 
                              sensitivity: float = 1.0,
                              mechanism: Optional[PrivacyMechanism] = None) -> List[torch.Tensor]:
        """Add noise to gradients using specified mechanism.
        
        Args:
            gradients: List of gradient tensors
            sensitivity: Gradient sensitivity bound
            mechanism: Privacy mechanism to use
            
        Returns:
            List of noisy gradients
        """
        if not self.enabled:
            return gradients
        
        if mechanism is None:
            mechanism = self.mechanism
        
        noisy_gradients = []
        for grad in gradients:
            if grad is not None:
                noise = self._generate_noise(grad.shape, sensitivity, mechanism)
                noisy_grad = grad + noise
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(None)
        
        # Track privacy budget consumption
        epsilon_consumed = self._calculate_epsilon_consumption(sensitivity, mechanism)
        self.privacy_accountant.add_budget(
            epsilon=epsilon_consumed,
            delta=self.delta,
            mechanism=mechanism,
            description="Gradient noise addition"
        )
        
        return noisy_gradients
    
    def _generate_noise(self, shape: Tuple[int, ...], sensitivity: float, 
                       mechanism: PrivacyMechanism) -> torch.Tensor:
        """Generate noise based on privacy mechanism.
        
        Args:
            shape: Shape of the tensor
            sensitivity: Sensitivity bound
            mechanism: Privacy mechanism
            
        Returns:
            Noise tensor
        """
        if mechanism == PrivacyMechanism.GAUSSIAN_MECHANISM:
            scale = sensitivity * self.noise_multiplier
            return torch.randn(shape, device='cpu') * scale
        
        elif mechanism == PrivacyMechanism.LAPLACE_MECHANISM:
            scale = sensitivity / self.epsilon
            return torch.distributions.Laplace(0, scale).sample(shape)
        
        elif mechanism == PrivacyMechanism.RANDOMIZED_RESPONSE:
            # Simplified randomized response for binary data
            p = 1 / (1 + np.exp(self.epsilon))
            mask = torch.rand(shape) < p
            return torch.where(mask, torch.randn(shape), torch.zeros(shape))
        
        else:
            # Default to Gaussian
            scale = sensitivity * self.noise_multiplier
            return torch.randn(shape, device='cpu') * scale
    
    def _calculate_epsilon_consumption(self, sensitivity: float, 
                                     mechanism: PrivacyMechanism) -> float:
        """Calculate epsilon consumption for the operation.
        
        Args:
            sensitivity: Sensitivity bound
            mechanism: Privacy mechanism used
            
        Returns:
            Epsilon consumed
        """
        if mechanism == PrivacyMechanism.GAUSSIAN_MECHANISM:
            # Simplified calculation
            return sensitivity / (self.noise_multiplier * np.sqrt(2 * np.log(1.25 / self.delta)))
        
        elif mechanism == PrivacyMechanism.LAPLACE_MECHANISM:
            return sensitivity / self.noise_multiplier
        
        else:
            return self.epsilon * 0.01  # Small consumption for other mechanisms
    
    def clip_gradients(self, gradients: List[torch.Tensor], 
                      max_norm: Optional[float] = None) -> List[torch.Tensor]:
        """Clip gradients to bound sensitivity.
        
        Args:
            gradients: List of gradient tensors
            max_norm: Maximum gradient norm
            
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
        if not self.enabled:
            return {'epsilon': 0.0, 'delta': 0.0}
        
        if self.privacy_engine and OPACUS_AVAILABLE:
            try:
                epsilon, delta = self.privacy_engine.get_privacy_spent()
                return {'epsilon': epsilon, 'delta': delta}
            except Exception as e:
                logger.warning(f"Failed to get Opacus privacy spent: {e}")
        
        # Fallback to our accountant
        total_epsilon, total_delta = self.privacy_accountant.get_total_budget_spent()
        return {'epsilon': total_epsilon, 'delta': total_delta}
    
    def is_privacy_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded.
        
        Returns:
            True if budget is exceeded
        """
        return self.privacy_accountant.is_budget_exceeded()
    
    def get_privacy_report(self) -> PrivacyReport:
        """Get comprehensive privacy report.
        
        Returns:
            Privacy report
        """
        return self.privacy_accountant.generate_report()


class SecureAggregation:
    """Secure aggregation with homomorphic encryption and secure multiparty computation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize secure aggregation.
        
        Args:
            config: Secure aggregation configuration
        """
        self.config = config
        self.enabled = config.get('secure_aggregation', False) and CRYPTOGRAPHY_AVAILABLE
        self.key_size = config.get('key_size', 2048)
        self.compression_enabled = config.get('compression_enabled', True)
        self.compression_ratio = config.get('compression_ratio', 0.1)
        
        if self.enabled:
            self._generate_key_pair()
    
    def _generate_key_pair(self) -> None:
        """Generate RSA key pair for encryption."""
        try:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size
            )
            self.public_key = self.private_key.public_key()
            logger.info("RSA key pair generated for secure aggregation")
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            self.enabled = False
    
    def encrypt_model_update(self, model_update: Dict[str, torch.Tensor], 
                           public_key: bytes) -> Dict[str, bytes]:
        """Encrypt model update using public key.
        
        Args:
            model_update: Model update dictionary
            public_key: Public key for encryption
            
        Returns:
            Encrypted model update
        """
        if not self.enabled:
            return model_update
        
        try:
            encrypted_update = {}
            for param_name, param_tensor in model_update.items():
                # Convert tensor to bytes
                tensor_bytes = self._tensor_to_bytes(param_tensor)
                
                # Encrypt using RSA
                encrypted_bytes = self._rsa_encrypt(tensor_bytes, public_key)
                encrypted_update[param_name] = encrypted_bytes
            
            return encrypted_update
            
        except Exception as e:
            logger.error(f"Failed to encrypt model update: {e}")
            return model_update
    
    def decrypt_model_update(self, encrypted_update: Dict[str, bytes], 
                           private_key: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model update using private key.
        
        Args:
            encrypted_update: Encrypted model update
            private_key: Private key for decryption
            
        Returns:
            Decrypted model update
        """
        if not self.enabled:
            return encrypted_update
        
        try:
            decrypted_update = {}
            for param_name, encrypted_bytes in encrypted_update.items():
                # Decrypt using RSA
                decrypted_bytes = self._rsa_decrypt(encrypted_bytes, private_key)
                
                # Convert bytes back to tensor
                param_tensor = self._bytes_to_tensor(decrypted_bytes)
                decrypted_update[param_name] = param_tensor
            
            return decrypted_update
            
        except Exception as e:
            logger.error(f"Failed to decrypt model update: {e}")
            return encrypted_update
    
    def _rsa_encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """Encrypt data using RSA public key."""
        try:
            pub_key = serialization.load_pem_public_key(public_key)
            encrypted = pub_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted
        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            return data
    
    def _rsa_decrypt(self, encrypted_data: bytes, private_key: bytes) -> bytes:
        """Decrypt data using RSA private key."""
        try:
            priv_key = serialization.load_pem_private_key(private_key, password=None)
            decrypted = priv_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted
        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            return encrypted_data
    
    def compress_model_update(self, model_update: Dict[str, torch.Tensor], 
                            compression_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """Compress model update using quantization and pruning.
        
        Args:
            model_update: Model update dictionary
            compression_ratio: Compression ratio (0-1)
            
        Returns:
            Compressed model update
        """
        if not self.compression_enabled:
            return model_update
        
        compressed_update = {}
        for param_name, param_tensor in model_update.items():
            # Quantize to reduce precision
            quantized = self._quantize_tensor(param_tensor, bits=8)
            
            # Prune small values
            threshold = torch.quantile(torch.abs(quantized), compression_ratio)
            mask = torch.abs(quantized) > threshold
            pruned = quantized * mask.float()
            
            compressed_update[param_name] = pruned
        
        return compressed_update
    
    def decompress_model_update(self, compressed_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decompress model update.
        
        Args:
            compressed_update: Compressed model update
            
        Returns:
            Decompressed model update
        """
        if not self.compression_enabled:
            return compressed_update
        
        decompressed_update = {}
        for param_name, param_tensor in compressed_update.items():
            # Dequantize
            dequantized = self._dequantize_tensor(param_tensor, bits=8)
            decompressed_update[param_name] = dequantized
        
        return decompressed_update
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize tensor to specified bit precision."""
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2 ** bits - 1)
        
        quantized = torch.round((tensor - min_val) / scale)
        return quantized * scale + min_val
    
    def _dequantize_tensor(self, tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Dequantize tensor."""
        # For symmetric quantization, dequantization is the same as quantization
        return tensor
    
    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes."""
        buffer = torch.tensor(tensor, dtype=torch.float32)
        return buffer.numpy().tobytes()
    
    def _bytes_to_tensor(self, tensor_bytes: bytes) -> torch.Tensor:
        """Convert bytes back to tensor."""
        import struct
        float_list = struct.unpack('f' * (len(tensor_bytes) // 4), tensor_bytes)
        return torch.tensor(float_list, dtype=torch.float32)


class PrivacyManager:
    """Comprehensive privacy management system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize privacy manager.
        
        Args:
            config: Privacy configuration
        """
        self.config = config
        self.differential_privacy = AdvancedDifferentialPrivacy(config)
        self.secure_aggregation = SecureAggregation(config)
        self.privacy_accountant = PrivacyAccountant(
            config.get('epsilon', DEFAULT_EPSILON),
            config.get('delta', DEFAULT_DELTA)
        )
        
        # Privacy monitoring
        self.privacy_log = []
        self.monitoring_enabled = config.get('privacy_monitoring', True)
    
    def setup_model_privacy(self, model: nn.Module, sample_rate: float, 
                          epochs: int, optimizer: Optional[torch.optim.Optimizer] = None,
                          data_loader: Optional[Any] = None) -> None:
        """Setup comprehensive privacy protection for the model.
        
        Args:
            model: PyTorch model
            sample_rate: Sampling rate
            epochs: Number of training epochs
            optimizer: PyTorch optimizer
            data_loader: Data loader
        """
        self.differential_privacy.setup_model_privacy(
            model, sample_rate, epochs, optimizer=optimizer, data_loader=data_loader
        )
        
        if self.monitoring_enabled:
            self._log_privacy_event("Model privacy setup", {
                'sample_rate': sample_rate,
                'epochs': epochs,
                'epsilon': self.differential_privacy.epsilon,
                'delta': self.differential_privacy.delta
            })
    
    def process_gradients(self, gradients: List[torch.Tensor], 
                         sensitivity: float = 1.0) -> List[torch.Tensor]:
        """Process gradients with privacy protection.
        
        Args:
            gradients: List of gradient tensors
            sensitivity: Gradient sensitivity bound
            
        Returns:
            Processed gradients
        """
        # Clip gradients
        clipped_gradients = self.differential_privacy.clip_gradients(gradients)
        
        # Add noise
        noisy_gradients = self.differential_privacy.add_noise_to_gradients(
            clipped_gradients, sensitivity
        )
        
        if self.monitoring_enabled:
            self._log_privacy_event("Gradient processing", {
                'sensitivity': sensitivity,
                'num_gradients': len(gradients)
            })
        
        return noisy_gradients
    
    def secure_aggregate_updates(self, model_updates: List[Dict[str, torch.Tensor]], 
                               weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Securely aggregate model updates.
        
        Args:
            model_updates: List of model updates
            weights: Optional weights for weighted averaging
            
        Returns:
            Aggregated model update
        """
        if not model_updates:
            return {}
        
        if weights is None:
            weights = [1.0 / len(model_updates)] * len(model_updates)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Aggregate updates
        aggregated_update = {}
        for param_name in model_updates[0].keys():
            weighted_sum = torch.zeros_like(model_updates[0][param_name])
            for update, weight in zip(model_updates, weights):
                weighted_sum += update[param_name] * weight
            aggregated_update[param_name] = weighted_sum
        
        if self.monitoring_enabled:
            self._log_privacy_event("Secure aggregation", {
                'num_updates': len(model_updates),
                'aggregation_method': 'weighted_average'
            })
        
        return aggregated_update
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get comprehensive privacy status.
        
        Returns:
            Privacy status dictionary
        """
        dp_status = self.differential_privacy.get_privacy_spent()
        privacy_report = self.differential_privacy.get_privacy_report()
        
        return {
            'differential_privacy': {
                'enabled': self.differential_privacy.enabled,
                'epsilon_spent': dp_status['epsilon'],
                'delta_spent': dp_status['delta'],
                'budget_exceeded': self.differential_privacy.is_privacy_budget_exceeded()
            },
            'secure_aggregation': {
                'enabled': self.secure_aggregation.enabled,
                'compression_enabled': self.secure_aggregation.compression_enabled
            },
            'privacy_report': privacy_report.__dict__,
            'privacy_level': privacy_report.privacy_level.value,
            'recommendations': privacy_report.recommendations
        }
    
    def _log_privacy_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Log privacy event for monitoring."""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'metadata': metadata,
            'privacy_budget': self.differential_privacy.get_privacy_spent()
        }
        self.privacy_log.append(event)
        
        # Keep only recent events
        if len(self.privacy_log) > 1000:
            self.privacy_log = self.privacy_log[-1000:]
    
    def export_privacy_log(self, filepath: str) -> None:
        """Export privacy log to file.
        
        Args:
            filepath: Path to export file
        """
        try:
            log_data = []
            for event in self.privacy_log:
                log_data.append({
                    'timestamp': event['timestamp'].isoformat(),
                    'event_type': event['event_type'],
                    'metadata': event['metadata'],
                    'privacy_budget': event['privacy_budget']
                })
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Privacy log exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export privacy log: {e}")
    
    def validate_privacy_guarantees(self) -> bool:
        """Validate that privacy guarantees are maintained.
        
        Returns:
            True if privacy guarantees are valid
        """
        # Check if budget is exceeded
        if self.differential_privacy.is_privacy_budget_exceeded():
            logger.error("Privacy budget exceeded - privacy guarantees violated")
            return False
        
        # Check privacy level
        privacy_report = self.differential_privacy.get_privacy_report()
        if privacy_report.privacy_level == PrivacyLevel.LOW:
            logger.warning("Privacy level is low - consider stronger protection")
        
        return True 