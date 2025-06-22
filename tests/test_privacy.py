"""
Tests for privacy mechanisms in federated learning.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from privacy.differential_privacy import (
    DifferentialPrivacy, 
    SecureAggregation, 
    PrivacyManager
)
from utils.config_loader import ConfigLoader


class TestDifferentialPrivacy:
    """Test differential privacy implementation."""
    
    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration for testing."""
        return {
            'differential_privacy': True,
            'epsilon': 0.5,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'max_grad_norm': 1.0
        }
    
    @pytest.fixture
    def dp(self, privacy_config):
        """Create differential privacy instance."""
        return DifferentialPrivacy(privacy_config)
    
    def test_initialization(self, dp, privacy_config):
        """Test differential privacy initialization."""
        assert dp.epsilon == privacy_config['epsilon']
        assert dp.delta == privacy_config['delta']
        assert dp.noise_multiplier == privacy_config['noise_multiplier']
        assert dp.max_grad_norm == privacy_config['max_grad_norm']
        assert dp.enabled == privacy_config['differential_privacy']
    
    def test_disabled_privacy(self):
        """Test behavior when privacy is disabled."""
        config = {'differential_privacy': False}
        dp = DifferentialPrivacy(config)
        
        # Test that no noise is added when disabled
        gradients = [torch.randn(10, 10) for _ in range(3)]
        original_gradients = [g.clone() for g in gradients]
        
        noisy_gradients = dp.add_noise_to_gradients(gradients)
        
        # Gradients should be unchanged
        for orig, noisy in zip(original_gradients, noisy_gradients):
            assert torch.allclose(orig, noisy)
    
    def test_gradient_clipping(self, dp):
        """Test gradient clipping functionality."""
        # Create gradients with large norm
        gradients = [torch.randn(10, 10) * 10 for _ in range(3)]
        
        clipped_gradients = dp.clip_gradients(gradients, max_norm=1.0)
        
        # Calculate total norm
        total_norm = 0.0
        for grad in clipped_gradients:
            total_norm += grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Total norm should be <= max_norm
        assert total_norm <= 1.0 + 1e-6
    
    def test_noise_addition(self, dp):
        """Test noise addition to gradients."""
        gradients = [torch.randn(5, 5) for _ in range(2)]
        original_gradients = [g.clone() for g in gradients]
        
        noisy_gradients = dp.add_noise_to_gradients(gradients, sensitivity=1.0)
        
        # Gradients should be different (noise added)
        for orig, noisy in zip(original_gradients, noisy_gradients):
            assert not torch.allclose(orig, noisy)
    
    def test_privacy_budget_calculation(self, dp):
        """Test privacy budget calculation."""
        # Mock privacy engine
        dp.privacy_engine = Mock()
        dp.privacy_engine.get_privacy_spent.return_value = (0.3, 1e-6)
        
        privacy_spent = dp.get_privacy_spent()
        
        assert privacy_spent['epsilon'] == 0.3
        assert privacy_spent['delta'] == 1e-6
    
    def test_privacy_budget_exceeded(self, dp):
        """Test privacy budget exceeded check."""
        # Mock privacy engine to return exceeded budget
        dp.privacy_engine = Mock()
        dp.privacy_engine.get_privacy_spent.return_value = (1.0, 1e-6)
        
        assert dp.is_privacy_budget_exceeded()
        
        # Mock privacy engine to return within budget
        dp.privacy_engine.get_privacy_spent.return_value = (0.1, 1e-6)
        
        assert not dp.is_privacy_budget_exceeded()


class TestSecureAggregation:
    """Test secure aggregation implementation."""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration for testing."""
        return {
            'encryption': True,
            'compression': True
        }
    
    @pytest.fixture
    def secure_agg(self, security_config):
        """Create secure aggregation instance."""
        return SecureAggregation(security_config)
    
    def test_initialization(self, secure_agg, security_config):
        """Test secure aggregation initialization."""
        assert secure_agg.encryption_enabled == security_config['encryption']
        assert secure_agg.compression_enabled == security_config['compression']
    
    def test_model_compression(self, secure_agg):
        """Test model update compression."""
        # Create model update
        model_update = {
            'layer1.weight': torch.randn(10, 10),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(5, 10),
            'layer2.bias': torch.randn(5)
        }
        
        compressed_update = secure_agg.compress_model_update(model_update, compression_ratio=0.5)
        
        # Check that compression was applied
        for param_name, param_tensor in compressed_update.items():
            # Most values should be zero (sparsified)
            sparsity = (param_tensor == 0).float().mean()
            assert sparsity > 0.3  # At least 30% sparsity
    
    def test_model_decompression(self, secure_agg):
        """Test model update decompression."""
        model_update = {
            'layer1.weight': torch.randn(10, 10),
            'layer2.weight': torch.randn(5, 10)
        }
        
        compressed_update = secure_agg.compress_model_update(model_update)
        decompressed_update = secure_agg.decompress_model_update(compressed_update)
        
        # For simple top-k sparsification, decompression should return the same
        for param_name in model_update:
            assert torch.allclose(compressed_update[param_name], decompressed_update[param_name])
    
    def test_tensor_serialization(self, secure_agg):
        """Test tensor serialization for encryption."""
        tensor = torch.randn(5, 5)
        
        # Serialize
        tensor_bytes = secure_agg._tensor_to_bytes(tensor)
        assert isinstance(tensor_bytes, bytes)
        
        # Deserialize
        reconstructed_tensor = secure_agg._bytes_to_tensor(tensor_bytes)
        assert torch.allclose(tensor, reconstructed_tensor)
    
    def test_disabled_encryption(self):
        """Test behavior when encryption is disabled."""
        config = {'encryption': False}
        secure_agg = SecureAggregation(config)
        
        model_update = {'layer1.weight': torch.randn(5, 5)}
        public_key = b"dummy_key"
        
        # Encryption should return original update
        encrypted_update = secure_agg.encrypt_model_update(model_update, public_key)
        assert encrypted_update == model_update
    
    def test_disabled_compression(self):
        """Test behavior when compression is disabled."""
        config = {'compression': False}
        secure_agg = SecureAggregation(config)
        
        model_update = {'layer1.weight': torch.randn(5, 5)}
        
        # Compression should return original update
        compressed_update = secure_agg.compress_model_update(model_update)
        assert compressed_update == model_update


class TestPrivacyManager:
    """Test privacy manager integration."""
    
    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration for testing."""
        return {
            'differential_privacy': True,
            'epsilon': 0.5,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'max_grad_norm': 1.0,
            'encryption': True,
            'compression': True
        }
    
    @pytest.fixture
    def privacy_manager(self, privacy_config):
        """Create privacy manager instance."""
        return PrivacyManager(privacy_config)
    
    def test_initialization(self, privacy_manager, privacy_config):
        """Test privacy manager initialization."""
        assert privacy_manager.dp.enabled == privacy_config['differential_privacy']
        assert privacy_manager.secure_agg.encryption_enabled == privacy_config['encryption']
        assert privacy_manager.secure_agg.compression_enabled == privacy_config['compression']
    
    def test_gradient_processing(self, privacy_manager):
        """Test gradient processing pipeline."""
        gradients = [torch.randn(5, 5) for _ in range(3)]
        original_gradients = [g.clone() for g in gradients]
        
        processed_gradients = privacy_manager.process_gradients(gradients, sensitivity=1.0)
        
        # Gradients should be processed (clipped and noisy)
        for orig, processed in zip(original_gradients, processed_gradients):
            assert not torch.allclose(orig, processed)
    
    def test_secure_aggregation(self, privacy_manager):
        """Test secure aggregation of model updates."""
        # Create multiple model updates
        model_updates = []
        for i in range(3):
            update = {
                'layer1.weight': torch.randn(5, 5) + i * 0.1,
                'layer1.bias': torch.randn(5) + i * 0.1
            }
            model_updates.append(update)
        
        # Aggregate updates
        aggregated_update = privacy_manager.secure_aggregate_updates(model_updates)
        
        # Check that aggregation worked
        assert 'layer1.weight' in aggregated_update
        assert 'layer1.bias' in aggregated_update
        
        # Check that aggregated values are reasonable
        for param_name in aggregated_update:
            param_tensor = aggregated_update[param_name]
            assert param_tensor.shape == model_updates[0][param_name].shape
    
    def test_weighted_aggregation(self, privacy_manager):
        """Test weighted aggregation of model updates."""
        model_updates = [
            {'layer1.weight': torch.ones(3, 3)},
            {'layer1.weight': torch.ones(3, 3) * 2},
            {'layer1.weight': torch.ones(3, 3) * 3}
        ]
        
        weights = [0.5, 0.3, 0.2]
        
        aggregated_update = privacy_manager.secure_aggregate_updates(model_updates, weights)
        
        # Expected result: 0.5*1 + 0.3*2 + 0.2*3 = 1.7
        expected_value = 0.5 * 1 + 0.3 * 2 + 0.2 * 3
        assert torch.allclose(aggregated_update['layer1.weight'], 
                            torch.ones(3, 3) * expected_value)
    
    def test_privacy_status(self, privacy_manager):
        """Test privacy status reporting."""
        status = privacy_manager.get_privacy_status()
        
        assert 'differential_privacy_enabled' in status
        assert 'encryption_enabled' in status
        assert 'compression_enabled' in status
        assert 'privacy_budget_spent' in status
        assert 'privacy_budget_exceeded' in status
        assert 'target_epsilon' in status
        assert 'target_delta' in status
        assert 'noise_multiplier' in status


class TestPrivacyIntegration:
    """Integration tests for privacy mechanisms."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock PyTorch model."""
        model = Mock()
        model.parameters.return_value = [torch.randn(5, 5), torch.randn(5)]
        return model
    
    def test_end_to_end_privacy_pipeline(self, mock_model):
        """Test end-to-end privacy pipeline."""
        config = {
            'differential_privacy': True,
            'epsilon': 0.5,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'max_grad_norm': 1.0,
            'encryption': True,
            'compression': True
        }
        
        privacy_manager = PrivacyManager(config)
        
        # Setup model privacy
        privacy_manager.setup_model_privacy(mock_model, sample_rate=0.1, epochs=10)
        
        # Process gradients
        gradients = [torch.randn(5, 5), torch.randn(5)]
        processed_gradients = privacy_manager.process_gradients(gradients)
        
        # Create model updates
        model_updates = [
            {'layer1.weight': torch.randn(5, 5)},
            {'layer1.weight': torch.randn(5, 5)}
        ]
        
        # Aggregate updates
        aggregated_update = privacy_manager.secure_aggregate_updates(model_updates)
        
        # Check results
        assert len(processed_gradients) == len(gradients)
        assert 'layer1.weight' in aggregated_update
        
        # Check privacy status
        status = privacy_manager.get_privacy_status()
        assert status['differential_privacy_enabled']
        assert status['encryption_enabled']
        assert status['compression_enabled']
    
    def test_privacy_config_validation(self):
        """Test privacy configuration validation."""
        # Valid configuration
        valid_config = {
            'differential_privacy': True,
            'epsilon': 0.5,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'max_grad_norm': 1.0
        }
        
        dp = DifferentialPrivacy(valid_config)
        assert dp.enabled
        
        # Invalid configuration (negative epsilon)
        invalid_config = {
            'differential_privacy': True,
            'epsilon': -0.5,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'max_grad_norm': 1.0
        }
        
        # Should not raise error but disable privacy
        dp = DifferentialPrivacy(invalid_config)
        # Note: In a real implementation, you might want to validate and raise errors


if __name__ == "__main__":
    pytest.main([__file__]) 