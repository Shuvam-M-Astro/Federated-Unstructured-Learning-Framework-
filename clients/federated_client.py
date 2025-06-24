"""
Federated learning client implementation.
"""

import asyncio
import json
import logging
import time
import argparse
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils.config_loader import config
from data_processing.data_loader import DataProcessor, UnstructuredDataset
from models.model_factory import ModelFactory
from privacy.differential_privacy import PrivacyManager
from communication.websocket_client import ConnectionManager

logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, data_path: str, data_type: str,
                 server_url: str = "ws://localhost:8000"):
        """Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            data_path: Path to local data
            data_type: Type of data ('text', 'image', 'tabular', 'mixed')
            server_url: WebSocket server URL
        """
        self.client_id = client_id
        self.data_path = data_path
        self.data_type = data_type
        self.server_url = server_url
        
        # Load configuration
        self.training_config = config.get_training_config()
        self.privacy_config = config.get_privacy_config()
        self.model_config = config.get_model_config()
        self.data_config = config.get_data_processing_config()
        
        # Initialize components
        self.data_processor = DataProcessor(self.data_config)
        self.privacy_manager = PrivacyManager(self.privacy_config)
        self.connection_manager = ConnectionManager(server_url, client_id)
        
        # Local model and data
        self.local_model = None
        self.dataloader = None
        self.optimizer = None
        self.criterion = None
        
        # Training state
        self.current_round = 0
        self.training_active = False
        self.model_received = False
        
        # Message handlers
        self.message_handlers = {
            'model_request': self._handle_model_request,
            'training_request': self._handle_training_request,
            'register_ack': self._handle_register_ack,
            'ping': self._handle_ping
        }
    
    async def start(self):
        """Start the federated client."""
        logger.info(f"Starting federated client {self.client_id}")
        
        # Validate and load data
        if not self.data_processor.validate_data(self.data_path, self.data_type):
            raise ValueError(f"Invalid data at {self.data_path}")
        
        # Get data information
        data_info = self.data_processor.get_data_info(self.data_path, self.data_type)
        logger.info(f"Data info: {data_info}")
        
        # Create dataloader
        self.dataloader = self.data_processor.create_dataloader(
            data_path=self.data_path,
            data_type=self.data_type,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=True
        )
        
        # Initialize local model (will be updated with global model)
        self.local_model = ModelFactory.create_model(
            self.model_config.get('type', 'cnn'),
            self.model_config
        )
        
        # Setup optimizer and loss function
        self.optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.training_config.get('learning_rate', 0.001),
            momentum=self.training_config.get('momentum', 0.9),
            weight_decay=self.training_config.get('weight_decay', 0.0001)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup privacy mechanisms (temporarily disabled for testing)
        try:
            batch_size = float(self.training_config.get('batch_size', 32))
            dataset_size = len(self.dataloader.dataset)
            sample_rate = batch_size / dataset_size if dataset_size > 0 else 0.1
            epochs = int(self.training_config.get('epochs', 10))
            self.privacy_manager.setup_model_privacy(
                self.local_model, 
                sample_rate, 
                epochs,
                optimizer=self.optimizer,
                data_loader=self.dataloader
            )
        except Exception as e:
            logger.warning(f"Privacy engine setup failed, continuing without privacy: {e}")
            # Disable privacy features
            self.privacy_config['differential_privacy'] = False
        
        # Start connection manager
        success = await self.connection_manager.start()
        if not success:
            raise ConnectionError("Failed to connect to server")
        
        # Register with server
        await self._register_with_server(data_info)
        
        # Start message processing loop
        await self._message_processing_loop()
    
    async def stop(self):
        """Stop the federated client."""
        logger.info(f"Stopping federated client {self.client_id}")
        await self.connection_manager.stop()
    
    async def _register_with_server(self, data_info: Dict[str, Any]):
        """Register with the federated server.
        
        Args:
            data_info: Information about local data
        """
        registration_data = {
            'data_info': data_info,
            'capabilities': {
                'data_type': self.data_type,
                'privacy_enabled': self.privacy_config.get('differential_privacy', True),
                'encryption_enabled': self.privacy_config.get('encryption', True)
            }
        }
        
        # Send registration message
        if self.connection_manager.is_connected():
            await self.connection_manager.client.send_message('register', registration_data)
        else:
            # Queue for later
            await self.connection_manager.message_queue.put({
                'type': 'register',
                'data': registration_data
            })
    
    async def _message_processing_loop(self):
        """Main message processing loop."""
        while True:
            try:
                # Get pending messages
                messages = await self.connection_manager.get_pending_messages()
                
                for message in messages:
                    message_type = message.get('type')
                    data = message.get('data', {})
                    
                    if message_type in self.message_handlers:
                        await self.message_handlers[message_type](data)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                
                # Check if we need to send status updates
                if self.connection_manager.is_connected():
                    await self._send_status_update()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _handle_model_request(self, data: Dict[str, Any]):
        """Handle model request from server.
        
        Args:
            data: Model request data
        """
        try:
            # Extract model state
            model_state = data.get('model_state', {})
            
            # Convert serialized model state back to tensors
            state_dict = {}
            for param_name, param_data in model_state.items():
                tensor_data = np.array(param_data['data'])
                original_shape = param_data['shape']
                
                # Reshape back to original shape
                tensor_data = tensor_data.reshape(original_shape)
                
                # Convert to torch tensor
                state_dict[param_name] = torch.from_numpy(tensor_data).float()
            
            # Load model state
            self.local_model.load_state_dict(state_dict, strict=False)
            self.model_received = True
            
            logger.info("Received global model from server")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    async def _handle_training_request(self, data: Dict[str, Any]):
        """Handle training request from server.
        
        Args:
            data: Training request data
        """
        if not self.model_received:
            logger.warning("No model received, cannot start training")
            return
        
        round_number = data.get('round', 0)
        training_config = data.get('training_config', {})
        
        logger.info(f"Starting training for round {round_number}")
        
        # Update training configuration
        self.training_config.update(training_config)
        
        # Start local training
        await self._train_local_model()
    
    async def _handle_register_ack(self, data: Dict[str, Any]):
        """Handle registration acknowledgment from server.
        
        Args:
            data: Registration acknowledgment data
        """
        logger.info("Registration acknowledged by server")
        
        # Update configuration if provided
        server_config = data.get('server_config', {})
        if 'training' in server_config:
            self.training_config.update(server_config['training'])
        if 'privacy' in server_config:
            self.privacy_config.update(server_config['privacy'])
    
    async def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping from server.
        
        Args:
            data: Ping data
        """
        # Respond with pong
        if self.connection_manager.is_connected():
            await self.connection_manager.client.send_message('pong', {
                'timestamp': time.time()
            })
    
    async def _train_local_model(self):
        """Train the local model on local data."""
        if self.dataloader is None:
            logger.error("No dataloader available")
            return
        
        try:
            self.local_model.train()
            total_loss = 0.0
            num_batches = 0
            
            logger.info("Starting local training")
            
            for batch_idx, (data, labels, metadata) in enumerate(self.dataloader):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.local_model(data)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply privacy mechanisms to gradients
                gradients = [p.grad for p in self.local_model.parameters()]
                processed_gradients = self.privacy_manager.process_gradients(gradients)
                
                # Update gradients
                for param, grad in zip(self.local_model.parameters(), processed_gradients):
                    if param.grad is not None and grad is not None:
                        param.grad.data = grad.data
                
                # Optimizer step
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Calculate training metrics
            metrics = {
                'loss': avg_loss,
                'num_batches': num_batches,
                'privacy_epsilon': self.privacy_manager.get_privacy_status()['privacy_budget_spent']['epsilon']
            }
            
            logger.info(f"Local training completed. Avg loss: {avg_loss:.4f}")
            
            # Send metrics to server
            await self.connection_manager.send_metrics(metrics)
            
            # Send model update to server
            await self._send_model_update()
            
        except Exception as e:
            logger.error(f"Error during local training: {e}")
    
    async def _send_model_update(self):
        """Send model update to server."""
        try:
            # Calculate model update (difference from received model)
            model_update = {}
            
            # For simplicity, we'll send the current model state with compression
            for param_name, param_tensor in self.local_model.state_dict().items():
                # Compress the tensor by keeping only significant values
                tensor_np = param_tensor.detach().cpu().numpy()
                flat_tensor = tensor_np.flatten()
                
                # Keep only top 10% of values by magnitude
                threshold = np.percentile(np.abs(flat_tensor), 90)
                mask = np.abs(flat_tensor) >= threshold
                compressed_tensor = flat_tensor * mask
                
                model_update[param_name] = torch.from_numpy(
                    compressed_tensor.reshape(tensor_np.shape)
                ).float()
            
            # Add metadata
            metadata = {
                'client_id': self.client_id,
                'round': self.current_round,
                'data_type': self.data_type,
                'num_samples': len(self.dataloader.dataset),
                'privacy_status': self.privacy_manager.get_privacy_status(),
                'compressed': True
            }
            
            # Send update
            success = await self.connection_manager.send_model_update(model_update, metadata)
            
            if success:
                logger.info("Model update sent to server")
            else:
                logger.warning("Failed to send model update")
            
        except Exception as e:
            logger.error(f"Error sending model update: {e}")
    
    async def _send_status_update(self):
        """Send status update to server."""
        status = 'ready' if self.model_received else 'waiting_for_model'
        
        details = {
            'client_id': self.client_id,
            'data_type': self.data_type,
            'num_samples': len(self.dataloader.dataset) if self.dataloader else 0,
            'privacy_enabled': self.privacy_config.get('differential_privacy', True)
        }
        
        if self.connection_manager.is_connected():
            await self.connection_manager.client.send_status_update(status, details)
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get current client status.
        
        Returns:
            Dictionary with client status
        """
        return {
            'client_id': self.client_id,
            'data_type': self.data_type,
            'data_path': self.data_path,
            'connected': self.connection_manager.is_connected(),
            'model_received': self.model_received,
            'current_round': self.current_round,
            'privacy_status': self.privacy_manager.get_privacy_status()
        }


async def main():
    """Main function to start the federated client."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--node-id', type=str, required=True, help='Unique client identifier')
    parser.add_argument('--data-path', type=str, required=True, help='Path to local data')
    parser.add_argument('--data-type', type=str, default='image', 
                       choices=['text', 'image', 'tabular', 'mixed'],
                       help='Type of data')
    parser.add_argument('--server-url', type=str, default='ws://localhost:8000',
                       help='WebSocket server URL')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config.validate_config()
    
    # Create and start client
    client = FederatedClient(
        client_id=args.node_id,
        data_path=args.data_path,
        data_type=args.data_type,
        server_url=args.server_url
    )
    
    try:
        await client.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main()) 