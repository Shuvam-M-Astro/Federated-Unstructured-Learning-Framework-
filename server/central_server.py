"""
Central server for federated learning coordination.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set
import websockets
from websockets.server import serve
import torch
import numpy as np
from collections import defaultdict, deque

from utils.config_loader import config
from models.model_factory import ModelFactory
from privacy.differential_privacy import PrivacyManager
from monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class FederatedServer:
    """Central server for federated learning."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize federated server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_info: Dict[str, Dict[str, Any]] = {}
        self.model_updates: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Rate limiting
        self.client_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_window = 50
        
        # Load configuration
        self.server_config = config.get_server_config()
        self.training_config = config.get_training_config()
        self.privacy_config = config.get_privacy_config()
        self.model_config = config.get_model_config()
        
        # Initialize components
        self.privacy_manager = PrivacyManager(self.privacy_config)
        self.metrics_collector = MetricsCollector()
        
        # Global model
        self.global_model = None
        self.current_round = 0
        self.max_rounds = self.training_config.get('max_rounds', 100)
        self.min_clients = self.server_config.get('min_clients', 2)
        
        # Training state
        self.training_active = False
        self.round_start_time = None
        self.round_timeout = self.server_config.get('timeout', 300)
        
        # Server start time for uptime tracking
        self.start_time = time.time()
        
        # Message handlers
        self.message_handlers = {
            'register': self._handle_register,
            'model_update': self._handle_model_update,
            'training_metrics': self._handle_training_metrics,
            'status_update': self._handle_status_update,
            'pong': self._handle_pong
        }
    
    async def start(self):
        """Start the federated server."""
        logger.info(f"Starting federated server on {self.host}:{self.port}")
        
        # Initialize global model
        self.global_model = ModelFactory.create_model(
            self.model_config.get('type', 'cnn'),
            self.model_config
        )
        
        # Start WebSocket server with increased message size limit
        async with serve(
            self._handle_client, 
            self.host, 
            self.port,
            max_size=100 * 1024 * 1024  # 100MB limit
        ):
            logger.info(f"Server listening on ws://{self.host}:{self.port}")
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._training_coordinator())
            
            # Keep server running
            await asyncio.Future()  # Run forever
    
    async def _handle_client(self, websocket, path):
        """Handle client connection.
        
        Args:
            websocket: WebSocket connection
            path: Request path
        """
        client_id = None
        
        try:
            # Get client ID from headers
            client_id = websocket.request_headers.get('Client-ID')
            if not client_id:
                client_id = f"client_{len(self.clients)}"
            
            logger.info(f"Client {client_id} connected")
            
            # Register client
            self.clients[client_id] = websocket
            self.client_info[client_id] = {
                'connected_at': time.time(),
                'last_heartbeat': time.time(),
                'status': 'connected',
                'data_info': None
            }
            
            # Send initial model
            await self._send_model_to_client(client_id)
            
            # Handle messages from client
            async for message in websocket:
                await self._handle_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if client_id:
                await self._handle_client_disconnect(client_id)
    
    async def _handle_message(self, client_id: str, message: str):
        """Handle message from client.
        
        Args:
            client_id: Client identifier
            message: Raw message string
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit(client_id):
                logger.warning(f"Rate limit exceeded for client {client_id}")
                await self._send_to_client(client_id, 'error', {
                    'error': 'Rate limit exceeded',
                    'retry_after': self.rate_limit_window
                })
                return
            
            parsed_message = json.loads(message)
            message_type = parsed_message.get('type')
            data = parsed_message.get('data', {})
            
            logger.debug(f"Received {message_type} from {client_id}")
            
            # Update last heartbeat
            self.client_info[client_id]['last_heartbeat'] = time.time()
            
            # Handle message
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](client_id, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message from {client_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if within rate limit, False otherwise
        """
        current_time = time.time()
        client_queue = self.client_requests[client_id]
        
        # Remove old requests outside the window
        while client_queue and current_time - client_queue[0] > self.rate_limit_window:
            client_queue.popleft()
        
        # Check if adding this request would exceed limit
        if len(client_queue) >= self.max_requests_per_window:
            return False
        
        # Add current request
        client_queue.append(current_time)
        return True
    
    async def _handle_register(self, client_id: str, data: Dict[str, Any]):
        """Handle client registration.
        
        Args:
            client_id: Client identifier
            data: Registration data
        """
        self.client_info[client_id].update({
            'data_info': data.get('data_info'),
            'capabilities': data.get('capabilities', {}),
            'status': 'ready'
        })
        
        logger.info(f"Client {client_id} registered with data info: {data.get('data_info')}")
        
        # Send acknowledgment
        await self._send_to_client(client_id, 'register_ack', {
            'client_id': client_id,
            'server_config': {
                'training': self.training_config,
                'privacy': self.privacy_config
            }
        })
    
    async def _handle_model_update(self, client_id: str, data: Dict[str, Any]):
        """Handle model update from client.
        
        Args:
            client_id: Client identifier
            data: Model update data
        """
        try:
            # Convert serialized model update back to tensors
            model_update = {}
            serializable_update = data.get('model_update', {})
            
            for param_name, param_data in serializable_update.items():
                tensor_data = np.array(param_data['data'])
                model_update[param_name] = torch.from_numpy(tensor_data)
            
            # Store model update
            self.model_updates[client_id] = model_update
            
            # Update client status
            self.client_info[client_id]['status'] = 'completed_training'
            
            logger.info(f"Received model update from {client_id}")
            
            # Check if we can proceed with aggregation
            await self._check_aggregation_ready()
            
        except Exception as e:
            logger.error(f"Error processing model update from {client_id}: {e}")
    
    async def _handle_training_metrics(self, client_id: str, data: Dict[str, Any]):
        """Handle training metrics from client.
        
        Args:
            client_id: Client identifier
            data: Training metrics
        """
        metrics = data.get('metrics', {})
        self.metrics_collector.add_client_metrics(client_id, metrics)
        
        logger.debug(f"Received metrics from {client_id}: {metrics}")
    
    async def _handle_status_update(self, client_id: str, data: Dict[str, Any]):
        """Handle status update from client.
        
        Args:
            client_id: Client identifier
            data: Status data
        """
        status = data.get('status')
        details = data.get('details', {})
        
        self.client_info[client_id].update({
            'status': status,
            'last_status_update': time.time(),
            'status_details': details
        })
        
        logger.debug(f"Client {client_id} status: {status}")
    
    async def _handle_pong(self, client_id: str, data: Dict[str, Any]):
        """Handle pong from client.
        
        Args:
            client_id: Client identifier
            data: Pong data
        """
        self.client_info[client_id]['last_heartbeat'] = time.time()
    
    async def _handle_client_disconnect(self, client_id: str):
        """Handle client disconnection.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.clients:
            del self.clients[client_id]
        
        if client_id in self.client_info:
            self.client_info[client_id]['status'] = 'disconnected'
        
        if client_id in self.model_updates:
            del self.model_updates[client_id]
        
        logger.info(f"Client {client_id} disconnected")
    
    async def _send_to_client(self, client_id: str, message_type: str, data: Dict[str, Any]):
        """Send message to specific client.
        
        Args:
            client_id: Client identifier
            message_type: Type of message
            data: Message data
        """
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not found")
            return
        
        try:
            message = {
                'type': message_type,
                'timestamp': time.time(),
                'data': data
            }
            
            await self.clients[client_id].send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
    
    async def _broadcast_to_clients(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients.
        
        Args:
            message_type: Type of message
            data: Message data
        """
        message = {
            'type': message_type,
            'timestamp': time.time(),
            'data': data
        }
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for client_id, websocket in self.clients.items():
            try:
                await websocket.send(message_str)
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self._handle_client_disconnect(client_id)
    
    async def _send_model_to_client(self, client_id: str):
        """Send current global model to client.
        
        Args:
            client_id: Client identifier
        """
        if self.global_model is None:
            return
        
        # Convert model to serializable format with compression
        model_state = {}
        for param_name, param_tensor in self.global_model.state_dict().items():
            # Convert to numpy and compress by keeping only significant values
            tensor_np = param_tensor.detach().cpu().numpy()
            
            # Simple compression: keep only top 10% of values by magnitude
            flat_tensor = tensor_np.flatten()
            threshold = np.percentile(np.abs(flat_tensor), 90)  # Keep top 10%
            mask = np.abs(flat_tensor) >= threshold
            compressed_tensor = flat_tensor * mask
            
            model_state[param_name] = {
                'data': compressed_tensor.tolist(),
                'shape': list(tensor_np.shape),
                'dtype': str(param_tensor.dtype),
                'compressed': True,
                'threshold': float(threshold)
            }
        
        await self._send_to_client(client_id, 'model_request', {
            'model_state': model_state,
            'round': self.current_round,
            'training_config': self.training_config
        })
    
    async def _training_coordinator(self):
        """Coordinate federated training rounds."""
        while True:
            try:
                # Wait for enough clients
                if len(self.clients) < self.min_clients:
                    await asyncio.sleep(5)
                    continue
                
                # Start new round
                if not self.training_active:
                    await self._start_training_round()
                
                # Check round timeout
                if self.training_active and self.round_start_time:
                    elapsed = time.time() - self.round_start_time
                    if elapsed > self.round_timeout:
                        logger.warning(f"Round {self.current_round} timed out")
                        await self._end_training_round()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in training coordinator: {e}")
                await asyncio.sleep(5)
    
    async def _start_training_round(self):
        """Start a new training round."""
        if self.current_round >= self.max_rounds:
            logger.info("Maximum rounds reached")
            return
        
        self.current_round += 1
        self.training_active = True
        self.round_start_time = time.time()
        self.model_updates.clear()
        
        logger.info(f"Starting training round {self.current_round}")
        
        # Send model to all clients
        await self._broadcast_to_clients('training_request', {
            'round': self.current_round,
            'training_config': self.training_config
        })
        
        # Record round start
        self.metrics_collector.record_round_start(self.current_round, len(self.clients))
    
    async def _end_training_round(self):
        """End current training round."""
        if not self.training_active:
            return
        
        self.training_active = False
        self.round_start_time = None
        
        logger.info(f"Ending training round {self.current_round}")
        
        # Aggregate model updates
        if len(self.model_updates) > 0:
            await self._aggregate_models()
        
        # Record round end
        self.metrics_collector.record_round_end(self.current_round)
    
    async def _check_aggregation_ready(self):
        """Check if we have enough model updates to proceed with aggregation."""
        if not self.training_active:
            return
        
        # Check if we have updates from all active clients
        active_clients = [
            client_id for client_id, info in self.client_info.items()
            if info['status'] in ['ready', 'training', 'completed_training']
        ]
        
        clients_with_updates = set(self.model_updates.keys())
        active_clients_set = set(active_clients)
        
        if clients_with_updates.issuperset(active_clients_set):
            logger.info("All active clients have submitted updates")
            await self._end_training_round()
    
    async def _aggregate_models(self):
        """Aggregate model updates from clients."""
        if len(self.model_updates) == 0:
            logger.warning("No model updates to aggregate")
            return
        
        try:
            # Get model updates as list
            updates_list = list(self.model_updates.values())
            
            # Aggregate updates
            aggregated_update = self.privacy_manager.secure_aggregate_updates(updates_list)
            
            # Apply aggregated update to global model
            with torch.no_grad():
                for param_name, param_tensor in self.global_model.named_parameters():
                    if param_name in aggregated_update:
                        param_tensor.data += aggregated_update[param_name]
            
            logger.info(f"Model aggregated from {len(updates_list)} clients")
            
            # Record aggregation metrics
            self.metrics_collector.record_aggregation(
                self.current_round,
                len(updates_list),
                self.privacy_manager.get_privacy_status()
            )
            
        except Exception as e:
            logger.error(f"Error aggregating models: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to clients."""
        while True:
            try:
                # Send ping to all clients
                await self._broadcast_to_clients('ping', {'timestamp': time.time()})
                
                # Check for stale clients
                current_time = time.time()
                stale_clients = []
                
                for client_id, info in self.client_info.items():
                    if current_time - info['last_heartbeat'] > 60:  # 60 second timeout
                        stale_clients.append(client_id)
                
                # Disconnect stale clients
                for client_id in stale_clients:
                    logger.warning(f"Client {client_id} is stale, disconnecting")
                    await self._handle_client_disconnect(client_id)
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status information.
        
        Returns:
            Server status dictionary
        """
        active_clients = len([c for c in self.client_info.values() if c.get('status') == 'ready'])
        total_clients = len(self.clients)
        
        return {
            'status': 'running' if self.training_active else 'idle',
            'current_round': self.current_round,
            'max_rounds': self.max_rounds,
            'active_clients': active_clients,
            'total_clients': total_clients,
            'min_clients': self.min_clients,
            'training_active': self.training_active,
            'uptime': time.time() - self.start_time,
            'memory_usage': self._get_memory_usage(),
            'model_updates_pending': len(self.model_updates)
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Memory usage information
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the server.
        
        Returns:
            Health check results
        """
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        # Check client connections
        active_clients = len([c for c in self.client_info.values() if c.get('status') == 'ready'])
        if active_clients < self.min_clients:
            health_status['status'] = 'degraded'
            health_status['checks']['clients'] = f'Insufficient clients: {active_clients}/{self.min_clients}'
        
        # Check memory usage
        memory_usage = self._get_memory_usage()
        if isinstance(memory_usage, dict) and 'percent' in memory_usage:
            if memory_usage['percent'] > 85:
                health_status['status'] = 'degraded'
                health_status['checks']['memory'] = f'High memory usage: {memory_usage["percent"]:.1f}%'
        
        return health_status


async def main():
    print(">>> main() in central_server.py is running")
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config.validate_config()
    
    # Create and start server
    server = FederatedServer(
        host=config.get('server.host', 'localhost'),
        port=config.get('server.port', 8000)
    )
    
    await server.start()


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Server crashed with exception: {e}") 