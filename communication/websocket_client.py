"""
WebSocket client for federated learning communication.
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Dict, Any, Optional, Callable
import websockets
import torch
import numpy as np

logger = logging.getLogger(__name__)


class FederatedWebSocketClient:
    """WebSocket client for federated learning communication."""
    
    def __init__(self, server_url: str, client_id: str, 
                 on_message: Optional[Callable] = None,
                 on_connect: Optional[Callable] = None,
                 on_disconnect: Optional[Callable] = None):
        """Initialize WebSocket client.
        
        Args:
            server_url: WebSocket server URL
            client_id: Unique client identifier
            on_message: Callback for incoming messages
            on_connect: Callback for connection events
            on_disconnect: Callback for disconnection events
        """
        self.server_url = server_url
        self.client_id = client_id
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        
        self.websocket = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        
        # Message handlers
        self.message_handlers = {
            'model_request': self._handle_model_request,
            'training_request': self._handle_training_request,
            'aggregation_request': self._handle_aggregation_request,
            'status_request': self._handle_status_request,
            'ping': self._handle_ping
        }
    
    async def connect(self) -> bool:
        """Connect to the WebSocket server.
        
        Returns:
            True if connection successful
        """
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                extra_headers={'Client-ID': self.client_id}
            )
            self.connected = True
            self.reconnect_attempts = 0
            
            logger.info(f"Connected to server: {self.server_url}")
            
            if self.on_connect:
                await self.on_connect()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from server")
            
            if self.on_disconnect:
                await self.on_disconnect()
    
    async def send_message(self, message_type: str, data: Dict[str, Any]) -> bool:
        """Send message to server.
        
        Args:
            message_type: Type of message
            data: Message data
            
        Returns:
            True if message sent successfully
        """
        if not self.connected:
            logger.warning("Not connected to server")
            return False
        
        try:
            message = {
                'type': message_type,
                'client_id': self.client_id,
                'timestamp': time.time(),
                'data': data
            }
            
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def send_model_update(self, model_update: Dict[str, torch.Tensor], 
                              metadata: Dict[str, Any]) -> bool:
        """Send model update to server.
        
        Args:
            model_update: Model parameter updates
            metadata: Additional metadata
            
        Returns:
            True if update sent successfully
        """
        # Convert tensors to serializable format
        serializable_update = {}
        for param_name, param_tensor in model_update.items():
            serializable_update[param_name] = {
                'data': param_tensor.detach().cpu().numpy().tolist(),
                'shape': list(param_tensor.shape),
                'dtype': str(param_tensor.dtype)
            }
        
        data = {
            'model_update': serializable_update,
            'metadata': metadata
        }
        
        return await self.send_message('model_update', data)
    
    async def send_training_metrics(self, metrics: Dict[str, float]) -> bool:
        """Send training metrics to server.
        
        Args:
            metrics: Training metrics
            
        Returns:
            True if metrics sent successfully
        """
        data = {
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        return await self.send_message('training_metrics', data)
    
    async def send_status_update(self, status: str, details: Dict[str, Any]) -> bool:
        """Send status update to server.
        
        Args:
            status: Status string
            details: Status details
            
        Returns:
            True if status sent successfully
        """
        data = {
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        
        return await self.send_message('status_update', data)
    
    async def listen(self) -> None:
        """Listen for incoming messages."""
        if not self.connected:
            logger.error("Not connected to server")
            return
        
        try:
            async for message in self.websocket:
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed by server")
            self.connected = False
            await self._handle_reconnection()
            
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            self.connected = False
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming message.
        
        Args:
            message: Raw message string
        """
        try:
            parsed_message = json.loads(message)
            message_type = parsed_message.get('type')
            data = parsed_message.get('data', {})
            
            logger.debug(f"Received message: {message_type}")
            
            # Call custom message handler if provided
            if self.on_message:
                await self.on_message(message_type, data)
            
            # Call specific handler if available
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_model_request(self, data: Dict[str, Any]) -> None:
        """Handle model request from server.
        
        Args:
            data: Request data
        """
        logger.info("Received model request from server")
        # This should be implemented by the federated client
    
    async def _handle_training_request(self, data: Dict[str, Any]) -> None:
        """Handle training request from server.
        
        Args:
            data: Request data
        """
        logger.info("Received training request from server")
        # This should be implemented by the federated client
    
    async def _handle_aggregation_request(self, data: Dict[str, Any]) -> None:
        """Handle aggregation request from server.
        
        Args:
            data: Request data
        """
        logger.info("Received aggregation request from server")
        # This should be implemented by the federated client
    
    async def _handle_status_request(self, data: Dict[str, Any]) -> None:
        """Handle status request from server.
        
        Args:
            data: Request data
        """
        logger.info("Received status request from server")
        # Send current status
        await self.send_status_update('ready', {'client_id': self.client_id})
    
    async def _handle_ping(self, data: Dict[str, Any]) -> None:
        """Handle ping from server.
        
        Args:
            data: Ping data
        """
        # Respond with pong
        await self.send_message('pong', {'timestamp': time.time()})
    
    async def _handle_reconnection(self) -> None:
        """Handle reconnection logic."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        
        logger.info(f"Attempting to reconnect in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        success = await self.connect()
        if success:
            logger.info("Reconnection successful")
        else:
            await self._handle_reconnection()


class MessageQueue:
    """Message queue for handling asynchronous communication."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize message queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self.queue = asyncio.Queue(maxsize=max_size)
        self.running = False
    
    async def put(self, message: Dict[str, Any]) -> bool:
        """Put message in queue.
        
        Args:
            message: Message to queue
            
        Returns:
            True if message queued successfully
        """
        try:
            await self.queue.put(message)
            return True
        except asyncio.QueueFull:
            logger.warning("Message queue full")
            return False
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get message from queue.
        
        Returns:
            Message from queue or None if empty
        """
        try:
            return await self.queue.get()
        except asyncio.QueueEmpty:
            return None
    
    def start(self) -> None:
        """Start the message queue."""
        self.running = True
    
    def stop(self) -> None:
        """Stop the message queue."""
        self.running = False
    
    def is_running(self) -> bool:
        """Check if queue is running.
        
        Returns:
            True if queue is running
        """
        return self.running


class ConnectionManager:
    """Manager for WebSocket connections."""
    
    def __init__(self, server_url: str, client_id: str):
        """Initialize connection manager.
        
        Args:
            server_url: WebSocket server URL
            client_id: Unique client identifier
        """
        self.server_url = server_url
        self.client_id = client_id
        self.client = None
        self.message_queue = MessageQueue()
        self.connection_task = None
    
    async def start(self) -> bool:
        """Start the connection manager.
        
        Returns:
            True if started successfully
        """
        try:
            self.client = FederatedWebSocketClient(
                server_url=self.server_url,
                client_id=self.client_id,
                on_message=self._on_message,
                on_connect=self._on_connect,
                on_disconnect=self._on_disconnect
            )
            
            success = await self.client.connect()
            if success:
                self.message_queue.start()
                self.connection_task = asyncio.create_task(self.client.listen())
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to start connection manager: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the connection manager."""
        if self.connection_task:
            self.connection_task.cancel()
        
        if self.client:
            await self.client.disconnect()
        
        self.message_queue.stop()
    
    async def send_model_update(self, model_update: Dict[str, torch.Tensor], 
                              metadata: Dict[str, Any]) -> bool:
        """Send model update through the connection manager.
        
        Args:
            model_update: Model parameter updates
            metadata: Additional metadata
            
        Returns:
            True if update sent successfully
        """
        if self.client and self.client.connected:
            return await self.client.send_model_update(model_update, metadata)
        else:
            # Queue the message for later
            message = {
                'type': 'model_update',
                'model_update': model_update,
                'metadata': metadata
            }
            return await self.message_queue.put(message)
    
    async def send_metrics(self, metrics: Dict[str, float]) -> bool:
        """Send metrics through the connection manager.
        
        Args:
            metrics: Training metrics
            
        Returns:
            True if metrics sent successfully
        """
        if self.client and self.client.connected:
            return await self.client.send_training_metrics(metrics)
        else:
            # Queue the message for later
            message = {
                'type': 'training_metrics',
                'metrics': metrics
            }
            return await self.message_queue.put(message)
    
    async def _on_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """Handle incoming messages.
        
        Args:
            message_type: Type of message
            data: Message data
        """
        # Queue the message for processing
        message = {
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }
        await self.message_queue.put(message)
    
    async def _on_connect(self) -> None:
        """Handle connection events."""
        logger.info("Connected to federated learning server")
    
    async def _on_disconnect(self) -> None:
        """Handle disconnection events."""
        logger.info("Disconnected from federated learning server")
    
    def is_connected(self) -> bool:
        """Check if connected to server.
        
        Returns:
            True if connected
        """
        return self.client and self.client.connected
    
    async def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get all pending messages from queue.
        
        Returns:
            List of pending messages
        """
        messages = []
        while not self.message_queue.queue.empty():
            message = await self.message_queue.get()
            if message:
                messages.append(message)
        return messages 