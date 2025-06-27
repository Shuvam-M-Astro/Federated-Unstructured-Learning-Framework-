#!/usr/bin/env python3
"""
Test server for validating federated learning server functionality.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any
import websockets
import torch
import numpy as np

logger = logging.getLogger(__name__)


class TestFederatedServer:
    """Test server for federated learning validation."""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        """Initialize test server.
        
        Args:
            host: Server host
            port: Server port (different from main server)
        """
        self.host = host
        self.port = port
        self.clients = {}
        self.test_results = {
            'connection_tests': [],
            'message_tests': [],
            'model_tests': [],
            'aggregation_tests': []
        }
    
    async def start(self):
        """Start the test server."""
        logger.info(f"Starting test server on {self.host}:{self.port}")
        
        try:
            async with websockets.serve(self._handle_client, self.host, self.port):
                logger.info(f"Test server listening on ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(f"Port {self.port} is already in use. Please use a different port.")
            else:
                logger.error(f"Failed to start test server: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error starting test server: {e}")
            raise
    
    async def _handle_client(self, websocket, path):
        """Handle test client connection."""
        client_id = f"test_client_{len(self.clients)}"
        
        try:
            logger.info(f"Test client {client_id} connected")
            self.clients[client_id] = websocket
            
            # Test basic message handling
            await self._test_message_handling(client_id, websocket)
            
            # Test model operations
            await self._test_model_operations(client_id, websocket)
            
            # Keep connection alive for additional tests
            async for message in websocket:
                await self._handle_test_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Test client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error in test client {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def _test_message_handling(self, client_id: str, websocket):
        """Test basic message handling functionality."""
        try:
            # Test registration message
            registration_msg = {
                'type': 'register',
                'data': {
                    'data_info': {
                        'num_samples': 1000,
                        'data_type': 'image',
                        'features': [3, 224, 224]
                    },
                    'capabilities': {
                        'privacy_enabled': True,
                        'encryption_enabled': True
                    }
                }
            }
            
            await websocket.send(json.dumps(registration_msg))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            if response_data.get('type') == 'register_ack':
                self.test_results['message_tests'].append({
                    'test': 'registration',
                    'status': 'passed',
                    'timestamp': time.time()
                })
                logger.info("Registration test passed")
            else:
                self.test_results['message_tests'].append({
                    'test': 'registration',
                    'status': 'failed',
                    'timestamp': time.time()
                })
                logger.error("Registration test failed")
                
        except Exception as e:
            self.test_results['message_tests'].append({
                'test': 'registration',
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            })
            logger.error(f"Registration test error: {e}")
    
    async def _test_model_operations(self, client_id: str, websocket):
        """Test model-related operations."""
        try:
            # Create dummy model update
            dummy_update = {
                'conv1.weight': {
                    'data': np.random.randn(64, 3, 7, 7).tolist(),
                    'shape': [64, 3, 7, 7]
                },
                'conv1.bias': {
                    'data': np.random.randn(64).tolist(),
                    'shape': [64]
                }
            }
            
            # Test model update message
            model_update_msg = {
                'type': 'model_update',
                'data': {
                    'model_update': dummy_update,
                    'round': 1,
                    'metrics': {
                        'loss': 0.5,
                        'accuracy': 0.85,
                        'num_batches': 10
                    }
                }
            }
            
            await websocket.send(json.dumps(model_update_msg))
            
            self.test_results['model_tests'].append({
                'test': 'model_update',
                'status': 'sent',
                'timestamp': time.time()
            })
            logger.info("Model update test sent")
            
        except Exception as e:
            self.test_results['model_tests'].append({
                'test': 'model_update',
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            })
            logger.error(f"Model update test error: {e}")
    
    async def _handle_test_message(self, client_id: str, message: str):
        """Handle test messages from client."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'test_result':
                test_name = data.get('data', {}).get('test_name')
                test_status = data.get('data', {}).get('status')
                
                self.test_results['message_tests'].append({
                    'test': test_name,
                    'status': test_status,
                    'timestamp': time.time()
                })
                
                logger.info(f"Test result: {test_name} - {test_status}")
                
        except Exception as e:
            logger.error(f"Error handling test message: {e}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test results summary."""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'test_categories': {}
        }
        
        for category, tests in self.test_results.items():
            category_summary = {
                'total': len(tests),
                'passed': len([t for t in tests if t.get('status') == 'passed']),
                'failed': len([t for t in tests if t.get('status') == 'failed']),
                'error': len([t for t in tests if t.get('status') == 'error'])
            }
            
            summary['test_categories'][category] = category_summary
            summary['total_tests'] += category_summary['total']
            summary['passed_tests'] += category_summary['passed']
            summary['failed_tests'] += category_summary['failed']
            summary['error_tests'] += category_summary['error']
        
        return summary


async def main():
    """Main function to run the test server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start test server
    test_server = TestFederatedServer()
    
    try:
        await test_server.start()
    except KeyboardInterrupt:
        logger.info("Test server stopped by user")
    except Exception as e:
        logger.error(f"Test server error: {e}")
    finally:
        # Print test summary
        summary = test_server.get_test_summary()
        logger.info("Test Summary:")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Errors: {summary['error_tests']}")


if __name__ == "__main__":
    asyncio.run(main()) 