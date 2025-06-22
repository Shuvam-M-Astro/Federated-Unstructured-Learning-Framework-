"""
Metrics collector for monitoring federated learning training.
"""

import time
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and manage training metrics for federated learning."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical records to keep
        """
        self.max_history = max_history
        
        # Round-level metrics
        self.round_metrics = {}
        self.round_timestamps = {}
        
        # Client-level metrics
        self.client_metrics = defaultdict(list)
        self.client_timestamps = defaultdict(list)
        
        # Aggregation metrics
        self.aggregation_metrics = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_rounds': 0,
            'total_clients': 0,
            'avg_round_time': 0.0,
            'avg_client_training_time': 0.0,
            'total_communication_bytes': 0,
            'privacy_budget_consumed': 0.0
        }
        
        # Real-time metrics (using deque for efficient sliding window)
        self.recent_metrics = deque(maxlen=max_history)
        
    def record_round_start(self, round_number: int, num_clients: int):
        """Record the start of a training round.
        
        Args:
            round_number: Current round number
            num_clients: Number of participating clients
        """
        timestamp = time.time()
        
        self.round_metrics[round_number] = {
            'start_time': timestamp,
            'num_clients': num_clients,
            'status': 'started'
        }
        
        self.round_timestamps[round_number] = timestamp
        
        logger.debug(f"Round {round_number} started with {num_clients} clients")
    
    def record_round_end(self, round_number: int):
        """Record the end of a training round.
        
        Args:
            round_number: Current round number
        """
        if round_number not in self.round_metrics:
            logger.warning(f"Round {round_number} not found in metrics")
            return
        
        end_time = time.time()
        start_time = self.round_metrics[round_number]['start_time']
        round_duration = end_time - start_time
        
        self.round_metrics[round_number].update({
            'end_time': end_time,
            'duration': round_duration,
            'status': 'completed'
        })
        
        # Update performance metrics
        self.performance_metrics['total_rounds'] += 1
        self.performance_metrics['avg_round_time'] = (
            (self.performance_metrics['avg_round_time'] * (self.performance_metrics['total_rounds'] - 1) + round_duration) 
            / self.performance_metrics['total_rounds']
        )
        
        logger.debug(f"Round {round_number} completed in {round_duration:.2f} seconds")
    
    def add_client_metrics(self, client_id: str, metrics: Dict[str, Any]):
        """Add metrics from a specific client.
        
        Args:
            client_id: Client identifier
            metrics: Client training metrics
        """
        timestamp = time.time()
        
        # Add timestamp to metrics
        metrics_with_timestamp = {
            **metrics,
            'timestamp': timestamp,
            'client_id': client_id
        }
        
        # Store client metrics
        self.client_metrics[client_id].append(metrics_with_timestamp)
        self.client_timestamps[client_id].append(timestamp)
        
        # Keep only recent history
        if len(self.client_metrics[client_id]) > self.max_history:
            self.client_metrics[client_id] = self.client_metrics[client_id][-self.max_history:]
            self.client_timestamps[client_id] = self.client_timestamps[client_id][-self.max_history:]
        
        # Add to recent metrics
        self.recent_metrics.append(metrics_with_timestamp)
        
        logger.debug(f"Added metrics from client {client_id}: {metrics}")
    
    def record_aggregation(self, round_number: int, num_updates: int, 
                          privacy_status: Dict[str, Any]):
        """Record model aggregation metrics.
        
        Args:
            round_number: Current round number
            num_updates: Number of model updates aggregated
            privacy_status: Privacy status information
        """
        timestamp = time.time()
        
        self.aggregation_metrics[round_number] = {
            'timestamp': timestamp,
            'num_updates': num_updates,
            'privacy_epsilon': privacy_status.get('privacy_budget_spent', {}).get('epsilon', 0.0),
            'privacy_delta': privacy_status.get('privacy_budget_spent', {}).get('delta', 0.0),
            'differential_privacy_enabled': privacy_status.get('differential_privacy_enabled', False),
            'encryption_enabled': privacy_status.get('encryption_enabled', False)
        }
        
        # Update performance metrics
        self.performance_metrics['privacy_budget_consumed'] = max(
            self.performance_metrics['privacy_budget_consumed'],
            privacy_status.get('privacy_budget_spent', {}).get('epsilon', 0.0)
        )
        
        logger.debug(f"Recorded aggregation for round {round_number} with {num_updates} updates")
    
    def get_round_metrics(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific round.
        
        Args:
            round_number: Round number
            
        Returns:
            Round metrics or None if not found
        """
        return self.round_metrics.get(round_number)
    
    def get_client_metrics(self, client_id: str, 
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics for a specific client.
        
        Args:
            client_id: Client identifier
            limit: Maximum number of recent metrics to return
            
        Returns:
            List of client metrics
        """
        metrics = self.client_metrics.get(client_id, [])
        if limit:
            metrics = metrics[-limit:]
        return metrics
    
    def get_recent_metrics(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent metrics from all clients.
        
        Args:
            limit: Maximum number of recent metrics to return
            
        Returns:
            List of recent metrics
        """
        metrics = list(self.recent_metrics)
        if limit:
            metrics = metrics[-limit:]
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Performance summary dictionary
        """
        return {
            'total_rounds': self.performance_metrics['total_rounds'],
            'avg_round_time': self.performance_metrics['avg_round_time'],
            'total_clients': len(self.client_metrics),
            'privacy_budget_consumed': self.performance_metrics['privacy_budget_consumed'],
            'total_communication_bytes': self.performance_metrics['total_communication_bytes']
        }
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get privacy summary.
        
        Returns:
            Privacy summary dictionary
        """
        if not self.aggregation_metrics:
            return {}
        
        # Calculate privacy statistics
        epsilons = [metrics['privacy_epsilon'] for metrics in self.aggregation_metrics.values()]
        deltas = [metrics['privacy_delta'] for metrics in self.aggregation_metrics.values()]
        
        return {
            'total_rounds_with_privacy': len(self.aggregation_metrics),
            'max_epsilon': max(epsilons) if epsilons else 0.0,
            'min_epsilon': min(epsilons) if epsilons else 0.0,
            'avg_epsilon': sum(epsilons) / len(epsilons) if epsilons else 0.0,
            'max_delta': max(deltas) if deltas else 0.0,
            'privacy_enabled_rounds': sum(1 for metrics in self.aggregation_metrics.values() 
                                        if metrics['differential_privacy_enabled'])
        }
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress summary.
        
        Returns:
            Training progress dictionary
        """
        if not self.recent_metrics:
            return {}
        
        # Calculate average loss from recent metrics
        recent_losses = [m.get('loss', 0.0) for m in self.recent_metrics if 'loss' in m]
        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
        
        # Get active clients
        active_clients = len(self.client_metrics)
        
        # Get current round
        current_round = max(self.round_metrics.keys()) if self.round_metrics else 0
        
        return {
            'current_round': current_round,
            'active_clients': active_clients,
            'avg_loss': avg_loss,
            'total_samples_processed': sum(m.get('num_batches', 0) * m.get('batch_size', 32) 
                                         for m in self.recent_metrics)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary.
        
        Returns:
            Complete metrics summary
        """
        return {
            'performance': self.get_performance_summary(),
            'privacy': self.get_privacy_summary(),
            'training_progress': self.get_training_progress(),
            'rounds': {
                'total': len(self.round_metrics),
                'completed': sum(1 for metrics in self.round_metrics.values() 
                               if metrics.get('status') == 'completed'),
                'in_progress': sum(1 for metrics in self.round_metrics.values() 
                                 if metrics.get('status') == 'started')
            },
            'clients': {
                'total': len(self.client_metrics),
                'active': len([client_id for client_id, metrics in self.client_metrics.items() 
                             if metrics])  # Clients with recent metrics
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
        """
        export_data = {
            'round_metrics': self.round_metrics,
            'client_metrics': dict(self.client_metrics),
            'aggregation_metrics': self.aggregation_metrics,
            'performance_metrics': self.performance_metrics,
            'summary': self.get_summary(),
            'export_timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.round_metrics.clear()
        self.round_timestamps.clear()
        self.client_metrics.clear()
        self.client_timestamps.clear()
        self.aggregation_metrics.clear()
        self.recent_metrics.clear()
        
        # Reset performance metrics
        self.performance_metrics = {
            'total_rounds': 0,
            'total_clients': 0,
            'avg_round_time': 0.0,
            'avg_client_training_time': 0.0,
            'total_communication_bytes': 0,
            'privacy_budget_consumed': 0.0
        }
        
        logger.info("All metrics cleared")


class RealTimeMonitor:
    """Real-time monitoring for federated learning."""
    
    def __init__(self, metrics_collector: MetricsCollector, update_interval: float = 5.0):
        """Initialize real-time monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            update_interval: Update interval in seconds
        """
        self.metrics_collector = metrics_collector
        self.update_interval = update_interval
        self.monitoring_active = False
        self.last_summary = {}
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring_active = True
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        logger.info("Real-time monitoring stopped")
    
    def get_status_update(self) -> Dict[str, Any]:
        """Get current status update.
        
        Returns:
            Status update dictionary
        """
        current_summary = self.metrics_collector.get_summary()
        
        # Calculate changes since last update
        changes = {}
        for key, value in current_summary.items():
            if key in self.last_summary:
                if isinstance(value, dict) and isinstance(self.last_summary[key], dict):
                    changes[key] = {k: v for k, v in value.items() 
                                  if k not in self.last_summary[key] or 
                                  self.last_summary[key][k] != v}
                elif value != self.last_summary[key]:
                    changes[key] = value
        
        self.last_summary = current_summary.copy()
        
        return {
            'timestamp': time.time(),
            'current_status': current_summary,
            'changes': changes
        }
    
    def print_status(self):
        """Print current status to console."""
        status = self.get_status_update()
        
        print("\n" + "="*50)
        print("FEDERATED LEARNING STATUS")
        print("="*50)
        
        # Training progress
        progress = status['current_status']['training_progress']
        print(f"Current Round: {progress.get('current_round', 0)}")
        print(f"Active Clients: {progress.get('active_clients', 0)}")
        print(f"Average Loss: {progress.get('avg_loss', 0.0):.4f}")
        
        # Performance
        performance = status['current_status']['performance']
        print(f"Total Rounds: {performance.get('total_rounds', 0)}")
        print(f"Average Round Time: {performance.get('avg_round_time', 0.0):.2f}s")
        
        # Privacy
        privacy = status['current_status']['privacy']
        if privacy:
            print(f"Privacy Budget Consumed: {privacy.get('max_epsilon', 0.0):.3f} ε")
            print(f"Privacy-Enabled Rounds: {privacy.get('privacy_enabled_rounds', 0)}")
        
        print("="*50)
    
    def monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.print_status()
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval) 