"""
Advanced web dashboard for monitoring federated learning training.

This module provides a comprehensive web-based monitoring interface with real-time
updates, interactive visualizations, alert management, and system health monitoring
for production federated learning systems.
"""

import dash
from dash import dcc, html, Input, Output, callback, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# Optional imports for advanced features
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from monitoring.metrics_collector import AdvancedMetricsCollector as MetricsCollector
except ImportError:
    from monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class AdvancedFederatedLearningDashboard:
    """Advanced web dashboard for federated learning monitoring."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector, 
                 port: int = 8050,
                 enable_alerts: bool = True,
                 enable_system_monitoring: bool = True,
                 refresh_interval: int = 5000):
        """Initialize advanced dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            port: Dashboard port
            enable_alerts: Whether to enable alert management
            enable_system_monitoring: Whether to enable system monitoring
            refresh_interval: Refresh interval in milliseconds
        """
        self.metrics_collector = metrics_collector
        self.port = port
        self.enable_alerts = enable_alerts
        self.enable_system_monitoring = enable_system_monitoring
        self.refresh_interval = refresh_interval
        
        # Initialize Dash app with advanced features
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME
            ],
            suppress_callback_exceptions=True
        )
        self.app.title = "Advanced Federated Learning Dashboard"
        
        # Setup layout
        self.app.layout = self._create_advanced_layout()
        
        # Setup callbacks
        self._setup_advanced_callbacks()
        
        # Data storage for plots
        self.plot_data = {
            'loss_history': [],
            'privacy_history': [],
            'client_activity': [],
            'round_progress': [],
            'system_metrics': [],
            'alert_history': []
        }
        
        # Enhanced performance tracking
        self.performance_predictions = {
            'loss_trend': None,
            'convergence_estimate': None,
            'resource_usage_forecast': None,
            'anomaly_scores': []
        }
        
        # Client analytics
        self.client_analytics = {
            'performance_ranking': [],
            'anomaly_detection': {},
            'contribution_metrics': {},
            'network_latency': {}
        }
        
        # Alert management
        self.active_alerts = []
        self.alert_history = []
        
        # System monitoring
        self.system_metrics = {}
        
        # Predictive models
        self.predictive_models = {}
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _create_advanced_layout(self):
        """Create advanced dashboard layout with multiple tabs."""
        return dbc.Container([
            # Header with navigation
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-brain me-2"),
                        "Advanced Federated Learning Dashboard"
                    ], className="text-center mb-3"),
                    html.P([
                        "Real-time monitoring and analytics for federated learning systems",
                        html.Br(),
                        html.Small(f"Last updated: ", id="last-updated")
                    ], className="text-center text-muted")
                ])
            ], className="mb-4"),
            
            # Alert banner
            html.Div(id="alert-banner"),
            
            # Main content with tabs
            dbc.Tabs([
                # Overview Tab
                dbc.Tab(self._create_overview_tab(), label="Overview", tab_id="overview"),
                
                # Training Tab
                dbc.Tab(self._create_training_tab(), label="Training", tab_id="training"),
                
                # Privacy Tab
                dbc.Tab(self._create_privacy_tab(), label="Privacy", tab_id="privacy"),
                
                # System Tab
                dbc.Tab(self._create_system_tab(), label="System", tab_id="system"),
                
                # Analytics Tab
                dbc.Tab(self._create_analytics_tab(), label="Analytics", tab_id="analytics"),
                
                # Alerts Tab
                dbc.Tab(self._create_alerts_tab(), label="Alerts", tab_id="alerts")
            ], id="main-tabs", active_tab="overview"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.refresh_interval,
                n_intervals=0
            ),
            
            # Store components for data persistence
            dcc.Store(id='session-store'),
            dcc.Store(id='local-store', storage_type='local')
        ], fluid=True)
    
    def _create_overview_tab(self):
        """Create overview tab with key metrics."""
        return dbc.Container([
            # Key Performance Indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-layer-group fa-2x text-primary mb-2"),
                                html.H4("Current Round", className="card-title"),
                                html.H2(id="current-round", children="0", className="text-primary")
                            ], className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-users fa-2x text-success mb-2"),
                                html.H4("Active Clients", className="card-title"),
                                html.H2(id="active-clients", children="0", className="text-success")
                            ], className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-chart-line fa-2x text-warning mb-2"),
                                html.H4("Average Loss", className="card-title"),
                                html.H2(id="avg-loss", children="0.0000", className="text-warning")
                            ], className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-shield-alt fa-2x text-info mb-2"),
                                html.H4("Privacy Budget", className="card-title"),
                                html.H2(id="privacy-budget", children="0.000", className="text-info")
                            ], className="text-center")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # System Health Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-heartbeat me-2"),
                            "System Health"
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("CPU Usage"),
                                        dbc.Progress(id="cpu-progress", value=0, className="mb-2"),
                                        html.Small(id="cpu-text", children="0%")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Memory Usage"),
                                        dbc.Progress(id="memory-progress", value=0, className="mb-2"),
                                        html.Small(id="memory-text", children="0%")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Network I/O"),
                                        html.H5(id="network-io", children="0 MB/s", className="text-info")
                                    ])
                                ], width=4)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Quick Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Progress"),
                        dbc.CardBody([
                            dcc.Graph(id="overview-training-chart", style={'height': '300px'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Client Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="overview-client-chart", style={'height': '300px'})
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_training_tab(self):
        """Create training monitoring tab."""
        return dbc.Container([
            # Training Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fas fa-play me-2"),
                                        "Start Training"
                                    ], id="start-training-btn", color="success", className="me-2"),
                                    dbc.Button([
                                        html.I(className="fas fa-pause me-2"),
                                        "Pause"
                                    ], id="pause-training-btn", color="warning", className="me-2"),
                                    dbc.Button([
                                        html.I(className="fas fa-stop me-2"),
                                        "Stop"
                                    ], id="stop-training-btn", color="danger")
                                ])
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Training Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Loss Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="training-loss-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Accuracy Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="training-accuracy-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Round Details
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Round Progress"),
                        dbc.CardBody([
                            dcc.Graph(id="round-progress-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Client Activity"),
                        dbc.CardBody([
                            dcc.Graph(id="client-activity-chart")
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_privacy_tab(self):
        """Create privacy monitoring tab."""
        return dbc.Container([
            # Privacy Status
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-shield-alt me-2"),
                            "Privacy Status"
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("Epsilon Budget"),
                                        dbc.Progress(id="epsilon-progress", value=0, className="mb-2"),
                                        html.Small(id="epsilon-text", children="0.0 / 0.0")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Delta Budget"),
                                        dbc.Progress(id="delta-progress", value=0, className="mb-2"),
                                        html.Small(id="delta-text", children="0.0 / 0.0")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Privacy Level"),
                                        html.H5(id="privacy-level", children="Unknown", className="text-info")
                                    ])
                                ], width=4)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Privacy Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Privacy Budget Consumption"),
                        dbc.CardBody([
                            dcc.Graph(id="privacy-budget-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Noise Addition Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="noise-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Privacy Settings
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Privacy Configuration"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Target Epsilon"),
                                    dbc.Input(id="target-epsilon", type="number", value=1.0, step=0.1)
                                ], width=4),
                                dbc.Col([
                                    html.Label("Target Delta"),
                                    dbc.Input(id="target-delta", type="number", value=1e-5, step=1e-6)
                                ], width=4),
                                dbc.Col([
                                    html.Label("Noise Multiplier"),
                                    dbc.Input(id="noise-multiplier", type="number", value=1.1, step=0.1)
                                ], width=4)
                            ], className="mt-3")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_system_tab(self):
        """Create system monitoring tab."""
        return dbc.Container([
            # System Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-server me-2"),
                            "System Resources"
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("CPU Usage"),
                                        dcc.Graph(id="cpu-gauge", style={'height': '200px'})
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Memory Usage"),
                                        dcc.Graph(id="memory-gauge", style={'height': '200px'})
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Disk Usage"),
                                        dcc.Graph(id="disk-gauge", style={'height': '200px'})
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Network I/O"),
                                        html.H4(id="network-display", children="0 MB/s")
                                    ])
                                ], width=3)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # System Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Metrics Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="system-metrics-chart")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Process Information
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Process Information"),
                        dbc.CardBody([
                            html.Div(id="process-info")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_analytics_tab(self):
        """Create analytics tab."""
        return dbc.Container([
            # Analytics Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analytics Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Time Range"),
                                    dcc.Dropdown(
                                        id="time-range-dropdown",
                                        options=[
                                            {'label': 'Last Hour', 'value': '1h'},
                                            {'label': 'Last 6 Hours', 'value': '6h'},
                                            {'label': 'Last 24 Hours', 'value': '24h'},
                                            {'label': 'Last Week', 'value': '7d'},
                                            {'label': 'Custom', 'value': 'custom'}
                                        ],
                                        value='24h'
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Metric Type"),
                                    dcc.Dropdown(
                                        id="metric-type-dropdown",
                                        options=[
                                            {'label': 'Training Metrics', 'value': 'training'},
                                            {'label': 'System Metrics', 'value': 'system'},
                                            {'label': 'Privacy Metrics', 'value': 'privacy'},
                                            {'label': 'Client Metrics', 'value': 'client'}
                                        ],
                                        value='training'
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Aggregation"),
                                    dcc.Dropdown(
                                        id="aggregation-dropdown",
                                        options=[
                                            {'label': 'Raw Data', 'value': 'raw'},
                                            {'label': 'Average', 'value': 'avg'},
                                            {'label': 'Min/Max', 'value': 'minmax'},
                                            {'label': 'Percentiles', 'value': 'percentiles'}
                                        ],
                                        value='avg'
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fas fa-download me-2"),
                                        "Export Data"
                                    ], id="export-btn", color="primary")
                                ], width=3)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Predictive Analytics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-crystal-ball me-2"),
                            "Predictive Analytics"
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("Loss Trend Prediction"),
                                        html.H4(id="loss-prediction", children="--", className="text-info"),
                                        html.Small("Next 5 rounds", className="text-muted")
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Convergence Estimate"),
                                        html.H4(id="convergence-estimate", children="--", className="text-success"),
                                        html.Small("Rounds to convergence", className="text-muted")
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Resource Forecast"),
                                        html.H4(id="resource-forecast", children="--", className="text-warning"),
                                        html.Small("Peak usage prediction", className="text-muted")
                                    ])
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Anomaly Score"),
                                        html.H4(id="anomaly-score", children="--", className="text-danger"),
                                        html.Small("Current system health", className="text-muted")
                                    ])
                                ], width=3)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Client Performance Analytics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-users me-2"),
                            "Client Performance Analytics"
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("Top Performers"),
                                        html.Div(id="top-performers", children="Loading...")
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Anomaly Detection"),
                                        html.Div(id="anomaly-detection", children="Loading...")
                                    ])
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Analytics Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analytics Dashboard"),
                        dbc.CardBody([
                            dcc.Graph(id="analytics-main-chart")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Statistical Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Statistical Summary"),
                        dbc.CardBody([
                            html.Div(id="statistical-summary")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_alerts_tab(self):
        """Create alerts management tab."""
        return dbc.Container([
            # Alert Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alert Management"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fas fa-plus me-2"),
                                        "Add Alert"
                                    ], id="add-alert-btn", color="success", className="me-2"),
                                    dbc.Button([
                                        html.I(className="fas fa-bell me-2"),
                                        "Test Alerts"
                                    ], id="test-alerts-btn", color="warning", className="me-2"),
                                    dbc.Button([
                                        html.I(className="fas fa-trash me-2"),
                                        "Clear All"
                                    ], id="clear-alerts-btn", color="danger")
                                ])
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Active Alerts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            "Active Alerts"
                        ]),
                        dbc.CardBody([
                            html.Div(id="active-alerts-list")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Alert History
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alert History"),
                        dbc.CardBody([
                            dcc.Graph(id="alert-history-chart")
                        ])
                    ])
                ])
            ])
        ])
    
    def _setup_advanced_callbacks(self):
        """Setup advanced dashboard callbacks."""
        
        # Overview callbacks
        @self.app.callback(
            [Output("current-round", "children"),
             Output("active-clients", "children"),
             Output("avg-loss", "children"),
             Output("privacy-budget", "children"),
             Output("last-updated", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_overview_metrics(n):
            """Update overview metrics."""
            try:
                current_metrics = self.metrics_collector.get_current_metrics()
                
                current_round = current_metrics.get('total_rounds', 0)
                active_clients = len(self.metrics_collector.client_metrics)
                avg_loss = f"{current_metrics.get('avg_loss', 0.0):.4f}"
                privacy_budget = f"{current_metrics.get('privacy_budget_consumed', 0.0):.3f}"
                last_updated = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                
                return current_round, active_clients, avg_loss, privacy_budget, last_updated
            except Exception as e:
                logger.error(f"Error updating overview metrics: {e}")
                return 0, 0, "0.0000", "0.000", "Error updating"
        
        # System health callbacks
        @self.app.callback(
            [Output("cpu-progress", "value"),
             Output("cpu-text", "children"),
             Output("memory-progress", "value"),
             Output("memory-text", "children"),
             Output("network-io", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_system_health(n):
            """Update system health metrics."""
            try:
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    network_io = psutil.net_io_counters()
                    network_mbps = (network_io.bytes_sent + network_io.bytes_recv) / 1024 / 1024
                    
                    return (
                        cpu_percent,
                        f"{cpu_percent:.1f}%",
                        memory_percent,
                        f"{memory_percent:.1f}%",
                        f"{network_mbps:.1f} MB/s"
                    )
                else:
                    return 0, "N/A", 0, "N/A", "N/A"
            except Exception as e:
                logger.error(f"Error updating system health: {e}")
                return 0, "Error", 0, "Error", "Error"
        
        # Training charts callbacks
        @self.app.callback(
            Output("training-loss-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_training_loss_chart(n):
            """Update training loss chart."""
            try:
                loss_metrics = self.metrics_collector.get_metric("federated.round.loss")
                
                if not loss_metrics:
                    return self._create_empty_figure("No loss data available")
                
                # Prepare data
                timestamps = [m.timestamp for m in loss_metrics]
                values = [m.value for m in loss_metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Training Loss Over Time",
                    xaxis_title="Time",
                    yaxis_title="Loss",
                    hovermode='x unified',
                    showlegend=True
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating training loss chart: {e}")
                return self._create_empty_figure("Error loading data")
        
        # Privacy charts callbacks
        @self.app.callback(
            Output("privacy-budget-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_privacy_budget_chart(n):
            """Update privacy budget chart."""
            try:
                epsilon_metrics = self.metrics_collector.get_metric("federated.privacy.epsilon")
                
                if not epsilon_metrics:
                    return self._create_empty_figure("No privacy data available")
                
                # Prepare data
                timestamps = [m.timestamp for m in epsilon_metrics]
                values = [m.value for m in epsilon_metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name='Privacy Budget (ε)',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Privacy Budget Consumption",
                    xaxis_title="Time",
                    yaxis_title="Epsilon (ε)",
                    hovermode='x unified',
                    showlegend=True
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating privacy budget chart: {e}")
                return self._create_empty_figure("Error loading data")
        
        # Alert banner callback
        @self.app.callback(
            Output("alert-banner", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_alert_banner(n):
            """Update alert banner."""
            try:
                if not self.enable_alerts or not self.metrics_collector.alert_manager:
                    return []
                
                current_metrics = self.metrics_collector.get_current_metrics()
                triggered_alerts = self.metrics_collector.alert_manager.check_alerts(current_metrics)
                
                if not triggered_alerts:
                    return []
                
                # Create alert banner
                alert_components = []
                for alert in triggered_alerts:
                    color_map = {
                        'info': 'info',
                        'warning': 'warning',
                        'error': 'danger',
                        'critical': 'danger'
                    }
                    
                    alert_component = dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong(f"[{alert.severity.value.upper()}] "),
                        alert.message
                    ], color=color_map.get(alert.severity.value, 'info'), dismissable=True)
                    
                    alert_components.append(alert_component)
                
                return alert_components
            except Exception as e:
                logger.error(f"Error updating alert banner: {e}")
                return []
        
        # Predictive analytics callbacks
        @self.app.callback(
            [Output("loss-prediction", "children"),
             Output("convergence-estimate", "children"),
             Output("resource-forecast", "children"),
             Output("anomaly-score", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_predictive_analytics(n):
            """Update predictive analytics."""
            try:
                # Get recent loss data
                loss_metrics = self.metrics_collector.get_metric("federated.round.loss")
                
                if not loss_metrics or len(loss_metrics) < 5:
                    return "--", "--", "--", "--"
                
                # Calculate loss trend prediction
                recent_losses = [m.value for m in loss_metrics[-10:]]
                if len(recent_losses) >= 5:
                    trend = self._calculate_loss_trend(recent_losses)
                    loss_prediction = f"{trend:.4f}"
                else:
                    loss_prediction = "--"
                
                # Calculate convergence estimate
                convergence_estimate = self._estimate_convergence(recent_losses)
                
                # Calculate resource forecast
                if PSUTIL_AVAILABLE:
                    cpu_history = [self.system_metrics.get('cpu_percent', 0)]
                    resource_forecast = f"{max(cpu_history) * 1.2:.1f}%"
                else:
                    resource_forecast = "--"
                
                # Calculate anomaly score
                anomaly_score = self._calculate_anomaly_score(recent_losses)
                
                return loss_prediction, convergence_estimate, resource_forecast, anomaly_score
                
            except Exception as e:
                logger.error(f"Error updating predictive analytics: {e}")
                return "--", "--", "--", "--"
        
        # Client performance analytics callbacks
        @self.app.callback(
            [Output("top-performers", "children"),
             Output("anomaly-detection", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_client_analytics(n):
            """Update client performance analytics."""
            try:
                # Get client metrics
                client_metrics = self.metrics_collector.client_metrics
                
                if not client_metrics:
                    return "No client data", "No anomalies detected"
                
                # Calculate top performers
                client_scores = {}
                for client_id, metrics in client_metrics.items():
                    if metrics:
                        # Calculate average loss for this client
                        avg_loss = sum(m.get('loss', 0) for m in metrics[-10:]) / len(metrics[-10:])
                        client_scores[client_id] = avg_loss
                
                # Sort by performance (lower loss is better)
                top_performers = sorted(client_scores.items(), key=lambda x: x[1])[:3]
                
                top_performers_html = []
                for i, (client_id, score) in enumerate(top_performers):
                    rank_icon = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
                    top_performers_html.append(
                        html.Div([
                            html.Span(rank_icon, className="me-2"),
                            html.Strong(f"Client {client_id}"),
                            html.Span(f": {score:.4f}", className="text-muted ms-2")
                        ], className="mb-1")
                    )
                
                # Anomaly detection
                anomalies = []
                for client_id, metrics in client_metrics.items():
                    if len(metrics) >= 5:
                        recent_losses = [m.get('loss', 0) for m in metrics[-5:]]
                        if self._detect_client_anomaly(recent_losses):
                            anomalies.append(client_id)
                
                if anomalies:
                    anomaly_html = [
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                            f"Anomalies detected in clients: {', '.join(anomalies)}"
                        ])
                    ]
                else:
                    anomaly_html = [
                        html.Div([
                            html.I(className="fas fa-check-circle text-success me-2"),
                            "No anomalies detected"
                        ])
                    ]
                
                return top_performers_html, anomaly_html
                
            except Exception as e:
                logger.error(f"Error updating client analytics: {e}")
                return "Error loading data", "Error detecting anomalies"
    
    def _create_empty_figure(self, message: str):
        """Create empty figure with message."""
        return {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': message,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            }
        }
    
    def _calculate_loss_trend(self, losses: List[float]) -> float:
        """Calculate predicted loss trend for next 5 rounds."""
        try:
            if len(losses) < 3:
                return losses[-1] if losses else 0.0
            
            # Simple linear regression for trend prediction
            x = list(range(len(losses)))
            y = losses
            
            # Calculate trend using simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Predict next 5 rounds
            current_loss = losses[-1]
            predicted_loss = current_loss + (slope * 5)
            
            return max(0.0, predicted_loss)  # Loss cannot be negative
            
        except Exception as e:
            logger.error(f"Error calculating loss trend: {e}")
            return losses[-1] if losses else 0.0
    
    def _estimate_convergence(self, losses: List[float]) -> str:
        """Estimate rounds to convergence."""
        try:
            if len(losses) < 5:
                return "--"
            
            # Calculate loss improvement rate
            recent_losses = losses[-5:]
            improvement_rate = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
            
            if improvement_rate <= 0.001:  # Very small improvement
                return "Converged"
            elif improvement_rate <= 0.01:  # Small improvement
                return "5-10 rounds"
            elif improvement_rate <= 0.05:  # Moderate improvement
                return "10-20 rounds"
            else:  # Large improvement
                return "20+ rounds"
                
        except Exception as e:
            logger.error(f"Error estimating convergence: {e}")
            return "--"
    
    def _calculate_anomaly_score(self, losses: List[float]) -> str:
        """Calculate anomaly score based on loss patterns."""
        try:
            if len(losses) < 3:
                return "0.0"
            
            # Calculate statistical measures
            mean_loss = sum(losses) / len(losses)
            variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
            std_dev = variance ** 0.5
            
            # Calculate anomaly score based on recent loss deviation
            recent_loss = losses[-1]
            z_score = abs(recent_loss - mean_loss) / std_dev if std_dev > 0 else 0
            
            # Normalize to 0-1 scale
            anomaly_score = min(1.0, z_score / 3.0)  # 3 sigma rule
            
            return f"{anomaly_score:.2f}"
            
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return "0.0"
    
    def _detect_client_anomaly(self, losses: List[float]) -> bool:
        """Detect anomalies in client loss patterns."""
        try:
            if len(losses) < 3:
                return False
            
            # Calculate statistical measures
            mean_loss = sum(losses) / len(losses)
            variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
            std_dev = variance ** 0.5
            
            # Check for anomalies (losses outside 2 standard deviations)
            recent_loss = losses[-1]
            z_score = abs(recent_loss - mean_loss) / std_dev if std_dev > 0 else 0
            
            return z_score > 2.0  # Anomaly if more than 2 standard deviations
            
        except Exception as e:
            logger.error(f"Error detecting client anomaly: {e}")
            return False
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        def background_monitor():
            while True:
                try:
                    # Update system metrics
                    if self.enable_system_monitoring and PSUTIL_AVAILABLE:
                        self.system_metrics = {
                            'cpu_percent': psutil.cpu_percent(),
                            'memory_percent': psutil.virtual_memory().percent,
                            'disk_percent': psutil.disk_usage('/').percent,
                            'network_io': psutil.net_io_counters()._asdict()
                        }
                    
                    # Update predictive analytics
                    self._update_predictive_analytics()
                    
                    # Update client analytics
                    self._update_client_analytics()
                    
                    # Update alert history
                    if self.enable_alerts and self.metrics_collector.alert_manager:
                        current_metrics = self.metrics_collector.get_current_metrics()
                        triggered_alerts = self.metrics_collector.alert_manager.check_alerts(current_metrics)
                        
                        for alert in triggered_alerts:
                            self.alert_history.append({
                                'timestamp': datetime.now(),
                                'alert': alert,
                                'metrics': current_metrics
                            })
                    
                    time.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error in background monitoring: {e}")
                    time.sleep(30)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=background_monitor, daemon=True)
        monitor_thread.start()
        logger.info("Background monitoring started")
    
    def _update_predictive_analytics(self):
        """Update predictive analytics in background."""
        try:
            # Get recent loss data
            loss_metrics = self.metrics_collector.get_metric("federated.round.loss")
            
            if loss_metrics and len(loss_metrics) >= 5:
                recent_losses = [m.value for m in loss_metrics[-10:]]
                
                # Update loss trend prediction
                self.performance_predictions['loss_trend'] = self._calculate_loss_trend(recent_losses)
                
                # Update convergence estimate
                self.performance_predictions['convergence_estimate'] = self._estimate_convergence(recent_losses)
                
                # Update anomaly scores
                self.performance_predictions['anomaly_scores'].append({
                    'timestamp': datetime.now(),
                    'score': float(self._calculate_anomaly_score(recent_losses))
                })
                
                # Keep only recent anomaly scores
                if len(self.performance_predictions['anomaly_scores']) > 50:
                    self.performance_predictions['anomaly_scores'] = self.performance_predictions['anomaly_scores'][-50:]
                
        except Exception as e:
            logger.error(f"Error updating predictive analytics: {e}")
    
    def _update_client_analytics(self):
        """Update client analytics in background."""
        try:
            client_metrics = self.metrics_collector.client_metrics
            
            if not client_metrics:
                return
            
            # Update client performance ranking
            client_scores = {}
            for client_id, metrics in client_metrics.items():
                if metrics:
                    # Calculate performance score based on recent metrics
                    recent_metrics = metrics[-10:] if len(metrics) >= 10 else metrics
                    avg_loss = sum(m.get('loss', 0) for m in recent_metrics) / len(recent_metrics)
                    client_scores[client_id] = avg_loss
            
            # Sort by performance (lower loss is better)
            self.client_analytics['performance_ranking'] = sorted(
                client_scores.items(), 
                key=lambda x: x[1]
            )
            
            # Update anomaly detection
            for client_id, metrics in client_metrics.items():
                if len(metrics) >= 5:
                    recent_losses = [m.get('loss', 0) for m in metrics[-5:]]
                    is_anomalous = self._detect_client_anomaly(recent_losses)
                    self.client_analytics['anomaly_detection'][client_id] = is_anomalous
            
        except Exception as e:
            logger.error(f"Error updating client analytics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for dashboard updates."""
        try:
            current_metrics = {
                'total_rounds': self.metrics_collector.performance_metrics.get('total_rounds', 0),
                'active_clients': len(self.metrics_collector.client_metrics),
                'avg_loss': 0.0,
                'privacy_budget_consumed': self.metrics_collector.performance_metrics.get('privacy_budget_consumed', 0.0),
                'system_metrics': self.system_metrics,
                'predictive_analytics': self.performance_predictions,
                'client_analytics': self.client_analytics
            }
            
            # Calculate average loss from recent metrics
            recent_metrics = self.metrics_collector.get_recent_metrics(limit=50)
            if recent_metrics:
                losses = [m.get('loss', 0) for m in recent_metrics if 'loss' in m]
                if losses:
                    current_metrics['avg_loss'] = sum(losses) / len(losses)
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {
                'total_rounds': 0,
                'active_clients': 0,
                'avg_loss': 0.0,
                'privacy_budget_consumed': 0.0,
                'system_metrics': {},
                'predictive_analytics': {},
                'client_analytics': {}
            }
    
    def start(self, debug: bool = False, host: str = "0.0.0.0"):
        """Start the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            host: Host address to bind to
        """
        try:
            logger.info(f"Starting dashboard on {host}:{self.port}")
            self.app.run_server(
                debug=debug,
                host=host,
                port=self.port,
                dev_tools_hot_reload=False
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            raise


def main():
    """Main function to run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--metrics-db", default="metrics.db", help="Metrics database path")
    
    args = parser.parse_args()
    
    # Create metrics collector
    metrics_collector = MetricsCollector(
        enable_database=True,
        enable_profiling=True,
        enable_alerts=True,
        db_path=args.metrics_db
    )
    
    # Create and start dashboard
    dashboard = AdvancedFederatedLearningDashboard(
        metrics_collector=metrics_collector,
        port=args.port,
        enable_alerts=True,
        enable_system_monitoring=True
    )
    
    dashboard.start(debug=args.debug, host=args.host)


if __name__ == "__main__":
    main() 