"""
Web dashboard for monitoring federated learning training.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import threading
from typing import Dict, Any, List
import logging

from monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class FederatedLearningDashboard:
    """Web dashboard for federated learning monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8050):
        """Initialize dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            port: Dashboard port
        """
        self.metrics_collector = metrics_collector
        self.port = port
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Federated Learning Dashboard"
        
        # Setup layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Data storage for plots
        self.plot_data = {
            'loss_history': [],
            'privacy_history': [],
            'client_activity': [],
            'round_progress': []
        }
    
    def _create_layout(self):
        """Create dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Federated Learning Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Status Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Current Round", className="card-title"),
                            html.H2(id="current-round", children="0", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Clients", className="card-title"),
                            html.H2(id="active-clients", children="0", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Average Loss", className="card-title"),
                            html.H2(id="avg-loss", children="0.0000", className="text-warning")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Privacy Budget", className="card-title"),
                            html.H2(id="privacy-budget", children="0.000", className="text-info")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                # Training Loss Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Loss Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="loss-chart")
                        ])
                    ])
                ], width=6),
                
                # Privacy Budget Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Privacy Budget Consumption"),
                        dbc.CardBody([
                            dcc.Graph(id="privacy-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Client Activity and Round Progress
            dbc.Row([
                # Client Activity
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Client Activity"),
                        dbc.CardBody([
                            dcc.Graph(id="client-activity-chart")
                        ])
                    ])
                ], width=6),
                
                # Round Progress
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Round Progress"),
                        dbc.CardBody([
                            dcc.Graph(id="round-progress-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Performance Metrics Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            html.Div(id="performance-table")
                        ])
                    ])
                ])
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("current-round", "children"),
             Output("active-clients", "children"),
             Output("avg-loss", "children"),
             Output("privacy-budget", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_status_cards(n):
            """Update status cards."""
            summary = self.metrics_collector.get_summary()
            
            current_round = summary.get('training_progress', {}).get('current_round', 0)
            active_clients = summary.get('training_progress', {}).get('active_clients', 0)
            avg_loss = f"{summary.get('training_progress', {}).get('avg_loss', 0.0):.4f}"
            privacy_budget = f"{summary.get('privacy', {}).get('max_epsilon', 0.0):.3f}"
            
            return current_round, active_clients, avg_loss, privacy_budget
        
        @self.app.callback(
            Output("loss-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_loss_chart(n):
            """Update training loss chart."""
            recent_metrics = self.metrics_collector.get_recent_metrics(limit=100)
            
            if not recent_metrics:
                return self._create_empty_figure("No training data available")
            
            # Extract loss data
            loss_data = []
            for metric in recent_metrics:
                if 'loss' in metric and 'timestamp' in metric:
                    loss_data.append({
                        'timestamp': pd.to_datetime(metric['timestamp'], unit='s'),
                        'loss': metric['loss'],
                        'client_id': metric.get('client_id', 'Unknown')
                    })
            
            if not loss_data:
                return self._create_empty_figure("No loss data available")
            
            df = pd.DataFrame(loss_data)
            
            fig = px.line(df, x='timestamp', y='loss', color='client_id',
                         title="Training Loss Over Time")
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Loss",
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output("privacy-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_privacy_chart(n):
            """Update privacy budget chart."""
            summary = self.metrics_collector.get_summary()
            privacy_data = summary.get('privacy', {})
            
            if not privacy_data:
                return self._create_empty_figure("No privacy data available")
            
            # Create privacy metrics
            metrics = ['max_epsilon', 'min_epsilon', 'avg_epsilon']
            values = [privacy_data.get(metric, 0.0) for metric in metrics]
            
            fig = go.Figure(data=[
                go.Bar(x=metrics, y=values, marker_color=['red', 'green', 'blue'])
            ])
            
            fig.update_layout(
                title="Privacy Budget Consumption",
                xaxis_title="Privacy Metric",
                yaxis_title="Epsilon Value",
                showlegend=False
            )
            
            return fig
        
        @self.app.callback(
            Output("client-activity-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_client_activity_chart(n):
            """Update client activity chart."""
            summary = self.metrics_collector.get_summary()
            client_metrics = summary.get('clients', {})
            
            if not client_metrics:
                return self._create_empty_figure("No client data available")
            
            # Count active vs total clients
            total_clients = client_metrics.get('total', 0)
            active_clients = client_metrics.get('active', 0)
            inactive_clients = total_clients - active_clients
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Active Clients', 'Inactive Clients'],
                    values=[active_clients, inactive_clients],
                    hole=0.4,
                    marker_colors=['green', 'gray']
                )
            ])
            
            fig.update_layout(
                title="Client Activity Distribution",
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output("round-progress-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_round_progress_chart(n):
            """Update round progress chart."""
            summary = self.metrics_collector.get_summary()
            round_metrics = summary.get('rounds', {})
            
            if not round_metrics:
                return self._create_empty_figure("No round data available")
            
            total_rounds = round_metrics.get('total', 0)
            completed_rounds = round_metrics.get('completed', 0)
            in_progress_rounds = round_metrics.get('in_progress', 0)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Completed', 'In Progress', 'Pending'],
                    y=[completed_rounds, in_progress_rounds, max(0, total_rounds - completed_rounds - in_progress_rounds)],
                    marker_color=['green', 'orange', 'gray']
                )
            ])
            
            fig.update_layout(
                title="Round Progress",
                xaxis_title="Round Status",
                yaxis_title="Number of Rounds",
                showlegend=False
            )
            
            return fig
        
        @self.app.callback(
            Output("performance-table", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_performance_table(n):
            """Update performance metrics table."""
            summary = self.metrics_collector.get_summary()
            performance = summary.get('performance', {})
            
            if not performance:
                return html.P("No performance data available")
            
            # Create table rows
            rows = []
            for key, value in performance.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                rows.append(html.Tr([
                    html.Td(key.replace('_', ' ').title()),
                    html.Td(formatted_value)
                ]))
            
            table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric"),
                        html.Th("Value")
                    ])
                ]),
                html.Tbody(rows)
            ], bordered=True, hover=True)
            
            return table
    
    def _create_empty_figure(self, message: str):
        """Create empty figure with message.
        
        Args:
            message: Message to display
            
        Returns:
            Empty plotly figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def start(self, debug: bool = False):
        """Start the dashboard.
        
        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard on port {self.port}")
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')


def main():
    """Main function to start the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create metrics collector (in a real scenario, this would be shared with the server)
    metrics_collector = MetricsCollector()
    
    # Create and start dashboard
    dashboard = FederatedLearningDashboard(metrics_collector, port=args.port)
    dashboard.start(debug=args.debug)


if __name__ == "__main__":
    main() 