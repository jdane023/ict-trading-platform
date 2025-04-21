"""
ICT Trading Strategy Web Application
A web-based interface for the ICT Trading Strategy
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import base64
import io
import json
import threading
import traceback

# Import the ICT Strategy class
# Assuming the ICTStrategy class is in a file called ict_strategy.py
from ict_strategy import ICTStrategy, fetch_historical_data

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server  # For Gunicorn deployment

# Store data in a global variable (in a production app, use a database)
global_store = {
    'data': None,
    'strategy': None,
    'backtest_results': None,
    'settings': {
        'sma1': 20,
        'sma2': 50,
        'sma3': 200,
        'atr_period': 14,
        'atr_stop': 2.0,
        'rr_ratio': 3.0,
        'structure_window': 10,
        'ote_lower': 0.65,
        'ote_upper': 0.79
    }
}

# Layout tabs
def create_data_tab():
    """Create the Data tab"""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Left side - Controls
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Data Import", className="card-title"),
                            
                            dbc.FormGroup([
                                dbc.Label("Symbol:"),
                                dbc.Input(id="symbol-input", type="text", value="EURUSD=X", placeholder="Enter symbol")
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("Timeframe:"),
                                dcc.Dropdown(
                                    id="timeframe-dropdown",
                                    options=[
                                        {'label': '1 Minute', 'value': '1m'},
                                        {'label': '5 Minutes', 'value': '5m'},
                                        {'label': '15 Minutes', 'value': '15m'},
                                        {'label': '30 Minutes', 'value': '30m'},
                                        {'label': '1 Hour', 'value': '1h'},
                                        {'label': '4 Hours', 'value': '4h'},
                                        {'label': '1 Day', 'value': '1d'},
                                        {'label': '1 Week', 'value': '1wk'},
                                        {'label': '1 Month', 'value': '1mo'}
                                    ],
                                    value='1d'
                                )
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("Start Date:"),
                                dcc.DatePickerSingle(
                                    id="start-date-picker",
                                    date=(datetime.now() - timedelta(days=365)).date(),
                                    display_format='YYYY-MM-DD'
                                )
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("End Date:"),
                                dcc.DatePickerSingle(
                                    id="end-date-picker",
                                    date=datetime.now().date(),
                                    display_format='YYYY-MM-DD'
                                )
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("Data Source:"),
                                dcc.Dropdown(
                                    id="source-dropdown",
                                    options=[
                                        {'label': 'Yahoo Finance', 'value': 'yahoo'},
                                        {'label': 'Alpha Vantage', 'value': 'alpha_vantage'}
                                    ],
                                    value='yahoo'
                                )
                            ]),
                            
                            dbc.Button("Import Data", id="import-button", color="primary", className="mr-2"),
                            dbc.Button("Upload CSV", id="upload-button", color="secondary"),
                            
                            # Hidden div for CSV upload
                            html.Div([
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    multiple=False
                                )
                            ], id="upload-div", style={"display": "none"})
                        ])
                    )
                ], width=4),
                
                # Right side - Data preview
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Data Preview", className="card-title"),
                            dash_table.DataTable(
                                id="data-table",
                                columns=[
                                    {"name": "Date", "id": "date"},
                                    {"name": "Open", "id": "open"},
                                    {"name": "High", "id": "high"},
                                    {"name": "Low", "id": "low"},
                                    {"name": "Close", "id": "close"},
                                    {"name": "Volume", "id": "volume"}
                                ],
                                page_size=15,
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '5px',
                                    'minWidth': '60px', 
                                    'maxWidth': '120px',
                                    'whiteSpace': 'normal'
                                },
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                }
                            ),
                            
                            html.Div(id="data-info", className="mt-3")
                        ])
                    )
                ], width=8)
            ])
        ])
    )

def create_backtest_tab():
    """Create the Backtest tab"""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Left side - Controls
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Backtest Configuration", className="card-title"),
                            
                            dbc.FormGroup([
                                dbc.Label("Initial Capital:"),
                                dbc.Input(id="capital-input", type="number", value=10000, min=100)
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("Risk Per Trade (%):"),
                                dbc.Input(id="risk-input", type="number", value=2, min=0.1, max=10, step=0.1)
                            ]),
                            
                            dbc.Button("Run Backtest", id="backtest-button", color="primary", className="mt-3")
                        ])
                    ),
                    
                    html.Div(id="backtest-status", className="mt-3")
                ], width=3),
                
                # Right side - Results
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Equity Curve", tab_id="tab-equity",
                                children=[
                                    dcc.Graph(id="equity-graph", style={"height": "60vh"})
                                ]),
                        
                        dbc.Tab(label="Trades", tab_id="tab-trades",
                                children=[
                                    dash_table.DataTable(
                                        id="trades-table",
                                        columns=[
                                            {"name": "#", "id": "trade_no"},
                                            {"name": "Direction", "id": "direction"},
                                            {"name": "Entry Date", "id": "entry_date"},
                                            {"name": "Entry Price", "id": "entry_price"},
                                            {"name": "Exit Date", "id": "exit_date"},
                                            {"name": "Exit Price", "id": "exit_price"},
                                            {"name": "Profit", "id": "profit"},
                                            {"name": "Return %", "id": "return_pct"},
                                            {"name": "Exit Reason", "id": "exit_reason"}
                                        ],
                                        page_size=15,
                                        style_table={'overflowX': 'auto'},
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '5px',
                                            'minWidth': '60px', 
                                            'maxWidth': '120px',
                                            'whiteSpace': 'normal'
                                        },
                                        style_header={
                                            'backgroundColor': 'rgb(230, 230, 230)',
                                            'fontWeight': 'bold'
                                        }
                                    )
                                ]),
                        
                        dbc.Tab(label="Metrics", tab_id="tab-metrics",
                                children=[
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H5("Performance Metrics", className="card-title"),
                                            
                                            html.Div([
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.P("Total Trades:"),
                                                        html.P("Win Rate:"),
                                                        html.P("Profit Factor:"),
                                                        html.P("Total Return:"),
                                                        html.P("Max Drawdown:"),
                                                    ], width=4),
                                                    dbc.Col([
                                                        html.P(id="metric-total-trades", children="-"),
                                                        html.P(id="metric-win-rate", children="-"),
                                                        html.P(id="metric-profit-factor", children="-"),
                                                        html.P(id="metric-total-return", children="-"),
                                                        html.P(id="metric-max-drawdown", children="-"),
                                                    ], width=2),
                                                    dbc.Col([
                                                        html.P("Winning Trades:"),
                                                        html.P("Losing Trades:"),
                                                        html.P("Average Win:"),
                                                        html.P("Average Loss:"),
                                                        html.P("Largest Win:"),
                                                        html.P("Largest Loss:"),
                                                    ], width=4),
                                                    dbc.Col([
                                                        html.P(id="metric-winning-trades", children="-"),
                                                        html.P(id="metric-losing-trades", children="-"),
                                                        html.P(id="metric-avg-win", children="-"),
                                                        html.P(id="metric-avg-loss", children="-"),
                                                        html.P(id="metric-largest-win", children="-"),
                                                        html.P(id="metric-largest-loss", children="-"),
                                                    ], width=2),
                                                ])
                                            ])
                                        ])
                                    )
                                ]),
                    ], id="backtest-tabs", active_tab="tab-equity")
                ], width=9)
            ])
        ])
    )

def create_live_trading_tab():
    """Create the Live Trading tab"""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Left side - Controls
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Live Trading Configuration", className="card-title"),
                            
                            dbc.FormGroup([
                                dbc.Label("Exchange:"),
                                dcc.Dropdown(
                                    id="exchange-dropdown",
                                    options=[
                                        {'label': 'Binance', 'value': 'binance'},
                                        {'label': 'Coinbase', 'value': 'coinbase'},
                                        {'label': 'Kraken', 'value': 'kraken'},
                                        {'label': 'KuCoin', 'value': 'kucoin'}
                                    ],
                                    value='binance'
                                )
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("Symbol:"),
                                dbc.Input(id="trading-symbol-input", type="text", value="BTC/USDT", placeholder="Enter trading pair")
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("Timeframe:"),
                                dcc.Dropdown(
                                    id="trading-timeframe-dropdown",
                                    options=[
                                        {'label': '1 Minute', 'value': '1m'},
                                        {'label': '5 Minutes', 'value': '5m'},
                                        {'label': '15 Minutes', 'value': '15m'},
                                        {'label': '30 Minutes', 'value': '30m'},
                                        {'label': '1 Hour', 'value': '1h'},
                                        {'label': '4 Hours', 'value': '4h'},
                                        {'label': '1 Day', 'value': '1d'}
                                    ],
                                    value='1h'
                                )
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("API Key:"),
                                dbc.Input(id="api-key-input", type="password", placeholder="Enter API key")
                            ]),
                            
                            dbc.FormGroup([
                                dbc.Label("API Secret:"),
                                dbc.Input(id="api-secret-input", type="password", placeholder="Enter API secret")
                            ]),
                            
                            dbc.Button("Connect", id="connect-button", color="primary", className="mr-2"),
                            
                            dbc.FormGroup([
                                dbc.Checkbox(id="trading-status-checkbox", className="mr-2"),
                                dbc.Label("Enable Trading", html_for="trading-status-checkbox")
                            ], className="mt-3")
                        ])
                    )
                ], width=4),
                
                # Right side - Live data and signals
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Latest Signal", className="card-title"),
                            html.Div(id="signal-div", className="p-3 mb-3 bg-light rounded")
                        ]),
                        className="mb-3"
                    ),
                    
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Live Chart", className="card-title"),
                            dcc.Graph(id="live-graph", style={"height": "40vh"})
                        ]),
                        className="mb-3"
                    ),
                    
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Activity Log", className="card-title"),
                            dbc.ListGroup(id="log-list", className="overflow-auto", style={"max-height": "200px"})
                        ])
                    )
                ], width=8)
            ])
        ])
    )

def create_settings_tab():
    """Create the Settings tab"""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Basic settings
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Strategy Parameters", className="card-title"),
                            
                            html.H6("Moving Averages", className="mt-3"),
                            dbc.FormGroup([
                                dbc.Label("SMA Period 1:"),
                                dbc.Input(id="sma1-input", type="number", value=global_store['settings']['sma1'], min=5, max=200, step=1)
                            ]),
                            dbc.FormGroup([
                                dbc.Label("SMA Period 2:"),
                                dbc.Input(id="sma2-input", type="number", value=global_store['settings']['sma2'], min=10, max=300, step=1)
                            ]),
                            dbc.FormGroup([
                                dbc.Label("SMA Period 3:"),
                                dbc.Input(id="sma3-input", type="number", value=global_store['settings']['sma3'], min=50, max=500, step=1)
                            ]),
                            
                            html.H6("ATR Settings", className="mt-3"),
                            dbc.FormGroup([
                                dbc.Label("ATR Period:"),
                                dbc.Input(id="atr-period-input", type="number", value=global_store['settings']['atr_period'], min=5, max=50, step=1)
                            ]),
                            dbc.FormGroup([
                                dbc.Label("ATR Stop Multiplier:"),
                                dbc.Input(id="atr-stop-input", type="number", value=global_store['settings']['atr_stop'], min=0.5, max=5, step=0.1)
                            ]),
                            
                            html.H6("Exit Settings", className="mt-3"),
                            dbc.FormGroup([
                                dbc.Label("Risk-Reward Ratio:"),
                                dbc.Input(id="rr-ratio-input", type="number", value=global_store['settings']['rr_ratio'], min=1, max=5, step=0.1)
                            ]),
                            dbc.FormGroup([
                                dbc.Checkbox(id="trailing-stop-checkbox", className="mr-2"),
                                dbc.Label("Enable Trailing Stop", html_for="trailing-stop-checkbox")
                            ]),
                            
                            dbc.Button("Save Settings", id="save-settings-button", color="primary", className="mt-3")
                        ])
                    )
                ], width=6),
                
                # Advanced settings
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Advanced Strategy Parameters", className="card-title"),
                            
                            html.H6("Market Structure Settings", className="mt-3"),
                            dbc.FormGroup([
                                dbc.Label("Structure Window:"),
                                dbc.Input(id="structure-window-input", type="number", value=global_store['settings']['structure_window'], min=5, max=30, step=1)
                            ]),
                            
                            html.H6("OTE Settings", className="mt-3"),
                            dbc.FormGroup([
                                dbc.Label("OTE Lower Level:"),
                                dbc.Input(id="ote-lower-input", type="number", value=global_store['settings']['ote_lower'], min=0.5, max=0.7, step=0.01)
                            ]),
                            dbc.FormGroup([
                                dbc.Label("OTE Upper Level:"),
                                dbc.Input(id="ote-upper-input", type="number", value=global_store['settings']['ote_upper'], min=0.7, max=0.9, step=0.01)
                            ]),
                            
                            dbc.Button("Reset to Defaults", id="reset-settings-button", color="secondary", className="mt-3")
                        ])
                    ),
                    
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Export/Import Settings", className="card-title"),
                            
                            dbc.Button("Export Settings", id="export-settings-button", color="info", className="mr-2"),
                            dbc.Button("Import Settings", id="import-settings-button", color="info"),
                            
                            # Hidden div for settings upload
                            html.Div([
                                dcc.Upload(
                                    id='upload-settings',
                                    children=html.Div(['Drag and Drop or ', html.A('Select Settings File')]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    multiple=False
                                )
                            ], id="upload-settings-div", style={"display": "none"})
                        ]),
                        className="mt-3"
                    )
                ], width=6)
            ])
        ])
    )

# Create main layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("ICT Trading Strategy", className="mt-3 mb-3"),
            dbc.Tabs([
                dbc.Tab(label="Data", tab_id="tab-data", children=[create_data_tab()]),
                dbc.Tab(label="Backtest", tab_id="tab-backtest", children=[create_backtest_tab()]),
                dbc.Tab(label="Live Trading", tab_id="tab-live-trading", children=[create_live_trading_tab()]),
                dbc.Tab(label="Settings", tab_id="tab-settings", children=[create_settings_tab()]),
            ], id="tabs", active_tab="tab-data")
        ])
    ]),
    
    # Notifications area
    html.Div(id="notifications-area"),
    
    # Hidden divs for storing data
    dcc.Store(id="backtest-store"),
    dcc.Store(id="settings-store", data=global_store['settings']),
    dcc.Store(id="log-store", data=[]),
    
    # Interval for live updates
    dcc.Interval(id="live-update-interval", interval=10000, n_intervals=0, disabled=True),
    
    # Download component for exporting settings
    dcc.Download(id="download-settings")
], fluid=True)

# Callbacks

# Data tab callbacks
@app.callback(
    [Output("data-table", "data"), Output("data-info", "children"), Output("notifications-area", "children")],
    [Input("import-button", "n_clicks"), Input("upload-data", "contents")],
    [State("symbol-input", "value"), State("timeframe-dropdown", "value"), 
     State("start-date-picker", "date"), State("end-date-picker", "date"), 
     State("source-dropdown", "value"), State("upload-data", "filename")]
)
def update_data(n_clicks, contents, symbol, timeframe, start_date, end_date, source, filename):
    ctx = dash.callback_context
    
    # Initialize outputs
    data_table = []
    data_info = ""
    notification = None
    
    if not ctx.triggered:
        return data_table, data_info, notification
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        if trigger_id == "import-button" and n_clicks:
            # Import data from online source
            df = fetch_historical_data(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                source=source
            )
            
            # Store data globally
            global_store['data'] = df
            
            # Success notification
            notification = dbc.Alert(
                f"Data imported successfully: {len(df)} rows",
                color="success",
                dismissable=True,
                duration=4000
            )
            
        elif trigger_id == "upload-data" and contents:
            # Parse uploaded CSV
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            try:
                # Try to read CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                
                # Check for date column
                if 'date' in df.columns:
                    date_col = 'date'
                elif 'datetime' in df.columns:
                    date_col = 'datetime'
                elif 'timestamp' in df.columns:
                    date_col = 'timestamp'
                else:
                    # Try to use the first column as the date
                    date_col = df.columns[0]
                
                # Convert date column to datetime and set as index
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                
                # Ensure all column names are lowercase
                df.columns = [col.lower() for col in df.columns]
                
                # Check for required columns
                required_cols = ['open', 'high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                
                # Add volume column if missing
                if 'volume' not in df.columns:
                    df['volume'] = 0
                
                # Store data globally
                global_store['data'] = df
                
                # Success notification
                notification = dbc.Alert(
                    f"CSV file '{filename}' loaded successfully: {len(df)} rows",
                    color="success",
                    dismissable=True,
                    duration=4000
                )
                
            except Exception as e:
                # Error notification
                notification = dbc.Alert(
                    f"Error loading CSV file: {str(e)}",
                    color="danger",
                    dismissable=True
                )
                return data_table, data_info, notification
        
        # Update data table and info
        if global_store['data'] is not None:
            df = global_store['data']
            
            # Prepare data for table
            data_table = []
            max_rows = min(100, len(df))
            for i in range(max_rows):
                date = df.index[i]
                row = df.iloc[i]
                
                # Format date
                date_str = date.strftime('%Y-%m-%d %H:%M')
                
                # Format values
                data_table.append({
                    "date": date_str,
                    "open": f"{row['open']:.4f}",
                    "high": f"{row['high']:.4f}",
                    "low": f"{row['low']:.4f}",
                    "close": f"{row['close']:.4f}",
                    "volume": f"{row['volume']:.0f}"
                })
            
            # Update info
            data_info = html.Div([
                html.P(f"Total rows: {len(df)}"),
                html.P(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"),
                html.P(f"Last close: {df['close'].iloc[-1]:.4f}")
            ])
    
    except Exception as e:
        # Error notification
        notification = dbc.Alert(
            f"Error: {str(e)}",
            color="danger",
            dismissable=True
        )
    
    return data_table, data_info, notification

# Show/hide CSV upload div
@app.callback(
    Output("upload-div", "style"),
    [Input("upload-button", "n_clicks")]
)
def toggle_upload_div(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}
    else:
        return {"display": "none"}

# Backtest tab callbacks
@app.callback(
    [Output("backtest-store", "data"), 
     Output("backtest-status", "children")],
    [Input("backtest-button", "n_clicks")],
    [State("capital-input", "value"), 
     State("risk-input", "value"),
     State("settings-store", "data")]
)
def run_backtest(n_clicks, capital, risk, settings):
    if not n_clicks:
        return None, ""
    
    if global_store['data'] is None:
        return None, dbc.Alert("No data available. Please import data first.", color="warning")
    
    status = dbc.Alert("Running backtest...", color="info")
    
    try:
        # Initialize the ICT strategy with current settings
        strategy = ICTStrategy(
            data=global_store['data'].copy(),
            capital=capital,
            risk_per_trade=risk/100
        )
        
        # Apply custom settings
        # (In a real implementation, you'd add methods to the ICTStrategy class
        # to allow changing these settings)
        
        # Run backtest
        results = strategy.backtest()
        
        # Store results globally
        global_store['strategy'] = strategy
        global_store['backtest_results'] = results
        
        # Prepare data to store
        store_data = {
            'equity_curve': [float(x) for x in strategy.equity_curve],
            'trades': strategy.trades,
            'performance': strategy.performance
        }
        
        # Update status
        status = dbc.Alert(f"Backtest completed successfully. {len(strategy.trades)} trades executed.", color="success")
        
        return store_data, status
        
    except Exception as e:
        status = dbc.Alert(f"Error running backtest: {str(e)}", color="danger")
        traceback.print_exc()
        return None, status

# Update equity chart
@app.callback(
    Output("equity-graph", "figure"),
    [Input("backtest-store", "data")]
)
def update_equity_chart(data):
    if not data:
        # Return empty figure if no data
        return {
            'data': [],
            'layout': {
                'title': 'Equity Curve',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Capital'}
            }
        }
    
    # Get data
    equity_curve = data['equity_curve']
    trades = data['trades']
    
    # Create dates array (simplified, in real app would use strategy's dates)
    dates = [str(i) for i in range(len(equity_curve))]
    
    # Create figure
    fig = {
        'data': [
            {
                'x': dates,
                'y': equity_curve,
                'type': 'line',
                'name': 'Equity'
            }
        ],
        'layout': {
            'title': 'Equity Curve',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Capital'}
        }
    }
    
    # Add trades
    if trades:
        # Buy signals
        buy_indices = [list(dates).index(str(t['entry_date'])) for t in trades if t['direction'] == 1]
        buy_values = [equity_curve[i] for i in buy_indices if i < len(equity_curve)]
        
        if buy_indices:
            fig['data'].append({
                'x': [dates[i] for i in buy_indices if i < len(equity_curve)],
                'y': buy_values,
                'mode': 'markers',
                'marker': {
                    'symbol': 'triangle-up',
                    'size': 12,
                    'color': 'green'
                },
                'name': 'Buy'
            })
        
        # Sell signals
        sell_indices = [list(dates).index(str(t['entry_date'])) for t in trades if t['direction'] == -1]
        sell_values = [equity_curve[i] for i in sell_indices if i < len(equity_curve)]
        
        if sell_indices:
            fig['data'].append({
                'x': [dates[i] for i in sell_indices if i < len(equity_curve)],
                'y': sell_values,
                'mode': 'markers',
                'marker': {
                    'symbol': 'triangle-down',
                    'size': 12,
                    'color': 'red'
                },
                'name': 'Sell'
            })
    
    return fig

# Update trades table
@app.callback(
    Output("trades-table", "data"),
    [Input("backtest-store", "data")]
)
def update_trades_table(data):
    if not data or 'trades' not in data:
        return []
    
    trades = data['trades']
    
    # Prepare data for table
    table_data = []
    for i, trade in enumerate(trades):
        direction = "LONG" if trade['direction'] == 1 else "SHORT"
        
        # Format values
        entry_date = str(trade['entry_date'])
        exit_date = str(trade['exit_date'])
        entry_price = f"{trade['entry_price']:.4f}"
        exit_price = f"{trade['exit_price']:.4f}"
        profit = f"{trade['profit']:.2f}"
        return_pct = f"{trade['return_pct']:.2f}%"
        
        # Add to table
        table_data.append({
            "trade_no": i+1,
            "direction": direction,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "profit": profit,
            "return_pct": return_pct,
            "exit_reason": trade['exit_reason']
        })
    
    return table_data

# Update metrics
@app.callback(
    [Output("metric-total-trades", "children"),
     Output("metric-win-rate", "children"),
     Output("metric-profit-factor", "children"),
     Output("metric-total-return", "children"),
     Output("metric-max-drawdown", "children"),
     Output("metric-winning-trades", "children"),
     Output("metric-losing-trades", "children"),
     Output("metric-avg-win", "children"),
     Output("metric-avg-loss", "children"),
     Output("metric-largest-win", "children"),
     Output("metric-largest-loss", "children")],
    [Input("backtest-store", "data")]
)
def update_metrics(data):
    if not data or 'performance' not in data:
        return ["-"] * 11
    
    perf = data['performance']
    
    return [
        str(perf['total_trades']),
        f"{perf['win_rate']:.2%}",
        f"{perf['profit_factor']:.2f}",
        f"{perf['total_return']:.2f}%",
        f"{perf['max_drawdown']:.2f}%",
        str(perf['winning_trades']),
        str(perf['losing_trades']),
        f"${perf['average_win']:.2f}",
        f"${perf['average_loss']:.2f}",
        f"${perf['largest_win']:.2f}",
        f"${perf['largest_loss']:.2f}"
    ]

# Settings callbacks
@app.callback(
    Output("settings-store", "data"),
    [Input("save-settings-button", "n_clicks"), 
     Input("reset-settings-button", "n_clicks"),
     Input("upload-settings", "contents")],
    [State("sma1-input", "value"),
     State("sma2-input", "value"),
     State("sma3-input", "value"),
     State("atr-period-input", "value"),
     State("atr-stop-input", "value"),
     State("rr-ratio-input", "value"),
     State("trailing-stop-checkbox", "checked"),
     State("structure-window-input", "value"),
     State("ote-lower-input", "value"),
     State("ote-upper-input", "value"),
     State("settings-store", "data"),
     State("upload-settings", "filename")]
)
def update_settings(save_clicks, reset_clicks, contents, 
                    sma1, sma2, sma3, atr_period, atr_stop, rr_ratio, 
                    trailing_stop, structure_window, ote_lower, ote_upper, 
                    current_settings, filename):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_settings
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "save-settings-button" and save_clicks:
        # Save new settings
        new_settings = {
            'sma1': sma1,
            'sma2': sma2,
            'sma3': sma3,
            'atr_period': atr_period,
            'atr_stop': atr_stop,
            'rr_ratio': rr_ratio,
            'trailing_stop': trailing_stop,
            'structure_window': structure_window,
            'ote_lower': ote_lower,
            'ote_upper': ote_upper
        }
        
        # Update global store
        global_store['settings'] = new_settings
        
        return new_settings
        
    elif trigger_id == "reset-settings-button" and reset_clicks:
        # Reset to default settings
        default_settings = {
            'sma1': 20,
            'sma2': 50,
            'sma3': 200,
            'atr_period': 14,
            'atr_stop': 2.0,
            'rr_ratio': 3.0,
            'trailing_stop': False,
            'structure_window': 10,
            'ote_lower': 0.65,
            'ote_upper': 0.79
        }
        
        # Update global store
        global_store['settings'] = default_settings
        
        return default_settings
        
    elif trigger_id == "upload-settings" and contents:
        # Parse uploaded settings file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            # Try to read JSON settings
            uploaded_settings = json.loads(decoded.decode('utf-8'))
            
            # Validate required keys
            required_keys = ['sma1', 'sma2', 'sma3', 'atr_period', 'atr_stop', 
                            'rr_ratio', 'structure_window', 'ote_lower', 'ote_upper']
            
            missing_keys = [key for key in required_keys if key not in uploaded_settings]
            
            if missing_keys:
                raise ValueError(f"Missing settings keys: {', '.join(missing_keys)}")
            
            # Update global store
            global_store['settings'] = uploaded_settings
            
            return uploaded_settings
            
        except Exception as e:
            # Return current settings if error
            return current_settings
    
    # If no trigger matched, return current settings
    return current_settings

# Export settings
@app.callback(
    Output("download-settings", "data"),
    [Input("export-settings-button", "n_clicks")],
    [State("settings-store", "data")]
)
def export_settings(n_clicks, settings):
    if not n_clicks:
        return None
    
    # Prepare settings for download
    return dict(
        content=json.dumps(settings, indent=4),
        filename="ict_strategy_settings.json"
    )

# Show/hide settings upload div
@app.callback(
    Output("upload-settings-div", "style"),
    [Input("import-settings-button", "n_clicks")]
)
def toggle_settings_upload_div(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}
    else:
        return {"display": "none"}

# Live trading callbacks
@app.callback(
    Output("signal-div", "children"),
    [Input("connect-button", "n_clicks"),
     Input("live-update-interval", "n_intervals")],
    [State("exchange-dropdown", "value"),
     State("trading-symbol-input", "value"),
     State("trading-timeframe-dropdown", "value"),
     State("api-key-input", "value"),
     State("api-secret-input", "value")]
)
def update_live_signal(connect_clicks, n_intervals, exchange, symbol, timeframe, api_key, api_secret):
    ctx = dash.callback_context
    
    if not ctx.triggered or not connect_clicks:
        return "No active signal"
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "connect-button":
        # On connect button click, we'd actually connect to the exchange API
        # For demo purposes, we'll just return a message
        return html.Div([
            html.H4("Connected to Exchange", className="text-success"),
            html.P(f"Exchange: {exchange}"),
            html.P(f"Symbol: {symbol}"),
            html.P(f"Timeframe: {timeframe}"),
            html.P("Waiting for signals...")
        ])
    
    elif trigger_id == "live-update-interval" and n_intervals:
        # In a real app, this would fetch data from the exchange
        # and run the strategy to generate signals
        
        # For demo purposes, randomly generate a signal
        signal_type = np.random.choice(["BUY", "SELL", "NONE"], p=[0.1, 0.1, 0.8])
        
        if signal_type == "BUY":
            return html.Div([
                html.H4("BUY SIGNAL", className="text-success"),
                html.P(f"Symbol: {symbol}"),
                html.P(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P("Entry Price: 50,000.00"),
                html.P("Stop Loss: 49,000.00"),
                html.P("Take Profit: 53,000.00")
            ])
        elif signal_type == "SELL":
            return html.Div([
                html.H4("SELL SIGNAL", className="text-danger"),
                html.P(f"Symbol: {symbol}"),
                html.P(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P("Entry Price: 50,000.00"),
                html.P("Stop Loss: 51,000.00"),
                html.P("Take Profit: 47,000.00")
            ])
        else:
            return html.Div([
                html.H4("No Signal", className="text-muted"),
                html.P(f"Symbol: {symbol}"),
                html.P(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                html.P("Waiting for trading setup...")
            ])
    
    # Default
    return "No active signal"

# Update live chart
@app.callback(
    Output("live-graph", "figure"),
    [Input("live-update-interval", "n_intervals")],
    [State("trading-symbol-input", "value")]
)
def update_live_chart(n_intervals, symbol):
    if not n_intervals:
        # Return empty figure
        return {
            'data': [],
            'layout': {
                'title': 'Live Chart',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Price'}
            }
        }
    
    # In a real app, this would use real market data
    # For demo purposes, generate random price data
    n_points = 100
    dates = [datetime.now() - timedelta(minutes=i) for i in range(n_points)]
    
    # Start with a base price
    base_price = 50000
    
    # Add some trend and randomness
    np.random.seed(42 + n_intervals)  # Change seed each update for variety
    trend = np.linspace(0, np.random.choice([-1000, 1000]), n_points)
    noise = np.random.normal(0, 100, n_points)
    
    # Calculate prices
    close_prices = base_price + trend + noise
    
    # Create figure
    fig = {
        'data': [
            {
                'x': dates,
                'y': close_prices,
                'type': 'line',
                'name': symbol
            }
        ],
        'layout': {
            'title': f'{symbol} - Live Chart',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price'},
            'uirevision': 'constant'  # Keep zoom level on updates
        }
    }
    
    return fig

# Update log
@app.callback(
    Output("log-list", "children"),
    [Input("connect-button", "n_clicks"),
     Input("trading-status-checkbox", "checked"),
     Input("live-update-interval", "n_intervals")],
    [State("log-list", "children")]
)
def update_log(connect_clicks, trading_enabled, n_intervals, current_log):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_log or []
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Initialize if None
    if current_log is None:
        current_log = []
    
    # Get timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if trigger_id == "connect-button" and connect_clicks:
        # Add connection log entry
        new_entry = dbc.ListGroupItem(f"[{timestamp}] Connected to exchange")
        return [new_entry] + current_log
    
    elif trigger_id == "trading-status-checkbox":
        # Add trading status log entry
        status = "enabled" if trading_enabled else "disabled"
        new_entry = dbc.ListGroupItem(f"[{timestamp}] Trading {status}")
        return [new_entry] + current_log
    
    elif trigger_id == "live-update-interval" and n_intervals and n_intervals % 5 == 0:
        # Add periodic update log (every 5 intervals)
        new_entry = dbc.ListGroupItem(f"[{timestamp}] Checking for signals...")
        return [new_entry] + current_log
    
    # Return current log if no new entry
    return current_log

# Enable/disable live updates based on trading status
@app.callback(
    Output("live-update-interval", "disabled"),
    [Input("trading-status-checkbox", "checked")]
)
def toggle_live_updates(trading_enabled):
    return not trading_enabled

# Main entry point
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
