"""
ICT Trading Strategy Automation
Based on Inner Circle Trader (ICT) methodology by Michael J. Huddleston

Key concepts implemented:
1. Order Blocks
2. Breaker Blocks
3. Liquidity Analysis
4. Market Structure Shifts
5. Optimal Trade Entry (OTE)

This script provides a backtesting framework for the ICT strategy
and can be connected to a broker API for live trading.

Requirements:
- pandas
- numpy
- matplotlib
- ta-lib (Technical Analysis Library)
- ccxt (for exchange connectivity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

# For Technical Analysis
try:
    import talib as ta
except ImportError:
    print("TA-Lib not installed. To install: pip install ta-lib")

# For trading execution (optional)
try:
    import ccxt
except ImportError:
    print("CCXT not installed. To install: pip install ccxt")

class ICTStrategy:
    def __init__(self, data, timeframe='1h', capital=10000, risk_per_trade=0.02):
        """
        Initialize the ICT Strategy
        
        Parameters:
        -----------
        data: pandas DataFrame with OHLCV data
            Must contain columns: 'open', 'high', 'low', 'close', 'volume'
        timeframe: str
            Timeframe of the data
        capital: float
            Initial capital
        risk_per_trade: float
            Risk per trade as a percentage of capital (0.02 = 2%)
        """
        self.data = data.copy()
        self.timeframe = timeframe
        self.capital = capital
        self.initial_capital = capital
        self.risk_per_trade = risk_per_trade
        self.positions = []
        self.trades = []
        self.current_position = None
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain column: {col}")
        
        # Ensure data is sorted by index (assumed to be datetime)
        self.data = self.data.sort_index()
        
        # Add indicator columns
        self._add_indicators()
    
    def _add_indicators(self):
        """Add all required indicators for the strategy"""
        # Basic indicators
        self._add_moving_averages()
        self._add_atr()
        
        # ICT specific
        self._identify_structure()
        self._identify_order_blocks()
        self._identify_breaker_blocks() 
        self._identify_liquidity_levels()
        self._calculate_ote_levels()
    
    def _add_moving_averages(self):
        """Add moving averages"""
        # Simple Moving Averages
        self.data['sma_20'] = ta.SMA(self.data['close'].values, timeperiod=20)
        self.data['sma_50'] = ta.SMA(self.data['close'].values, timeperiod=50)
        self.data['sma_200'] = ta.SMA(self.data['close'].values, timeperiod=200)
        
        # Exponential Moving Averages
        self.data['ema_8'] = ta.EMA(self.data['close'].values, timeperiod=8)
        self.data['ema_21'] = ta.EMA(self.data['close'].values, timeperiod=21)
    
    def _add_atr(self, period=14):
        """Add Average True Range indicator"""
        self.data['atr'] = ta.ATR(self.data['high'].values, 
                                  self.data['low'].values, 
                                  self.data['close'].values, 
                                  timeperiod=period)
    
    def _identify_structure(self, window=10):
        """
        Identify market structure (higher highs, lower lows, etc.)
        
        Market structure shifts occur when a trend changes direction
        - Bearish shift: previous uptrend breaks a significant low
        - Bullish shift: previous downtrend breaks a significant high
        """
        self.data['highest_high'] = self.data['high'].rolling(window=window).max()
        self.data['lowest_low'] = self.data['low'].rolling(window=window).min()
        
        # Initialize structure columns
        self.data['structure'] = 0  # 1 for bullish, -1 for bearish, 0 for neutral
        self.data['structure_shift'] = 0  # 1 for bullish shift, -1 for bearish shift
        
        # Loop through data starting from window size
        for i in range(window, len(self.data)):
            # Get previous structure
            prev_structure = self.data['structure'].iloc[i-1]
            
            # Check for higher highs and higher lows (bullish structure)
            if (self.data['high'].iloc[i] > self.data['highest_high'].iloc[i-1] and 
                self.data['low'].iloc[i] > self.data['lowest_low'].iloc[i-window]):
                self.data.loc[self.data.index[i], 'structure'] = 1
                
                # Check for structure shift
                if prev_structure == -1:
                    self.data.loc[self.data.index[i], 'structure_shift'] = 1
            
            # Check for lower highs and lower lows (bearish structure)
            elif (self.data['high'].iloc[i] < self.data['highest_high'].iloc[i-window] and 
                  self.data['low'].iloc[i] < self.data['lowest_low'].iloc[i-1]):
                self.data.loc[self.data.index[i], 'structure'] = -1
                
                # Check for structure shift
                if prev_structure == 1:
                    self.data.loc[self.data.index[i], 'structure_shift'] = -1
            
            # Maintain previous structure if no clear change
            else:
                self.data.loc[self.data.index[i], 'structure'] = prev_structure
    
    def _identify_order_blocks(self, window=20):
        """
        Identify potential order blocks
        
        Order blocks are candles that precede a strong move in the opposite direction.
        - Bullish order block: a down candle that precedes a strong bullish move
        - Bearish order block: an up candle that precedes a strong bearish move
        """
        # Initialize columns
        self.data['bullish_ob'] = 0
        self.data['bearish_ob'] = 0
        self.data['ob_strength'] = 0
        
        # Loop through data starting from a few candles in
        for i in range(3, len(self.data) - 1):
            # Check for bullish order block (down candle before up move)
            if (self.data['close'].iloc[i-1] < self.data['open'].iloc[i-1] and  # down candle
                self.data['high'].iloc[i] > self.data['high'].iloc[i-1] and     # next candle breaks high
                self.data['close'].iloc[i] > self.data['open'].iloc[i] and      # next candle is bullish
                self.data['close'].iloc[i] > self.data['close'].iloc[i-1]):     # significant move up
                
                # Calculate strength based on the following move
                strength = (self.data['close'].iloc[i] - self.data['low'].iloc[i-1]) / self.data['low'].iloc[i-1]
                
                # Mark order block if strong enough
                if strength > 0.002:  # 0.2% minimum move
                    self.data.loc[self.data.index[i-1], 'bullish_ob'] = 1
                    self.data.loc[self.data.index[i-1], 'ob_strength'] = strength
            
            # Check for bearish order block (up candle before down move)
            elif (self.data['close'].iloc[i-1] > self.data['open'].iloc[i-1] and  # up candle
                  self.data['low'].iloc[i] < self.data['low'].iloc[i-1] and       # next candle breaks low
                  self.data['close'].iloc[i] < self.data['open'].iloc[i] and      # next candle is bearish
                  self.data['close'].iloc[i] < self.data['close'].iloc[i-1]):     # significant move down
                
                # Calculate strength based on the following move
                strength = (self.data['high'].iloc[i-1] - self.data['close'].iloc[i]) / self.data['high'].iloc[i-1]
                
                # Mark order block if strong enough
                if strength > 0.002:  # 0.2% minimum move
                    self.data.loc[self.data.index[i-1], 'bearish_ob'] = 1
                    self.data.loc[self.data.index[i-1], 'ob_strength'] = strength
        
        # Add columns for order block zones (price areas)
        self.data['bullish_ob_low'] = 0.0
        self.data['bullish_ob_high'] = 0.0
        self.data['bearish_ob_low'] = 0.0
        self.data['bearish_ob_high'] = 0.0
        
        # Fill in order block price levels
        for i in range(len(self.data)):
            if self.data['bullish_ob'].iloc[i] == 1:
                self.data.loc[self.data.index[i], 'bullish_ob_low'] = self.data['low'].iloc[i]
                self.data.loc[self.data.index[i], 'bullish_ob_high'] = self.data['open'].iloc[i]
            
            if self.data['bearish_ob'].iloc[i] == 1:
                self.data.loc[self.data.index[i], 'bearish_ob_low'] = self.data['close'].iloc[i]
                self.data.loc[self.data.index[i], 'bearish_ob_high'] = self.data['high'].iloc[i]
    
    def _identify_breaker_blocks(self):
        """
        Identify breaker blocks
        
        A breaker block is formed when price breaks through a previous order block
        and the zone flips from resistance to support (or vice versa).
        """
        # Initialize columns
        self.data['bullish_breaker'] = 0
        self.data['bearish_breaker'] = 0
        
        # Track recent order blocks
        recent_bullish_obs = []
        recent_bearish_obs = []
        
        # Loop through data
        for i in range(len(self.data)):
            idx = self.data.index[i]
            
            # Add new order blocks to tracking
            if self.data['bullish_ob'].iloc[i] == 1:
                ob_data = {
                    'index': i,
                    'date': idx,
                    'low': self.data['bullish_ob_low'].iloc[i],
                    'high': self.data['bullish_ob_high'].iloc[i],
                    'broken': False
                }
                recent_bullish_obs.append(ob_data)
            
            if self.data['bearish_ob'].iloc[i] == 1:
                ob_data = {
                    'index': i,
                    'date': idx,
                    'low': self.data['bearish_ob_low'].iloc[i],
                    'high': self.data['bearish_ob_high'].iloc[i],
                    'broken': False
                }
                recent_bearish_obs.append(ob_data)
            
            # Check if price breaks bullish order blocks (creating bearish breakers)
            for ob in recent_bullish_obs:
                if not ob['broken'] and self.data['low'].iloc[i] < ob['low']:
                    # Price broke below the bullish order block
                    ob['broken'] = True
                    
                    # Look for retest of the broken zone
                    if i > ob['index'] + 2:  # Ensure there's a gap between break and retest
                        self.data.loc[idx, 'bearish_breaker'] = 1
            
            # Check if price breaks bearish order blocks (creating bullish breakers)
            for ob in recent_bearish_obs:
                if not ob['broken'] and self.data['high'].iloc[i] > ob['high']:
                    # Price broke above the bearish order block
                    ob['broken'] = True
                    
                    # Look for retest of the broken zone
                    if i > ob['index'] + 2:  # Ensure there's a gap between break and retest
                        self.data.loc[idx, 'bullish_breaker'] = 1
            
            # Remove old order blocks (older than 50 candles)
            recent_bullish_obs = [ob for ob in recent_bullish_obs if i - ob['index'] < 50]
            recent_bearish_obs = [ob for ob in recent_bearish_obs if i - ob['index'] < 50]
    
    def _identify_liquidity_levels(self, window=20):
        """
        Identify liquidity levels where stop orders might be clustered
        
        This includes:
        - Swing highs (potential sell-side liquidity)
        - Swing lows (potential buy-side liquidity)
        - Recent session highs/lows
        - Round numbers
        """
        # Initialize columns
        self.data['buy_liquidity'] = 0
        self.data['sell_liquidity'] = 0
        
        # Find swing points
        for i in range(window, len(self.data) - window):
            # Check for swing high
            if all(self.data['high'].iloc[i] > self.data['high'].iloc[i-j] for j in range(1, window)) and \
               all(self.data['high'].iloc[i] > self.data['high'].iloc[i+j] for j in range(1, window)):
                self.data.loc[self.data.index[i], 'sell_liquidity'] = 1
            
            # Check for swing low
            if all(self.data['low'].iloc[i] < self.data['low'].iloc[i-j] for j in range(1, window)) and \
               all(self.data['low'].iloc[i] < self.data['low'].iloc[i+j] for j in range(1, window)):
                self.data.loc[self.data.index[i], 'buy_liquidity'] = 1
    
    def _calculate_ote_levels(self):
        """
        Calculate Optimal Trade Entry (OTE) levels using Fibonacci retracements
        
        In ICT methodology, OTE levels are commonly at the 0.65-0.79 retracement zone
        """
        # Initialize columns
        self.data['ote_bullish'] = 0
        self.data['ote_bearish'] = 0
        
        # ICT OTE levels (0.65-0.79 is the primary OTE zone)
        ote_lower = 0.65
        ote_upper = 0.79
        
        # Find significant moves to calculate OTE levels
        for i in range(20, len(self.data) - 1):
            # Find bullish moves (for bearish OTE)
            if (self.data['close'].iloc[i] > self.data['open'].iloc[i] and
                self.data['close'].iloc[i] - self.data['open'].iloc[i] > 1.5 * self.data['atr'].iloc[i]):
                
                # Calculate the Fibonacci levels
                high = self.data['high'].iloc[i]
                low = min(self.data['low'].iloc[max(0, i-10):i])
                range_size = high - low
                
                # Calculate OTE zone
                ote_low = high - range_size * ote_upper
                ote_high = high - range_size * ote_lower
                
                # Mark OTE zone for next few candles (potential bearish entry)
                for j in range(1, min(10, len(self.data) - i - 1)):
                    if ote_low <= self.data['low'].iloc[i+j] <= ote_high:
                        self.data.loc[self.data.index[i+j], 'ote_bearish'] = 1
            
            # Find bearish moves (for bullish OTE)
            if (self.data['close'].iloc[i] < self.data['open'].iloc[i] and
                self.data['open'].iloc[i] - self.data['close'].iloc[i] > 1.5 * self.data['atr'].iloc[i]):
                
                # Calculate the Fibonacci levels
                low = self.data['low'].iloc[i]
                high = max(self.data['high'].iloc[max(0, i-10):i])
                range_size = high - low
                
                # Calculate OTE zone
                ote_low = low + range_size * ote_lower
                ote_high = low + range_size * ote_upper
                
                # Mark OTE zone for next few candles (potential bullish entry)
                for j in range(1, min(10, len(self.data) - i - 1)):
                    if ote_low <= self.data['high'].iloc[i+j] <= ote_high:
                        self.data.loc[self.data.index[i+j], 'ote_bullish'] = 1
    
    def generate_signals(self):
        """
        Generate trading signals based on ICT method
        
        Combines multiple ICT concepts:
        1. Market structure to determine trend direction
        2. Order blocks and breaker blocks for entry zones
        3. OTE (Optimal Trade Entry) for precise entries
        4. Liquidity levels for targets
        """
        # Initialize signal column
        self.data['signal'] = 0  # 1 for buy, -1 for sell, 0 for no signal
        
        for i in range(20, len(self.data)):
            # Only generate signals if not already in a position
            if self.current_position is None:
                # Bullish signal conditions:
                if (
                    # 1. Overall trend is bullish (market structure)
                    self.data['structure'].iloc[i] == 1 and
                    
                    # 2. Either at a bullish order block or breaker block
                    (self.data['bullish_ob'].iloc[i] == 1 or self.data['bullish_breaker'].iloc[i] == 1) and
                    
                    # 3. In an optimal trade entry zone
                    self.data['ote_bullish'].iloc[i] == 1 and
                    
                    # 4. Price is above key moving average
                    self.data['close'].iloc[i] > self.data['sma_50'].iloc[i]
                ):
                    self.data.loc[self.data.index[i], 'signal'] = 1
                
                # Bearish signal conditions:
                elif (
                    # 1. Overall trend is bearish (market structure)
                    self.data['structure'].iloc[i] == -1 and
                    
                    # 2. Either at a bearish order block or breaker block
                    (self.data['bearish_ob'].iloc[i] == 1 or self.data['bearish_breaker'].iloc[i] == 1) and
                    
                    # 3. In an optimal trade entry zone
                    self.data['ote_bearish'].iloc[i] == 1 and
                    
                    # 4. Price is below key moving average
                    self.data['close'].iloc[i] < self.data['sma_50'].iloc[i]
                ):
                    self.data.loc[self.data.index[i], 'signal'] = -1
        
        return self.data
    
    def backtest(self):
        """
        Run a backtest of the strategy
        
        This simulates trading based on the generated signals.
        """
        # Generate signals if not already generated
        if 'signal' not in self.data.columns:
            self.generate_signals()
        
        # Reset simulation variables
        self.capital = self.initial_capital
        self.equity_curve = [self.capital]
        self.trades = []
        self.current_position = None
        
        # Loop through data to simulate trading
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            current_price = self.data['close'].iloc[i]
            
            # Check for entry signals
            if self.data['signal'].iloc[i] != 0 and self.current_position is None:
                signal = self.data['signal'].iloc[i]
                
                # Calculate position size based on risk
                atr = self.data['atr'].iloc[i]
                stop_distance = 2 * atr  # Use 2 ATR for stop loss distance
                
                if signal == 1:  # Buy signal
                    stop_price = current_price - stop_distance
                else:  # Sell signal
                    stop_price = current_price + stop_distance
                
                risk_amount = self.capital * self.risk_per_trade
                position_size = risk_amount / abs(current_price - stop_price)
                
                # Enter position
                self.current_position = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'direction': signal,
                    'stop_price': stop_price,
                    'position_size': position_size,
                    'initial_risk': risk_amount
                }
            
            # Check for exit conditions if in a position
            elif self.current_position is not None:
                # Define take profit levels (3:1 risk-reward)
                if self.current_position['direction'] == 1:  # Long position
                    take_profit = self.current_position['entry_price'] + 3 * abs(self.current_position['entry_price'] - self.current_position['stop_price'])
                    
                    # Exit if stop hit
                    if self.data['low'].iloc[i] <= self.current_position['stop_price']:
                        exit_price = self.current_position['stop_price']
                        profit = (exit_price - self.current_position['entry_price']) * self.current_position['position_size']
                        exit_reason = 'Stop Loss'
                        self._close_position(current_date, exit_price, profit, exit_reason)
                    
                    # Exit if take profit hit
                    elif self.data['high'].iloc[i] >= take_profit:
                        exit_price = take_profit
                        profit = (exit_price - self.current_position['entry_price']) * self.current_position['position_size']
                        exit_reason = 'Take Profit'
                        self._close_position(current_date, exit_price, profit, exit_reason)
                    
                    # Exit if market structure shifts bearish
                    elif self.data['structure_shift'].iloc[i] == -1:
                        exit_price = current_price
                        profit = (exit_price - self.current_position['entry_price']) * self.current_position['position_size']
                        exit_reason = 'Structure Shift'
                        self._close_position(current_date, exit_price, profit, exit_reason)
                
                else:  # Short position
                    take_profit = self.current_position['entry_price'] - 3 * abs(self.current_position['entry_price'] - self.current_position['stop_price'])
                    
                    # Exit if stop hit
                    if self.data['high'].iloc[i] >= self.current_position['stop_price']:
                        exit_price = self.current_position['stop_price']
                        profit = (self.current_position['entry_price'] - exit_price) * self.current_position['position_size']
                        exit_reason = 'Stop Loss'
                        self._close_position(current_date, exit_price, profit, exit_reason)
                    
                    # Exit if take profit hit
                    elif self.data['low'].iloc[i] <= take_profit:
                        exit_price = take_profit
                        profit = (self.current_position['entry_price'] - exit_price) * self.current_position['position_size']
                        exit_reason = 'Take Profit'
                        self._close_position(current_date, exit_price, profit, exit_reason)
                    
                    # Exit if market structure shifts bullish
                    elif self.data['structure_shift'].iloc[i] == 1:
                        exit_price = current_price
                        profit = (self.current_position['entry_price'] - exit_price) * self.current_position['position_size']
                        exit_reason = 'Structure Shift'
                        self._close_position(current_date, exit_price, profit, exit_reason)
            
            # Update equity curve
            self.equity_curve.append(self.capital)
        
        # Calculate performance metrics
        self.calculate_performance()
        
        return {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'performance': self.performance
        }
    
    def _close_position(self, exit_date, exit_price, profit, reason):
        """Helper method to close a position and record the trade"""
        self.capital += profit
        
        # Complete trade data
        trade = self.current_position.copy()
        trade['exit_date'] = exit_date
        trade['exit_price'] = exit_price
        trade['profit'] = profit
        trade['return_pct'] = (profit / self.capital) * 100
        trade['exit_reason'] = reason
        
        # Add to trades list
        self.trades.append(trade)
        
        # Reset current position
        self.current_position = None
    
    def calculate_performance(self):
        """Calculate performance metrics from backtest results"""
        if not self.trades:
            self.performance = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        gross_profits = sum([t['profit'] for t in winning_trades])
        gross_losses = abs(sum([t['profit'] for t in losing_trades]))
        
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate max drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        self.performance = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'average_win': gross_profits / len(winning_trades) if winning_trades else 0,
            'average_loss': gross_losses / len(losing_trades) if losing_trades else 0,
            'largest_win': max([t['profit'] for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t['profit'] for t in losing_trades]) if losing_trades else 0
        }
    
    def plot_results(self, show_trades=True):
        """
        Plot backtest results including equity curve and trades
        """
        if not hasattr(self, 'equity_curve'):
            print("Run backtest first before plotting results")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        dates = self.data.index[1:]  # Equity curve starts from index 1
        ax1.plot(dates, self.equity_curve[1:], label='Equity', color='blue')
        ax1.set_title('ICT Strategy Backtest Results')
        ax1.set_ylabel('Capital')
        ax1.grid(True)
        ax1.legend()
        
        # Add trades if requested
        if show_trades and self.trades:
            # Mark buy signals with green arrows
            buy_dates = [t['entry_date'] for t in self.trades if t['direction'] == 1]
            buy_prices = [t['entry_price'] for t in self.trades if t['direction'] == 1]
            ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy')
            
            # Mark sell signals with red arrows
            sell_dates = [t['entry_date'] for t in self.trades if t['direction'] == -1]
            sell_prices = [t['entry_price'] for t in self.trades if t['direction'] == -1]
            ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell')
            
            # Mark exits with x
            exit_dates = [t['exit_date'] for t in self.trades]
            exit_prices = [t['exit_price'] for t in self.trades]
            ax1.scatter(exit_dates, exit_prices, marker='x', color='black', s=50, label='Exit')
        
        # Plot price chart
        ax2.plot(self.data.index, self.data['close'], label='Price', color='black')
        
        # Add order blocks
        for i in range(len(self.data)):
            if self.data['bullish_ob'].iloc[i] == 1:
                # Draw bullish order block
                rect = plt.Rectangle(
                    (self.data.index[i], self.data['bullish_ob_low'].iloc[i]),
                    width=(self.data.index[-1] - self.data.index[0])/50,  # Fixed width
                    height=self.data['bullish_ob_high'].iloc[i] - self.data['bullish_ob_low'].iloc[i],
                    color='green',
                    alpha=0.3
                )
                ax2.add_patch(rect)
            
            if self.data['bearish_ob'].iloc[i] == 1:
                # Draw bearish order block
                rect = plt.Rectangle(
                    (self.data.index[i], self.data['bearish_ob_low'].iloc[i]),
                    width=(self.data.index[-1] - self.data.index[0])/50,  # Fixed width
                    height=self.data['bearish_ob_high'].iloc[i] - self.data['bearish_ob_low'].iloc[i],
                    color='red',
                    alpha=0.3
                )
                ax2.add_patch(rect)
        
        ax2.set_ylabel('Price')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance statistics
        self._print_performance()
    
    def _print_performance(self):
        """Print performance metrics in a readable format"""
        if not hasattr(self, 'performance'):
            print("Run backtest first before printing performance metrics")
            return
        
        print("\n" + "="*50)
        print("ICT STRATEGY PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Trades: {self.performance['total_trades']}")
        print(f"Win Rate: {self.performance['win_rate']:.2%}")
        print(f"Profit Factor: {self.performance['profit_factor']:.2f}")
        print(f"Total Return: {self.performance['total_return']:.2f}%")
        print(f"Max Drawdown: {self.performance['max_drawdown']:.2f}%")
        print("-"*50)
        print(f"Winning Trades: {self.performance['winning_trades']}")
        print(f"Losing Trades: {self.performance['losing_trades']}")
        print(f"Average Win: ${self.performance['average_win']:.2f}")
        print(f"Average Loss: ${self.performance['average_loss']:.2f}")
        print(f"Largest Win: ${self.performance['largest_win']:.2f}")
        print(f"Largest Loss: ${self.performance['largest_loss']:.2f}")
        print("="*50)
        
        # Print trade details
        print("\nTRADE DETAILS:")
        print("-"*100)
        for i, trade in enumerate(self.trades):
            direction = "LONG" if trade['direction'] == 1 else "SHORT"
            print(f"Trade {i+1}: {direction} | Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f} | " + 
                  f"Exit: {trade['exit_date'].date()} @ ${trade['exit_price']:.2f} | " +
                  f"P/L: ${trade['profit']:.2f} ({trade['return_pct']:.2f}%) | Reason: {trade['exit_reason']}")
        print("-"*100)

    def live_trading_connector(self, exchange_id='binance', symbol='BTC/USDT', timeframe='1h'):
        """
        Connect to exchange API for live trading
        
        Parameters:
        -----------
        exchange_id: str
            ID of the exchange to connect to (default: 'binance')
        symbol: str
            Trading pair symbol (default: 'BTC/USDT')
        timeframe: str
            Timeframe to use (default: '1h')
        """
        try:
            # Initialize exchange
            exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
                # Add your API credentials here
                'apiKey': 'YOUR_API_KEY',
                'secret': 'YOUR_API_SECRET'
            })
            
            print(f"Connected to {exchange_id} exchange")
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Initialize ICT Strategy with live data
            live_strategy = ICTStrategy(df, timeframe=timeframe)
            live_strategy.generate_signals()
            
            # Check for signals in the most recent candle
            latest_signal = live_strategy.data['signal'].iloc[-1]
            
            if latest_signal == 1:  # Buy signal
                print(f"BUY SIGNAL detected for {symbol}")
                # Implement order placement logic here
                
            elif latest_signal == -1:  # Sell signal
                print(f"SELL SIGNAL detected for {symbol}")
                # Implement order placement logic here
                
            else:
                print(f"No signal for {symbol}")
            
            return live_strategy
            
        except Exception as e:
            print(f"Error connecting to exchange: {e}")
            return None


def fetch_historical_data(symbol, interval, start_date=None, end_date=None, source='yahoo'):
    """
    Fetch historical price data for backtesting
    
    Parameters:
    -----------
    symbol: str
        Trading symbol (e.g., 'AAPL' for Apple stock)
    interval: str
        Data interval (e.g., '1d' for daily, '1h' for hourly)
    start_date: str, optional
        Start date in 'YYYY-MM-DD' format
    end_date: str, optional
        End date in 'YYYY-MM-DD' format
    source: str
        Data source ('yahoo' or 'alpha_vantage')
    
    Returns:
    --------
    pandas.DataFrame: OHLCV data
    """
    if source == 'yahoo':
        try:
            import yfinance as yf
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            
            # Ensure column names are lowercase
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            return None
            
    elif source == 'alpha_vantage':
        try:
            from alpha_vantage.timeseries import TimeSeries
            
            # Initialize Alpha Vantage API
            ts = TimeSeries(key='YOUR_ALPHA_VANTAGE_API_KEY', output_format='pandas')
            
            if interval == '1d':
                data, _ = ts.get_daily(symbol=symbol, outputsize='full')
            elif interval == '1h':
                data, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')
            
            # Rename columns to match expected format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Sort by date (oldest first)
            data = data.sort_index()
            
            return data
            
        except ImportError:
            print("alpha_vantage not installed. Install with: pip install alpha_vantage")
            return None
    
    else:
        print(f"Unsupported data source: {source}")
        return None


def run_ict_backtest_example():
    """
    Example function to run ICT strategy backtest on historical data
    """
    # Fetch historical data
    symbol = 'EURUSD=X'  # EUR/USD forex pair
    data = fetch_historical_data(symbol, interval='1d', start_date='2023-01-01')
    
    if data is None:
        print("Failed to fetch data")
        return
    
    # Initialize strategy
    ict = ICTStrategy(data, capital=10000, risk_per_trade=0.02)
    
    # Run backtest
    results = ict.backtest()
    
    # Plot results
    ict.plot_results()
    
    return ict


if __name__ == "__main__":
    # Run example backtest
    strategy = run_ict_backtest_example()
        