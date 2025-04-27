import pandas as pd
import numpy as np

class MomentumMeanReversionStrategy:
    def __init__(self, data, short_window=20, long_window=50, std_dev=2.0, rsi_period=14):
        """
        Initialize data and parameters
        """
        # Data preprocessing
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
            
        self.original_data = data.copy()
        
        # Close data processing
        try:
            # MultiIndex DataFrame processing
            if isinstance(self.original_data.columns, pd.MultiIndex):
                print("Processing MultiIndex DataFrame")
                # Extract Close price from ('Close', 'AAPL') format columns
                price_data = self.original_data.xs('Close', axis=1, level=0)
                if isinstance(price_data, pd.Series):
                    close_series = price_data
                else:
                    # Select the first ticker if multiple tickers exist
                    close_series = price_data.iloc[:, 0]
            else:
                # General DataFrame processing
                if 'Close' not in self.original_data.columns:
                    raise ValueError("No Close price column found")
                close_series = self.original_data['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
            
            # Create a new DataFrame
            self.data = pd.DataFrame(index=self.original_data.index)
            
            # Add Close price and preprocessing
            self.data['Close'] = pd.to_numeric(close_series, errors='coerce')
            self.data['Close'] = self.data['Close'].ffill().bfill()
            
            if self.data['Close'].isna().any():
                raise ValueError("Unable to handle NaN values in Close prices")
                
            print(f"Processed Close data shape: {self.data.shape}")
            print(f"Close data sample:\n{self.data['Close'].head()}")
            print(f"Close data statistics:\n{self.data['Close'].describe()}")
                
        except Exception as e:
            print(f"Original data structure:")
            print(f"Columns: {self.original_data.columns}")
            print(f"Index levels: {[name for name in self.original_data.columns.names]}")
            print(f"Data types:\n{self.original_data.dtypes}")
            raise ValueError(f"Error processing Close price data: {str(e)}")

        # Parameter setting and validation
        try:
            self.short_window = int(short_window)
            self.long_window = int(long_window)
            self.std_dev = float(std_dev)
            self.rsi_period = int(rsi_period)
            
            if self.short_window <= 0 or self.long_window <= 0:
                raise ValueError("Window sizes must be positive")
            if self.short_window >= self.long_window:
                raise ValueError("Short window must be smaller than long window")
            if self.std_dev <= 0:
                raise ValueError("Standard deviation must be positive")
            if self.rsi_period <= 0:
                raise ValueError("RSI period must be positive")
                
            print(f"Parameters validated: short={self.short_window}, long={self.long_window}, std={self.std_dev}, rsi={self.rsi_period}")
            
        except Exception as e:
            raise ValueError(f"Error validating parameters: {str(e)}")

    def compute_indicators(self):
        """Calculate technical indicators"""
        try:
            df = self.data.copy()
            
            # Calculate moving averages
            df['short_ma'] = df['Close'].rolling(window=self.short_window, min_periods=1).mean()
            df['long_ma'] = df['Close'].rolling(window=self.long_window, min_periods=1).mean()

            # Calculate Bollinger Bands
            rolling_std = df['Close'].rolling(window=self.long_window, min_periods=1).std()
            df['middle_band'] = df['long_ma']
            df['upper_band'] = df['middle_band'] + (rolling_std * self.std_dev)
            df['lower_band'] = df['middle_band'] - (rolling_std * self.std_dev)

            # Calculate RSI
            close_diff = df['Close'].diff()
            gains = close_diff.where(close_diff > 0, 0.0)
            losses = -close_diff.where(close_diff < 0, 0.0)
            
            avg_gains = gains.rolling(window=self.rsi_period, min_periods=1).mean()
            avg_losses = losses.rolling(window=self.rsi_period, min_periods=1).mean()
            
            rs = avg_gains / avg_losses.replace(0, np.inf)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # NaN and infinite value handling
            df['RSI'] = df['RSI'].fillna(50).clip(0, 100)
            
            # NaN handling for all indicators
            for col in ['short_ma', 'long_ma', 'middle_band', 'upper_band', 'lower_band']:
                df[col] = df[col].fillna(method='ffill').fillna(df['Close'])

            # Data verification
            print("\nIndicator Statistics:")
            for col in df.columns:
                print(f"\n{col}:")
                print(df[col].describe())
                print(f"NaN count: {df[col].isna().sum()}")
                print(f"Inf count: {np.isinf(df[col]).sum()}")

            self.data = df
            
        except Exception as e:
            print(f"Error in compute_indicators: {str(e)}")
            print(f"Data types: {df.dtypes}")
            print(f"Data shape: {df.shape}")
            raise

    def generate_signals(self):
        """Generate trading signals"""
        try:
            df = self.data.copy()
            
            # Convert to numpy array
            close_prices = df['Close'].values
            lower_bands = df['lower_band'].values
            upper_bands = df['upper_band'].values
            rsi_values = df['RSI'].values
            positions = np.zeros(len(df))
            
            # Relax trading conditions
            buy_signals = (close_prices < lower_bands) & (rsi_values < 40)  # Relax RSI condition (30 -> 40)
            sell_signals = (close_prices > upper_bands) & (rsi_values > 60)  # Relax RSI condition (70 -> 60)
            
            # Additional trading conditions: moving average based
            short_ma = df['short_ma'].values
            long_ma = df['long_ma'].values
            
            # Add moving average crossover signal
            buy_signals = buy_signals | ((short_ma > long_ma) & (rsi_values < 50))
            sell_signals = sell_signals | ((short_ma < long_ma) & (rsi_values > 50))
            
            # Signal application
            positions[buy_signals] = 1
            positions[sell_signals] = -1
            
            # Maintain position (forward fill)
            for i in range(1, len(positions)):
                if positions[i] == 0:
                    positions[i] = positions[i-1]
            
            df['position'] = positions
            self.data = df
            
        except Exception as e:
            print(f"Error in generate_signals: {str(e)}")
            raise

    def backtest(self):
        try:
            df = self.data.copy()

            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Strategy'] = df['position'].shift(1) * df['Returns']
            
            # NaN value handling
            df['Strategy'] = df['Strategy'].fillna(0)
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['Strategy']).cumprod() - 1
            df['benchmark'] = (1 + df['Returns']).cumprod() - 1  # Add benchmark calculation

            # Generate trade log
            position_changes = df['position'].diff().fillna(0)
            trades = df[position_changes != 0].copy()
            trades['Trade Type'] = trades['position'].map({1: 'Buy', -1: 'Sell'})
            trades['Price'] = trades['Close']

            # Calculate performance metrics
            strategy_returns = df['Strategy'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate annualization factor (trading days per year / total days)
            annualization = np.sqrt(252 / len(df))
            
            # Performance metrics
            total_return = df['cumulative_returns'].iloc[-1]
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            
            # Sharpe Ratio (calculated only if volatility is not zero)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Maximum drawdown
            cumulative = df['cumulative_returns']
            rolling_max = cumulative.expanding().max()
            drawdowns = cumulative - rolling_max
            max_drawdown = abs(drawdowns.min())

            # Return metrics as a dictionary (column name consistency)
            metrics = {
                'return': round(float(total_return) * 100, 2),  # 'Total Return (%)' -> 'return'
                'sharpe': round(float(sharpe_ratio), 2),        # 'Sharpe Ratio' -> 'sharpe'
                'max_drawdown': round(float(max_drawdown) * 100, 2),  # 'Max Drawdown (%)' -> 'max_drawdown'
                'num_trades': len(trades)                       # 'Number of Trades' -> 'num_trades'
            }

            return {
                'metrics': metrics,
                'trade_log': trades[['Trade Type', 'Price', 'RSI', 'upper_band', 'lower_band']].copy(),
                'chart': df['cumulative_returns'],
                'benchmark': df['benchmark']  # Add benchmark to return dictionary
            }
            
        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            print(f"Data shape: {df.shape}")
            print(f"Available columns: {df.columns}")
            return {
                'metrics': {
                    'return': 0.0,
                    'sharpe': 0.0,
                    'max_drawdown': 0.0,
                    'num_trades': 0
                },
                'trade_log': pd.DataFrame(),
                'chart': None,  # chart is None if error occurs
                'benchmark': None  # Add benchmark to return dictionary
            }

    def evaluate(self):
        self.compute_indicators()
        self.generate_signals()
        return self.backtest()
