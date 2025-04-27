import pandas as pd
import numpy as np

class StatisticalArbitrageStrategy:
    def __init__(self, data_x, data_y, lookback=30, entry_z=1.0, exit_z=0.5):
        """
        Initialize data and parameters
        """
        # If data_x is a DataFrame, select the first column
        if isinstance(data_x, pd.DataFrame):
            print(f"Converting data_x from DataFrame to Series (columns: {data_x.columns})")
            data_x = data_x.iloc[:, 0]
        if isinstance(data_y, pd.DataFrame):
            print(f"Converting data_y from DataFrame to Series (columns: {data_y.columns})")
            data_y = data_y.iloc[:, 0]
            
        self.data_x = data_x.copy()
        self.data_y = data_y.copy()
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        
        # Data validation
        if not isinstance(self.data_x, pd.Series) or not isinstance(self.data_y, pd.Series):
            raise ValueError("Both data_x and data_y must be pandas Series")
        
        print(f"Data initialized successfully:")
        print(f"data_x type: {type(self.data_x)}, length: {len(self.data_x)}")
        print(f"data_y type: {type(self.data_y)}, length: {len(self.data_y)}")

    def compute_spread(self):
        """
        Calculate spread and generate statistics
        """
        try:
            # Re-validate data
            if not isinstance(self.data_x, pd.Series) or not isinstance(self.data_y, pd.Series):
                raise ValueError("Data must be pandas Series")
            
            # Create DataFrame and calculate spread
            df = pd.DataFrame(index=self.data_x.index)
            df['spread'] = self.data_x - self.data_y
            
            # Calculate moving average and standard deviation
            df['mean'] = df['spread'].rolling(window=self.lookback).mean()
            df['std'] = df['spread'].rolling(window=self.lookback).std()
            df['zscore'] = (df['spread'] - df['mean']) / df['std']
            
            # Remove NaN
            self.df = df.dropna()
            
            print(f"Spread computation completed:")
            print(f"Original length: {len(df)}")
            print(f"After dropna: {len(self.df)}")
            
        except Exception as e:
            print(f"Error in compute_spread: {str(e)}")
            print(f"data_x type: {type(self.data_x)}")
            print(f"data_y type: {type(self.data_y)}")
            raise

    def generate_signals(self):
        df = self.df
        df['position'] = 0
        df.loc[df['zscore'] > self.entry_z, 'position'] = -1
        df.loc[df['zscore'] < -self.entry_z, 'position'] = 1
        df.loc[df['zscore'].abs() < self.exit_z, 'position'] = 0
        df['position'] = df['position'].ffill()
        self.df = df

    def backtest(self):
        df = self.df
        df['return'] = (self.data_x.pct_change().loc[df.index] - self.data_y.pct_change().loc[df.index]) * df['position'].shift(1)
        df['strategy'] = df['return'].fillna(0)
        df['cumulative_returns'] = (1 + df['strategy']).cumprod() - 1
        
        # Calculate benchmark (equal-weighted portfolio)
        df['benchmark_returns'] = (self.data_x.pct_change().loc[df.index] + self.data_y.pct_change().loc[df.index]) / 2
        df['benchmark'] = (1 + df['benchmark_returns']).cumprod() - 1

        metrics = {
            'return': round(df['cumulative_returns'].iloc[-1] * 100, 2),
            'sharpe': round(df['strategy'].mean() / df['strategy'].std() * np.sqrt(252), 2) if df['strategy'].std() > 0 else 0,
            'max_drawdown': round(((df['cumulative_returns'].cummax() - df['cumulative_returns']).max()) * 100, 2)
        }

        return {
            'metrics': metrics,
            'chart': df['cumulative_returns'],
            'benchmark': df['benchmark']
        }

    def evaluate(self):
        self.compute_spread()
        self.generate_signals()
        return self.backtest()
