import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class FactorModelStrategy:
    def __init__(self, data, lookback_period=126, num_factors=5):
        self.data = data.copy()
        self.lookback_period = lookback_period
        self.num_factors = num_factors

    def apply_factor_model(self):
        df = self.data
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        returns = df['Return'].rolling(window=self.lookback_period).mean().dropna()
        rolling_matrix = pd.concat([returns.shift(i) for i in range(self.lookback_period)], axis=1).dropna()
        rolling_matrix.columns = [f"lag_{i}" for i in range(self.lookback_period)]

        pca = PCA(n_components=self.num_factors)
        factors = pca.fit_transform(rolling_matrix)

        explained_var = pca.explained_variance_ratio_.sum()
        signal = pd.Series(factors[:, 0], index=rolling_matrix.index)

        df = df.loc[rolling_matrix.index].copy()
        df['signal'] = signal
        df['position'] = np.where(df['signal'] > 0, 1, -1)

        df['Strategy'] = df['position'].shift(1) * df['Return']
        df['cumulative_returns'] = (1 + df['Strategy']).cumprod() - 1
        df['benchmark'] = (1 + df['Return']).cumprod() - 1

        trades = df[df['position'].diff() != 0].copy()
        trades['Trade Type'] = trades['position'].map({1: 'Buy', -1: 'Sell'})
        trades['Price'] = trades['Close']

        metrics = {
            'Total Return (%)': round(df['cumulative_returns'].iloc[-1] * 100, 2),
            'Sharpe Ratio': round(df['Strategy'].mean() / df['Strategy'].std() * np.sqrt(252), 2),
            'Max Drawdown (%)': round(((df['cumulative_returns'].cummax() - df['cumulative_returns']).max()) * 100, 2),
            'Explained Variance (%)': round(explained_var * 100, 2)
        }

        return {
            'metrics': metrics,
            'trade_log': trades[['Trade Type', 'Price', 'signal']].copy(),
            'chart': df['cumulative_returns'],
            'benchmark': df['benchmark']
        }

    def evaluate(self):
        return self.apply_factor_model()