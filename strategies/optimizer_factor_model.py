import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Factor Model strategy optimization class
# -----------------------------
class FactorModelOptimizer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._load_data()

    def _load_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df['Return'] = df['Close'].pct_change()
        return df.dropna()

    def evaluate_strategy(self, lookback_period, num_factors):
        df = self.data.copy()

        try:
            returns = df['Return'].rolling(window=lookback_period).mean().dropna()
            rolling_matrix = pd.concat([returns.shift(i) for i in range(lookback_period)], axis=1).dropna()

            if rolling_matrix.shape[0] == 0:
                return None

            pca = PCA(n_components=num_factors)
            factors = pca.fit_transform(rolling_matrix)
            signal = pd.Series(factors[:, 0], index=rolling_matrix.index)

            df = df.loc[rolling_matrix.index].copy()
            df['signal'] = signal
            df['position'] = np.where(df['signal'] > 0, 1, -1)
            df['Strategy'] = df['position'].shift(1) * df['Return']
            df['cumulative_returns'] = (1 + df['Strategy']).cumprod() - 1
            df['benchmark'] = (1 + df['Return']).cumprod() - 1

            total_return = df['cumulative_returns'].iloc[-1]
            max_dd = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
            sharpe = df['Strategy'].mean() / df['Strategy'].std() * np.sqrt(252)

            return {
                'lookback': lookback_period,
                'factors': num_factors,
                'return': total_return,
                'max_drawdown': max_dd,
                'sharpe': sharpe,
                'curve': df['cumulative_returns'],
                'benchmark': df['benchmark']
            }
        except Exception as e:
            return None

    def run_grid_search(self, lookback_range, factor_range):
        results = []
        for lookback, n_factors in product(lookback_range, factor_range):
            result = self.evaluate_strategy(lookback, n_factors)
            if result:
                results.append(result)
        return pd.DataFrame(results)

    def plot_heatmap(self, result_df):
        pivot_table = result_df.pivot("lookback", "factors", "return")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn")
        plt.title(f"Total Return Heatmap for {self.ticker}")
        plt.xlabel("Number of Factors")
        plt.ylabel("Lookback Period")
        plt.tight_layout()
        plt.show()

# -----------------------------
# Example usage (run module for testing)
# -----------------------------
if __name__ == "__main__":
    optimizer = FactorModelOptimizer("AAPL", "2022-01-01", "2025-01-01")
    lookbacks = range(30, 151, 30)       # ex: 30, 60, 90, 120, 150
    factors = range(3, 9)                # ex: 3~8

    df_results = optimizer.run_grid_search(lookbacks, factors)
    print(df_results.sort_values("return", ascending=False).head())
    optimizer.plot_heatmap(df_results)
