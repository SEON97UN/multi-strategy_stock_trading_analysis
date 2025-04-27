import pandas as pd
import numpy as np
from strategies.statistical_arbitrage import StatisticalArbitrageStrategy

class StatisticalArbitrageOptimizer:
    def __init__(self, data_x, data_y, ticker_x="AAPL", ticker_y="MSFT"):
        self.data_x = data_x
        self.data_y = data_y
        self.ticker_x = ticker_x
        self.ticker_y = ticker_y

    def run_grid_search(self, lookback_range, entry_range, exit_range):
        results = []
        total = len(lookback_range) * len(entry_range) * len(exit_range)
        count = 0

        for lookback in lookback_range:
            for entry_z in entry_range:
                for exit_z in exit_range:
                    count += 1
                    print(f"\rTesting {count}/{total} combinations...", end="")

                    try:
                        strategy = StatisticalArbitrageStrategy(
                            data_x=self.data_x,
                            data_y=self.data_y,
                            lookback=lookback,
                            entry_z=entry_z,
                            exit_z=exit_z
                        )

                        result = strategy.evaluate()
                        metrics = result['metrics']
                        curve = result.get('chart', None)
                        benchmark = result.get('benchmark', None)

                        if any(pd.isna(list(metrics.values()))) or any(np.isinf(list(metrics.values()))):
                            continue

                        results.append({
                            'lookback': lookback,
                            'entry_z': entry_z,
                            'exit_z': exit_z,
                            'return': metrics['return'],
                            'sharpe': metrics['sharpe'],
                            'max_drawdown': metrics['max_drawdown'],
                            'curve': curve,
                            'benchmark': benchmark
                        })
                    except Exception as e:
                        print(f" Error: {e}")
                        continue

        print("\nGrid search completed.")
        return pd.DataFrame(results)
