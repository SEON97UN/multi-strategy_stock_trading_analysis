import pandas as pd
import numpy as np
from strategies.momentum_mean_reversion import MomentumMeanReversionStrategy

class MomentumOptimizer:
    def __init__(self, ticker, start, end, data):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.df = data

    def run_grid_search(self, short_range, long_range, std_range, rsi_range):
        """
        Run grid search to find the optimal parameter combination.
        """
        try:
            # Define required metrics
            required_metrics = ['return', 'sharpe', 'max_drawdown']
            
            # List to store results
            results = []
            
            # Iterate over all parameter combinations
            total_combinations = (len(short_range) * 
                                len(long_range) * 
                                len(std_range) * 
                                len(rsi_range))
            
            print(f"\nStarting grid search with {total_combinations} combinations...")
            
            processed = 0
            for short_window in short_range:
                for long_window in long_range:
                    # Only execute if short_window is less than long_window
                    if short_window >= long_window:
                        processed += len(std_range) * len(rsi_range)
                        continue
                        
                    for std_dev in std_range:
                        for rsi_period in rsi_range:
                            processed += 1
                            print(f"\rProgress: {processed}/{total_combinations} combinations tested ({(processed/total_combinations)*100:.1f}%)", end="")
                            
                            try:
                                # Create strategy instance and execute
                                strategy = MomentumMeanReversionStrategy(
                                    data=self.df,
                                    short_window=short_window,
                                    long_window=long_window,
                                    std_dev=std_dev,
                                    rsi_period=rsi_period
                                )
                                
                                result = strategy.evaluate()
                                metrics = result['metrics']
                                
                                # Check if all required metrics are present
                                if not all(metric in metrics for metric in required_metrics):
                                    continue
                                    
                                # Extract metrics
                                total_return = float(metrics['return'])
                                sharpe = float(metrics['sharpe'])
                                max_drawdown = float(metrics['max_drawdown'])
                                
                                # Check if the result is valid
                                if (pd.isna(total_return) or pd.isna(sharpe) or pd.isna(max_drawdown) or
                                    np.isinf(total_return) or np.isinf(sharpe) or np.isinf(max_drawdown)):
                                    continue
                                
                                # Store result
                                results.append({
                                    'short_window': short_window,
                                    'long_window': long_window,
                                    'std_dev': std_dev,
                                    'rsi_period': rsi_period,
                                    'return': total_return,
                                    'sharpe': sharpe,
                                    'max_drawdown': max_drawdown,
                                    'curve': result.get('chart'),  # Store cumulative return curve
                                    'benchmark': result.get('benchmark')  # Add benchmark
                                })
                                
                            except Exception as e:
                                print(f"Error with parameters (short={short_window}, long={long_window}, std={std_dev}, rsi={rsi_period}): {str(e)}")
                                continue
            
            print("\nGrid search completed.")
            
            # If no results are found
            if not results:
                print("\nNo valid results found. Possible reasons:")
                print("1. Parameter ranges may be inappropriate")
                print("2. Data quality issues (check for gaps, outliers)")
                print("3. Strategy constraints may be too strict")
                return pd.DataFrame()
                
            # Convert results to DataFrame and sort by return
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('return', ascending=False)
            
            # Print top results
            print("\nTop 5 parameter combinations:")
            print(results_df.head().to_string(index=False))
            
            return results_df
            
        except Exception as e:
            print(f"Error in grid search: {str(e)}")
            return pd.DataFrame()
