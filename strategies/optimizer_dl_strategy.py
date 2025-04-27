import pandas as pd
import numpy as np
from .dl_strategy import DLStrategy

class DLStrategyOptimizer:
    """Optimizer for Deep Learning strategy using grid search or random search."""
    def __init__(self, ticker, start_date, end_date, global_seed=42):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.global_seed = global_seed
        np.random.seed(self.global_seed)  # Set global seed
        self.strategy = DLStrategy(ticker, start_date, end_date, random_state=self.global_seed)
        
    def run_grid_search(self):
        """Run grid search for strategy optimization."""
        param_grid = {
            'sequence_length': [30, 60, 90],
            'hidden_size': [32, 128, 256],
            'num_layers': [1, 2, 3],
            'batch_size': [16, 64, 128],
            'epochs': [50, 100]
        }
        
        results = []
        total_combinations = (len(param_grid['sequence_length']) * 
                            len(param_grid['hidden_size']) * 
                            len(param_grid['num_layers']) * 
                            len(param_grid['batch_size']) * 
                            len(param_grid['epochs']))
        current_combination = 0
        
        # Iterate over all parameter combinations
        for seq_len in param_grid['sequence_length']:
            for hidden_size in param_grid['hidden_size']:
                for n_layers in param_grid['num_layers']:
                    for batch_size in param_grid['batch_size']:
                        for epochs in param_grid['epochs']:
                            current_combination += 1
                            print(f"Processing combination {current_combination}/{total_combinations}")
                            try:
                                # Create new strategy instance
                                strategy = DLStrategy(
                                    self.ticker, 
                                    self.start_date, 
                                    self.end_date, 
                                    random_state=self.global_seed
                                )
                                # Train model
                                strategy.train_model(
                                    sequence_length=seq_len,
                                    hidden_size=hidden_size,
                                    num_layers=n_layers,
                                    batch_size=batch_size,
                                    epochs=epochs
                                )
                                # Run backtest
                                backtest_result = strategy.backtest()
                                # Store result
                                result = {
                                    'sequence_length': seq_len,
                                    'hidden_size': hidden_size,
                                    'num_layers': n_layers,
                                    'batch_size': batch_size,
                                    'epochs': epochs,
                                    'return': backtest_result['metrics']['return'],
                                    'sharpe': backtest_result['metrics']['sharpe'],
                                    'max_drawdown': backtest_result['metrics']['max_drawdown'],
                                    'curve': backtest_result['chart'],
                                    'benchmark': backtest_result['benchmark'],
                                    'predictions': backtest_result.get('predictions', None),
                                    'actual': backtest_result.get('actual', None),
                                    'loss_history': backtest_result.get('loss_history', None)
                                }
                                results.append(result)
                                print(f"Completed: seq_len={seq_len}, hidden={hidden_size}, "
                                      f"layers={n_layers}, batch={batch_size}, epochs={epochs}, "
                                      f"return={result['return']:.2f}%")
                            except Exception as e:
                                print(f"Error in combination {current_combination}: {str(e)}")
                                continue
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            # Sort by return
            results_df = results_df.sort_values('return', ascending=False)
            best_params = results_df.iloc[0].to_dict()
        else:
            best_params = None
        return best_params, results_df

    def optimize_strategy(self):
        """Run strategy optimization (grid search)."""
        return self.run_grid_search()

    def get_best_parameters(self, results_df):
        best_result = results_df.iloc[0]
        return {
            'sequence_length': best_result['sequence_length'],
            'hidden_size': best_result['hidden_size'],
            'num_layers': best_result['num_layers'],
            'batch_size': best_result['batch_size'],
            'epochs': best_result['epochs'],
            'expected_return': best_result['return']
        }
    
    def run_random_search(self, n_trials=20):
        """Run random search for strategy optimization."""
        results = []
        np.random.seed(self.global_seed)
        random_params = []
        for trial in range(n_trials):
            params = {
                'seq_len': int(np.random.choice([20, 30, 45, 60, 90])),
                'hidden_size': int(np.random.choice([32, 64, 128, 256])),
                'n_layers': int(np.random.choice([1, 2, 3, 4])),
                'batch_size': int(np.random.choice([16, 32, 64, 128])),
                'epoch': int(np.random.choice([50, 100, 150])),
                'trial_seed': self.global_seed + trial  # Unique but reproducible seed for each trial
            }
            random_params.append(params)
        for trial, params in enumerate(random_params):
            try:
                # Initialize strategy with reproducible seed
                strategy = DLStrategy(self.ticker, self.start_date, self.end_date, 
                                   random_state=params['trial_seed'])
                # Train model
                strategy.train_model(
                    sequence_length=params['seq_len'],
                    hidden_size=params['hidden_size'],
                    num_layers=params['n_layers'],
                    batch_size=params['batch_size'],
                    epochs=params['epoch']
                )
                # Run backtest
                backtest_result = strategy.backtest()
                # Store result
                result = {
                    'sequence_length': params['seq_len'],
                    'hidden_size': params['hidden_size'],
                    'num_layers': params['n_layers'],
                    'batch_size': params['batch_size'],
                    'epochs': params['epoch'],
                    'trial_seed': params['trial_seed'],
                    'return': backtest_result['metrics']['return'],
                    'sharpe': backtest_result['metrics']['sharpe'],
                    'max_drawdown': backtest_result['metrics']['max_drawdown'],
                    'curve': backtest_result['chart'],
                    'benchmark': backtest_result['benchmark'],
                    'predictions': backtest_result.get('predictions', None),
                    'actual': backtest_result.get('actual', None),
                    'loss_history': backtest_result.get('loss_history', None)
                }
                results.append(result)
                print(f"Completed trial {trial+1}/{n_trials}: seq_len={params['seq_len']}, "
                      f"hidden={params['hidden_size']}, layers={params['n_layers']}, "
                      f"batch={params['batch_size']}, epochs={params['epoch']}, "
                      f"return={result['return']:.2f}%")
            except Exception as e:
                print(f"Error in trial {trial+1}: {str(e)}")
                continue
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            best_params = results_df.sort_values('return', ascending=False).iloc[0].to_dict()
        else:
            best_params = None
        return best_params, results_df 