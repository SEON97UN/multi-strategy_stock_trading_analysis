import pandas as pd
import itertools
from strategies.ml_strategy import MachineLearningStrategy

class MLStrategyOptimizer:
    def __init__(self, data):
        self.data = data

    def run_grid_search(self, window_range, max_depth_range, n_estimators_range):
        model_types = ['RandomForest', 'XGBoost']
        feature_sets = [
            ['ma', 'std', 'momentum'],
            ['ma', 'momentum', 'volatility'],
            ['std', 'momentum', 'volatility'],
            ['ma', 'std', 'momentum', 'volatility']
        ]

        all_combinations = list(itertools.product(
            window_range,
            max_depth_range,
            n_estimators_range,
            model_types,
            feature_sets
        ))

        results = []
        total = len(all_combinations)

        for i, (window, depth, n_estimators, model_type, features) in enumerate(all_combinations, start=1):
            print(f"Testing {i}/{total}: window={window}, depth={depth}, estimators={n_estimators}, model={model_type}, features={features}")

            try:
                strategy = MachineLearningStrategy(
                    data=self.data,
                    window=window,
                    max_depth=depth,
                    n_estimaters=n_estimators,
                    model_type=model_type,
                    feature_set=features
                )
                result = strategy.evaluate()

                if result and result.get('metrics'):
                    m = result['metrics']
                    results.append({
                        "window": window,
                        "max_depth": depth,
                        "n_estimators": n_estimators,
                        "model": model_type,
                        "features": features,
                        "return": m.get('return'),
                        "sharpe": m.get('sharpe'),
                        "accuracy": m.get('accuracy'),
                        "max_drawdown": m.get('max_drawdown'),
                        "curve": result.get('chart'),
                        "benchmark": result.get('benchmark'),
                        "buy_dates": result.get('buy_dates'),
                        "walk_forward_return": result.get('walk_forward_return'),
                        "feature_importance": result.get('feature_importance'),
                        "confusion_matrix": result.get('confusion_matrix'),
                        "classification_report": result.get('classification_report'),
                    })

            except Exception as e:
                print(f"Error: {e}")
                continue

        df = pd.DataFrame(results)
        if "return" not in df.columns:
            print("No 'return' column found in results.")
            return df
        return df.sort_values("return", ascending=False)
