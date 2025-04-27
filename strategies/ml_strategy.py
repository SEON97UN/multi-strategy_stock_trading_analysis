import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

class MachineLearningStrategy:
    def __init__(self, data, window=5, max_depth=None, n_estimaters=100, model_type='RandomForest', feature_set=None):
        self.data = data.copy()
        self.window = window
        self.max_depth = max_depth
        self.n_estimaters = n_estimaters
        self.model_type = model_type

        if feature_set:
            self.feature_set = [str(f).replace(',', '').strip() for f in feature_set]
        else:
            self.feature_set = ['ma', 'std', 'momentum', 'volatility']

    def generate_features(self):
        df = self.data.copy()
        df['return'] = df['Close'].pct_change()
        df['ma'] = df['Close'].rolling(window=self.window).mean()
        df['std'] = df['Close'].rolling(window=self.window).std()
        df['momentum'] = df['Close'] / df['Close'].shift(self.window) - 1
        df['volatility'] = df['return'].rolling(window=self.window).std()
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()

        self.df = df
        self.features = df[self.feature_set]
        self.labels = df['target']

    def walk_forward_validation(self, step=50):
        df = self.df.copy()
        returns = []

        for i in range(0, len(df) - step, step):
            train = df.iloc[i:i+step]
            test = df.iloc[i+step:i+2*step]
            if len(test) < step:
                break

            X_train = train[self.feature_set]
            y_train = train['target']
            X_test = test[self.feature_set]
            y_test = test['target']

            model = self._get_model()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            strat_ret = pred[:-1] * test['return'].iloc[1:].values
            if len(strat_ret) > 0:
                returns.append(np.mean(strat_ret))

        return round(np.mean(returns) * 252 * 100, 2) if returns else 0

    def _get_model(self):
        if self.model_type == 'XGBoost':
            return XGBClassifier(n_estimators=self.n_estimaters, max_depth=self.max_depth, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=self.n_estimaters, max_depth=self.max_depth, random_state=42)

    def train_model(self):
        split = int(len(self.features) * 0.7)
        X_train, X_test = self.features.iloc[:split], self.features.iloc[split:]
        y_train, y_test = self.labels.iloc[:split], self.labels.iloc[split:]

        model = self._get_model()
        model.fit(X_train, y_train)

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.df['prediction'] = np.nan
        self.df.loc[self.X_test.index, 'prediction'] = model.predict(X_test)
        self.df['prediction'] = self.df['prediction'].ffill()

        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(X_train.columns))
        }).sort_values(by='importance', ascending=False)

    def backtest(self):
        df = self.df.copy()
        
        # 테스트 기간의 시작점 찾기
        split = int(len(self.features) * 0.7)
        test_start = self.features.index[split]
        
        # 테스트 기간의 데이터만 사용
        df = df.loc[test_start:].copy()
        
        df['strategy'] = df['prediction'].shift(1) * df['return']
        df['strategy'] = df['strategy'].fillna(0)
        
        # 누적 수익률 계산 (테스트 기간만)
        df['cumulative_returns'] = (1 + df['strategy']).cumprod() - 1
        df['benchmark'] = (1 + df['return']).cumprod() - 1

        accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        cr = classification_report(self.y_test, self.model.predict(self.X_test), output_dict=True)

        buy_dates = df[df['prediction'].shift(1) == 1].index.strftime('%Y-%m-%d').tolist()

        metrics = {
            'return': round(df['cumulative_returns'].iloc[-1] * 100, 2),
            'sharpe': round(df['strategy'].mean() / df['strategy'].std() * np.sqrt(252), 2) if df['strategy'].std() > 0 else 0,
            'accuracy': round(accuracy * 100, 2),
            'max_drawdown': round(((df['cumulative_returns'].cummax() - df['cumulative_returns']).max()) * 100, 2)
        }

        return {
            'metrics': metrics,
            'chart': df['cumulative_returns'],
            'benchmark': df['benchmark'],
            'buy_dates': buy_dates,
            'feature_importance': self.feature_importance,
            'confusion_matrix': cm,
            'classification_report': cr,
            'walk_forward_return': self.walk_forward_validation()
        }

    def evaluate(self):
        self.generate_features()
        self.train_model()
        return self.backtest()
