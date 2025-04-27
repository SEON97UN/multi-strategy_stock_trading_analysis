import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class StockDataset(Dataset):
    """Dataset for stock time series data."""
    def __init__(self, data, sequence_length=60):
        self.data = data
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(data)
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        x = self.scaled_data[idx:idx + self.sequence_length]
        y = self.scaled_data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMAttention(nn.Module):
    """LSTM with Attention mechanism for time series prediction."""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        out = self.fc(context)
        return out
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if needed
            return self(x).numpy()

class DLStrategy:
    """Deep Learning strategy for stock prediction using LSTM with Attention."""
    def __init__(self, ticker, start_date, end_date, random_state=42):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.price_scaler = None  # Separate scaler for price only
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def prepare_data(self):
        """Download and preprocess data, add technical indicators, and fit scalers."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = self._calculate_macd(data['Close'])
        
        # Remove missing values
        data = data.dropna()
        
        # Select features for training
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'SMA_20', 'SMA_50', 'RSI', 'MACD']
        self.data = data[features]
        
        # Fit scalers
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)
        self.price_scaler = MinMaxScaler()
        self.price_scaler.fit(data[['Close']])
        
        return self.data
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def train_model(self, sequence_length=60, batch_size=32, epochs=50, hidden_size=64, num_layers=2):
        """Train the LSTM-Attention model."""
        data = self.prepare_data()
        scaled_data = self.scaler.transform(data)
        
        dataset = StockDataset(scaled_data, sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              generator=torch.Generator().manual_seed(self.random_state))
        
        self.model = LSTMAttention(
            input_size=data.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        patience = 5
        best_loss = float('inf')
        patience_counter = 0
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y[:, 3:4])  # Use only Close price for prediction
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            avg_epoch_loss = epoch_loss / batch_count
            self.loss_history.append(avg_epoch_loss)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        print(f'Training completed. Best loss: {best_loss:.4f}')
    
    def predict(self, data):
        """Predict using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        if self.scaler is None or self.price_scaler is None:
            raise ValueError("Scalers not initialized. Call prepare_data() first.")
        self.model.eval()
        with torch.no_grad():
            scaled_data = self.scaler.transform(data)
            x = torch.FloatTensor(scaled_data).unsqueeze(0)
            prediction = self.model(x)
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.numpy()
            prediction_reshaped = prediction.reshape(-1, 1)
            original_prediction = self.price_scaler.inverse_transform(prediction_reshaped)
        return original_prediction[0][0]
    
    def backtest(self):
        """Backtest the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call prepare_data() first.")
        data = self.data.copy()
        scaled_data = self.scaler.transform(data)
        X = []
        y = []
        sequence_length = 60
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 3])  # Use only Close price
        X = np.array(X)
        y = np.array(y)
        predictions = self.model.predict(X)
        predictions = predictions.flatten()
        dummy = np.zeros((len(predictions), data.shape[1]))
        dummy[:, 3] = predictions
        predictions = self.scaler.inverse_transform(dummy)[:, 3]
        dummy = np.zeros((len(y), data.shape[1]))
        dummy[:, 3] = y
        actual = self.scaler.inverse_transform(dummy)[:, 3]
        predictions_returns = np.diff(predictions) / predictions[:-1]
        actual_returns = np.diff(actual) / actual[:-1]
        cumulative_returns = pd.Series((1 + predictions_returns).cumprod(), 
                                     index=data.index[sequence_length+1:])
        benchmark_returns = pd.Series((1 + actual_returns).cumprod(), 
                                    index=data.index[sequence_length+1:])
        metrics = {
            'return': (cumulative_returns.iloc[-1] - 1) * 100,
            'sharpe': self._calculate_sharpe_ratio(predictions_returns),
            'max_drawdown': self._calculate_max_drawdown(cumulative_returns)
        }
        return {
            'metrics': metrics,
            'chart': cumulative_returns,
            'benchmark': benchmark_returns,
            'predictions': pd.Series(predictions, index=data.index[sequence_length:]),
            'actual': pd.Series(actual, index=data.index[sequence_length:]),
            'loss_history': self.loss_history
        }
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        trading_days = len(returns)
        if trading_days < 2:
            return 0.0
        annual_factor = np.sqrt(trading_days)
        annual_return = np.mean(returns) * trading_days
        annual_volatility = np.std(returns) * annual_factor
        if annual_volatility == 0:
            return 0.0
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown (percentage)."""
        if len(cumulative_returns) == 0:
            return 0.0
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        return max_drawdown * 100 