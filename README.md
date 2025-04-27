# Multi-Strategy Stock Trading Analysis

<div align="center">
<img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python">
<img src="https://img.shields.io/badge/Streamlit-Enabled-brightgreen?logo=streamlit">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</div>

## 📈 Project Overview

**Multi-Strategy Stock Trading Analysis** is a quantitative research platform designed to implement, evaluate, and compare multiple systematic trading strategies using financial time series data.  
This project integrates factor-based models, momentum and mean reversion logic, statistical arbitrage, and machine learning/deep learning methods into a cohesive and extensible framework.  
An interactive dashboard built with Streamlit enables effective strategy testing, model validation, and result visualization.

The project aims to bridge academic research and practical application, providing a robust environment for both theoretical exploration and empirical backtesting.

---

## 🛠️ Key Features

- Comprehensive implementation of five quantitative trading strategies
- Empirical backtesting and walk-forward validation with performance metrics
- Portfolio optimization and risk management tools
- Visualization of cumulative returns, benchmarks, and model diagnostics
- Modular, extensible codebase for reproducible research and experimentation
- Interactive Streamlit web interface for user-friendly analysis

---

## 🧠 Strategy Model Descriptions

### 1. Factor Model Strategy (`factor_model.py`)

This strategy employs a **Principal Component Analysis (PCA)-based factor model** to extract latent risk factors from historical rolling returns.  
The methodology replaces traditional predefined factors (e.g., market, size, value, momentum) with **data-driven statistical factors**, allowing flexible factor discovery across various market environments.

- **Factor Extraction:** PCA on lagged return matrices (rolling window)
- **Signal Generation:** First principal component used as the signal for position allocation  
- **Positioning:** Long/short based on factor signal direction  
- **Performance Metrics:** Total Return, Sharpe Ratio, Maximum Drawdown, Explained Variance  

> 📌 *Unlike traditional factor models that rely on economic intuition, this approach uses unsupervised learning (PCA) to identify dominant return drivers from the data itself.*

---

### 2. Momentum & Mean Reversion Strategy (`momentum_mean_reversion.py`)

This hybrid strategy integrates **momentum (trend-following)** and **mean reversion (contrarian)** techniques using technical indicators.

- **Indicators Used:**  
  - Moving Averages (short-term and long-term crossovers)  
  - Bollinger Bands (price deviations from moving average ± standard deviation)  
  - Relative Strength Index (RSI)  

- **Signal Logic:**  
  - Momentum: Buy when short-term MA crosses above long-term MA + RSI supports bullish momentum  
  - Mean Reversion: Buy near lower Bollinger Band with oversold RSI (relaxed condition: RSI < 40), Sell near upper band with overbought RSI (RSI > 60)  
- **Position Persistence:** Positions are maintained via forward-filling until signal reversal  
- **Performance Metrics:** Cumulative Return, Sharpe Ratio, Max Drawdown, Number of Trades  

---

### 3. Statistical Arbitrage Strategy (`statistical_arbitrage.py`)

This strategy performs **pair trading** using spread reversion between two co-moving assets.

- **Spread Calculation:** Difference between the prices of two correlated assets  
- **Z-score Normalization:** Standardization of spread based on rolling mean and standard deviation  
- **Entry/Exit Conditions:**  
  - Enter long/short when z-score crosses entry threshold (±entry_z)  
  - Exit position when z-score reverts within the exit band (±exit_z)  
- **Market Neutrality:** Designed to be delta-neutral between the two assets  
- **Performance Metrics:** Total Return, Sharpe Ratio, Maximum Drawdown  

> 📌 *Focuses on mean reversion opportunities in statistically identified relationships between asset pairs, independent of broader market direction.*

---

### 4. Machine Learning Strategy (`ml_strategy.py`)

Applies **supervised classification models** to predict the next-day stock movement (up/down) based on technical features.

- **Models Supported:**  
  - Random Forest Classifier  
  - XGBoost Classifier (Gradient Boosted Trees)  

- **Feature Engineering:**  
  - Moving Average (MA)  
  - Standard Deviation (Volatility)  
  - Momentum (relative price change over window)  
  - Return Volatility  

- **Labeling:** Binary classification (1 if next day's close > today's close, else 0)  
- **Model Evaluation:**  
  - Train/Test Split (70/30)  
  - Walk-forward validation (step-based rolling re-training)  
- **Performance Metrics:** Accuracy, Sharpe Ratio, Max Drawdown, Feature Importance, Confusion Matrix, Classification Report  

---

### 5. Deep Learning Strategy (`dl_strategy.py`)

Utilizes **LSTM (Long Short-Term Memory)** networks with an integrated **Attention Mechanism** to model nonlinear temporal dependencies in stock price sequences.

- **Architecture:**  
  - Multi-layer LSTM to process sequential time series data  
  - Attention layer to focus on important time steps in the sequence  
  - Fully connected output layer for predicting future closing prices  

- **Feature Set:**  
  - OHLCV (Open, High, Low, Close, Volume)  
  - SMA (Simple Moving Averages: 20, 50 periods)  
  - RSI (Relative Strength Index)  
  - MACD (Moving Average Convergence Divergence)  

- **Loss Function:** Mean Squared Error (MSE)  
- **Optimization:** Adam optimizer, Early Stopping based on validation loss  
- **Backtesting Approach:** Regression-based price forecasting with cumulative return calculation and benchmark comparison  
- **Performance Metrics:** Total Return, Sharpe Ratio, Maximum Drawdown, Training Loss History  

> 📌 *The attention-enhanced LSTM improves interpretability by highlighting the contribution of different time steps to the prediction.*

---

## 🏗️ Project Structure

multi-strategy_stock_trading_analysis/
├── app.py
├── requirements.txt
├── strategies/
│   ├── dl_strategy.py
│   ├── factor_model.py
│   ├── ml_strategy.py
│   ├── momentum_mean_reversion.py
│   ├── optimizer_dl_strategy.py
│   ├── optimizer_factor_model.py
│   ├── optimizer_ml_strategy.py
│   ├── optimizer_momentum.py
│   ├── optimizer_stat_arb.py
│   └── statistical_arbitrage.py
├── .gitignore
└── LICENSE


---

## ⚡️ Quick Start

### Install dependencies:
```bash
pip install -r requirements.txt
```

###Run the Streamlit app
```bash
streamlit run app.py
```

---

## 💡 Use Cases
Academic research on quantitative trading and time series forecasting

Empirical evaluation and comparison of trading strategies using real financial data

Exploration of machine learning and deep learning techniques in algorithmic trading

Portfolio construction and systematic risk management analysis

---

## 🙋‍♂️ Author
Name: [Seong Jun Chang]
Email: [sjchang.stats@gmail.com]

---

## 📄 License
This project is licensed under the MIT License.

---

⭐️ Portfolio Statement for Master’s Program Application
“This project demonstrates my ability to apply data science methodologies to financial market analysis, integrating quantitative modeling, machine learning, and deep learning approaches for systematic trading strategy development.
Through rigorous feature engineering, model implementation, and empirical performance evaluation, I built an end-to-end research framework that bridges academic insights with practical application.
The project reflects my commitment to data-driven decision-making, predictive modeling, and risk management—key areas I aim to further explore through graduate study in Data Science.”
