import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from strategies.optimizer_factor_model import FactorModelOptimizer
from strategies.optimizer_momentum import MomentumOptimizer
from strategies.optimizer_dl_strategy import DLStrategyOptimizer
import os
import joblib
import json
import torch
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

# Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Multi-Strategy Stock Analysis", layout="wide")

# Add Bootstrap Icons CDN
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">', unsafe_allow_html=True)

# Define icon style
icon_style = "font-size: 1.2em; margin-right: 0.5em;"

# Title (with icon in HTML)
st.markdown(f"""
<h1><i class='bi bi-graph-up-arrow' style='{icon_style}'></i> Multi-Strategy Stock Analysis Dashboard</h1>
""", unsafe_allow_html=True)
st.markdown("Now with Automated Strategy Optimization!")
st.markdown("---")

# --------------------------
# 1. Select Ticker and Date Range
# --------------------------
st.sidebar.markdown(f"<h3><i class='bi bi-calendar-range' style='{icon_style}'></i> 1. Select Ticker and Date Range</h3>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 2)
start_date = st.sidebar.date_input("Start Date", value=start_date)
end_date = st.sidebar.date_input("End Date", value=end_date)

# --------------------------
# 2. Select Strategy
# --------------------------
st.sidebar.markdown(f"<h3><i class='bi bi-gear' style='{icon_style}'></i> 2. Select Strategy</h3>", unsafe_allow_html=True)
strategy_options = [
    "Factor Model",
    "Momentum/Mean Reversion",
    "Statistical Arbitrage",
    "Machine Learning",
    "Deep Learning"
]
selected_strategy = st.sidebar.selectbox("Select a strategy to analyze", options=strategy_options)

# --------------------------
# 3. Run Optimized Analysis
# --------------------------
st.sidebar.markdown(f"<h3><i class='bi bi-play-fill' style='{icon_style}'></i> Run Optimized Analysis</h3>", unsafe_allow_html=True)
if st.sidebar.button("Run Optimized Analysis"):
    st.success("✅ Running optimization and analysis...")

    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)

    # ------------------------
    # Factor Model
    # ------------------------
    if selected_strategy == "Factor Model":
        optimizer = FactorModelOptimizer(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        lookbacks = range(30, 181, 30)
        factors = range(3, 9)
        result_df = optimizer.run_grid_search(lookbacks, factors)

        if result_df.empty:
            st.error("Optimization failed: No valid results found.")
        else:
            best_result = result_df.sort_values("return", ascending=False).iloc[0]
            st.markdown(f"<h4><i class='bi bi-search' style='{icon_style}'></i> Best Parameter Combination</h4>", unsafe_allow_html=True)

            formatted_result = best_result.copy()
            formatted_result['return'] = f"{best_result['return']:.2f}%"
            formatted_result['max_drawdown'] = f"{best_result['max_drawdown']:.2f}%"
            formatted_result['sharpe'] = f"{best_result['sharpe']:.2f}"
            st.dataframe(formatted_result.drop('curve').to_frame().T.reset_index(drop=True), use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-graph-up' style='{icon_style}'></i> Cumulative Return Curve</h4>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=best_result['curve'].index, y=best_result['curve'], mode='lines', name='Strategy'))
            fig.add_trace(go.Scatter(x=best_result['curve'].index, y=best_result['benchmark'], mode='lines', name='Benchmark'))
            fig.update_layout(title='Best Factor Model Return Curve', xaxis_title='Date', yaxis_title='Cumulative Return')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-grid-3x3' style='{icon_style}'></i> Return Heatmap</h4>", unsafe_allow_html=True)
            pivot_table = result_df.pivot_table(index="lookback", columns="factors", values="return")
            heatmap_fig = px.imshow(pivot_table.values,
                                    labels=dict(x="# of Factors", y="Lookback", color="Total Return"),
                                    x=pivot_table.columns,
                                    y=pivot_table.index,
                                    text_auto=True,
                                    color_continuous_scale="RdYlGn")
            st.plotly_chart(heatmap_fig, use_container_width=True)

    # ------------------------
    # Momentum/Mean Reversion
    # ------------------------
    elif selected_strategy == "Momentum/Mean Reversion":
        optimizer = MomentumOptimizer(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), data)
        short_range = range(5, 21, 5)
        long_range = range(30, 91, 30)
        std_range = [1.0, 1.5, 2.0, 2.5]
        rsi_range = range(7, 22, 7)
        result_df = optimizer.run_grid_search(short_range, long_range, std_range, rsi_range)

        if result_df.empty:
            st.error("Optimization failed: No valid results found.")
        else:
            best_result = result_df.sort_values("return", ascending=False).iloc[0]

            st.markdown(f"<h4><i class='bi bi-search' style='{icon_style}'></i> Best Parameters Summary</h4>", unsafe_allow_html=True)
            display_result = best_result.drop("curve") if "curve" in best_result else best_result
            st.dataframe(display_result.to_frame().T.reset_index(drop=True).style.format({
                "return": "{:.2f}%",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.2f}%"
            }), use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-graph-up' style='{icon_style}'></i> Cumulative Return Curve</h4>", unsafe_allow_html=True)
            curve_data = best_result.get('curve', best_result.get('chart'))
            if curve_data is not None and hasattr(curve_data, 'index'):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_data.index,
                    y=curve_data,
                    mode='lines',
                    name='Strategy'
                ))
                fig.add_trace(go.Scatter(
                    x=curve_data.index,
                    y=best_result.get('benchmark'),
                    mode='lines',
                    name='Benchmark'
                ))
                fig.update_layout(
                    title='Momentum/Mean Reversion: Best Return Curve',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return'
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-grid-3x3' style='{icon_style}'></i> Return Heatmap (Short vs Long)</h4>", unsafe_allow_html=True)
            heatmap_df = result_df.pivot_table(index="short_window", columns="long_window", values="return")
            heatmap_fig = px.imshow(
                heatmap_df.values,
                labels=dict(x="Long Window", y="Short Window", color="Return (%)"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                text_auto=True,
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

            with st.expander("Show All Optimization Results"):
                st.dataframe(result_df.sort_values("return", ascending=False).style.format({
                    "return": "{:.2f}%", "sharpe": "{:.2f}", "max_drawdown": "{:.2f}%"
                }), use_container_width=True)

    # ------------------------
    # Statistical Arbitrage
    # ------------------------
    elif selected_strategy == "Statistical Arbitrage":
        st.markdown(f"<div style='background-color:#eaf2fb; padding: 0.7em 1em; border-radius: 0.5em;'><i class='bi bi-arrow-left-right' style='{icon_style}'></i> This strategy compares two stocks for statistical arbitrage (pair trading).</div>", unsafe_allow_html=True)

        ticker_y = st.sidebar.text_input("Enter Paired Ticker", value="MSFT")

        data_x = yf.download(ticker, start=start_date, end=end_date)['Close']
        data_y = yf.download(ticker_y, start=start_date, end=end_date)['Close']
        data_x.name = ticker
        data_y.name = ticker_y

        if data_x.empty or data_y.empty:
            st.warning("One of the tickers returned empty data. Please check ticker symbols.")
        else:
            from strategies.optimizer_stat_arb import StatisticalArbitrageOptimizer

            optimizer = StatisticalArbitrageOptimizer(data_x, data_y, ticker, ticker_y)
            lookback_range = [20, 30, 40]
            entry_range = [1.0, 1.5, 2.0]
            exit_range = [0.3, 0.5, 0.7]

            result_df = optimizer.run_grid_search(lookback_range, entry_range, exit_range)

            if result_df.empty:
                st.error("Optimization failed: No valid results found.")
            else:
                best_result = result_df.sort_values("return", ascending=False).iloc[0]
                st.markdown(f"<h4><i class='bi bi-search' style='{icon_style}'></i> Best Parameter Combination</h4>", unsafe_allow_html=True)

                formatted = best_result.copy()
                formatted['return'] = f"{formatted['return']:.2f}%"
                formatted['max_drawdown'] = f"{formatted['max_drawdown']:.2f}%"
                formatted['sharpe'] = f"{formatted['sharpe']:.2f}"
                st.dataframe(formatted.drop("curve").to_frame().T.reset_index(drop=True), use_container_width=True)

                st.markdown(f"<h4><i class='bi bi-graph-up' style='{icon_style}'></i> Cumulative Return Curve</h4>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=best_result['curve'].index, y=best_result['curve'], mode='lines', name='Strategy'))
                fig.add_trace(go.Scatter(x=best_result['curve'].index, y=best_result['benchmark'], mode='lines', name='Benchmark'))
                fig.update_layout(title="Best Statistical Arbitrage Strategy", xaxis_title="Date", yaxis_title="Cumulative Return")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"<h4><i class='bi bi-grid-3x3' style='{icon_style}'></i> Return Heatmap</h4>", unsafe_allow_html=True)
                heatmap_df = result_df.pivot_table(index="lookback", columns="entry_z", values="return")
                heatmap_fig = px.imshow(heatmap_df.values,
                                        labels=dict(x="Entry Z", y="Lookback", color="Return"),
                                        x=heatmap_df.columns,
                                        y=heatmap_df.index,
                                        text_auto=True,
                                        color_continuous_scale="RdYlGn")
                st.plotly_chart(heatmap_fig, use_container_width=True)

                st.markdown(f"<h4><i class='bi bi-table' style='{icon_style}'></i> Full Optimization Result</h4>", unsafe_allow_html=True)
                st.dataframe(result_df.sort_values("return", ascending=False).reset_index(drop=True), use_container_width=True)

    # ------------------------
    # Machine Learning
    # ------------------------
    elif selected_strategy == "Machine Learning":
        st.markdown(f"<div style='background-color:#eaf2fb; padding: 0.7em 1em; border-radius: 0.5em;'><i class='bi bi-cpu' style='{icon_style}'></i> This strategy uses Random Forest / XGBoost to predict tomorrow's return direction.</div>", unsafe_allow_html=True)

        from strategies.optimizer_ml_strategy import MLStrategyOptimizer

        optimizer = MLStrategyOptimizer(data)
        window_range = [3, 5, 10, 15]
        max_depth_range = [3, 5, 10]
        n_estimators_range = [50, 100, 200]

        result_df = optimizer.run_grid_search(window_range, max_depth_range, n_estimators_range)

        if result_df.empty:
            st.error("Optimization failed: No valid results found.")
        else:
            best_result = result_df.iloc[0]

            st.markdown(f"<h4><i class='bi bi-search' style='{icon_style}'></i> Best Parameter Combination</h4>", unsafe_allow_html=True)
            display_metrics = {
                'model': best_result.get('model', 'RandomForest'),
                'window': best_result.get('window', 0),
                'max_depth': best_result.get('max_depth', 0),
                'n_estimators': best_result.get('n_estimators', 0),
                'return': f"{best_result.get('return', 0):.2f}%",
                'accuracy': f"{best_result.get('accuracy', 0):.2f}%",
                'sharpe': f"{best_result.get('sharpe', 0):.2f}",
            }
            st.dataframe(pd.DataFrame([display_metrics]).T, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-graph-up' style='{icon_style}'></i> Cumulative Return Curve</h4>", unsafe_allow_html=True)
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=best_result['curve'].index,
                y=best_result['curve'],
                mode='lines',
                name='Strategy'
            ))
            fig_returns.add_trace(go.Scatter(
                x=best_result['curve'].index,
                y=best_result['benchmark'],
                mode='lines',
                name='Benchmark'
            ))
            fig_returns.update_layout(
                title='Machine Learning Strategy: Best Return Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative Return'
            )
            st.plotly_chart(fig_returns, use_container_width=True)

            if 'confusion_matrix' in best_result:
                st.markdown(f"<h4><i class='bi bi-grid-3x3' style='{icon_style}'></i> Confusion Matrix</h4>", unsafe_allow_html=True)
                cm = best_result['confusion_matrix']
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual Down', 'Actual Up'],
                    columns=['Predicted Down', 'Predicted Up']
                )
                st.dataframe(cm_df, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-speedometer2' style='{icon_style}'></i> Performance Metrics</h4>", unsafe_allow_html=True)
            metrics = {
                'Average Return': f"{result_df['return'].mean() * 100:.2f}%",
                'Max Return': f"{result_df['return'].max() * 100:.2f}%",
                'Min Return': f"{result_df['return'].min() * 100:.2f}%",
                'Average Accuracy': f"{result_df['accuracy'].mean() * 100:.2f}%"
            }
            st.dataframe(pd.DataFrame([metrics]).T, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-bar-chart' style='{icon_style}'></i> Parameter Impact Analysis</h4>", unsafe_allow_html=True)
            
            fig_window = px.box(result_df, x='window', y='return',
                              title='Impact of Window Size on Returns')
            st.plotly_chart(fig_window, use_container_width=True)

            fig_depth = px.box(result_df, x='max_depth', y='return',
                              title='Impact of Max Depth on Returns')
            st.plotly_chart(fig_depth, use_container_width=True)

            fig_model = px.box(result_df, x='model', y='return',
                              title='Return by Model Type')
            st.plotly_chart(fig_model, use_container_width=True)

            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            # Save best ML model
            if hasattr(best_result, 'model'):
                joblib.dump(best_result['model'], os.path.join(models_dir, f'{ticker}_best_ml_model.pkl'))
            # Save best ML parameters
            ml_params = {k: v for k, v in best_result.items() if k in ['model', 'window', 'max_depth', 'n_estimators']}
            with open(os.path.join(models_dir, f'{ticker}_best_ml_params.json'), 'w') as f:
                json.dump(ml_params, f, indent=2, cls=NumpyEncoder)

    # ------------------------
    # Deep Learning
    # ------------------------
    elif selected_strategy == "Deep Learning":
        st.markdown(f"<div style='background-color:#eaf2fb; padding: 0.7em 1em; border-radius: 0.5em;'><i class='bi bi-diagram-3' style='{icon_style}'></i> This strategy uses Deep Learning with LSTM and Attention mechanism to predict price movements.</div>", unsafe_allow_html=True)
        
        global_seed = 42
        
        with st.spinner('Running grid search optimization... This may take a while (162 combinations)'):
            optimizer = DLStrategyOptimizer(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), global_seed=global_seed)
            best_params, results_df = optimizer.optimize_strategy()
        
        if results_df.empty:
            st.error("Optimization failed: No valid results found.")
        else:
            st.markdown(f"<h4><i class='bi bi-search' style='{icon_style}'></i> Best Parameters</h4>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([best_params]).T, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-speedometer2' style='{icon_style}'></i> Performance Metrics</h4>", unsafe_allow_html=True)
            metrics = {
                'Best Return': f"{best_params['return']:.2f}%",
                'Average Return': f"{results_df['return'].mean():.2f}%",
                'Max Return': f"{results_df['return'].max():.2f}%",
                'Min Return': f"{results_df['return'].min():.2f}%"
            }
            st.dataframe(pd.DataFrame([metrics]).T, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-graph-up' style='{icon_style}'></i> Cumulative Return Curve</h4>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=best_params['curve'].index,
                y=best_params['curve'],
                mode='lines',
                name='Strategy'
            ))
            fig.add_trace(go.Scatter(
                x=best_params['curve'].index,
                y=best_params['benchmark'],
                mode='lines',
                name='Benchmark'
            ))
            fig.update_layout(
                title='Deep Learning Strategy: Best Return Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative Return'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-graph-down' style='{icon_style}'></i> Training Loss Curve</h4>", unsafe_allow_html=True)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(len(best_params['loss_history']))),
                y=best_params['loss_history'],
                mode='lines',
                name='Training Loss'
            ))
            fig_loss.update_layout(
                title='Model Training Loss Over Time',
                xaxis_title='Epoch',
                yaxis_title='Loss'
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            if 'attention_weights' in best_params:
                st.markdown(f"<h4><i class='bi bi-eye' style='{icon_style}'></i> Attention Weights</h4>", unsafe_allow_html=True)
                fig_attention = go.Figure(data=go.Heatmap(
                    z=best_params['attention_weights'],
                    x=list(range(best_params['attention_weights'].shape[1])),
                    y=list(range(best_params['attention_weights'].shape[0])),
                    colorscale='Viridis'
                ))
                fig_attention.update_layout(
                    title='Attention Weights Heatmap',
                    xaxis_title='Time Steps',
                    yaxis_title='Sequence Length'
                )
                st.plotly_chart(fig_attention, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-bar-chart' style='{icon_style}'></i> Prediction Error Distribution</h4>", unsafe_allow_html=True)
            errors = best_params['actual'] - best_params['predictions']
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                name='Error Distribution'
            ))
            fig_error.update_layout(
                title='Distribution of Prediction Errors',
                xaxis_title='Prediction Error',
                yaxis_title='Frequency'
            )
            st.plotly_chart(fig_error, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-bar-chart' style='{icon_style}'></i> Parameter Impact Analysis</h4>", unsafe_allow_html=True)
            
            fig_seq = px.box(results_df, x='sequence_length', y='return',
                            title='Impact of Sequence Length on Returns')
            st.plotly_chart(fig_seq, use_container_width=True)

            fig_hidden = px.box(results_df, x='hidden_size', y='return',
                              title='Impact of Hidden Size on Returns')
            st.plotly_chart(fig_hidden, use_container_width=True)

            st.markdown(f"<h4><i class='bi bi-table' style='{icon_style}'></i> Full Optimization Results</h4>", unsafe_allow_html=True)
            st.dataframe(results_df.sort_values('return', ascending=False), use_container_width=True)

            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            # Save best DL model
            if 'model' in best_params:
                torch.save(best_params['model'].state_dict(), os.path.join(models_dir, f'{ticker}_best_dl_model.pth'))
            dl_params = {k: v for k, v in best_params.items() if k not in ['curve', 'benchmark', 'loss_history', 'attention_weights', 'actual', 'predictions']}
            with open(os.path.join(models_dir, f'{ticker}_best_dl_params.json'), 'w') as f:
                json.dump(dl_params, f, indent=2, cls=NumpyEncoder)

    else:
        st.warning("Optimization currently only available for Factor Model and Momentum Strategy. Other strategies will be supported soon.")