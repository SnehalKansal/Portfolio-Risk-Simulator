import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns # For correlation heatmap
import plotly.graph_objects as go # For interactive charts
import plotly.express as px
import warnings
from portfolio_core import (
    ANNUALIZATION_FACTOR,
    fetch_adjusted_close_prices,
    compute_log_returns,
    run_monte_carlo_simulation,
    calculate_risk_metrics,
    get_portfolio_metrics,
    optimize_weights,
)

# Suppress specific warnings from yfinance or pandas that might clutter the output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning) # Often seen with chained assignments

# Set page configuration
st.set_page_config(layout="wide", page_title="Portfolio Risk Simulator & Optimizer 💸")

## --- Constants ---
# ANNUALIZATION_FACTOR imported from portfolio_core

## --- Functions ---

@st.cache_data
def get_data_cached(tickers, start_date, end_date):
    """Streamlit-cached wrapper around core data fetch."""
    return fetch_adjusted_close_prices(tickers, start_date, end_date)


## Simulation is imported from portfolio_core

## Risk metrics imported from portfolio_core

def calculate_component_var(log_returns, weights, confidence_level=0.95):
    """
    Calculates the contribution of each asset to the total portfolio VaR using an analytical approximation.
    """
    if log_returns.empty or weights.size == 0:
        return pd.Series(dtype=float) 

    cov_matrix = log_returns.cov()
    
    # Handle cases where cov_matrix might be problematic (e.g., single asset, all 0 returns)
    if cov_matrix.empty:
        return pd.Series(0, index=log_returns.columns) # Return 0 for all if no cov data
    if np.linalg.det(cov_matrix) == 0 and len(weights) > 1: # For single asset, det will be the variance, potentially 0
        st.warning("Covariance matrix is singular, component VaR approximation might be unreliable.")
        
    portfolio_std_daily = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    if portfolio_std_daily == 0:
        return pd.Series(0, index=log_returns.columns) 

    from scipy.stats import norm
    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha) 

    portfolio_mean_daily = np.sum(log_returns.mean() * weights)
    
    marginal_contributions_to_std = np.dot(cov_matrix, weights.T) / portfolio_std_daily
    total_var_return_daily = -(portfolio_mean_daily + z_score * portfolio_std_daily)
    
    component_var_return_daily = marginal_contributions_to_std * total_var_return_daily
    component_var_loss_pct = -component_var_return_daily * np.sqrt(ANNUALIZATION_FACTOR)

    return pd.Series(component_var_loss_pct, index=log_returns.columns)


## Metrics imported from portfolio_core

## Optimization handled via optimize_weights from portfolio_core

# --- Streamlit UI (remains largely the same, but with updated function calls and error handling) ---

st.title("💸 Portfolio Risk Simulator & Optimizer")
st.markdown("Analyze and optimize investment portfolios using Monte Carlo simulations and modern portfolio theory.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Portfolio Configuration")
    tickers_input = st.text_input("Ticker Symbols (comma-separated)", "MSFT,AAPL,GOOGL,AMZN,NVDA", help="e.g., MSFT, AAPL, GOOGL")
    weights_input = st.text_input("Initial Portfolio Weights (comma-separated, optional)", "0.2,0.2,0.2,0.2,0.2", help="Must sum to 1. If empty, equal weights are used.")
    
    st.header("2. Simulation & Optimization Parameters")
    data_col, sim_col = st.columns(2)
    with data_col:
        start_date = st.date_input("Data Start Date", pd.to_datetime('2020-01-01'))
    with sim_col:
        end_date = st.date_input("Data End Date", pd.Timestamp.now().date())

    num_simulations = st.slider("Number of Monte Carlo Simulations", 10000, 100000, 50000, step=10000)
    forecast_days = st.slider("Forecast Horizon (Trading Days)", 30, ANNUALIZATION_FACTOR * 3, ANNUALIZATION_FACTOR, help=f"{ANNUALIZATION_FACTOR} days is approximately 1 year.")
    risk_free_rate = st.slider("Risk-Free Rate (Annual)", 0.00, 0.10, 0.02, step=0.005, format="%.3f", help="Used for Sharpe Ratio calculation.")

    run_simulation_button = st.button("Run Simulation & Optimization 🚀", type="primary")

# --- Main Content ---
if run_simulation_button:
    try:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            st.stop()

        initial_weights = None
        if weights_input:
            try:
                input_weights = np.array([float(w.strip()) for w in weights_input.split(',')])
                if len(input_weights) != len(tickers):
                    st.error("Number of initial weights must match the number of tickers.")
                    st.stop()
                if not np.isclose(np.sum(input_weights), 1.0):
                    st.warning(f"Initial portfolio weights do not sum to 1 ({np.sum(input_weights):.2f}). Adjusting them to sum to 1.")
                    initial_weights = input_weights / np.sum(input_weights)
                else:
                    initial_weights = input_weights
            except ValueError:
                st.error("Invalid weight input. Please enter numbers separated by commas.")
                st.stop()
        else:
            initial_weights = np.array([1/len(tickers)] * len(tickers))

        # --- Data Fetching and Preprocessing ---
        with st.spinner("Fetching historical data..."):
            data = get_data_cached(tickers, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            
            if data.empty:
                st.error("No valid data retrieved for any of the tickers. Please check ticker symbols and date range.")
                st.stop()
            
            downloaded_tickers = data.columns.tolist()

            if len(downloaded_tickers) != len(tickers):
                missing_tickers = set(tickers) - set(downloaded_tickers)
                st.warning(f"Could not retrieve data for: {', '.join(missing_tickers)}. Proceeding with available tickers: {', '.join(downloaded_tickers)}.")
                tickers = downloaded_tickers # Update tickers list to only those with data
                
                if not tickers: 
                    st.error("No valid tickers found after data retrieval. Please re-check.")
                    st.stop()
                
                # Re-adjust initial weights based on available tickers
                initial_weights = np.array([1/len(tickers)] * len(tickers)) 
                
            log_returns = compute_log_returns(data)
            
            if log_returns.empty:
                st.error("Not enough historical data to calculate returns after dropping NaNs. Try a wider date range or fewer tickers.")
                st.stop()
            if len(log_returns) < ANNUALIZATION_FACTOR:
                st.warning(f"Less than {ANNUALIZATION_FACTOR} days of data available. Annualized metrics may be less reliable.")
            
            if set(log_returns.columns) != set(tickers):
                st.error("Internal error: Mismatch between log_returns columns and active tickers after processing.")
                st.stop()
                
        st.success("Data loaded successfully!")

        # --- Display Historical Data & Correlation ---
        st.header("Historical Data Analysis")
        
        hist_tab, corr_tab = st.tabs(["Historical Prices", "Correlation Matrix"])

        with hist_tab:
            st.subheader("Adjusted Close Prices")
            fig_hist = px.line(data, title="Historical Adjusted Close Prices")
            fig_hist.update_layout(hovermode="x unified", legend_title_text='Asset')
            st.plotly_chart(fig_hist, use_container_width=True)

        with corr_tab:
            st.subheader("Asset Correlation Matrix")
            if len(tickers) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, linewidths=.5)
                ax_corr.set_title("Log Returns Correlation Matrix")
                st.pyplot(fig_corr)
            else:
                st.info("Correlation matrix requires at least two assets.")

        # --- Run Monte Carlo Simulation ---
        st.subheader("Monte Carlo Simulation")
        with st.spinner(f"Running {num_simulations} Monte Carlo Simulations for {forecast_days} days..."):
            initial_portfolio_value_for_sim = 100 
            final_values, simulated_price_paths = run_monte_carlo_simulation(log_returns, initial_weights, num_simulations, forecast_days, initial_portfolio_value=initial_portfolio_value_for_sim)

        st.success("Simulation complete!")

        # --- Results Tabs ---
        sim_vis_tab, risk_metrics_tab, opt_tab = st.tabs(["Simulation Visuals", "Risk Metrics", "Portfolio Optimization"])

        with sim_vis_tab:
            st.header("Monte Carlo Simulation Results Visuals")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sample Simulated Price Paths")
                if simulated_price_paths.size > 0:
                    fig = go.Figure()
                    for i in range(simulated_price_paths.shape[1]):
                        fig.add_trace(go.Scatter(y=simulated_price_paths[:, i], mode='lines', name=f'Path {i+1}', opacity=0.3, showlegend=False))
                    fig.update_layout(
                        title=f"100 Sample Simulated Price Paths for {forecast_days} Days",
                        xaxis_title="Trading Days",
                        yaxis_title=f"Portfolio Value (Initial ${initial_portfolio_value_for_sim})",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No simulated price paths to display.")
            
            with col2:
                st.subheader("Final Portfolio Value Distribution")
                if final_values.size > 0:
                    fig_hist_final = px.histogram(x=final_values, nbins=50, marginal="box",
                                                title="Distribution of Final Portfolio Values",
                                                labels={'x': f"Final Portfolio Value (Initial ${initial_portfolio_value_for_sim})", 'count': "Frequency"})
                    fig_hist_final.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig_hist_final, use_container_width=True)
                else:
                    st.info("No final portfolio values to display.")
                
        with risk_metrics_tab:
            st.header("Portfolio Risk Analysis")
            
            st.subheader("Current Portfolio Performance (Initial Weights)")
            initial_metrics = get_portfolio_metrics(initial_weights, log_returns, risk_free_rate)
            initial_ann_return = initial_metrics.annual_return
            initial_ann_vol = initial_metrics.annual_volatility
            initial_sharpe = initial_metrics.sharpe_ratio
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Annualized Return", f"{initial_ann_return:.2%}")
            with metric_cols[1]:
                st.metric("Annualized Volatility", f"{initial_ann_vol:.2%}")
            with metric_cols[2]:
                st.metric("Sharpe Ratio", f"{initial_sharpe:.2f}")

            st.subheader("Value-at-Risk (VaR) & Conditional VaR (CVaR)")
            risk_cols = st.columns(3)
            
            if final_values.size > 0:
                VaR_95_loss, CVaR_95_loss = calculate_risk_metrics(final_values, 0.95, initial_portfolio_value=initial_portfolio_value_for_sim)
                VaR_99_loss, CVaR_99_loss = calculate_risk_metrics(final_values, 0.99, initial_portfolio_value=initial_portfolio_value_for_sim)
            else:
                VaR_95_loss, CVaR_95_loss = 0, 0
                VaR_99_loss, CVaR_99_loss = 0, 0

            with risk_cols[0]:
                st.metric("95% VaR (Loss %)", f"{VaR_95_loss:.2%}")
                st.markdown("*(Max expected loss with 95% confidence over forecast horizon)*")
            
            with risk_cols[1]:
                st.metric("99% VaR (Loss %)", f"{VaR_99_loss:.2%}")
                st.markdown("*(Max expected loss with 99% confidence over forecast horizon)*")
                
            with risk_cols[2]:
                st.metric("95% CVaR (Loss %)", f"{CVaR_95_loss:.2%}")
                st.markdown("*(Average loss given it exceeds 95% VaR)*")

            st.subheader("Component VaR (Contribution to Portfolio VaR)")
            if len(tickers) > 0 and not log_returns.empty:
                component_var_df = calculate_component_var(log_returns, initial_weights, confidence_level=0.95)
                st.dataframe(component_var_df.to_frame(name='Contribution to 95% VaR (Annualized Loss %)').style.format("{:.2%}"))
                st.info("*(Approximated contribution of each asset to the total portfolio VaR. Sum of contributions might not exactly equal total VaR due to non-linearity and approximation methods.)*")
            else:
                st.info("Component VaR requires at least one asset with valid data.")


        with opt_tab:
            st.header("Portfolio Optimization")
            st.markdown("Finds optimal weights based on different objectives and plots the efficient frontier.")
            
            opt_goal = st.radio("Optimization Goal:", ["Maximize Sharpe Ratio", "Minimize Volatility"], index=0, horizontal=True)

            if len(tickers) == 0:
                st.warning("Cannot optimize an empty portfolio. Please provide valid tickers.")
                optimized_weights = np.array([])
            elif len(tickers) == 1:
                st.info("Optimization is not meaningful for a single asset. The optimal weight will be 100% for that asset.")
                optimized_weights = np.array([1.0])
            else:
                with st.spinner(f"Finding Optimal Portfolio to {opt_goal}..."):
                    goal = "sharpe" if opt_goal == "Maximize Sharpe Ratio" else "volatility"
                    optimized_weights = optimize_weights(initial_weights, log_returns, risk_free_rate, goal=goal)

            optimal_metrics = get_portfolio_metrics(optimized_weights, log_returns, risk_free_rate)

            st.subheader("Optimal Portfolio Metrics")
            opt_metric_cols = st.columns(3)
            with opt_metric_cols[0]:
                st.metric("Optimal Annualized Return", f"{optimal_metrics.annual_return:.2%}")
            with opt_metric_cols[1]:
                st.metric("Optimal Annualized Volatility", f"{optimal_metrics.annual_volatility:.2%}")
            with opt_metric_cols[2]:
                st.metric(f"Optimal Sharpe Ratio", f"{optimal_metrics.sharpe_ratio:.2f}")

            if optimized_weights.size > 0:
                st.subheader("Optimal Portfolio Weights")
                weights_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Initial Weight': initial_weights,
                    'Optimal Weight': optimized_weights
                })
                weights_df = weights_df.set_index('Ticker')
                st.dataframe(weights_df.style.format({'Optimal Weight': "{:.2%}", 'Initial Weight': "{:.2%}"}))
            else:
                st.info("No optimal weights to display as optimization could not be performed.")


            st.subheader("Efficient Frontier")
            st.markdown("The Efficient Frontier represents portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given level of expected return.")

            if len(tickers) < 2:
                st.info("Efficient Frontier requires at least two assets to visualize diversification benefits.")
            else:
                with st.spinner("Calculating Efficient Frontier... This may take a moment for many assets."):
                    num_portfolios = 5000 
                    random_portfolio_data = []

                    for i in range(num_portfolios):
                        weights = np.random.random(len(tickers))
                        weights /= np.sum(weights)
                        
                        m = get_portfolio_metrics(weights, log_returns, risk_free_rate)
                        random_portfolio_data.append({'Return': m.annual_return, 'Volatility': m.annual_volatility, 'Sharpe': m.sharpe_ratio})

                    df_random_portfolios = pd.DataFrame(random_portfolio_data)

                    min_vol_idx = df_random_portfolios['Volatility'].idxmin()
                    min_vol_portfolio = df_random_portfolios.loc[min_vol_idx]

                    max_sharpe_idx = df_random_portfolios['Sharpe'].idxmax()
                    max_sharpe_portfolio = df_random_portfolios.loc[max_sharpe_idx]

                    fig_frontier = go.Figure()

                    fig_frontier.add_trace(go.Scatter(
                        x=df_random_portfolios['Volatility'], y=df_random_portfolios['Return'], mode='markers',
                        marker=dict(size=5, color=df_random_portfolios['Sharpe'], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
                        name='Random Portfolios',
                        hoverinfo='text',
                        text=[f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.2f}" for r, v, s in zip(df_random_portfolios['Return'], df_random_portfolios['Volatility'], df_random_portfolios['Sharpe'])]
                    ))

                    fig_frontier.add_trace(go.Scatter(
                        x=[min_vol_portfolio['Volatility']], y=[min_vol_portfolio['Return']], mode='markers',
                        marker=dict(color='red', size=12, symbol='star', line=dict(width=1, color='black')),
                        name='Min Volatility Portfolio',
                        hoverinfo='text',
                        text=f"Min Volatility<br>Return: {min_vol_portfolio['Return']:.2%}<br>Volatility: {min_vol_portfolio['Volatility']:.2%}<br>Sharpe: {min_vol_portfolio['Sharpe']:.2f}"
                    ))

                    fig_frontier.add_trace(go.Scatter(
                        x=[max_sharpe_portfolio['Volatility']], y=[max_sharpe_portfolio['Return']], mode='markers',
                        marker=dict(color='green', size=12, symbol='star', line=dict(width=1, color='black')),
                        name='Max Sharpe Portfolio',
                        hoverinfo='text',
                        text=f"Max Sharpe<br>Return: {max_sharpe_portfolio['Return']:.2%}<br>Volatility: {max_sharpe_portfolio['Volatility']:.2%}<br>Sharpe: {max_sharpe_portfolio['Sharpe']:.2f}"
                    ))

                    fig_frontier.add_trace(go.Scatter(
                        x=[initial_ann_vol], y=[initial_ann_return], mode='markers',
                        marker=dict(color='blue', size=10, symbol='circle', line=dict(width=1, color='black')),
                        name='Initial Portfolio',
                         hoverinfo='text',
                        text=f"Initial Portfolio<br>Return: {initial_ann_return:.2%}<br>Volatility: {initial_ann_vol:.2%}<br>Sharpe: {initial_sharpe:.2f}"
                    ))
                    
                    fig_frontier.add_trace(go.Scatter(
                        x=[optimal_metrics.annual_volatility], y=[optimal_metrics.annual_return], mode='markers',
                        marker=dict(color='purple', size=10, symbol='circle', line=dict(width=1, color='black')),
                        name=f'Optimized ({opt_goal.replace("Maximize ", "").replace("Minimize ", "")}) Portfolio',
                         hoverinfo='text',
                        text=f"Optimized Portfolio<br>Return: {optimal_metrics.annual_return:.2%}<br>Volatility: {optimal_metrics.annual_volatility:.2%}<br>Sharpe: {optimal_metrics.sharpe_ratio:.2f}"
                    ))

                    fig_frontier.update_layout(
                        title='Efficient Frontier with Monte Carlo Random Portfolios',
                        xaxis_title='Annualized Volatility',
                        yaxis_title='Annualized Return',
                        hovermode='closest',
                        showlegend=True,
                        height=600
                    )
                    fig_frontier.update_xaxes(tickformat=".2%")
                    fig_frontier.update_yaxes(tickformat=".2%")
                    st.plotly_chart(fig_frontier, use_container_width=True)


    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please check your inputs. Common issues include invalid ticker symbols, insufficient historical data for the selected date range, or internet connectivity problems.")

st.markdown("---")
st.markdown("Developed by [Snehal Kansal](https://github.com/SnehalKansal) • [LinkedIn](https://www.linkedin.com/in/snehalkansal/)")