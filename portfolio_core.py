import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import minimize
from datetime import datetime, timedelta


ANNUALIZATION_FACTOR: int = 252


@dataclass
class PortfolioMetrics:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float


def fetch_adjusted_close_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    # yfinance expects end to be exclusive; ensure end > start
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(days=1)
            end_date = end_dt.strftime("%Y-%m-%d")
    except Exception:
        # If parsing fails, proceed with given strings
        pass
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        # Try preferred path: history() API
        try:
            h = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d', auto_adjust=False)
            if not h.empty:
                col = 'Adj Close' if 'Adj Close' in h.columns else ('Close' if 'Close' in h.columns else None)
                if col:
                    series_map[ticker] = h[col]
                    continue
        except Exception:
            pass
        # Fallback #1: direct download with explicit interval
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            if not df.empty:
                col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if col:
                    series_map[ticker] = df[col]
                    continue
        except Exception:
            pass
        # Fallback #2: long period then slice
        try:
            df_fb = yf.download(ticker, period='10y', interval='1d', progress=False)
            if not df_fb.empty:
                col = 'Adj Close' if 'Adj Close' in df_fb.columns else ('Close' if 'Close' in df_fb.columns else None)
                if col:
                    df_fb = df_fb.loc[(df_fb.index >= start_date) & (df_fb.index < end_date)]
                    if not df_fb.empty:
                        series_map[ticker] = df_fb[col]
        except Exception:
            pass
    if not series_map:
        return pd.DataFrame()
    combined = pd.DataFrame(series_map).dropna(how='all')
    # Keep only tickers with at least some data
    valid_cols = [c for c in combined.columns if combined[c].notna().any()]
    return combined[valid_cols]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return np.log(1 + prices.pct_change()).dropna()


def run_monte_carlo_simulation(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    num_simulations: int,
    forecast_days: int,
    initial_portfolio_value: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if log_returns.empty or weights.size == 0:
        return np.array([]), np.array([])
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    if cov_matrix.empty or np.linalg.det(cov_matrix) == 0:
        return np.array([]), np.array([])
    try:
        L = np.linalg.cholesky(cov_matrix)
    except Exception:
        return np.array([]), np.array([])
    simulated_returns = np.zeros((forecast_days, num_simulations))
    simulated_price_paths_for_plot = np.zeros((forecast_days + 1, min(num_simulations, 100)))
    for i in range(num_simulations):
        random_returns = np.dot(L, np.random.normal(0, 1, size=(len(weights), forecast_days)))
        portfolio_daily_returns = np.sum(mean_returns.values.reshape(1, -1) + random_returns.T, axis=1)
        simulated_returns[:, i] = portfolio_daily_returns
        if i < 100:
            simulated_price_paths_for_plot[0, i] = initial_portfolio_value
            simulated_price_paths_for_plot[1:, i] = initial_portfolio_value * np.exp(np.cumsum(portfolio_daily_returns))
    final_values = initial_portfolio_value * np.exp(np.sum(simulated_returns, axis=0))
    return final_values, simulated_price_paths_for_plot


def calculate_risk_metrics(
    final_values: np.ndarray, confidence_level: float, initial_portfolio_value: float = 100.0
) -> Tuple[float, float]:
    if final_values.size == 0:
        return 0.0, 0.0
    var_value = np.percentile(final_values, (1 - confidence_level) * 100)
    cvar_value = final_values[final_values <= var_value].mean()
    var_loss = (initial_portfolio_value - var_value) / initial_portfolio_value if var_value < initial_portfolio_value else 0.0
    cvar_loss = (initial_portfolio_value - cvar_value) / initial_portfolio_value if cvar_value < initial_portfolio_value else 0.0
    return float(var_loss), float(cvar_loss)


def get_portfolio_metrics(weights: np.ndarray, log_returns: pd.DataFrame, risk_free_rate: float) -> PortfolioMetrics:
    if log_returns.empty or weights.size == 0:
        return PortfolioMetrics(0.0, 0.0, 0.0)
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    portfolio_return_daily = float(np.sum(mean_returns * weights))
    if cov_matrix.empty or np.linalg.det(cov_matrix) == 0:
        portfolio_std_daily = 0.0
    else:
        portfolio_std_daily = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    annual_return = portfolio_return_daily * ANNUALIZATION_FACTOR
    annual_vol = portfolio_std_daily * np.sqrt(ANNUALIZATION_FACTOR)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0.0
    return PortfolioMetrics(annual_return, annual_vol, sharpe)


def optimize_weights(
    initial_weights: np.ndarray,
    log_returns: pd.DataFrame,
    risk_free_rate: float,
    goal: str = "sharpe",
) -> np.ndarray:
    if log_returns.empty or len(initial_weights) == 0:
        return np.array([])
    bounds = tuple((0, 1) for _ in range(len(initial_weights)))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)

    def objective_sharpe(w: np.ndarray) -> float:
        return -get_portfolio_metrics(w, log_returns, risk_free_rate).sharpe_ratio

    def objective_vol(w: np.ndarray) -> float:
        return get_portfolio_metrics(w, log_returns, risk_free_rate).annual_volatility

    objective = objective_sharpe if goal == "sharpe" else objective_vol
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = result.x
    w_opt = np.maximum(0, w_opt)
    w_opt /= np.sum(w_opt)
    return w_opt


def random_portfolios(
    num_portfolios: int,
    num_assets: int,
    log_returns: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    records = []
    for _ in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        m = get_portfolio_metrics(w, log_returns, risk_free_rate)
        records.append({'Return': m.annual_return, 'Volatility': m.annual_volatility, 'Sharpe': m.sharpe_ratio})
    return pd.DataFrame(records)


