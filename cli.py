import argparse
import numpy as np
from datetime import date
from portfolio_core import (
    fetch_adjusted_close_prices,
    compute_log_returns,
    run_monte_carlo_simulation,
    calculate_risk_metrics,
    get_portfolio_metrics,
    optimize_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio Monte Carlo Simulator & Optimizer (CLI)")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g., MSFT,AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--simulations", type=int, default=20000, help="Number of Monte Carlo simulations")
    parser.add_argument("--days", type=int, default=252, help="Forecast horizon (trading days)")
    parser.add_argument("--risk_free", type=float, default=0.02, help="Annual risk-free rate")
    parser.add_argument("--goal", type=str, choices=["sharpe", "volatility"], default="sharpe", help="Optimization goal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    prices = fetch_adjusted_close_prices(tickers, args.start, args.end)
    if prices.empty:
        print("No price data fetched. Check tickers or dates.")
        return
    log_returns = compute_log_returns(prices)
    if log_returns.empty:
        print("Insufficient data to compute returns.")
        return
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    final_values, _ = run_monte_carlo_simulation(log_returns, initial_weights, args.simulations, args.days)
    var95, cvar95 = calculate_risk_metrics(final_values, 0.95)
    metrics_initial = get_portfolio_metrics(initial_weights, log_returns, args.risk_free)
    goal = "sharpe" if args.goal == "sharpe" else "volatility"
    optimized_w = optimize_weights(initial_weights, log_returns, args.risk_free, goal=goal)
    metrics_opt = get_portfolio_metrics(optimized_w, log_returns, args.risk_free)

    print("Tickers:", tickers)
    print(f"Initial Weights: {np.round(initial_weights, 4)}")
    print(f"Initial Annual Return: {metrics_initial.annual_return:.4f}")
    print(f"Initial Annual Volatility: {metrics_initial.annual_volatility:.4f}")
    print(f"Initial Sharpe: {metrics_initial.sharpe_ratio:.4f}")
    print(f"VaR95 Loss %: {var95:.4f}  CVaR95 Loss %: {cvar95:.4f}")
    print(f"Optimized Weights: {np.round(optimized_w, 4)}")
    print(f"Optimized Annual Return: {metrics_opt.annual_return:.4f}")
    print(f"Optimized Annual Volatility: {metrics_opt.annual_volatility:.4f}")
    print(f"Optimized Sharpe: {metrics_opt.sharpe_ratio:.4f}")


if __name__ == "__main__":
    main()


