# Portfolio Risk Simulator & Optimizer

Interactive Streamlit app plus reusable core library, CLI, tests, and Docker for simulating portfolio risk (Monte Carlo), computing VaR/CVaR, and optimizing weights (Sharpe or volatility).

## Features
- Streamlit UI: historical prices, correlation heatmap, Monte Carlo visuals, VaR/CVaR, optimization, efficient frontier
- Core Python module: clean, testable functions for fetching data, returns, simulation, metrics, optimization
- CLI: run simulations headlessly and print key metrics
- Tests: unit tests for core logic
- Docker: reproducible, one-command run

## Quickstart (UI)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI Usage
```bash
python cli.py --tickers MSFT,AAPL,GOOGL --start 2020-01-01 --end 2024-12-31 --simulations 20000 --days 252 --risk_free 0.02 --goal sharpe
```

## Tests
```bash
pip install -r requirements.txt
pip install pytest
pytest -q monte_carlo/tests
```

## Docker
```bash
docker build -t portfolio-simulator ./monte_carlo
docker run -p 8501:8501 portfolio-simulator
# Open http://localhost:8501
```

## Project Structure
```
monte_carlo/
  app.py                # Streamlit UI
  portfolio_core.py     # Reusable core logic (fetch, returns, sim, metrics, optimize)
  cli.py                # Headless CLI
  tests/                # Unit tests for core
  requirements.txt
  README.md
  Dockerfile
```

## Notes
- Data sourced via yfinance; ensure internet access.
- Educational use; not financial advice.


