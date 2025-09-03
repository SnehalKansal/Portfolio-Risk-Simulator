import numpy as np
import pandas as pd
from portfolio_core import (
    compute_log_returns,
    run_monte_carlo_simulation,
    calculate_risk_metrics,
    get_portfolio_metrics,
)


def test_compute_log_returns_empty():
    assert compute_log_returns(pd.DataFrame()).empty


def test_compute_log_returns_nonempty():
    df = pd.DataFrame({"A": [100, 101, 102], "B": [50, 49, 51]})
    logs = compute_log_returns(df)
    assert not logs.empty
    assert set(logs.columns) == {"A", "B"}


def test_run_monte_carlo_shapes():
    rng = np.random.default_rng(0)
    # Create synthetic log returns for 2 assets and 300 days
    data = rng.normal(0.0005, 0.01, size=(300, 2))
    log_returns = pd.DataFrame(data, columns=["A", "B"])
    weights = np.array([0.6, 0.4])
    final_values, paths = run_monte_carlo_simulation(log_returns, weights, 2000, 60)
    assert final_values.size == 2000
    assert paths.shape[0] == 61


def test_metrics_basic():
    rng = np.random.default_rng(1)
    data = rng.normal(0.0003, 0.01, size=(252, 3))
    logs = pd.DataFrame(data, columns=["X", "Y", "Z"])
    w = np.array([1/3, 1/3, 1/3])
    m = get_portfolio_metrics(w, logs, 0.02)
    assert isinstance(m.annual_return, float)
    assert isinstance(m.annual_volatility, float)
    assert isinstance(m.sharpe_ratio, float)


def test_var_cvar_bounds():
    vals = np.array([90, 95, 100, 105, 110])
    var95, cvar95 = calculate_risk_metrics(vals, 0.95, 100)
    assert 0.0 <= var95 <= 1.0
    assert 0.0 <= cvar95 <= 1.0


