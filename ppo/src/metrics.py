"""Risk-adjusted performance metrics shared by baselines and PPO evaluation."""

import numpy as np


def portfolio_returns(portfolio_values):
    values = np.asarray(portfolio_values, dtype=np.float64)
    return values[1:] / values[:-1] - 1.0


def max_drawdown(portfolio_values):
    values = np.asarray(portfolio_values, dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    return float(drawdown.min())


def sharpe_ratio(returns, periods_per_year=24 * 365, risk_free_rate=0.0):
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year=24 * 365, risk_free_rate=0.0):
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(portfolio_values, periods_per_year=24 * 365):
    values = np.asarray(portfolio_values, dtype=np.float64)
    n_periods = len(values) - 1
    if n_periods <= 0:
        return 0.0
    total_return = values[-1] / values[0] - 1.0
    years = n_periods / periods_per_year
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    mdd = abs(max_drawdown(values))
    if mdd == 0:
        return 0.0
    return float(annualized_return / mdd)


def summarize(portfolio_values, periods_per_year=24 * 365, risk_free_rate=0.0):
    values = np.asarray(portfolio_values, dtype=np.float64)
    returns = portfolio_returns(values)
    return {
        "total_return_pct": float((values[-1] / values[0] - 1.0) * 100),
        "sharpe": sharpe_ratio(returns, periods_per_year, risk_free_rate),
        "sortino": sortino_ratio(returns, periods_per_year, risk_free_rate),
        "max_drawdown_pct": float(max_drawdown(values) * 100),
        "calmar": calmar_ratio(values, periods_per_year),
    }
