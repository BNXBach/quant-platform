from __future__ import annotations
import numpy as np
import pandas as pd

def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n

def turnover(w_prev: np.ndarray | None, w_new: np.ndarray) -> float:
    if w_prev is None:
        return 0.0
    return float(np.sum(np.abs(w_new - w_prev)))

def max_drawdown(rets: pd.Series) -> float:
    equity = pd.Series(np.exp(rets.cumsum()))
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())

def realized_portfolio_return(rets: np.ndarray, w: np.ndarray) -> float:
    return float(rets @ w)

def realized_var_cvar(rets: pd.Series, alpha: float = 0.95) -> tuple:
    loss = -rets.dropna().to_numpy()
    var = np.quantile(loss, alpha)
    cvar = loss[loss >= var].mean()
    return (float(var), float(cvar))