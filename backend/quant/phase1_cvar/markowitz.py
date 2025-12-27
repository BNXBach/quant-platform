from __future__ import annotations
import json  
import numpy as np  
import cvxpy as cp

def markowitz_portfolio(rets: np.ndarray, target_return: float, w_max: float = 1.0, l2_tau: float = 0.0, scenarios: np.ndarray = None, alpha: float = 0.95) -> str:
    N = rets.shape[1]  # Number of assets
    mu = np.mean(scenarios, axis=0)  # Expected returns
    Sigma = np.cov(scenarios, rowvar=False)  # Covariance matrix

    # Define optimization variables
    w = cp.Variable(N)

    # Define objective function: minimize variance + L2 regularization
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # Define constraints
    constraints = [
        cp.sum(w) == 1,  # Weights sum to 1
        target_return <= mu @ w,
        mu @ w <= target_return + 1e-7,
        w >= 0,  # No short selling
        w <= w_max  # Max weight constraint
    ]
    if l2_tau is not None and l2_tau > 0.0:
        constraints.append(cp.sum_squares(w) <= l2_tau)
        
    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    ok = prob.status in ("optimal", "optimal_inaccurate")
    weights = np.array(w.value).reshape(-1) if ok and w.value is not None else np.zeros(N) 

    if scenarios is not None:
        loss = -(scenarios @ weights)
        var = float(np.quantile(loss, alpha))
        cvar = float(loss[loss >= var].mean())
    else:
        var = None
        cvar = None

    expected_return = mu @ weights
    variance = weights.T @ Sigma @ weights
    return json.dumps({
        "target_return": target_return,
        "w_max": w_max,
        "l2_tau": l2_tau,
        "weights": weights.tolist(),
        "expected_return": expected_return,
        "variance": variance,   
        "var": var,
        "cvar": cvar,
        "status": prob.status   
    }   , indent=2)