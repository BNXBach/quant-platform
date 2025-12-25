from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.quant.phase1_cvar.config import p1_config
from backend.quant.phase1_cvar.logging_utils import setup_logger
from backend.quant.phase1_cvar.markowitz import markowitz_portfolio
from backend.quant.phase1_cvar.cvar import cvar_portfolio

# If you have scenario generators as functions, import them here.
# Otherwise load pre-generated scenario files per seed.
from backend.quant.phase1_cvar.scenarios import generate_scenarios, ScenarioSpec  # adjust to your repo


def concentration_metrics(w: np.ndarray) -> dict:
    hhi = float(np.sum(w**2))
    return {
        "max_weight": float(np.max(w)),
        "hhi": hhi,
        "n_eff": float(1.0 / hhi) if hhi > 0 else None,
    }


def distances(w: np.ndarray, w_ref: np.ndarray) -> dict:
    d = w - w_ref
    return {
        "d1": float(np.sum(np.abs(d))),
        "d2": float(np.sqrt(np.sum(d**2))),
        "dinf": float(np.max(np.abs(d))),
    }


def main():
    logger = setup_logger(
        name="phase1",
        level=logging.INFO,
        log_file="backend/reports/phase1/logs/phase1_optimize_stability.log",
    )

    # --- settings (match Step 3) ---
    w_max = p1_config.w_max          # Example max weight per asset
    l2_tau = p1_config.l2_tau
    alpha = p1_config.alpha

    # choose which methods to test
    methods = ["historical", "mbb"]

    # choose seeds
    seeds = list(range(10))  # 10 runs is enough for a strong stability section

    # paths
    out_dir = Path("backend/reports/phase1/optimize")
    out_dir.mkdir(parents=True, exist_ok=True)

    # load historical returns used for mu/Sigma
    rets = pd.read_parquet("backend/data/returns.parquet")
    rets_matrix = rets.to_numpy()
    assets = list(rets.columns)

    rows = []

    for method in methods:
        logger.info("Running stability for method=%s seeds=%s", method, seeds)

        # First pass: solve CVaR for each seed, store, and pick reference
        cvar_runs = []
        for seed in seeds:
            spec = ScenarioSpec(method=method, n_scenarios=2000, seed=seed, block_len=5)
            X = generate_scenarios(rets, spec)  # (S, N)

            cvar_json = cvar_portfolio(rets_matrix, w_max=w_max, l2_tau=l2_tau, scenarios=X, alpha=alpha)
            cvar_res = json.loads(cvar_json)
            w_cvar = np.array(cvar_res["weights"], dtype=float)

            cvar_runs.append((seed, X, cvar_res, w_cvar))

        # reference = mean weights across seeds (more stable than picking seed 0)
        w_ref_cvar = np.mean([w for _, _, _, w in cvar_runs], axis=0)

        # Second pass: run Markowitz at iso-return and compute stability metrics
        mv_runs = []
        for seed, _, cvar_res, w_cvar in cvar_runs:
            mu_star = float(cvar_res["expected_return"])

            mv_json = markowitz_portfolio(
                rets_matrix,
                target_return=mu_star,
                w_max=w_max,
                l2_tau=l2_tau,
                scenarios=X,
                alpha=alpha,
            )
            mv_res = json.loads(mv_json)
            w_mv = np.array(mv_res["weights"], dtype=float)
            mv_runs.append((seed, mv_res, w_mv))

        w_ref_mv = np.mean([w for _, _, w in mv_runs], axis=0)
        for (seed, X, cvar_res, w_cvar), (_, mv_res, w_mv) in zip(cvar_runs, mv_runs):
            row = {
                "method": method,
                "seed": seed,
                "cvar_prob_status": cvar_res["status"],
                "mv_prob_status": mv_res["status"],

                # ---- CVaR ----
                "cvar_value": float(cvar_res.get("cvar") or cvar_res.get("cvar_tail_mean")),
                "cvar_var": float(cvar_res.get("var") or cvar_res.get("var_quantile")),
                "cvar_variance": float(cvar_res["variance"]),
                "cvar_exp_return": float(cvar_res["expected_return"]),
                **{f"cvar_{k}": v for k, v in concentration_metrics(w_cvar).items()},
                **{f"cvar_{k}": v for k, v in distances(w_cvar, w_ref_cvar).items()},

                # ---- Markowitz ----
                "mv_value": float(mv_res["cvar"]),          # scenario-evaluated CVaR for Markowitz portfolio
                "mv_var": float(mv_res["var"]),
                "mv_variance": float(mv_res["variance"]),
                "mv_exp_return": float(mv_res["expected_return"]),
                **{f"mv_{k}": v for k, v in concentration_metrics(w_mv).items()},
                **{f"mv_{k}": v for k, v in distances(w_mv, w_ref_mv).items()},
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stability_comparison.csv", index=False)
    print(df.head())
    # summary table
    summary = df.groupby("method").agg(
        cvar_d1_mean=("cvar_d1", "mean"),
        cvar_d1_std=("cvar_d1", "std"),
        mv_d1_mean=("mv_d1", "mean"),
        mv_d1_std=("mv_d1", "std"),
        cvar_risk_std=("cvar_value", "std"),
        mv_risk_std=("mv_value", "std"),
    )
    summary.to_csv(out_dir / "stability_comparison_summary.csv")

    logger.info("Saved stability outputs to %s", out_dir.resolve())
    logger.info("Summary:\n%s", summary)


if __name__ == "__main__":
    main()