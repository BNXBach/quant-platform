from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import json

from backend.quant.phase1_cvar.config import p1_config
from backend.quant.phase1_cvar.logging_utils import setup_logger
from backend.quant.phase1_cvar.markowitz import markowitz_portfolio
from backend.quant.phase1_cvar.cvar import cvar_portfolio
from backend.quant.phase1_cvar.scenarios import ScenarioSpec, generate_scenarios, wassertein_dist


def main():
    logger = setup_logger(
        name="phase1",
        level=logging.INFO,
        log_file="backend/reports/phase1/logs/phase1_optimize.log",
    )

    scen_dir = Path("backend/reports/phase1/scenarios")
    scen_files = {
        "historical": scen_dir / "scenarios_historical.npy",
        "mbb": scen_dir / "scenarios_mbb.npy",
    }
    data_dir = p1_config.data_path
    rets_path = data_dir / "returns.parquet"
    out_dir = Path("backend/reports/phase1/optimize")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rets_path.exists():
        raise FileNotFoundError(f"Missing {rets_path}. Run data.py first.")

    rets = pd.read_parquet(rets_path)
    assets = list(rets.columns)
    logger.info("Loaded returns: shape=%s date_range=%s→%s", rets.shape, rets.index.min().date(), rets.index.max().date())
    logger.info("Assets: %s", list(rets.columns))

    rets_matrix = rets.to_numpy()
    w_max = p1_config.w_max          # Example max weight per asset
    l2_tau = p1_config.l2_tau
    alpha = p1_config.alpha         # Example L2 regularization

    all_results = {
        "config": {
            "w_max": w_max,
            "l2_tau": l2_tau,
            "assets": assets,
            "n_obs": int(rets.shape[0]),
        },
        "by_scenario_method": {}
    }

    for name, path in scen_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing scenario file: {path}. Run generate_scenarios.py first.")

        scenarios = np.load(path)
        logger.info("Loaded scenarios [%s]: shape=%s", name, scenarios.shape)

        # 1) Solve CVaR first
        cvar_json = cvar_portfolio(rets_matrix, w_max, l2_tau, scenarios, alpha)
        logger.info("CVaR Optimization result (%s): %s", name, cvar_json)

        cvar_res = json.loads(cvar_json)
        target_return = float(cvar_res.get("expected_return"))

        # 2) Solve Markowitz at the same expected return
        mv_json = markowitz_portfolio(rets_matrix, target_return, w_max, l2_tau, scenarios, alpha)
        logger.info("Markowitz Optimization result (%s): %s", name, mv_json)

        mv_res = json.loads(mv_json)

        all_results["by_scenario_method"][name] = {
            "cvar": cvar_res,
            "markowitz": mv_res,
            "iso_return_target": target_return,
        }

    out_path = out_dir / "optimize_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Saved optimization results to %s", out_path.resolve())
if __name__ == "__main__":
    main()