from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backend.quant.phase1_cvar.config import p1_config
from backend.quant.phase1_cvar.logging_utils import setup_logger
from backend.quant.phase1_cvar.scenarios import ScenarioSpec, generate_scenarios, summarize_matrix, corr_matrix, plot_hist_vs_scen, plot_corr_heatmap

def main():
    logger = setup_logger(
        name="phase1",
        level=logging.INFO,
        log_file="backend/reports/phase1/logs/phase1_scenarios.log",
    )

    data_dir = p1_config.data_path
    out_dir = Path("backend/reports/phase1/scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)

    rets_path = data_dir / "returns.parquet"
    if not rets_path.exists():
        raise FileNotFoundError(f"Missing {rets_path}. Run data.py first.")

    rets = pd.read_parquet(rets_path)
    logger.info("Loaded returns: shape=%s date_range=%s→%s", rets.shape, rets.index.min().date(), rets.index.max().date())
    logger.info("Assets: %s", list(rets.columns))
    corr_rets = plot_corr_heatmap(corr_matrix(rets.to_numpy()),labels=list(rets.columns),title=f"data", out_dir=out_dir)
    corr_rets.show()
    # ---- Choose scenario specs to compare ----
    specs = [
        ScenarioSpec(method="historical", n_scenarios=2000, seed=1),
        ScenarioSpec(method="mbb", n_scenarios=2000, seed=1, block_len=5),
    ]

    results = {}

    for spec in specs:
        logger.info("Generating scenarios: method=%s S=%d seed=%d block_len=%s",
                    spec.method, spec.n_scenarios, spec.seed, spec.block_len)

        X = generate_scenarios(rets, spec)  # (S, N)
        logger.info("Scenario matrix shape=%s", X.shape)

        summ = summarize_matrix(X)
        results[spec.method] = summ

        # Save scenarios
        np.save(out_dir / f"scenarios_{spec.method}.npy", X)

        # Save summary JSON
        with open(out_dir / f"summary_{spec.method}.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)

        # Basic validation plots
        fig_hist = plot_hist_vs_scen(rets, X, title=f"{spec.method} scenarios", out_dir=out_dir)
        fig_corr = plot_corr_heatmap(corr_matrix(X),labels=list(rets.columns),title=f"{spec.method}", out_dir=out_dir)

        fig_hist.show()
        fig_corr.show()
    # Also log historical summary for comparison
    hist_summ = summarize_matrix(rets.to_numpy())
    with open(out_dir / "summary_historical_data.json", "w", encoding="utf-8") as f:
        json.dump(hist_summ, f, indent=2)

    logger.info("Saved Step 2 outputs to %s", out_dir.resolve())
    logger.info("Done.")

if __name__ == "__main__":
    main()