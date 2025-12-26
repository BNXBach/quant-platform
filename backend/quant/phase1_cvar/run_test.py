from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.quant.phase1_cvar.logging_utils import setup_logger
from backend.quant.phase1_cvar.config import p1_config
from backend.quant.phase1_cvar.cvar import cvar_portfolio
from backend.quant.phase1_cvar.markowitz import markowitz_portfolio

from backend.quant.phase1_cvar.backtest_utils import turnover, realized_portfolio_return, equal_weight, max_drawdown, realized_var_cvar


from backend.quant.phase1_cvar.scenarios import generate_scenarios, ScenarioSpec  


def summarize(df: pd.DataFrame, ptype: str) -> dict:
    ret = df[f"r_{ptype}"]
    ret = ret.dropna()
    mu = float(ret.mean())
    vol = float(ret.std())
    ann_ret = float(mu * 52)
    ann_vol = float(vol * np.sqrt(52))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else None
    avg_turnover = float(df[f"turnover_{ptype}"].mean())
    max_dd = max_drawdown(ret)
    var, cvar = realized_var_cvar(ret, alpha=p1_config.alpha)
    return {"mean_weekly": mu, "vol_weekly": vol, "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "avg_turnover": avg_turnover, "max_drawdown": max_dd, "var": var, "cvar": cvar}


def main():
    logger = setup_logger(
        name="phase1",
        level=logging.INFO,
        log_file="backend/reports/phase1/logs/phase1_backtest.log",
    )

    out_dir = Path("backend/reports/phase1/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    rets = pd.read_parquet("backend/data/returns.parquet")
    assets = list(rets.columns)
    N = len(assets)


    logger.info("Step4Config=%s", p1_config)
    logger.info("Assets=%s", assets)
    logger.info("Returns shape=%s date_range=%s→%s", rets.shape, rets.index.min().date(), rets.index.max().date())

    # Rebalance dates: we need train_weeks history AND a next week to evaluate
    start_idx = p1_config.train_weeks
    end_idx = len(rets) - 1  # because we look at t+1 realized

    ptypes = ["cvar_historical", "mv_historical", "cvar_mbb", "mv_mbb", "ew"]
    stypes = ["historical", "mbb"]

    prev_weights = dict.fromkeys(ptypes)
    weights = dict.fromkeys(ptypes)
    realized_rets = dict.fromkeys(ptypes)
    turnovers = dict.fromkeys(ptypes)
    scens = dict.fromkeys(stypes)
    

    rows = []
    weights_rows = []  # store weights time series

    for t in range(start_idx, end_idx):
        train = rets.iloc[t - p1_config.train_weeks : t]  # up to t-1 inclusive
        next_ret = rets.iloc[t].to_numpy()          # realized at t (next period)

        date = rets.index[t]
        logger.info("Rebalance date=%s train_window=[%s→%s] eval_next=%s",
                    train.index.max().date(), train.index.min().date(), train.index.max().date(), date.date())

        # Generate scenarios from TRAIN only
        for method in scens.keys():
            scens[method] = generate_scenarios(train, ScenarioSpec(method=method, n_scenarios=p1_config.n_scenarios, seed=t, block_len=p1_config.mbb_block_len))

        for method, scen in scens.items():
            cvar_res = json.loads(cvar_portfolio(train.to_numpy(), w_max=p1_config.w_max, l2_tau=p1_config.l2_tau,
                                                                scenarios=scen, alpha=p1_config.alpha))
            weights[f"cvar_{method}"] = cvar_res['weights']
            mu_star = float(cvar_res["expected_return"])
            weights[f"mv_{method}"] = json.loads(markowitz_portfolio(train.to_numpy(), target_return=mu_star, w_max=p1_config.w_max,
                                                                   l2_tau=p1_config.l2_tau, scenarios=scen, alpha=p1_config.alpha))['weights']
        weights["ew"] = equal_weight(N)


            
        unit_row = {"date": date,}
        unit_weight_row = {"date": date,}
        # Turnover + realized returns
        for portfolio_type in ptypes:
            realized_rets[portfolio_type] = realized_portfolio_return(next_ret, np.array(weights[portfolio_type]))
            turnovers[portfolio_type] = turnover(prev_weights[portfolio_type], np.array(weights[portfolio_type]))
            prev_weights[portfolio_type] = np.array(weights[portfolio_type])
            r_key = f"r_{portfolio_type}"
            to_key = f"turnover_{portfolio_type}"
            unit_row[r_key] = realized_rets[portfolio_type]
            unit_row[to_key] = turnovers[portfolio_type]
            unit_weight_row[portfolio_type] = weights[portfolio_type]
        rows.append(unit_row)
        weights_rows.append(unit_weight_row)




    df = pd.DataFrame(rows).set_index("date")
    df.to_csv(out_dir / "backtest_results.csv")

    wdf = pd.DataFrame(weights_rows).set_index("date")
    wdf.to_parquet(out_dir / "weights_timeseries.parquet")
    wdf.to_csv(out_dir / "weights_timeseries.csv")
    # Summary stats (log returns → annualization approx: 52 weeks)
    summary = {}
    for portfolio_type in ptypes:
        summary[portfolio_type] = summarize(df, portfolio_type)


    with open(out_dir / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved Step 4 outputs to %s", out_dir.resolve())


if __name__ == "__main__":
    main()