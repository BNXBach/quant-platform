from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json
import plotly.graph_objects as go
from backend.quant.phase1_cvar.logging_utils import setup_logger

def compute_log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff().dropna() # type: ignore

def resample_prices(prices: pd.DataFrame | pd.Series, frequency: str) -> pd.DataFrame | pd.Series:
    return prices.resample(frequency).last() # type: ignore

def load_prices_yfinance(tickers: list[str], start: str, end: str | None, renew: bool = False) -> pd.Series:
    import yfinance as yf
    if renew:
        df = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
        )
        df.to_parquet("backend/data/raw_yf.parquet") # type: ignore
    else:
        df = pd.read_parquet("backend/data/raw_yf.parquet", engine = "pyarrow")

    if df is None or df.empty:
        raise ValueError("No data downloaded from yfinance.")
    if isinstance(df.columns, pd.MultiIndex):
        price_types = df.columns.get_level_values(0)
        
        if "Adj Close" in price_types:
            prices = df["Adj Close"].copy()
        else:
            prices = df["Close"].copy()

    prices.index = pd.to_datetime(prices.index)
    return prices

def save_data(prices: pd.DataFrame | pd.Series, rets: pd.DataFrame | pd.Series, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_dir/"prices.parquet")
    rets.to_parquet(out_dir/"returns.parquet")

    prices.to_csv(out_dir/"prices.csv")
    rets.to_csv(out_dir/"returns.csv")

def plot_log_returns(prices: pd.DataFrame | pd.Series, rets: pd.DataFrame | pd.Series) -> tuple:
    fig_price = go.Figure()
    fig_rets = go.Figure()

    for ticker in rets.columns:
        fig_price.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices[ticker],
                mode="lines",
                name=ticker
            )
        )
        fig_rets.add_trace(
            go.Scatter(
                x=rets.index,
                y=rets[ticker],
                mode="lines",
                name=ticker
            )
        )

    fig_price.update_layout(
        title="Price (Interactive)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        xaxis=dict(
            hoverformat="%b-%d-%Y"
        )
    )
    fig_rets.update_layout(
        title="Log Returns (Interactive)",
        xaxis_title="Date",
        yaxis_title="Log Return",
        hovermode="x unified",
        xaxis=dict(
            hoverformat="%b-%d-%Y"
        )
    )

    return (fig_price,fig_rets)

def main():
    from backend.quant.phase1_cvar.config import p1_config

    logger = setup_logger(
        name="phase1",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        log_file="backend/reports/phase1/logs/phase1_data.log",
    )

    logger.info("Loading data with P1Config:")
    logger.info("Tickers: %s", p1_config.tickers)
    logger.info("Start=%s End=%s Freq=%s", p1_config.start, p1_config.end, p1_config.frequency)

    raw_prices = load_prices_yfinance(
        tickers=p1_config.tickers,
        start=p1_config.start,
        end=p1_config.end,
        renew=p1_config.renew_data,
    )
    logger.info("Downloaded prices: shape=%s", raw_prices.shape)
    logger.info("First raw price date: %s", raw_prices.index.min().date())
    logger.info("Last raw price date: %s", raw_prices.index.max().date())

    prices = resample_prices(raw_prices, p1_config.frequency)
    logger.info("Resampled prices: shape=%s", prices.shape)
    logger.info("First resampled price date: %s", prices.index.min().date())
    logger.info("Last resampled price date: %s", prices.index.max().date())
    # prices = clean_prices(prices)
    # logger.info("Cleaned prices: shape=%s", prices.shape)

    logger.info("Final tickers kept (%d): %s", prices.shape[1], list(prices.columns))

    rets = compute_log_returns(prices)
    logger.info("Returns: shape=%s, date_range=%s -> %s", rets.shape, rets.index.min().date(), rets.index.max().date())
    logger.info("Last return date: %s", rets.index.max().date())
    logger.info("Returns summary: mean=%s", rets.mean().round(6).to_dict())
    logger.info("Returns summary: vol=%s", rets.std().round(6).to_dict())

    save_data(prices, rets, p1_config.data_path)
    logger.info("Saved data to %s", p1_config.data_path.resolve())

    fig_price, fig_rets = plot_log_returns(prices, rets)
    fig_price.write_html(Path("backend/reports/phase1/data/price.html"))  
    fig_rets.write_html(Path("backend/reports/phase1/data/log_returns.html"))
    logger.info("Saved interactive plots to %s", Path("backend/reports/phase1/data/").resolve())
    
    
    rets = compute_log_returns(prices)
    save_data(prices, rets, p1_config.data_path)

    # Save metadata for future reference
    meta = {
        "tickers": list(prices.columns),
        "start": p1_config.start,
        "end": p1_config.end,
        "freq": p1_config.frequency,
        "n_prices": int(prices.shape[0]),
        "n_returns": int(rets.shape[0]),
        "last_raw_price_date": str(raw_prices.index.max().date()),
        "last_resampled_price_date": str(prices.index.max().date()),
        "last_return_date": str(rets.index.max().date()),
    }

    (p1_config.data_path).mkdir(parents=True, exist_ok=True)
    with open(p1_config.data_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fig_price.show()
    fig_rets.show()
if __name__ == "__main__":
    main()

