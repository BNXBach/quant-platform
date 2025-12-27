from dataclasses import dataclass
from pathlib import Path

@dataclass
class P1Config:
    tickers: list[str]
    data_path: Path = Path("backend/data")
    start: str = "2005-01-01"
    end: str| None = "2025-11-30"
    frequency: str = "W-FRI"
    renew_data: bool = True
    alpha: float = 0.95
    w_max: float = 0.35
    l2_tau: float = 0.5
    train_weeks: int = 156          # 3 years weekly
    n_scenarios: int = 5000
    mbb_block_len: int = 5

p1_config = P1Config(
    #tickers=["SPY", "QQQ", "IWM", "EFA", "EEM"]
    #tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "UNH", "XOM", "PG", "KO"]
    tickers = ["SPY", "QQQ", "HYG", "EMB", "XLE", "EEM", "TLT", "IEF", "GLD"]
)