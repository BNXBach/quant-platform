from dataclasses import dataclass
from pathlib import Path

@dataclass
class P1Config:
    tickers: list[str]
    data_path: Path = Path("backend/data")
    start: str = "2020-01-01"
    end: str| None = None
    frequency: str = "W-FRI"
    alpha: float = 0.95
    w_max: float = 0.35
    l2_tau: float = 0.5

p1_config = P1Config(
    tickers=["SPY", "QQQ", "IWM", "EFA", "EEM"]
)