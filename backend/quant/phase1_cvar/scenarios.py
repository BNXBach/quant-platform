from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

@dataclass
class ScenarioSpec:
    method: str
    n_scenarios: int = 2000
    seed: int = 1234
    block_len: int = 5

def historical_sample(rets: pd.DataFrame | pd.Series, spec: ScenarioSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    T = rets.shape[0]
    idx = rng.integers(0, T, size=spec.n_scenarios)
    print(idx)
    return rets.iloc[idx].to_numpy()

def moving_block_bootstrap(rets: pd.DataFrame | pd.Series, spec: ScenarioSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    T,N = rets.shape
    L = spec.block_len
    n_blocks = int(np.ceil(spec.n_scenarios / L))
    starts = rng.integers(0, T - L + 1, size=n_blocks)
    blocks = [rets.iloc[start:start+L].to_numpy() for start in starts]
    boot = np.vstack(blocks)[:spec.n_scenarios]
    return boot

def generate_scenarios(rets: pd.DataFrame | pd.Series, spec: ScenarioSpec) -> np.ndarray:
    if spec.method == "historical":
        return historical_sample(rets, spec)
    elif spec.method == "mbb":
        return moving_block_bootstrap(rets, spec)
    raise ValueError(f"Unknown scenario generation method: {spec.method}")

def summarize_matrix(X: np.ndarray) -> dict:
    return {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "mean": np.mean(X, axis=0).tolist(),
        "std": np.std(X, axis=0, ddof=1 ).tolist(),
    }

def corr_matrix(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def plot_hist_vs_scen(    rets: pd.DataFrame, scen: np.ndarray, title: str, max_assets: int = 5, bins: int = 100, out_dir: Path = None) -> go.Figure:
    cols = list(rets.columns)[:max_assets]
    n = len(cols)

    fig = make_subplots(
        rows=int(max_assets/3)+1,
        cols=3,
        subplot_titles=[str(c) for c in cols],
        shared_yaxes=True
    )

    for j, c in enumerate(cols):
        hist_data = rets[c].to_numpy()
        scen_data = scen[:, j]

        fig.add_trace(
            go.Histogram(
                x=hist_data,
                nbinsx=bins,
                histnorm="probability density",
                opacity=0.45,
                name=f"hist {c}",
                showlegend=True,
            ),
            row=int(j/3)+1, col=(j%3)+1
        )

        fig.add_trace(
            go.Histogram(
                x=scen_data,
                nbinsx=bins,
                histnorm="probability density",
                opacity=0.45,
                name=f"scen {c}",
                showlegend=True,
            ),
            row=int(j/3)+1, col=(j%3)+1
        )

        fig.update_xaxes(title_text="Return", row=int(j/3)+1, col=(j%3)+1)

    fig.update_layout(
        title=title,
        barmode="overlay",
        bargap=0.05,
        hovermode="closest",
        legend_title_text="Series",
    )
    for i in range(1, int(max_assets/3)+2):
        fig.update_yaxes(title_text="Density", row=i, col=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_dir / f"data_vs_scen_{title.replace(' ','_')}.html"))
    return fig

def plot_corr_heatmap(mat: np.ndarray, labels: list[str], title: str, zmin: float | None = -1.0, zmax: float | None = 1.0, out_dir: Path = None) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Corr"),
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.5f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_dir / f"{title.replace(' ','_')}_correlation_heatmap.html"))
    return fig