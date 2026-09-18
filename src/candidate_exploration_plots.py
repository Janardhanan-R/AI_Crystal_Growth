"""
candidate_exploration_plots.py
================================
Plots for the candidate-exploration module. Kept in a separate file so
`candidate_exploration.py` can be imported (e.g. for tests) without
importing matplotlib. Reuses the same color palette / style conventions as
src/evaluation.py so figures look consistent across the project.

Every plot here is a SCREENING visualization over surrogate predictions on
VIRTUAL candidates -- none of it implies CrysMAS or experimental
validation. Candidates with a support warning (support_category !=
"in_range") are always visually distinguished (marker shape/alpha), never
silently plotted identically to fully-supported candidates.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SUPPORTED_COLOR = "#4a3aa7"
UNSUPPORTED_COLOR = "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.size": 9,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _supported_mask(df: pd.DataFrame) -> pd.Series:
    return df["support_category"] == "in_range"


def plot_D_distribution(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    supported = _supported_mask(df)
    ax.hist(df.loc[supported, "pred_D"], bins=30, alpha=0.75, color=SUPPORTED_COLOR,
            label=f"Supported (n={int(supported.sum())})")
    if (~supported).any():
        ax.hist(df.loc[~supported, "pred_D"], bins=30, alpha=0.55, color=UNSUPPORTED_COLOR,
                label=f"Support warning (n={int((~supported).sum())})")
    ax.axvline(0, color=INK, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Predicted D (m) -- interface deflection, NOT diameter")
    ax.set_ylabel("Candidate count")
    ax.set_title("Distribution of predicted D across virtual candidates\n"
                  "(surrogate screening only -- not CrysMAS/experimental output)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_D_distribution.png", dpi=150)
    plt.close(fig)


def plot_absD_vs_uncertainty(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    supported = _supported_mask(df)
    ax.scatter(df.loc[supported, "abs_D"], df.loc[supported, "aggregate_uncertainty"],
               s=14, alpha=0.5, color=SUPPORTED_COLOR, label="Supported", edgecolor="none")
    if (~supported).any():
        ax.scatter(df.loc[~supported, "abs_D"], df.loc[~supported, "aggregate_uncertainty"],
                   s=16, alpha=0.6, color=UNSUPPORTED_COLOR, marker="^",
                   label="Support warning", edgecolor="none")
    if "high_uncertainty_flag" in df.columns and df["high_uncertainty_flag"].any():
        flagged = df["high_uncertainty_flag"].fillna(False)
        ax.scatter(df.loc[flagged, "abs_D"], df.loc[flagged, "aggregate_uncertainty"],
                   s=40, facecolors="none", edgecolors=INK, linewidths=0.8,
                   label="High-uncertainty flag (relative screening threshold)")
    ax.set_xlabel("|Predicted D| (m)")
    ax.set_ylabel("Aggregate uncertainty (normalized within this candidate set, 0-1)")
    ax.set_title("Predicted |D| vs. surrogate uncertainty\n"
                  "GPR std is a model-internal signal, not a calibrated physical confidence")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / "absD_vs_uncertainty.png", dpi=150)
    plt.close(fig)


def plot_top_candidates(df: pd.DataFrame, out_dir: Path, by: str = "abs_D", n: int = 20):
    from .candidate_exploration import rank_candidates
    ranked = rank_candidates(df, by=by, supported_only=False).head(n)
    if ranked.empty:
        return
    col = {"abs_D": "abs_D", "uncertainty": "aggregate_uncertainty", "screening_score": "screening_score"}[by]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(ranked))))
    colors = [SUPPORTED_COLOR if s == "in_range" else UNSUPPORTED_COLOR for s in ranked["support_category"]]
    y_pos = np.arange(len(ranked))
    ax.barh(y_pos, ranked[col], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{int(cid)}" for cid in ranked["candidate_id"]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(col)
    ax.set_title(f"Top {len(ranked)} candidates ranked by {by}\n"
                 "(purple = fully supported, pink = has a support warning -- see legend note)")
    fig.tight_layout()
    fig.savefig(out_dir / f"top_candidates_by_{by}.png", dpi=150)
    plt.close(fig)


def plot_temperature_pairwise(df: pd.DataFrame, out_dir: Path, sample_n: int = 400, seed: int = 42):
    cols = ["T1", "T2", "T3", "T4", "T5"]
    sample = df.sample(n=min(sample_n, len(df)), random_state=seed)
    supported = _supported_mask(sample)

    fig, axes = plt.subplots(len(cols), len(cols), figsize=(11, 11))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(sample[ci], bins=15, color=SUPPORTED_COLOR, alpha=0.7)
            else:
                ax.scatter(sample.loc[supported, cj], sample.loc[supported, ci],
                           s=5, alpha=0.35, color=SUPPORTED_COLOR, edgecolor="none")
                if (~supported).any():
                    ax.scatter(sample.loc[~supported, cj], sample.loc[~supported, ci],
                               s=6, alpha=0.45, color=UNSUPPORTED_COLOR, marker="^", edgecolor="none")
            if i == len(cols) - 1:
                ax.set_xlabel(cj, fontsize=8)
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Candidate heater-temperature pairwise relationships (sample of {len(sample)})")
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_pairwise.png", dpi=150)
    plt.close(fig)


def make_all_plots(df: pd.DataFrame, plots_dir: str):
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_D_distribution(df, out_dir)
    plot_absD_vs_uncertainty(df, out_dir)
    plot_top_candidates(df, out_dir, by="abs_D", n=20)
    if not df["screening_score"].isna().all():
        plot_top_candidates(df, out_dir, by="screening_score", n=20)
    plot_temperature_pairwise(df, out_dir)
