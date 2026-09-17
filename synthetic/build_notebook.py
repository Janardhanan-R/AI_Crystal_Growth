"""
build_notebook.py
==================
Hand-builds 10_data_validation_comparison.ipynb (nbformat v4 JSON) directly,
without depending on the `nbformat` package (not installable in this
sandboxed environment -- no PyPI network access). The notebook is a
runnable, self-contained walkthrough of the validation comparison between
original and synthetic data, reading the CSV/JSON outputs already produced
by 01_*.py through 06_*.py rather than recomputing everything from scratch
(so it opens and runs in seconds).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "10_data_validation_comparison.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


cells = [
    md("# Data Validation & Comparison Notebook\n\n"
       "Original (`data/Table.csv`, n=96) vs. synthetic "
       "(`synthetic/synthetic_crystal_growth_dataset.csv`, n=2000) data.\n\n"
       "This notebook reads the outputs already produced by "
       "`01_dataset_analysis.py` .. `06_ml_evaluation.py` -- it does not "
       "regenerate data or refit models, so it runs in seconds. Re-run those "
       "scripts first (see `08_REPRODUCIBILITY_INSTRUCTIONS.md`) if you want "
       "fresh numbers.\n\n"
       "**Statistical similarity shown here is not evidence of physical "
       "validity** -- see `05_VALIDATION_REPORT.md` and "
       "`07_ASSUMPTIONS_AND_LIMITATIONS.md` for the full, honest scope of "
       "these results."),

    code("import json\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\n"
         "from IPython.display import Image, display\n\n"
         "ROOT = Path('..').resolve() if Path('data').exists() is False else Path('.').resolve()\n"
         "SYN = Path('.').resolve()\n"
         "original = pd.read_csv(SYN.parent / 'data' / 'Table.csv')\n"
         "synthetic = pd.read_csv(SYN / 'synthetic_crystal_growth_dataset.csv')\n"
         "print('original:', original.shape, ' synthetic:', synthetic.shape)"),

    md("## 1. Original dataset analysis (recap)\n\nFull output: `results/dataset_analysis.json`."),
    code("with open('results/dataset_analysis.json') as f:\n"
         "    analysis = json.load(f)\n"
         "print('Shape:', analysis['shape'])\n"
         "print('Duplicate rows:', analysis['duplicate_rows'])\n"
         "print('Ordering constraint violations:', analysis['ordering_constraint_T1_to_T5_monotone_nonincreasing']['n_violations'])\n"
         "print('Derived-variable identities:', analysis['derived_variable_identities'])"),

    md("## 2. Kolmogorov–Smirnov and Wasserstein-distance comparison\n\n"
       "Full output: `results/ks_wasserstein_tests.csv`."),
    code("ks_df = pd.read_csv('results/ks_wasserstein_tests.csv')\nks_df"),

    md("## 3. Correlation-structure comparison\n\n"
       "Full output: `results/correlation_comparison.csv`. Largest divergences shown first "
       "(all involve `D (m)` -- see `05_VALIDATION_REPORT.md` §2 for discussion)."),
    code("corr_df = pd.read_csv('results/correlation_comparison.csv')\ncorr_df.head(10)"),
    code("display(Image(filename='plots/correlation_comparison.png'))"),

    md("## 4. Distribution comparison (example: Lc and D)\n\n"
       "All columns are plotted individually in `plots/distribution_comparison_<col>.png`."),
    code("display(Image(filename='plots/distribution_comparison_Lc_m.png'))"),
    code("display(Image(filename='plots/distribution_comparison_D_m.png'))"),

    md("## 5. Pair plot (Lc, Le, D)"),
    code("display(Image(filename='plots/pairplot_comparison.png'))"),

    md("## 6. Duplicate / near-duplicate and outlier check\n\n"
       "Recomputed directly here (cheap) rather than re-reading a saved report, "
       "for a self-contained sanity check."),
    code("input_cols = ['T1 (K)', 'T2 (K)', 'T3 (K)', 'T4 (K)', 'T5 (K)']\n"
         "exact_dup = synthetic[input_cols].duplicated().sum()\n"
         "print('Exact duplicate synthetic input rows:', exact_dup)\n\n"
         "ref = original[input_cols].values\n"
         "ref_std = ref.std(axis=0); ref_std[ref_std == 0] = 1.0\n"
         "syn_scaled = synthetic[input_cols].values / ref_std\n"
         "ref_scaled = ref / ref_std\n"
         "min_dists = np.array([np.sqrt(((ref_scaled - row) ** 2).sum(axis=1)).min() for row in syn_scaled])\n"
         "print('Synthetic rows within 0.05 normalized distance of an original row:', int((min_dists < 0.05).sum()))"),

    md("## 7. Predictive-relationship (linear coefficient sign) comparison\n\n"
       "Full output: `results/predictive_relationship_comparison.csv`."),
    code("pred_df = pd.read_csv('results/predictive_relationship_comparison.csv')\n"
         "print(f\"Sign agreement: {pred_df['same_sign'].sum()}/{len(pred_df)}\")\n"
         "pred_df"),

    md("## 8. ML evaluation: original-only vs. synthetic-augmented (CV summary)\n\n"
       "Full outputs: `results/cv_results_summary.csv`, `results/final_test_results.csv`. "
       "See `06_MODEL_EVALUATION_REPORT.md` for the full narrative -- headline finding: "
       "augmentation does NOT improve the single best model (GaussianProcess) on the "
       "untouched original test set, though it helps weaker models (e.g. RandomForest) "
       "and can hurt others (SVR) substantially."),
    code("cv_summary = pd.read_csv('results/cv_results_summary.csv')\n"
         "cv_summary.sort_values(['target', 'regime', 'R2_mean'], ascending=[True, True, False])"),
    code("final_df = pd.read_csv('results/final_test_results.csv')\n"
         "final_df.sort_values(['target', 'regime', 'R2'], ascending=[True, True, False])"),

    md("## 9. GPR prediction-interval coverage (uncertainty-aware model)\n\n"
       "Full output: `results/gpr_prediction_intervals.csv`."),
    code("pd.read_csv('results/gpr_prediction_intervals.csv')"),

    md("## Summary\n\n"
       "This notebook reproduces, interactively, the same validation and evaluation results "
       "narrated in `05_VALIDATION_REPORT.md` and `06_MODEL_EVALUATION_REPORT.md`. See "
       "`07_ASSUMPTIONS_AND_LIMITATIONS.md` for what these results do and do not establish."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT}")
