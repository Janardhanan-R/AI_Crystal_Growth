# experiments/

Ad-hoc investigations that sit outside the main `python3 main.py` pipeline
-- things tried to answer a specific question, with honestly-reported
results (including negative ones), not folded into `src/` unless they
actually worked.

| File | What it is |
|---|---|
| `IMPROVE_D_REPORT.md` | **Start here.** Can the D (interface deflection) metric be improved? Answer: no, not meaningfully, and why. |
| `01_improve_D_metrics.py` | Script: 9 real variants (alternate GPR kernels, feature subsets, interaction terms, stacking) run through the project's own CV code. |
| `01_RESULTS_improve_D.csv` | Output of the above. |
| `02_residual_diagnostic.py` | Script: identifies which specific rows drive D's error (answers "why can't we improve it"). |

Run either script from the project root once `requirements.txt` is
installed, e.g.: `python3 experiments/01_improve_D_metrics.py`.
