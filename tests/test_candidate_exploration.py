"""
tests/test_candidate_exploration.py
=====================================
Tests for src/candidate_exploration.py.

Written as unittest.TestCase classes deliberately: this sandbox's system
Python has pandas/scikit-learn/scipy but no network access to install
pytest (same restriction documented elsewhere in this project for
xgboost/shap), so these tests must run with the stdlib `unittest` module
here. unittest.TestCase classes are also natively discovered and run by
pytest with no changes, so this works identically in an environment (e.g.
your local venv) that does have pytest installed.

Run:
    python3 -m unittest discover tests -v      # works anywhere (stdlib only)
    pytest tests/test_candidate_exploration.py -v   # also works, if pytest is installed

Uses the project's real, already-trained surrogate
(models/fast_surrogate.joblib) rather than a mock, since the whole point of
this module is to evaluate candidates with that exact surrogate. If it
isn't present, the whole module is skipped (via unittest.skipUnless) with a
clear reason, not silently passed -- run `python3 main.py` first.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)  # every relative path in candidate_exploration.py (e.g. "data/Table.csv")
                      # is relative to the repo root, matching how main.py is run

from src.surrogate import FastSurrogate
from src.candidate_exploration import (
    ExplorationConfig,
    RAW_ORDER,
    SUPPORT_IN_RANGE,
    check_candidate_support,
    compute_objectives,
    evaluate_candidates,
    export_candidates,
    filter_candidates,
    generate_candidates,
    rank_candidates,
    run_candidate_exploration,
    top_n_views,
)

SURROGATE_PATH = REPO_ROOT / "models" / "fast_surrogate.joblib"
SURROGATE_AVAILABLE = SURROGATE_PATH.exists()


@unittest.skipUnless(SURROGATE_AVAILABLE, f"{SURROGATE_PATH} not found -- run `python3 main.py` first")
class CandidateExplorationTestCase(unittest.TestCase):
    """Base class: loads the real surrogate once and builds one shared,
    fully-evaluated candidate set for tests that don't need a fresh one."""

    @classmethod
    def setUpClass(cls):
        cls.surrogate = FastSurrogate.load(str(SURROGATE_PATH))
        cls.small_config = ExplorationConfig(n_samples=200, random_state=42)
        df = generate_candidates(cls.surrogate, cls.small_config)
        df = evaluate_candidates(df, cls.surrogate)
        df = check_candidate_support(df, cls.surrogate)
        cls.evaluated = compute_objectives(df, cls.small_config)

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="candidate_exploration_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestGeneration(CandidateExplorationTestCase):

    def test_respects_bounds(self):
        df = generate_candidates(self.surrogate, self.small_config)
        for f in RAW_ORDER:
            lo, hi = self.surrogate.feature_ranges[f]
            self.assertTrue((df[f] >= lo - 1e-9).all(), f"{f} below training bound")
            self.assertTrue((df[f] <= hi + 1e-9).all(), f"{f} above training bound")

    def test_ordering_constraint(self):
        df = generate_candidates(self.surrogate, self.small_config)
        ordering = np.all(np.diff(df[RAW_ORDER].values, axis=1) <= 1e-9, axis=1)
        self.assertTrue(ordering.all(), "Some candidates violate T1>=T2>=T3>=T4>=T5")

    def test_no_duplicates(self):
        df = generate_candidates(self.surrogate, self.small_config)
        rounded = df[RAW_ORDER].round(self.small_config.dedupe_decimals)
        self.assertFalse(rounded.duplicated().any(), "Duplicate candidates found")

    def test_requested_count_and_not_validated_flag(self):
        df = generate_candidates(self.surrogate, self.small_config)
        self.assertLessEqual(len(df), self.small_config.n_samples)
        self.assertEqual(df.attrs["n_generated"], len(df))
        self.assertFalse(df.attrs["is_physically_validated"])

    def test_reproducible_same_seed(self):
        df1 = generate_candidates(self.surrogate, self.small_config)
        df2 = generate_candidates(self.surrogate, ExplorationConfig(n_samples=self.small_config.n_samples, random_state=42))
        pd.testing.assert_frame_equal(
            df1[RAW_ORDER].reset_index(drop=True), df2[RAW_ORDER].reset_index(drop=True)
        )

    def test_different_seed_differs(self):
        df1 = generate_candidates(self.surrogate, self.small_config)
        df2 = generate_candidates(self.surrogate, ExplorationConfig(n_samples=self.small_config.n_samples, random_state=123))
        self.assertFalse(df1[RAW_ORDER].reset_index(drop=True).equals(df2[RAW_ORDER].reset_index(drop=True)))

    def test_lhs_strategy(self):
        cfg = ExplorationConfig(n_samples=100, random_state=7, sampling_strategy="lhs")
        df = generate_candidates(self.surrogate, cfg)
        self.assertGreater(len(df), 0)
        ordering = np.all(np.diff(df[RAW_ORDER].values, axis=1) <= 1e-9, axis=1)
        self.assertTrue(ordering.all())

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            generate_candidates(self.surrogate, ExplorationConfig(sampling_strategy="not_a_strategy"))

    def test_invalid_n_samples_raises(self):
        with self.assertRaises(ValueError):
            generate_candidates(self.surrogate, ExplorationConfig(n_samples=0))

    def test_insufficient_valid_candidates_handled_gracefully(self):
        """A near-infeasible, extremely narrow bounds box should return
        fewer rows than requested, with the shortfall recorded -- not
        raise, hang, or silently pad with duplicates."""
        narrow_bounds = {
            "T1": (1200.0, 1200.001), "T2": (1200.0, 1200.001), "T3": (1200.0, 1200.001),
            "T4": (1200.0, 1200.001), "T5": (1200.0, 1200.001),
        }
        cfg = ExplorationConfig(n_samples=50, random_state=1, max_generation_attempts_multiplier=3)
        df = generate_candidates(self.surrogate, cfg, bounds=narrow_bounds)
        self.assertLessEqual(len(df), 50)
        self.assertIn("shortfall", df.attrs)


class TestEvaluation(CandidateExplorationTestCase):

    def test_feature_engineering_consistency(self):
        from src.feature_engineering import add_engineered_features
        df = generate_candidates(self.surrogate, self.small_config)
        evaluated = evaluate_candidates(df, self.surrogate)
        expected = add_engineered_features(df[RAW_ORDER])
        for col in ["dT_12", "dT_23", "dT_34", "dT_45", "dT_15", "T_mean"]:
            np.testing.assert_allclose(evaluated[col].values, expected[col].values)

    def test_prediction_columns_present_and_populated(self):
        for t in ["Lc", "Le", "D"]:
            self.assertIn(f"pred_{t}", self.evaluated.columns)
            self.assertIn(f"unc_{t}", self.evaluated.columns)
            self.assertTrue(self.evaluated[f"pred_{t}"].notna().all())

    def test_D_is_independent_model_prediction(self):
        """D is a directly-modeled surrogate target (README Section 0), not
        algebraically derived here -- pred_D should not exactly equal
        pred_Lc - pred_Le for every row (unlike the original simulation
        data, where the identity is exact by construction of CrysMAS)."""
        diff = (self.evaluated["pred_D"] - (self.evaluated["pred_Lc"] - self.evaluated["pred_Le"])).abs()
        self.assertGreater(diff.max(), 0)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            evaluate_candidates(pd.DataFrame(columns=["candidate_id"] + RAW_ORDER), self.surrogate)


class TestSupportChecks(CandidateExplorationTestCase):

    def test_support_categories_are_valid_labels(self):
        valid_labels = {"in_range", "outside_range", "outside_convex_hull", "invalid_constraint"}
        self.assertLessEqual(set(self.evaluated["support_category"].unique()), valid_labels)

    def test_flags_out_of_range_candidate(self):
        df = generate_candidates(self.surrogate, self.small_config)
        evaluated = evaluate_candidates(df, self.surrogate)
        evaluated.loc[0, "T1"] = self.surrogate.feature_ranges["T1"][1] + 1000.0
        checked = check_candidate_support(evaluated, self.surrogate)
        self.assertFalse(checked.loc[0, "all_in_range"])
        self.assertIn(checked.loc[0, "support_category"], ("outside_range", "invalid_constraint"))

    def test_flags_ordering_violation(self):
        df = generate_candidates(self.surrogate, self.small_config)
        evaluated = evaluate_candidates(df, self.surrogate)
        evaluated.loc[0, ["T1", "T2", "T3", "T4", "T5"]] = [1035.0, 1055.0, 1195.0, 1255.0, 1275.0]
        checked = check_candidate_support(evaluated, self.surrogate)
        self.assertFalse(checked.loc[0, "ordering_satisfied"])
        self.assertEqual(checked.loc[0, "support_category"], "invalid_constraint")

    def test_convex_hull_screening_available_on_real_dataset(self):
        """Verified to succeed on this project's actual 96-row dataset
        (see module docstring) -- confirm it actually ran."""
        checked = check_candidate_support(self.evaluated, self.surrogate)
        self.assertTrue(checked.attrs.get("convex_hull_screening_available"))
        self.assertTrue(checked["convex_hull_checked"].all())
        self.assertTrue(checked["inside_convex_hull"].isin([True, False]).all())


class TestObjectivesAndUncertainty(CandidateExplorationTestCase):

    def test_objective_columns_present(self):
        for col in ["abs_D", "aggregate_uncertainty", "support_penalty", "high_uncertainty_flag"]:
            self.assertIn(col, self.evaluated.columns)
        self.assertTrue((self.evaluated["abs_D"] >= 0).all())

    def test_uncertainty_flag_is_percentile_based(self):
        cfg = ExplorationConfig(n_samples=self.small_config.n_samples, random_state=42, uncertainty_flag_percentile=90.0)
        df = generate_candidates(self.surrogate, cfg)
        df = evaluate_candidates(df, self.surrogate)
        df = check_candidate_support(df, self.surrogate)
        df = compute_objectives(df, cfg)
        n_flagged = df["high_uncertainty_flag"].sum()
        self.assertGreater(n_flagged, 0)
        self.assertLessEqual(n_flagged, max(1, int(0.20 * len(df))))

    def test_multi_objective_weights_change_ranking(self):
        cfg_a = ExplorationConfig(n_samples=self.small_config.n_samples, random_state=42, objective="multi_objective",
                                   multi_objective_weights={"abs_D": 1.0, "uncertainty": 0.0, "support_penalty": 0.0})
        cfg_b = ExplorationConfig(n_samples=self.small_config.n_samples, random_state=42, objective="multi_objective",
                                   multi_objective_weights={"abs_D": 0.0, "uncertainty": 1.0, "support_penalty": 0.0})
        df = generate_candidates(self.surrogate, cfg_a)
        df = evaluate_candidates(df, self.surrogate)
        df = check_candidate_support(df, self.surrogate)
        df_a = compute_objectives(df, cfg_a)
        df_b = compute_objectives(df, cfg_b)
        self.assertFalse(df_a["screening_score"].equals(df_b["screening_score"]))


class TestFilterAndRank(CandidateExplorationTestCase):

    def test_filter_supported_only(self):
        filtered = filter_candidates(self.evaluated, supported_only=True)
        self.assertTrue((filtered["support_category"] == SUPPORT_IN_RANGE).all())

    def test_rank_ascending_and_rank_field(self):
        ranked = rank_candidates(self.evaluated, by="abs_D")
        self.assertTrue(ranked["abs_D"].is_monotonic_increasing)
        self.assertEqual(ranked["rank_abs_D"].tolist(), list(range(1, len(ranked) + 1)))

    def test_rank_invalid_by_raises(self):
        with self.assertRaises(ValueError):
            rank_candidates(self.evaluated, by="not_a_real_column")

    def test_top_n_views_not_padded(self):
        tiny = pd.DataFrame({
            "candidate_id": [0, 1, 2],
            "T1": [1270, 1265, 1260], "T2": [1240, 1235, 1230], "T3": [1210, 1215, 1220],
            "T4": [1080, 1075, 1070], "T5": [1040, 1045, 1050],
            "pred_Lc": [-0.21, -0.212, -0.214], "pred_Le": [-0.211, -0.213, -0.215], "pred_D": [0.001, 0.001, 0.001],
            "unc_Lc": [0.0004, 0.0004, 0.0004], "unc_Le": [0.0004, 0.0004, 0.0004], "unc_D": [0.0006, 0.0006, 0.0006],
            "abs_D": [0.001, 0.001, 0.001], "aggregate_uncertainty": [0.1, 0.2, 0.3],
            "support_category": ["in_range", "in_range", "outside_range"],
            "screening_score": [np.nan, np.nan, np.nan],
        })
        views = top_n_views(tiny, n_values=(5, 10, 20))
        self.assertEqual(len(views["abs_D__all_candidates__top5"]), 3)


class TestExport(CandidateExplorationTestCase):

    def test_writes_three_files_with_expected_content(self):
        paths = export_candidates(self.evaluated, output_dir=self.tmp_dir)
        self.assertTrue(Path(paths["all_csv"]).exists())
        self.assertTrue(Path(paths["top10_csv"]).exists())
        self.assertTrue(Path(paths["summary_json"]).exists())

        all_df = pd.read_csv(paths["all_csv"])
        self.assertEqual(len(all_df), len(self.evaluated))
        top10_df = pd.read_csv(paths["top10_csv"])
        self.assertEqual(len(top10_df), min(10, len(self.evaluated)))
        self.assertTrue(top10_df["abs_D"].is_monotonic_increasing)

        with open(paths["summary_json"]) as f:
            summary = json.load(f)
        self.assertFalse(summary["is_physically_validated"])
        self.assertIn("support_category_counts", summary)

    def test_does_not_overwrite_optimization_results(self):
        sentinel = Path(self.tmp_dir) / "optimization_demo.csv"
        sentinel.write_text("untouched_marker\n1\n")
        export_candidates(self.evaluated, output_dir=self.tmp_dir)
        self.assertEqual(sentinel.read_text(), "untouched_marker\n1\n")


class TestEndToEnd(CandidateExplorationTestCase):

    def test_run_candidate_exploration_end_to_end(self):
        cfg = ExplorationConfig(n_samples=150, random_state=42, objective="multi_objective")
        results_dir = Path(self.tmp_dir) / "results"
        plots_dir = Path(self.tmp_dir) / "plots"
        df = run_candidate_exploration(self.surrogate, cfg, output_dir=str(results_dir), plots_dir=str(plots_dir))
        self.assertGreater(len(df), 0)
        self.assertTrue((results_dir / "candidate_exploration_all.csv").exists())
        self.assertTrue((results_dir / "candidate_exploration_top10.csv").exists())
        self.assertTrue((results_dir / "candidate_exploration_summary.json").exists())
        self.assertTrue((plots_dir / "predicted_D_distribution.png").exists())
        self.assertTrue((plots_dir / "absD_vs_uncertainty.png").exists())

    def test_does_not_mutate_surrogate(self):
        before = {t: pipe.named_steps["model"] for t, pipe in self.surrogate.models.items()}
        run_candidate_exploration(self.surrogate, ExplorationConfig(n_samples=50),
                                   output_dir=str(Path(self.tmp_dir) / "r2"), plots_dir=None)
        after = {t: pipe.named_steps["model"] for t, pipe in self.surrogate.models.items()}
        for t in before:
            self.assertIs(before[t], after[t], f"Surrogate model for {t} was replaced/refit during exploration")


if __name__ == "__main__":
    unittest.main()
