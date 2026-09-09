# -*- coding: utf-8 -*-
"""Build an isolated, common-fold comparison package for Section 4.

This is a retrospective CV5 robustness lane, not the registered primary split.
Every trained system is evaluated out-of-fold under ``data/splits/cv5_folds.csv``.
Zero-shot systems have no fitted fold state, so their one-pass predictions are
partitioned by the same fold labels (mathematically identical to scoring them
five times).  The paper-facing comparison is restricted to the original 3,468
official-test cases so every system and every 33-scenario response uses the
same cases.

Prediction and the main conventional response comparison follow the active
CASE generator's D250 weighted lane. ``Human CV5 H-MNL`` remains an unweighted
behavioural panel-logit trained on four folds and applied to the held-out fold;
it is not a copy of the observed labels. ``Panel Logit`` is the full-feature
predictive DCM, keeping the existing distinction between the behavioural
reference and predictive baseline. A tuned-base sensitivity is also exported;
the historical filename ``unweighted`` is not used as a claim because SVM's
tuning selected a balanced model in all five folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


TRC = Path(r"C:\SP_LLM\TRC")
REAL = TRC / "Real_exp"
DATA = TRC / "data"
OUT_LEGACY = REAL / "pdf" / "SECTION4_RESULTS_CV5_ALL_MODELS_20260907"
OUT_FULLGRID = REAL / "pdf" / "SECTION4_RESULTS_CV5_ALL_MODELS_FULLGRID_20260907"
OUT = OUT_LEGACY
FIG = OUT / "figures"

FOLDS = (1, 2, 3, 4, 5)
SCENARIOS = {
    "Fare": ["F_M3", "F_M2", "F_M1", "BASE", "F_P1", "F_P2", "F_P3"],
    "IVT": ["R_M3", "R_M2", "R_M1", "BASE", "R_P1", "R_P2", "R_P3"],
    "Wait": ["W_M3", "W_M2", "W_M1", "BASE", "W_P1", "W_P2", "W_P3"],
}
RESPONSE_SCENARIOS = {
    **SCENARIOS,
    "Reliability": ["A_M3", "A_M2", "A_M1", "BASE", "A_P1", "A_P2", "A_P3"],
}
XCOL = {"Fare": "fare_cf", "IVT": "ride_cf", "Wait": "wait_cf", "Reliability": "rel_cf"}
RAW_FARE_THRESHOLD = 1e-4

TABULAR_LABEL = {
    "panel_logit": "Human CV5 H-MNL",
    "panel_logit_full": "Panel Logit",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "ffnn": "FFNN",
    "dnn": "DNN",
}
PREDICTION_ROWS = [
    ("Human CV5 H-MNL", "unweighted"),
    ("Panel Logit", "weighted"),
    ("SVM", "weighted"),
    ("Random Forest", "weighted"),
    ("XGBoost", "weighted"),
    ("FFNN", "weighted"),
    ("DNN", "weighted"),
    ("Qwen 2B ZS", "zero_shot"),
    ("Qwen 2B SFT", "sft_aaai_composite"),
    ("Qwen 9B ZS", "zero_shot"),
    ("Qwen 9B SFT", "sft_aaai_composite"),
]
BEHAVIOUR_ORDER = [
    "Human CV5 H-MNL",
    "Panel Logit",
    "SVM",
    "Random Forest",
    "XGBoost",
    "FFNN",
    "DNN",
    "Qwen 2B ZS",
    "Qwen 2B SFT",
    "Qwen 9B ZS",
    "Qwen 9B SFT",
]
MAIN_BEHAVIOUR_LANE = {
    "Human CV5 H-MNL": "unweighted",
    "Panel Logit": "weighted",
    "SVM": "weighted",
    "Random Forest": "weighted",
    "XGBoost": "weighted",
    "FFNN": "weighted",
    "DNN": "weighted",
    "Qwen 2B ZS": "zero_shot",
    "Qwen 2B SFT": "sft_aaai_composite",
    "Qwen 9B ZS": "zero_shot",
    "Qwen 9B SFT": "sft_aaai_composite",
}
MODEL_STYLE = {
    "Human CV5 H-MNL": ("#202020", "o", "-"),
    "Panel Logit": ("#64748B", "s", "-"),
    "SVM": ("#A1A1AA", "^", "--"),
    "Random Forest": ("#71717A", "v", "--"),
    "XGBoost": ("#52525B", "D", "--"),
    "FFNN": ("#B8861B", "P", ":"),
    "DNN": ("#76539B", "X", ":"),
    "Qwen 2B ZS": ("#DB5A91", "o", "--"),
    "Qwen 2B SFT": ("#C93F7C", "o", "-"),
    "Qwen 9B ZS": ("#168C97", "s", "--"),
    "Qwen 9B SFT": ("#007F8B", "s", "-"),
}


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(DATA / "splits" / "cv5_folds.csv", dtype={"respondent_id": str})
    assert folds.respondent_id.is_unique and set(folds.fold) == set(FOLDS)
    parts = []
    for partition in ("development", "test"):
        d = pd.read_parquet(DATA / "model_inputs" / f"HUMAN_retro_{partition}.parquet")
        d["partition_source"] = partition
        parts.append(d)
    pool = pd.concat(parts, ignore_index=True)
    pool["case_id"] = pool.case_id.astype(str)
    pool["respondent_id"] = pool.respondent_id.astype(str)
    pool = pool.merge(folds, on="respondent_id", how="left", validate="many_to_one")
    assert len(pool) == 11584 and pool.case_id.is_unique and pool.fold.notna().all()
    test = pool.loc[pool.partition_source.eq("test")].copy()
    assert len(test) == 3468 and test.respondent_id.nunique() == 651
    grid = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")
    grid["case_id"] = grid.case_id.astype(str)
    assert grid.duplicated(["case_id", "scenario_id"]).sum() == 0
    assert grid.case_id.nunique() == 11584 and grid.scenario_id.nunique() == 33
    return pool, test, grid


def _load_tabular(pool: pd.DataFrame) -> pd.DataFrame:
    frames = []
    pat = re.compile(r"f(\d+)_(.+)_(unweighted|weighted)\.parquet$")
    for path in sorted((TRC / "predictions" / "cv5_tabular").glob("*.parquet")):
        match = pat.match(path.name)
        if not match:
            continue
        fold, model, lane = int(match.group(1)), match.group(2), match.group(3)
        d = pd.read_parquet(path)
        d["case_id"] = d.case_id.astype(str)
        expected = set(pool.loc[pool.fold.eq(fold), "case_id"])
        assert set(d.case_id) == expected, path.name
        assert len(d) == len(expected) * 33 and not d.p.isna().any()
        assert not d.duplicated(["case_id", "scenario_id"]).any()
        assert d.fold.eq(fold).all() and d.model.eq(model).all() and d.lane.eq(lane).all()
        d["system"] = TABULAR_LABEL[model]
        d["hard"] = (d.p > 0.5).astype(int)
        frames.append(d[["case_id", "scenario_id", "p", "hard", "fold", "system", "lane"]])
    out = pd.concat(frames, ignore_index=True)
    assert len(frames) == 70
    return out


def _load_sft_scenarios(test_ids: set[str]) -> pd.DataFrame:
    frames = []
    specs = [
        ("Qwen 2B SFT", TRC / "predictions" / "cv5_llm", "qwen_mid_f*_scenarios.parquet"),
        ("Qwen 9B SFT", TRC / "predictions" / "cv5_llm_grid", "qwen_f*.parquet"),
    ]
    for system, root, glob in specs:
        d = pd.concat([pd.read_parquet(p) for p in sorted(root.glob(glob))], ignore_index=True)
        d["case_id"] = d.case_id.astype(str)
        d = d.loc[d.case_id.isin(test_ids)].copy()
        d["p"] = expit(d.margin.to_numpy(float))
        tie = np.abs(d.margin.to_numpy(float)) < 1e-9
        d["hard"] = np.where(tie, d.free_argmax.to_numpy() == "AV", d.margin.to_numpy(float) > 0).astype(int)
        d["system"] = system
        d["lane"] = "sft_aaai_composite"
        assert len(d) == 3468 * 33 and d.case_id.nunique() == 3468
        assert d.scenario_id.nunique() == 33 and not d.p.isna().any()
        assert not d.duplicated(["case_id", "scenario_id"]).any()
        frames.append(d[["case_id", "scenario_id", "p", "hard", "fold", "system", "lane"]])
    return pd.concat(frames, ignore_index=True)


def _standardize_zs(d: pd.DataFrame, system: str, test: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["case_id"] = d.case_id.astype(str)
    d = d.loc[d.case_id.isin(set(test.case_id))].copy()
    expected_fold = d.case_id.map(test.set_index("case_id").fold)
    if "fold" in d:
        assert d.fold.astype(int).eq(expected_fold.astype(int)).all(), system
    else:
        d["fold"] = expected_fold.astype(int)
    d["p"] = expit(d.margin.to_numpy(float))
    tie = np.abs(d.margin.to_numpy(float)) < 1e-9
    d["hard"] = np.where(
        tie, d.free_argmax.to_numpy() == "AV", d.margin.to_numpy(float) > 0
    ).astype(int)
    d["system"], d["lane"] = system, "zero_shot"
    assert len(d) == 3468 * 33 and d.case_id.nunique() == 3468, system
    assert d.scenario_id.nunique() == 33 and not d.p.isna().any(), system
    assert not d.duplicated(["case_id", "scenario_id"]).any(), system
    return d[["case_id", "scenario_id", "p", "hard", "fold", "system", "lane"]]


def _load_zeroshot_scenarios(
    test: pd.DataFrame, grid: pd.DataFrame, full_grid: bool = False
) -> pd.DataFrame:
    """Load both ZS scenario grids and attach/verify the common CV5 folds."""
    if full_grid:
        mid_files = [
            TRC / "predictions" / "cv5_llm" / f"qwen_mid_zs_f{fold}_scenarios.parquet"
            for fold in FOLDS
        ]
        big_files = [
            TRC / "predictions" / "cv5_llm_zs_grid" / f"qwen_p{part}.parquet"
            for part in (1, 2)
        ]
        missing = [str(p) for p in [*mid_files, *big_files] if not p.exists()]
        if missing:
            raise FileNotFoundError("Incomplete full-grid ZS inputs: " + "; ".join(missing))
        sources = [
            ("Qwen 2B ZS", pd.concat([pd.read_parquet(p) for p in mid_files], ignore_index=True)),
            ("Qwen 9B ZS", pd.concat([pd.read_parquet(p) for p in big_files], ignore_index=True)),
        ]
        out = pd.concat([_standardize_zs(d, system, test) for system, d in sources], ignore_index=True)
        expected = pd.MultiIndex.from_frame(
            grid.loc[grid.case_id.isin(set(test.case_id)), ["case_id", "scenario_id"]]
            .astype({"case_id": str})
            .drop_duplicates()
        )
        for system, d in out.groupby("system"):
            actual = pd.MultiIndex.from_frame(d[["case_id", "scenario_id"]])
            assert len(actual) == len(expected) and actual.difference(expected).empty, system
            assert expected.difference(actual).empty, system
        return out

    bank = pd.read_parquet(
        REAL / "data" / "06_predictions_llm" / "llm_scores_cv5_2b_9b.parquet",
        columns=["case_id", "scenario_id", "model", "margin", "free_argmax"],
    )
    labels = {"qwen_mid_zeroshot": "Qwen 2B ZS", "qwen_zeroshot": "Qwen 9B ZS"}
    bank["case_id"] = bank.case_id.astype(str)
    bank = bank.loc[bank.model.isin(labels)].copy()
    parts = [_standardize_zs(d, labels[model], test) for model, d in bank.groupby("model")]
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == 2 * 3468 * 33
    assert not out.duplicated(["system", "case_id", "scenario_id"]).any()
    return out


def _raw_llm_fullgrid_inventory(test: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Audit the four Qwen size/state grids at their raw-score grain."""
    specs = [
        (
            "Qwen 2B ZS",
            [TRC / "predictions" / "cv5_llm" / f"qwen_mid_zs_f{f}_scenarios.parquet" for f in FOLDS],
        ),
        (
            "Qwen 2B SFT",
            [TRC / "predictions" / "cv5_llm" / f"qwen_mid_f{f}_scenarios.parquet" for f in FOLDS],
        ),
        (
            "Qwen 9B ZS",
            [TRC / "predictions" / "cv5_llm_zs_grid" / f"qwen_p{p}.parquet" for p in (1, 2)],
        ),
        (
            "Qwen 9B SFT",
            [TRC / "predictions" / "cv5_llm_grid" / f"qwen_f{f}.parquet" for f in FOLDS],
        ),
    ]
    expected = pd.MultiIndex.from_frame(
        grid.loc[grid.case_id.isin(set(test.case_id)), ["case_id", "scenario_id"]]
        .astype({"case_id": str})
        .drop_duplicates()
    )
    fold_map = test.set_index("case_id").fold
    rows = []
    for system, files in specs:
        assert all(p.exists() for p in files), system
        d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        d["case_id"] = d.case_id.astype(str)
        d = d.loc[d.case_id.isin(set(test.case_id))].copy()
        if "fold" not in d:
            d["fold"] = d.case_id.map(fold_map).astype(int)
        actual = pd.MultiIndex.from_frame(d[["case_id", "scenario_id"]])
        required = ["q_av", "q_keep", "margin", "candidate_mass", "free_argmax"]
        nonfinite = int((~np.isfinite(d[["q_av", "q_keep", "margin", "candidate_mass"]])).sum().sum())
        key_exact = len(actual) == len(expected) and actual.difference(expected).empty and expected.difference(actual).empty
        fold_exact = d.fold.astype(int).eq(d.case_id.map(fold_map).astype(int)).all()
        rows.append({
            "system": system,
            "source_files": len(files),
            "rows": len(d),
            "cases": d.case_id.nunique(),
            "scenarios": d.scenario_id.nunique(),
            "folds_present": ",".join(map(str, sorted(d.fold.astype(int).unique()))),
            "duplicate_keys": int(d.duplicated(["case_id", "scenario_id"]).sum()),
            "missing_required": int(d[required].isna().sum().sum()),
            "nonfinite_scores": nonfinite,
            "exact_common_grid": bool(key_exact),
            "exact_fold_assignment": bool(fold_exact),
            "candidate_mass_min": float(d.candidate_mass.min()),
            "candidate_mass_mean": float(d.candidate_mass.mean()),
            "candidate_mass_below_0_5": int((d.candidate_mass < 0.5).sum()),
            "exact_margin_ties": int((np.abs(d.margin.to_numpy(float)) < 1e-9).sum()),
            "free_argmax_keep": int(d.free_argmax.eq("Keep").sum()),
            "free_argmax_av": int(d.free_argmax.eq("AV").sum()),
        })
    out = pd.DataFrame(rows)
    assert out.rows.eq(3468 * 33).all() and out.cases.eq(3468).all()
    assert out.scenarios.eq(33).all() and out.duplicate_keys.eq(0).all()
    assert out.missing_required.eq(0).all() and out.nonfinite_scores.eq(0).all()
    assert out.exact_common_grid.all() and out.exact_fold_assignment.all()
    assert out.candidate_mass_below_0_5.eq(0).all()
    return out


def _qwen_hard_choice_tables(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce strict-margin rates while retaining the registered tie rule."""
    specs = [
        (
            "Qwen 2B ZS",
            [TRC / "predictions" / "cv5_llm" / f"qwen_mid_zs_f{f}_scenarios.parquet" for f in FOLDS],
        ),
        (
            "Qwen 2B SFT",
            [TRC / "predictions" / "cv5_llm" / f"qwen_mid_f{f}_scenarios.parquet" for f in FOLDS],
        ),
        (
            "Qwen 9B ZS",
            [TRC / "predictions" / "cv5_llm_zs_grid" / f"qwen_p{p}.parquet" for p in (1, 2)],
        ),
        (
            "Qwen 9B SFT",
            [TRC / "predictions" / "cv5_llm_grid" / f"qwen_f{f}.parquet" for f in FOLDS],
        ),
    ]
    test_ids = set(test.case_id)
    raw = []
    for system, files in specs:
        d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        d["case_id"] = d.case_id.astype(str)
        d = d.loc[d.case_id.isin(test_ids)].copy()
        d["system"] = system
        d["strict_positive_margin_av"] = d.margin.to_numpy(float) > 0
        tie = np.abs(d.margin.to_numpy(float)) < 1e-9
        d["tie_resolved_av"] = np.where(
            tie, d.free_argmax.to_numpy() == "AV", d.margin.to_numpy(float) > 0
        )
        d["exact_tie"] = tie
        d["pair_normalized_p_av"] = expit(d.margin.to_numpy(float))
        raw.append(d)
    raw = pd.concat(raw, ignore_index=True)
    rows = []
    for attribute, sids in RESPONSE_SCENARIOS.items():
        for system in ("Qwen 2B ZS", "Qwen 2B SFT", "Qwen 9B ZS", "Qwen 9B SFT"):
            d = raw.loc[raw.system.eq(system) & raw.scenario_id.isin(sids)]
            for rung, sid in enumerate(sids, start=-3):
                s = d.loc[d.scenario_id.eq(sid)]
                rows.append({
                    "Model": system,
                    "Attribute": attribute,
                    "rung": rung,
                    "scenario_id": sid,
                    "N cases": len(s),
                    "Strict positive-margin AV rate": float(s.strict_positive_margin_av.mean()),
                    "Tie-resolved AV rate": float(s.tie_resolved_av.mean()),
                    "Pair-normalized mean P(AV)": float(s.pair_normalized_p_av.mean()),
                    "Exact ties": int(s.exact_tie.sum()),
                })
    response = pd.DataFrame(rows)
    spans = []
    for (system, attribute), d in response.groupby(["Model", "Attribute"], sort=False):
        x = d.set_index("rung")
        spans.append({
            "Model": system,
            "Size": "2B" if "2B" in system else "9B",
            "State": "ZS" if system.endswith("ZS") else "SFT",
            "Attribute": attribute,
            "Strict AV-rate span (-3 minus +3)": float(
                x.loc[-3, "Strict positive-margin AV rate"]
                - x.loc[3, "Strict positive-margin AV rate"]
            ),
            "Tie-resolved AV-rate span (-3 minus +3)": float(
                x.loc[-3, "Tie-resolved AV rate"] - x.loc[3, "Tie-resolved AV rate"]
            ),
            "Pair-normalized P(AV) span (-3 minus +3)": float(
                x.loc[-3, "Pair-normalized mean P(AV)"]
                - x.loc[3, "Pair-normalized mean P(AV)"]
            ),
        })
    return response, pd.DataFrame(spans)


def _load_new_qwen9b_zs_base() -> pd.DataFrame | None:
    """Load the new all-11,584 one-pass CV5 ZS run after both remote parts arrive."""
    root = TRC / "predictions" / "cv5_llm_zs"
    files = [root / "qwen_base_p1.parquet", root / "qwen_base_p2.parquet"]
    if not all(p.exists() for p in files):
        return None
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    d["case_id"] = d.case_id.astype(str)
    assert len(d) == 11584 and d.case_id.is_unique and d.scenario_id.eq("BASE").all()
    assert set(d.fold) == set(FOLDS) and not d.margin.isna().any()
    d["p"] = expit(d.margin.to_numpy(float))
    tie = np.abs(d.margin.to_numpy(float)) < 1e-9
    d["hard"] = np.where(tie, d.free_argmax.to_numpy() == "AV", d.margin.to_numpy(float) > 0).astype(int)
    d["system"], d["lane"] = "Qwen 9B ZS", "zero_shot"
    return d


def _qwen9b_zs_base_audit(
    frozen_scenarios: pd.DataFrame, new_base: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for fold, f in new_base.groupby("fold"):
        rows.append({"scope": "fold", "fold": int(fold), "n": len(f),
                     **_metrics(f.y.to_numpy(int), f.p.to_numpy(float), f.hard.to_numpy(int))})
    rows.append({"scope": "pooled", "fold": 0, "n": len(new_base),
                 **_metrics(new_base.y.to_numpy(int), new_base.p.to_numpy(float), new_base.hard.to_numpy(int))})
    old = frozen_scenarios.loc[
        frozen_scenarios.system.eq("Qwen 9B ZS") & frozen_scenarios.scenario_id.eq("BASE"),
        ["case_id", "p", "hard"],
    ].rename(columns={"p": "p_old", "hard": "hard_old"})
    new = new_base.loc[new_base.case_id.isin(set(test.case_id)), ["case_id", "p", "hard", "y"]].rename(
        columns={"p": "p_new", "hard": "hard_new"}
    )
    pair = old.merge(new, on="case_id", validate="one_to_one")
    audit = {
        "official_test_pairs": len(pair),
        "max_abs_probability_difference": float(np.abs(pair.p_new - pair.p_old).max()),
        "mean_abs_probability_difference": float(np.abs(pair.p_new - pair.p_old).mean()),
        "hard_label_mismatches": int((pair.hard_new != pair.hard_old).sum()),
        "old_official_test_metrics": _metrics(pair.y.to_numpy(int), pair.p_old.to_numpy(float), pair.hard_old.to_numpy(int)),
        "new_official_test_metrics": _metrics(pair.y.to_numpy(int), pair.p_new.to_numpy(float), pair.hard_new.to_numpy(int)),
    }
    return pd.DataFrame(rows), audit


def _metrics(y: np.ndarray, p: np.ndarray, hard: np.ndarray | None = None) -> dict[str, float]:
    if hard is None:
        hard = (p > 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "av_f1": float(f1_score(y, hard, zero_division=0)),
        "macro_f1": float(f1_score(y, hard, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, hard)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "pred_av_rate": float(hard.mean()),
        "true_av_rate": float(y.mean()),
    }


def _prediction_tables(pred: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pred.loc[pred.scenario_id.eq("BASE")].merge(
        test[["case_id", "y", "fold"]], on=["case_id", "fold"], how="inner", validate="many_to_one"
    )
    rows = []
    for (system, lane), g in base.groupby(["system", "lane"], sort=False):
        for fold, f in g.groupby("fold"):
            rows.append({"system": system, "lane": lane, "scope": "fold", "fold": int(fold),
                         "n": len(f), **_metrics(f.y.to_numpy(int), f.p.to_numpy(float), f.hard.to_numpy(int))})
        rows.append({"system": system, "lane": lane, "scope": "pooled", "fold": 0,
                     "n": len(g), **_metrics(g.y.to_numpy(int), g.p.to_numpy(float), g.hard.to_numpy(int))})
    detailed = pd.DataFrame(rows)
    selected = []
    for system, lane in PREDICTION_ROWS:
        p = detailed.loc[(detailed.system.eq(system)) & detailed.lane.eq(lane)]
        pooled = p.loc[p.scope.eq("pooled")].iloc[0]
        folds = p.loc[p.scope.eq("fold")]
        selected.append({
            "Model": system,
            "Lane": lane,
            "N": int(pooled.n),
            "AV-F1 pooled": pooled.av_f1,
            "AV-F1 fold mean": folds.av_f1.mean(),
            "AV-F1 fold SD": folds.av_f1.std(ddof=1),
            "Accuracy pooled": pooled.accuracy,
            "Macro-F1 pooled": pooled.macro_f1,
            "ROC-AUC pooled": pooled.roc_auc,
            "PR-AUC pooled": pooled.pr_auc,
            "Log loss pooled": pooled.log_loss,
            "Predicted AV rate": pooled.pred_av_rate,
            "Observed AV rate": pooled.true_av_rate,
        })
    return detailed, pd.DataFrame(selected)


def _six_pairs(pmat: np.ndarray, xmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pl, pu = pmat[:, :-1], pmat[:, 1:]
    xl, xu = xmat[:, :-1], xmat[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        elasticity = ((pu - pl) / ((pu + pl) / 2)) / ((xu - xl) / ((xu + xl) / 2))
        slope = (pu - pl) / (xu - xl)
    return elasticity, slope


def _select_behaviour(pred: pd.DataFrame, sensitivity: bool = False) -> pd.DataFrame:
    parts = []
    for system in BEHAVIOUR_ORDER:
        if system.startswith("Qwen"):
            lane = MAIN_BEHAVIOUR_LANE[system]
        elif sensitivity:
            lane = "unweighted"
        else:
            lane = MAIN_BEHAVIOUR_LANE[system]
        d = pred.loc[pred.system.eq(system) & pred.lane.eq(lane)].copy()
        d["analysis_lane"] = "tuned_base_sensitivity" if sensitivity else "case_generator_weighted"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _behaviour_tables(
    pred: pd.DataFrame, test: pd.DataFrame, grid: pd.DataFrame, sensitivity: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = _select_behaviour(pred, sensitivity=sensitivity)
    # One row per common system-case-scenario.
    assert selected.groupby("system").size().eq(3468 * 33).all()
    ids = test.case_id.astype(str).tolist()
    grid = grid.loc[grid.case_id.isin(set(ids))].copy()
    case_rows, summary_rows, fold_rows = [], [], []
    for system in BEHAVIOUR_ORDER:
        d = selected.loc[selected.system.eq(system)]
        model_lane = str(d.lane.iloc[0])
        assert d.lane.nunique() == 1
        mats = {}
        for attribute in ("Fare", "IVT"):
            sids = SCENARIOS[attribute]
            pmat = d.loc[d.scenario_id.isin(sids)].pivot(
                index="case_id", columns="scenario_id", values="p"
            )[sids].loc[ids].to_numpy(float)
            xmat = grid.loc[grid.scenario_id.isin(sids)].pivot(
                index="case_id", columns="scenario_id", values=XCOL[attribute]
            )[sids].loc[ids].to_numpy(float)
            mats[attribute] = (*_six_pairs(pmat, xmat), pmat, xmat)
        fare_e, fare_d, fare_p, _ = mats["Fare"]
        _, ivt_d, _, _ = mats["IVT"]
        raw_df, raw_dt = fare_d.mean(axis=1), ivt_d.mean(axis=1)
        finite = np.isfinite(raw_df) & np.isfinite(raw_dt)
        computable = finite & (np.abs(raw_df) >= RAW_FARE_THRESHOLD)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_vot = np.where(computable, 60 * raw_dt / raw_df, np.nan)
        c = pd.DataFrame({
            "system": system,
            "analysis_lane": "tuned_base_sensitivity" if sensitivity else "case_generator_weighted",
            "model_prediction_lane": model_lane,
            "case_id": ids,
            "respondent_id": test.set_index("case_id").loc[ids, "respondent_id"].to_numpy(),
            "fold": test.set_index("case_id").loc[ids, "fold"].to_numpy(int),
            "elasticity": fare_e.mean(axis=1),
            "monotone_pairs": (fare_e < 0).sum(axis=1),
            "delta_p_base_p1": fare_p[:, 4] - fare_p[:, 3],
            "raw_DF": raw_df,
            "raw_DT": raw_dt,
            "raw_finite": finite,
            "raw_computable": computable,
            "raw_vot": raw_vot,
        })
        for k in range(6):
            c[f"fare_elasticity_pair_{k + 1}"] = fare_e[:, k]
        case_rows.append(c)
        valid = c.loc[c.raw_computable]
        fin = c.loc[c.raw_finite]
        all_intervals = fare_e[np.isfinite(fare_e)]
        per_fold = []
        for fold, f in c.groupby("fold"):
            ff = f.loc[f.raw_finite]
            fv = f.loc[f.raw_computable]
            row = {
                "Model": system,
                "Analysis lane": "tuned_base_sensitivity" if sensitivity else "case_generator_weighted",
                "Model prediction lane": model_lane,
                "fold": int(fold),
                "N cases": len(f),
                "Mean case Fare elasticity": f.elasticity.mean(),
                "Population Direct VOT (kKRW/h)": 60 * ff.raw_DT.mean() / ff.raw_DF.mean(),
                "Joint conventional sign (%)": 100 * ((fv.raw_DF < 0) & (fv.raw_DT < 0)).mean(),
            }
            per_fold.append(row)
            fold_rows.append(row)
        pf = pd.DataFrame(per_fold)
        summary_rows.append({
            "Model": system,
            "Analysis lane": "tuned_base_sensitivity" if sensitivity else "case_generator_weighted",
            "Model prediction lane": model_lane,
            "N cases": len(c),
            "N respondents": c.respondent_id.nunique(),
            "Mean case Fare elasticity": c.elasticity.mean(),
            "Fare elasticity fold mean": pf["Mean case Fare elasticity"].mean(),
            "Fare elasticity fold SD": pf["Mean case Fare elasticity"].std(ddof=1),
            "Median case Fare elasticity": c.elasticity.median(),
            "Mean interval Fare elasticity": all_intervals.mean(),
            "Negative Fare intervals (%)": 100 * (all_intervals < 0).mean(),
            "Fare monotone pairs (%)": 100 * c.monotone_pairs.sum() / (6 * len(c)),
            "Computable VOT (%)": 100 * c.raw_computable.mean(),
            "Fare-negative among computable (%)": 100 * (valid.raw_DF < 0).mean(),
            "IVT-negative among computable (%)": 100 * (valid.raw_DT < 0).mean(),
            "Joint conventional sign (%)": 100 * ((valid.raw_DF < 0) & (valid.raw_DT < 0)).mean(),
            "Population Direct VOT (kKRW/h)": 60 * fin.raw_DT.mean() / fin.raw_DF.mean(),
            "Population VOT fold mean": pf["Population Direct VOT (kKRW/h)"].mean(),
            "Population VOT fold SD": pf["Population Direct VOT (kKRW/h)"].std(ddof=1),
            "Median case VOT (kKRW/h)": valid.raw_vot.median(),
        })
    return pd.concat(case_rows, ignore_index=True), pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def _response_table(
    pred: pd.DataFrame,
    grid: pd.DataFrame,
    sensitivity: bool = False,
    scenarios: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    keep = _select_behaviour(pred, sensitivity=sensitivity)
    scenarios = SCENARIOS if scenarios is None else scenarios
    rows = []
    for attribute, sids in scenarios.items():
        x = grid.loc[grid.scenario_id.isin(sids)].groupby("scenario_id")[XCOL[attribute]].mean()
        for system in BEHAVIOUR_ORDER:
            d = keep.loc[keep.system.eq(system) & keep.scenario_id.isin(sids)]
            mean = d.groupby("scenario_id").p.mean()
            fold = d.groupby(["fold", "scenario_id"]).p.mean().unstack(0)
            for rung, sid in enumerate(sids, start=-3):
                vals = fold.loc[sid]
                rows.append({"analysis_lane": "tuned_base_sensitivity" if sensitivity else "case_generator_weighted",
                             "attribute": attribute, "system": system, "rung": rung,
                             "scenario_id": sid, "mean_physical_value": x.loc[sid],
                             "mean_p_av": mean.loc[sid], "sd_across_folds": vals.std(ddof=1),
                             "n_cases": d.loc[d.scenario_id.eq(sid), "case_id"].nunique()})
    return pd.DataFrame(rows)


def _response_span_tables(response: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-system spans and matched Qwen SFT-vs-ZS contrasts."""
    rows = []
    for (system, attribute), d in response.groupby(["system", "attribute"], sort=False):
        rung = d.set_index("rung")
        rows.append({
            "Model": system,
            "Attribute": attribute,
            "P(AV) at -3": float(rung.loc[-3, "mean_p_av"]),
            "P(AV) at BASE": float(rung.loc[0, "mean_p_av"]),
            "P(AV) at +3": float(rung.loc[3, "mean_p_av"]),
            "Response span P(-3)-P(+3)": float(
                rung.loc[-3, "mean_p_av"] - rung.loc[3, "mean_p_av"]
            ),
            "N cases": int(rung.loc[0, "n_cases"]),
        })
    all_spans = pd.DataFrame(rows)
    contrast = []
    for size in ("2B", "9B"):
        for attribute in RESPONSE_SCENARIOS:
            zs = all_spans.loc[
                all_spans.Model.eq(f"Qwen {size} ZS") & all_spans.Attribute.eq(attribute)
            ].iloc[0]
            sft = all_spans.loc[
                all_spans.Model.eq(f"Qwen {size} SFT") & all_spans.Attribute.eq(attribute)
            ].iloc[0]
            zspan = float(zs["Response span P(-3)-P(+3)"])
            sspan = float(sft["Response span P(-3)-P(+3)"])
            contrast.append({
                "Size": size,
                "Attribute": attribute,
                "ZS P(AV) at BASE": float(zs["P(AV) at BASE"]),
                "SFT P(AV) at BASE": float(sft["P(AV) at BASE"]),
                "ZS response span": zspan,
                "SFT response span": sspan,
                "SFT minus ZS span": sspan - zspan,
                "Absolute sensitivity change vs ZS (%)": (
                    100 * (abs(sspan) / abs(zspan) - 1) if abs(zspan) >= 0.01 else np.nan
                ),
            })
    return all_spans, pd.DataFrame(contrast)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction(table: pd.DataFrame) -> None:
    d = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    y = np.arange(len(d))
    for i, row in d.iterrows():
        color, marker, _ = MODEL_STYLE[row.Model]
        ax.errorbar(row["AV-F1 fold mean"], i, xerr=row["AV-F1 fold SD"], fmt=marker,
                    color=color, ecolor=color, capsize=3, markersize=6, linewidth=1.3)
        ax.text(row["AV-F1 fold mean"] + 0.014, i, f"{row['AV-F1 pooled']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, d.Model)
    ax.set_xlim(0, max(0.58, d["AV-F1 fold mean"].max() + 0.09))
    ax.set_xlabel("AV-class F1 (fold mean ± fold SD; label = pooled value)")
    ax.set_title("CV5 prediction performance on the common official-test cases", loc="left", weight="bold")
    ax.text(0, 1.01, "N = 3,468 cases; weighted predictive lane, except Human H-MNL; all trained rows are out-of-fold",
            transform=ax.transAxes, fontsize=8.5, color="#444444")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "Figure_01_cv5_prediction_av_f1")


def _plot_response(table: pd.DataFrame) -> None:
    attributes = [a for a in RESPONSE_SCENARIOS if a in set(table.attribute)]
    fig, axes = plt.subplots(1, len(attributes), figsize=(4.55 * len(attributes), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, attribute in zip(axes, attributes):
        for system in BEHAVIOUR_ORDER:
            d = table.loc[table.attribute.eq(attribute) & table.system.eq(system)].sort_values("rung")
            color, marker, ls = MODEL_STYLE[system]
            ax.plot(d.rung, d.mean_p_av, color=color, marker=marker, linestyle=ls,
                    linewidth=1.25, markersize=3.7, label=system)
        ax.axvline(0, color="#9CA3AF", linewidth=0.8)
        ax.set_xticks(range(-3, 4), ["−3", "−2", "−1", "BASE", "+1", "+2", "+3"])
        ax.set_xlabel(f"{attribute} scenario rung")
        ax.set_title(attribute, weight="bold")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean P(AV)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False, fontsize=8)
    title_attributes = ", ".join(attributes[:-1]) + (f", and {attributes[-1]}" if len(attributes) > 1 else attributes[0])
    fig.suptitle(f"CV5 probability response under {title_attributes} interventions", x=0.04, ha="left", weight="bold")
    fig.text(0.04, 0.925, "Common 3,468 official-test cases; conventional lane matches the active CASE generator (D250 weighted)",
             fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.13, 1, 0.91))
    _save(fig, "Figure_02_cv5_probability_response")


def _plot_qwen_response(table: pd.DataFrame) -> None:
    qwen = table.loc[table.system.str.startswith("Qwen")].copy()
    attributes = list(RESPONSE_SCENARIOS)
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.65), sharey=True)
    for ax, attribute in zip(axes, attributes):
        for system in ("Qwen 2B ZS", "Qwen 2B SFT", "Qwen 9B ZS", "Qwen 9B SFT"):
            d = qwen.loc[qwen.attribute.eq(attribute) & qwen.system.eq(system)].sort_values("rung")
            color, marker, ls = MODEL_STYLE[system]
            fill = "white" if system.endswith("ZS") else color
            ax.plot(
                d.rung,
                d.mean_p_av,
                color=color,
                marker=marker,
                markerfacecolor=fill,
                markeredgecolor=color,
                linestyle=ls,
                linewidth=1.6,
                markersize=5.0,
                label=system,
            )
        ax.axvline(0, color="#9CA3AF", linewidth=0.8)
        ax.set_xticks(range(-3, 4), ["-3", "-2", "-1", "BASE", "+1", "+2", "+3"])
        ax.set_xlabel(f"{attribute} scenario rung")
        ax.set_title(attribute, weight="bold")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean P(AV)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False)
    fig.suptitle("Qwen CV5 response functions by model size and alignment state", x=0.04, ha="left", weight="bold")
    fig.text(
        0.04,
        0.92,
        "Full 33-scenario grids on the same 3,468 cases; open/dashed = zero-shot, filled/solid = SFT",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    _save(fig, "Figure_04_qwen_fullgrid_response")


def _plot_qwen_span_contrast(contrast: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), sharex=True, sharey=True)
    attributes = list(RESPONSE_SCENARIOS)[::-1]
    for ax, size in zip(axes, ("2B", "9B")):
        d = contrast.loc[contrast.Size.eq(size)].set_index("Attribute").loc[attributes]
        color = MODEL_STYLE[f"Qwen {size} SFT"][0]
        for i, (_, row) in enumerate(d.iterrows()):
            zs, sft = row["ZS response span"], row["SFT response span"]
            ax.plot([zs, sft], [i, i], color="#9CA3AF", linewidth=1.2, zorder=1)
            ax.scatter(zs, i, facecolors="white", edgecolors=color, marker="o", s=40, linewidths=1.3, zorder=3)
            ax.scatter(sft, i, color=color, marker="s", s=38, zorder=3)
            ax.text(zs, i + 0.17, f"{zs:+.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(sft, i - 0.17, f"{sft:+.3f}", ha="center", va="top", fontsize=7)
        ax.axvline(0, color="#4B5563", linewidth=0.9)
        ax.set_title(f"Qwen {size}", weight="bold")
        ax.set_yticks(range(len(attributes)))
        ax.set_xlabel("Response span: mean P(AV,-3) - mean P(AV,+3)")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticklabels(attributes)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].scatter([], [], facecolors="white", edgecolors="#4B5563", marker="o", label="Zero-shot")
    axes[0].scatter([], [], color="#4B5563", marker="s", label="SFT")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Qwen CV5 response-span change after SFT", x=0.06, ha="left", weight="bold")
    fig.text(
        0.06,
        0.91,
        "Positive values mean greater AV response when Fare/IVT/Wait is lower; Reliability uses the same signed -3 minus +3 convention",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "Figure_05_qwen_response_span_sft_vs_zs")


def _plot_qwen_hard_span_contrast(spans: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), sharex=True, sharey=True)
    attributes = list(RESPONSE_SCENARIOS)[::-1]
    col = "Strict AV-rate span (-3 minus +3)"
    for ax, size in zip(axes, ("2B", "9B")):
        d = spans.loc[spans.Size.eq(size)].set_index(["State", "Attribute"])
        color = MODEL_STYLE[f"Qwen {size} SFT"][0]
        for i, attribute in enumerate(attributes):
            zs = float(d.loc[("ZS", attribute), col])
            sft = float(d.loc[("SFT", attribute), col])
            ax.plot([zs, sft], [i, i], color="#9CA3AF", linewidth=1.2, zorder=1)
            ax.scatter(zs, i, facecolors="white", edgecolors=color, marker="o", s=40, linewidths=1.3, zorder=3)
            ax.scatter(sft, i, color=color, marker="s", s=38, zorder=3)
            ax.text(zs, i + 0.17, f"{zs:+.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(sft, i - 0.17, f"{sft:+.3f}", ha="center", va="top", fontsize=7)
        ax.axvline(0, color="#4B5563", linewidth=0.9)
        ax.set_title(f"Qwen {size}", weight="bold")
        ax.set_yticks(range(len(attributes)))
        ax.set_xlabel("Strict AV-rate span: rate(-3) - rate(+3)")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticklabels(attributes)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].scatter([], [], facecolors="white", edgecolors="#4B5563", marker="o", label="Zero-shot")
    axes[0].scatter([], [], color="#4B5563", marker="s", label="SFT")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Qwen CV5 hard-choice response-span change after SFT", x=0.06, ha="left", weight="bold")
    fig.text(
        0.06,
        0.91,
        "Diagnostic uses margin > 0 exactly; exact ties are reported separately and Direct elasticity/VOT use pair-normalized probabilities",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "Figure_06_qwen_hard_choice_response_span")


def _plot_behaviour(table: pd.DataFrame) -> None:
    d = table.set_index("Model").loc[BEHAVIOUR_ORDER].iloc[::-1].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0))
    specs = [
        ("Fare elasticity fold mean", "Fare elasticity fold SD", "Mean case Fare elasticity",
         "Mean Direct Fare elasticity", 0.0),
        ("Population VOT fold mean", "Population VOT fold SD", "Population Direct VOT (kKRW/h)",
         "Population Direct VOT-equivalent (kKRW/h)", None),
    ]
    for ax, (point_col, sd_col, label_col, xlabel, zero) in zip(axes, specs):
        for i, row in d.iterrows():
            color, marker, _ = MODEL_STYLE[row.Model]
            ax.errorbar(row[point_col], i, xerr=row[sd_col], color=color, ecolor=color,
                        marker=marker, markersize=5.5, capsize=2.5, linewidth=1.0, zorder=3)
            ax.text(row[point_col], i + 0.19, f"{row[label_col]:.2f}", ha="center", va="bottom", fontsize=7)
        ref = float(d.loc[d.Model.eq("Human CV5 H-MNL"), label_col].iloc[0])
        ax.axvline(ref, color="#202020", linestyle="--", linewidth=1.0, label="Human CV5 H-MNL")
        if zero is not None:
            ax.axvline(zero, color="#9CA3AF", linewidth=0.8)
        ax.set_yticks(range(len(d)), d.Model if ax is axes[0] else [])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CV5 Direct behavioural quantities on common held-out cases", x=0.08, ha="left", weight="bold")
    fig.text(0.08, 0.93, "Fold mean +/- fold SD; labels and dashed Human H-MNL reference are pooled CASE estimates",
             fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "Figure_03_cv5_direct_elasticity_vot")


def _inventory(pool: pd.DataFrame, test: pd.DataFrame, tab: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in BEHAVIOUR_ORDER:
        behaviour_lane = MAIN_BEHAVIOUR_LANE[system]
        d = pred.loc[pred.system.eq(system) & pred.lane.eq(behaviour_lane)]
        rows.append({
            "system": system,
            "trained_or_zero_shot": "zero-shot" if system.endswith("ZS") else "trained",
            "folds_present": ",".join(map(str, sorted(d.fold.unique()))),
            "official_test_cases": d.case_id.nunique(),
            "scenarios": d.scenario_id.nunique(),
            "rows": len(d),
            "duplicate_system_case_scenario": int(d.duplicated(["case_id", "scenario_id"]).sum()),
            "missing_probability": int(d.p.isna().sum()),
            "behaviour_lane": behaviour_lane,
        })
    out = pd.DataFrame(rows)
    assert out.official_test_cases.eq(len(test)).all() and out.scenarios.eq(33).all()
    assert out.duplicate_system_case_scenario.eq(0).all() and out.missing_probability.eq(0).all()
    return out


def _write_readme(inventory: pd.DataFrame, prediction: pd.DataFrame, behaviour: pd.DataFrame) -> None:
    h = behaviour.set_index("Model").loc["Human CV5 H-MNL"]
    best = prediction.sort_values("AV-F1 pooled", ascending=False).iloc[0]
    text = f"""# CV5 all-model robustness package

Generated 2026-09-07 (Asia/Seoul). This package is add-only and does not modify
the active English/Korean manuscripts.

## Scope and interpretation

All trained systems use the same respondent-level five-fold assignment. Each
official-test case is evaluated only by the model trained without its fold.
The common paper-facing denominator is 3,468 cases (651 respondents) and 33
BASE/counterfactual scenarios. Zero-shot has no training state, so one scoring
pass is partitioned by the same fold labels; repeating inference five times
would not create different fold models.

`Human CV5 H-MNL` is the out-of-fold behavioural panel logit trained on the
other four folds. It is not an artificial perfect-prediction row. `Panel Logit`
is the full-feature predictive DCM. Prediction uses the weighted DCM/ML/DL lane
used by the manuscript. The main Direct response comparison also uses the
weighted conventional files read by the active CASE generator, while Human CV5
H-MNL remains unweighted. A tuned-base sensitivity is exported separately. The
historical `unweighted` filename is not interpreted literally because SVM
tuning selected a balanced specification in all five folds.

This is a retrospective CV5 robustness analysis over the combined development
and official-test pool. Because official-test respondents can occur in the
training portion of a *different* CV fold, it must not replace or be described
as the registered primary held-out analysis.

## Input provenance

- Fold assignment: `data/splits/cv5_folds.csv`.
- DCM/ML/DL: `predictions/cv5_tabular/f{{1..5}}_*_{{weighted,unweighted}}.parquet`
  (70 complete 33-scenario files).
- Qwen 2B SFT: `predictions/cv5_llm/qwen_mid_f{{1..5}}_scenarios.parquet`.
- Qwen 9B SFT: `predictions/cv5_llm_grid/qwen_f{{1..5}}.parquet`.
- Fold-invariant ZS scenarios: `Real_exp/data/06_predictions_llm/llm_scores_cv5_2b_9b.parquet`.
- New Qwen 9B ZS all-case BASE run, when present:
  `predictions/cv5_llm_zs/qwen_base_p{{1,2}}.parquet`.

## Current result snapshot

- Best pooled official-test AV-F1: {best['Model']} = {best['AV-F1 pooled']:.4f}.
- Human CV5 H-MNL mean case Fare elasticity: {h['Mean case Fare elasticity']:.4f}.
- Human CV5 H-MNL population Direct VOT: {h['Population Direct VOT (kKRW/h)']:.3f} kKRW/h.
- Human H-MNL VOT is fold-unstable (fold estimates include a negative value), so
  this cross-fitted reference is a robustness diagnostic rather than a safe
  drop-in replacement for the registered Human reference.
- Complete system cells: {len(inventory)}/{len(BEHAVIOUR_ORDER)}, each with 3,468 cases x 33 scenarios.

## Files

- `CV5_INPUT_INVENTORY.csv`: model/fold/scenario completeness gate.
- `Table_01_prediction_performance_selected_lanes.csv`: paper-rule comparison.
- `Table_S01_prediction_performance_all_lanes_and_folds.csv`: pooled and fold-level metrics.
- `Table_02_direct_behavior_summary.csv`: six-pair Fare elasticity and Direct VOT.
- `Table_S02_direct_behavior_case.parquet`: case-grain raw derivatives/quantities.
- `Table_S02b_direct_behavior_by_fold.csv`: fold-specific behavioural quantities.
- `Table_S03_probability_response.csv`: rung-level mean probability and fold SD.
- `Table_S04_*`: tuned-base behavioural sensitivity outputs.
- `Table_S05_qwen9b_zs_all_cv5_cases.csv` and `QWEN9B_ZS_BASE_REPRODUCIBILITY.json`:
  present when the new 11,584-case remote zero-shot run has been retrieved.
- `CHANGES_vs_CASE_CV5_2B_9B_20260907.csv`: old candidate versus all-model CV5 deltas.
- `figures/`: three PDF + 600-dpi PNG figure pairs.
- `validation_summary.json`: machine-readable QA.
- `manifest.csv`: SHA-256 and sizes for all package files except itself.

## Figure contract

1. Figure 1 asks how predictive AV-F1 varies across the shared folds. It uses a
   horizontal dot-and-interval display (fold mean +/- fold SD) with the pooled
   value printed beside each row.
2. Figure 2 asks whether systems move P(AV) coherently over the same seven-rung
   Fare, IVT, and Wait ladders. It uses three aligned line panels and the same
   weighted conventional lane as the active CASE generator.
3. Figure 3 compares the CASE six-pair Direct Fare elasticity and population
   VOT-equivalent against the cross-fitted Human H-MNL reference. Dots and
   intervals show fold mean +/- fold SD; labels show pooled estimates.

Model identity is encoded by marker and line style as well as the manuscript
palette, so the figures remain interpretable without relying on color alone.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def _write_fullgrid_readme(
    inventory: pd.DataFrame,
    llm_inventory: pd.DataFrame,
    prediction: pd.DataFrame,
    behaviour: pd.DataFrame,
    qwen_contrast: pd.DataFrame,
    qwen_hard_spans: pd.DataFrame,
    grid_base_audit: dict[str, object] | None,
) -> None:
    best = prediction.sort_values("AV-F1 pooled", ascending=False).iloc[0]
    h = behaviour.set_index("Model").loc["Human CV5 H-MNL"]
    q9 = qwen_contrast.loc[qwen_contrast.Size.eq("9B")].set_index("Attribute")
    q9h = qwen_hard_spans.loc[qwen_hard_spans.Size.eq("9B")].set_index(["State", "Attribute"])
    audit_note = "No standalone BASE audit was available."
    if grid_base_audit is not None:
        audit_note = (
            "The full-grid and standalone 9B ZS BASE runs share all 3,468 cases; "
            f"they differ on {grid_base_audit['hard_label_mismatches']} hard label(s), with mean absolute "
            f"probability difference {grid_base_audit['mean_abs_probability_difference']:.6f}. "
            "The full-grid BASE is used throughout this package to keep every response curve internally consistent."
        )
    text = f"""# CV5 all-model full-grid robustness package

Generated 2026-09-07 (Asia/Seoul). This is an add-only Fold-5 analysis package;
it does not modify the active English or Korean manuscripts.

## Scope

The package places Human H-MNL, Panel Logit, five ML/DL baselines, and all four
Qwen size/state cells on the same respondent-level five-fold assignment and the
same 3,468 official-test cases x 33 BASE/counterfactual scenarios. Every trained
row is out-of-fold. Zero-shot has no trained fold state, so its single scoring
grid is partitioned by the common respondent folds for fold summaries.

`Human CV5 H-MNL` is a cross-fitted behavioural panel-logit prediction, not the
observed human label. `Panel Logit` is the full-feature predictive DCM. The main
conventional lane matches the weighted lane used by the active CASE generator;
the tuned-base sensitivity remains separate.

This is retrospective robustness evidence. Its fold pool combines development
and official-test respondents, so it cannot replace the registered primary split.

## Completed Qwen matrix

All four cells pass the raw-score gate: 114,444 rows, 3,468 cases, 33 scenarios,
folds 1--5, zero duplicate keys, zero missing/non-finite required scores, exact
common-grid coverage, and no candidate-mass row below 0.5. The four cells total
{int(llm_inventory.rows.sum()):,} raw scenario rows.

- Qwen 2B ZS: `predictions/cv5_llm/qwen_mid_zs_f{{1..5}}_scenarios.parquet`.
- Qwen 2B SFT: `predictions/cv5_llm/qwen_mid_f{{1..5}}_scenarios.parquet`.
- Qwen 9B ZS: `predictions/cv5_llm_zs_grid/qwen_p{{1,2}}.parquet`.
- Qwen 9B SFT: `predictions/cv5_llm_grid/qwen_f{{1..5}}.parquet`.

{audit_note}

## Result snapshot

- Best pooled official-test AV-F1: {best['Model']} = {best['AV-F1 pooled']:.4f}.
- Human CV5 H-MNL Fare elasticity: {h['Mean case Fare elasticity']:.4f};
  population Direct VOT-equivalent: {h['Population Direct VOT (kKRW/h)']:.3f} kKRW/h.
- Qwen 9B Fare response span changes from {q9.loc['Fare', 'ZS response span']:.4f}
  (ZS) to {q9.loc['Fare', 'SFT response span']:.4f} (SFT).
- Qwen 9B IVT response span changes from {q9.loc['IVT', 'ZS response span']:.4f}
  to {q9.loc['IVT', 'SFT response span']:.4f}; Wait changes from
  {q9.loc['Wait', 'ZS response span']:.4f} to {q9.loc['Wait', 'SFT response span']:.4f}.
- On the separate strict `margin > 0` hard-choice diagnostic, Qwen 9B IVT
  changes from {q9h.loc[('ZS', 'IVT'), 'Strict AV-rate span (-3 minus +3)']:.4f}
  to {q9h.loc[('SFT', 'IVT'), 'Strict AV-rate span (-3 minus +3)']:.4f}; Wait changes from
  {q9h.loc[('ZS', 'Wait'), 'Strict AV-rate span (-3 minus +3)']:.4f} to
  {q9h.loc[('SFT', 'Wait'), 'Strict AV-rate span (-3 minus +3)']:.4f}.
- Human H-MNL VOT remains fold-unstable and includes a negative fold estimate;
  this package therefore remains a robustness diagnostic, not a manuscript promotion.

## Tables

- `CV5_INPUT_INVENTORY.csv`: all 11 system/fold/scenario completeness checks.
- `QWEN_FULLGRID_INPUT_INVENTORY.csv`: raw four-cell Qwen quality audit.
- `Table_01_prediction_performance_selected_lanes.csv`: pooled and fold-summary prediction metrics.
- `Table_S01_prediction_performance_all_lanes_and_folds.csv`: every lane and fold.
- `Table_02_direct_behavior_summary.csv`: six-pair Fare elasticity and Direct VOT.
- `Table_S02_direct_behavior_case.parquet`: case-grain derivatives and validity fields.
- `Table_S02b_direct_behavior_by_fold.csv`: fold-level behavioural quantities.
- `Table_S03_probability_response.csv`: four-attribute response curves for all systems.
- `Table_03_response_span_all_models.csv`: signed -3 minus +3 response spans.
- `Table_04_qwen_sft_vs_zs_response_span.csv`: matched 2B/9B SFT-vs-ZS contrasts.
- `Table_05_qwen_hard_choice_response.csv`: strict-margin, tie-resolved, and soft response rates.
- `Table_06_qwen_hard_choice_response_span.csv`: strict and tie-resolved hard-choice spans.
- `Table_S04_*`: tuned-base conventional sensitivity outputs.
- `Table_S05_qwen9b_zs_all_cv5_cases.csv`: standalone 11,584-case 9B ZS BASE metrics.
- `QWEN9B_ZS_GRID_VS_STANDALONE_BASE.json`: repeated-inference reproducibility check.

## Figures

1. `Figure_01_cv5_prediction_av_f1`: fold mean +/- fold SD with pooled AV-F1 labels.
2. `Figure_02_cv5_probability_response`: all-model Fare, IVT, Wait, and Reliability curves.
3. `Figure_03_cv5_direct_elasticity_vot`: Direct Fare elasticity and VOT-equivalent.
4. `Figure_04_qwen_fullgrid_response`: matched 2B/9B ZS/SFT response functions.
5. `Figure_05_qwen_response_span_sft_vs_zs`: signed response-span change after SFT.
6. `Figure_06_qwen_hard_choice_response_span`: strict `margin > 0` hard-choice span diagnostic.

Figures use marker, fill, and line style as well as colour. PDF and 600-dpi PNG
versions are generated from the same reviewed tables. `validation_summary.json`
and `manifest.csv` provide machine-readable QA and file hashes.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def _changes_vs_previous(prediction: pd.DataFrame, behaviour: pd.DataFrame) -> pd.DataFrame:
    old_root = REAL / "pdf" / "SECTION4_RESULTS_CASE_CV5_2B_9B_20260907"
    old_p = pd.read_csv(old_root / "Table_01_prediction_performance.csv").set_index("Model")
    old_e = pd.read_csv(old_root / "Table_02_direct_fare_response_interval.csv").set_index("Model")
    old_v = pd.read_csv(old_root / "Table_03_direct_vot_validity.csv").set_index("Model")
    new_p = prediction.set_index("Model")
    new_b = behaviour.set_index("Model")
    rows = []
    for model in [m for m in prediction.Model if m != "Human CV5 H-MNL"]:
        for old_col, new_col, metric in [
            ("AV-F1", "AV-F1 pooled", "AV-F1"),
            ("Accuracy", "Accuracy pooled", "Accuracy"),
            ("Macro-F1", "Macro-F1 pooled", "Macro-F1"),
            ("Log Loss", "Log loss pooled", "Log loss"),
        ]:
            old, new = float(old_p.loc[model, old_col]), float(new_p.loc[model, new_col])
            rows.append({"section": "prediction", "Model": model, "metric": metric,
                         "previous": old, "all_model_cv5": new, "delta": new - old})
    for model in BEHAVIOUR_ORDER:
        old_em = "Human reference" if model == "Human CV5 H-MNL" else model
        old_vm = "Human same-CF reference" if model == "Human CV5 H-MNL" else model
        old, new = float(old_e.loc[old_em, "Mean"]), float(new_b.loc[model, "Mean interval Fare elasticity"])
        rows.append({"section": "direct_behavior", "Model": model, "metric": "Mean interval Fare elasticity",
                     "previous": old, "all_model_cv5": new, "delta": new - old})
        old, new = float(old_v.loc[old_vm, "Population Direct VOT-eq."]), float(new_b.loc[model, "Population Direct VOT (kKRW/h)"])
        rows.append({"section": "direct_behavior", "Model": model, "metric": "Population Direct VOT (kKRW/h)",
                     "previous": old, "all_model_cv5": new, "delta": new - old})
    return pd.DataFrame(rows)


def _manifest() -> pd.DataFrame:
    rows = []
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "manifest.csv"):
        rows.append({"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "manifest.csv", index=False)
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use the completed Qwen 2B/9B ZS/SFT 33-scenario grids and write the full-grid package.",
    )
    args = parser.parse_args(argv)
    global OUT, FIG
    OUT = OUT_FULLGRID if args.full_grid else OUT_LEGACY
    FIG = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pool, test, grid = _read_inputs()
    tab = _load_tabular(pool)
    test_ids = set(test.case_id)
    tab_test = tab.loc[tab.case_id.isin(test_ids)].copy()
    sft = _load_sft_scenarios(test_ids)
    zs = _load_zeroshot_scenarios(test, grid, full_grid=args.full_grid)
    pred = pd.concat([tab_test, sft, zs], ignore_index=True)
    pred_for_prediction = pred
    new_zs_base = _load_new_qwen9b_zs_base()
    zs_audit = None
    grid_base_audit = None
    if new_zs_base is not None:
        zs_all_metrics, zs_audit = _qwen9b_zs_base_audit(zs, new_zs_base, test)
        zs_all_metrics.to_csv(OUT / "Table_S05_qwen9b_zs_all_cv5_cases.csv", index=False)
        if args.full_grid:
            grid_base_audit = {
                "official_test_pairs": zs_audit["official_test_pairs"],
                "max_abs_probability_difference": zs_audit["max_abs_probability_difference"],
                "mean_abs_probability_difference": zs_audit["mean_abs_probability_difference"],
                "hard_label_mismatches": zs_audit["hard_label_mismatches"],
                "full_grid_base_metrics": zs_audit["old_official_test_metrics"],
                "standalone_base_metrics": zs_audit["new_official_test_metrics"],
                "selected_source": "full_grid_base_for_internal_33_scenario_consistency",
            }
            (OUT / "QWEN9B_ZS_GRID_VS_STANDALONE_BASE.json").write_text(
                json.dumps(grid_base_audit, indent=2), encoding="utf-8"
            )
        else:
            (OUT / "QWEN9B_ZS_BASE_REPRODUCIBILITY.json").write_text(
                json.dumps(zs_audit, indent=2), encoding="utf-8"
            )
            replacement = new_zs_base.loc[new_zs_base.case_id.isin(test_ids),
                                          ["case_id", "scenario_id", "p", "hard", "fold", "system", "lane"]]
            pred_for_prediction = pd.concat([
                pred.loc[~(pred.system.eq("Qwen 9B ZS") & pred.scenario_id.eq("BASE"))],
                replacement,
            ], ignore_index=True)

    inventory = _inventory(pool, test, tab, pred)
    llm_inventory = _raw_llm_fullgrid_inventory(test, grid) if args.full_grid else None
    detailed, selected = _prediction_tables(pred_for_prediction, test)
    cases, behaviour, behaviour_by_fold = _behaviour_tables(pred, test, grid)
    response_scenarios = RESPONSE_SCENARIOS if args.full_grid else SCENARIOS
    response = _response_table(
        pred, grid.loc[grid.case_id.isin(test_ids)], scenarios=response_scenarios
    )
    sensitivity_cases, sensitivity_behaviour, sensitivity_by_fold = _behaviour_tables(
        pred, test, grid, sensitivity=True
    )
    sensitivity_response = _response_table(
        pred,
        grid.loc[grid.case_id.isin(test_ids)],
        sensitivity=True,
        scenarios=response_scenarios,
    )
    all_spans, qwen_contrast = _response_span_tables(response) if args.full_grid else (None, None)
    qwen_hard_response, qwen_hard_spans = (
        _qwen_hard_choice_tables(test) if args.full_grid else (None, None)
    )

    inventory.to_csv(OUT / "CV5_INPUT_INVENTORY.csv", index=False)
    if llm_inventory is not None:
        llm_inventory.to_csv(OUT / "QWEN_FULLGRID_INPUT_INVENTORY.csv", index=False)
    detailed.to_csv(OUT / "Table_S01_prediction_performance_all_lanes_and_folds.csv", index=False)
    selected.to_csv(OUT / "Table_01_prediction_performance_selected_lanes.csv", index=False)
    behaviour.to_csv(OUT / "Table_02_direct_behavior_summary.csv", index=False)
    cases.to_parquet(OUT / "Table_S02_direct_behavior_case.parquet", index=False)
    behaviour_by_fold.to_csv(OUT / "Table_S02b_direct_behavior_by_fold.csv", index=False)
    response.to_csv(OUT / "Table_S03_probability_response.csv", index=False)
    sensitivity_behaviour.to_csv(OUT / "Table_S04_direct_behavior_tuned_base_summary.csv", index=False)
    sensitivity_cases.to_parquet(OUT / "Table_S04_direct_behavior_tuned_base_case.parquet", index=False)
    sensitivity_by_fold.to_csv(OUT / "Table_S04_direct_behavior_tuned_base_by_fold.csv", index=False)
    sensitivity_response.to_csv(OUT / "Table_S04_probability_response_tuned_base.csv", index=False)
    if all_spans is not None and qwen_contrast is not None:
        all_spans.to_csv(OUT / "Table_03_response_span_all_models.csv", index=False)
        qwen_contrast.to_csv(OUT / "Table_04_qwen_sft_vs_zs_response_span.csv", index=False)
    if qwen_hard_response is not None and qwen_hard_spans is not None:
        qwen_hard_response.to_csv(OUT / "Table_05_qwen_hard_choice_response.csv", index=False)
        qwen_hard_spans.to_csv(OUT / "Table_06_qwen_hard_choice_response_span.csv", index=False)
    _changes_vs_previous(selected, behaviour).to_csv(
        OUT / "CHANGES_vs_CASE_CV5_2B_9B_20260907.csv", index=False
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlepad": 12})
    _plot_prediction(selected)
    _plot_response(response)
    _plot_behaviour(behaviour)
    if qwen_contrast is not None:
        _plot_qwen_response(response)
        _plot_qwen_span_contrast(qwen_contrast)
    if qwen_hard_spans is not None:
        _plot_qwen_hard_span_contrast(qwen_hard_spans)

    human_fold = behaviour_by_fold.loc[behaviour_by_fold.Model.eq("Human CV5 H-MNL")]
    checks = {
        "pool_cases": int(pool.case_id.nunique()),
        "pool_respondents": int(pool.respondent_id.nunique()),
        "official_test_cases": int(test.case_id.nunique()),
        "official_test_respondents": int(test.respondent_id.nunique()),
        "folds": sorted(map(int, pool.fold.unique())),
        "tabular_files": 70,
        "systems": int(inventory.system.nunique()),
        "complete_systems": int((inventory.official_test_cases.eq(3468) & inventory.scenarios.eq(33)).sum()),
        "duplicate_prediction_keys": int(inventory.duplicate_system_case_scenario.sum()),
        "missing_probabilities": int(inventory.missing_probability.sum()),
        "case_behavior_rows": int(len(cases)),
        "case_behavior_duplicate_keys": int(cases.duplicated(["system", "case_id"]).sum()),
        "full_grid_mode": bool(args.full_grid),
        "qwen_fullgrid_complete_cells": (
            None if llm_inventory is None else int(
                (
                    llm_inventory.rows.eq(3468 * 33)
                    & llm_inventory.exact_common_grid
                    & llm_inventory.exact_fold_assignment
                ).sum()
            )
        ),
        "qwen_fullgrid_total_rows": None if llm_inventory is None else int(llm_inventory.rows.sum()),
        "qwen_fullgrid_duplicate_keys": None if llm_inventory is None else int(llm_inventory.duplicate_keys.sum()),
        "qwen_fullgrid_missing_required": None if llm_inventory is None else int(llm_inventory.missing_required.sum()),
        "qwen_fullgrid_exact_margin_ties": (
            None if llm_inventory is None else int(llm_inventory.exact_margin_ties.sum())
        ),
        "qwen9b_zs_prediction_source": (
            "full_grid_base" if args.full_grid else "standalone_all_case_base"
        ),
        "new_qwen9b_zs_all_case_base_used_for_prediction": bool(new_zs_base is not None and not args.full_grid),
        "new_qwen9b_zs_official_hard_label_mismatches_vs_frozen": (
            None if args.full_grid or zs_audit is None else zs_audit["hard_label_mismatches"]
        ),
        "qwen9b_zs_fullgrid_vs_standalone_base_hard_label_mismatches": (
            None if not args.full_grid or grid_base_audit is None
            else grid_base_audit["hard_label_mismatches"]
        ),
        "data_validation_status": "PASS",
        "analytical_readiness": "ROBUSTNESS_ONLY_NOT_PRIMARY_PROMOTION_READY",
        "human_cv5_vot_fold_min": float(human_fold["Population Direct VOT (kKRW/h)"].min()),
        "human_cv5_vot_fold_max": float(human_fold["Population Direct VOT (kKRW/h)"].max()),
        "human_cv5_has_negative_vot_fold": bool((human_fold["Population Direct VOT (kKRW/h)"] < 0).any()),
        "warnings": [
            "Retrospective CV5 combines development and official-test respondents; it cannot replace the registered primary split.",
            "Human CV5 H-MNL VOT varies materially across folds and includes a negative fold estimate.",
            "The historical tabular 'unweighted' filename is tuned-base; SVM selected balancing in every fold.",
        ],
        "primary_analysis_replacement_allowed": False,
    }
    (OUT / "validation_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if args.full_grid:
        assert llm_inventory is not None and qwen_contrast is not None and qwen_hard_spans is not None
        _write_fullgrid_readme(
            inventory,
            llm_inventory,
            selected,
            behaviour,
            qwen_contrast,
            qwen_hard_spans,
            grid_base_audit,
        )
    else:
        _write_readme(inventory, selected, behaviour)
    manifest = _manifest()
    print(json.dumps(checks, indent=2))
    print(f"[ok] {OUT} ({len(manifest)} manifested files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
