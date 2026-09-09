# -*- coding: utf-8 -*-
"""Build an add-only Section 4 score bank with both Qwen SFT cells replaced by CV5 predictions.

The source bank is never modified. Qwen 2B and 9B zero-shot rows stay unchanged.
For each SFT scale, every official test case is scored by the adapter for the
fold that holds out that respondent. The output retains the original score-bank
schema and adds a provenance ``fold`` column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRC = Path(r"C:\SP_LLM\TRC")
REAL = TRC / "Real_exp"
DEFAULT_SRC = REAL / "data" / "06_predictions_llm" / "llm_scores.parquet"
DEFAULT_DST = REAL / "data" / "06_predictions_llm" / "llm_scores_cv5_2b_9b.parquet"
DEFAULT_TEST = REAL / "data" / "03_model_inputs" / "HUMAN_retro_test.parquet"
DEFAULT_CV5_2B = TRC / "predictions" / "cv5_llm"
DEFAULT_CV5_9B = TRC / "predictions" / "cv5_llm_grid"
KEYS = ["case_id", "scenario_id"]
VALUE_COLUMNS = [
    "q_av", "q_keep", "margin", "p_av_raw", "p_keep_raw",
    "candidate_mass", "raw_argmax", "free_argmax",
]


def load_cv5_2b(root: Path, test_ids: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        frame = pd.read_parquet(root / f"qwen_mid_f{fold}_scenarios.parquet")
        frame["case_id"] = frame.case_id.astype(str)
        frame = frame.loc[frame.case_id.isin(test_ids)].copy()
        frame["fold"] = fold
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def load_cv5_9b(root: Path, test_ids: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        frame = pd.read_parquet(root / f"qwen_f{fold}.parquet")
        frame["case_id"] = frame.case_id.astype(str)
        unexpected = set(frame.case_id) - test_ids
        assert not unexpected, f"9B fold {fold} contains {len(unexpected)} non-test cases"
        frame["fold"] = fold
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def validate_cv(frame: pd.DataFrame, model: str, test_ids: set[str]) -> None:
    missing = set(VALUE_COLUMNS + KEYS + ["fold"]) - set(frame.columns)
    assert not missing, f"{model}: missing columns {sorted(missing)}"
    assert len(frame) == 3_468 * 33, (model, len(frame))
    assert frame.case_id.nunique() == 3_468
    assert frame.scenario_id.nunique() == 33
    assert set(frame.case_id) == test_ids
    assert not frame.duplicated(KEYS).any()
    assert frame[VALUE_COLUMNS].notna().all().all()
    assert set(frame.fold.astype(int)) == {1, 2, 3, 4, 5}


def bank_rows(frame: pd.DataFrame, old: pd.DataFrame, model: str) -> pd.DataFrame:
    template = old.iloc[0]
    new = pd.DataFrame({
        "case_id": frame.case_id,
        "scenario_id": frame.scenario_id,
        "q_av": frame.q_av,
        "q_keep": frame.q_keep,
        "margin": frame.margin,
        "p_av_raw": frame.p_av_raw,
        "p_keep_raw": frame.p_keep_raw,
        "candidate_mass": frame.candidate_mass,
        "raw_argmax": frame.raw_argmax,
        "free_argmax": frame.free_argmax,
        "world": template["world"],
        "information_condition": template["information_condition"],
        "model_family": template["model_family"],
        "model_state": template["model_state"],
        "model": model,
        "seed": template["seed"],
        "calibrated_probability": np.nan,
        "pred_label": frame.raw_argmax,
        "fold": frame.fold.astype(int),
    })
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--cv5-2b", type=Path, default=DEFAULT_CV5_2B)
    parser.add_argument("--cv5-9b", type=Path, default=DEFAULT_CV5_9B)
    args = parser.parse_args()

    bank = pd.read_parquet(args.src)
    bank["case_id"] = bank.case_id.astype(str)
    test_ids = set(pd.read_parquet(args.test, columns=["case_id"]).case_id.astype(str))
    assert len(test_ids) == 3_468

    replacements: dict[str, pd.DataFrame] = {
        "qwen_mid_sft": load_cv5_2b(args.cv5_2b, test_ids),
        "qwen_sft": load_cv5_9b(args.cv5_9b, test_ids),
    }
    new_parts: list[pd.DataFrame] = []
    for model, frame in replacements.items():
        validate_cv(frame, model, test_ids)
        old = bank.loc[bank.model.eq(model)].copy()
        assert len(old) == 3_468 * 33
        assert set(map(tuple, old[KEYS].to_numpy())) == set(map(tuple, frame[KEYS].to_numpy()))
        new_parts.append(bank_rows(frame, old, model))

    keep = bank.loc[~bank.model.isin(replacements)].copy()
    keep["fold"] = np.nan
    columns = list(bank.columns) + ["fold"]
    out = pd.concat([keep] + [part[columns] for part in new_parts], ignore_index=True)
    out = out.sort_values(["model", "scenario_id", "case_id"]).reset_index(drop=True)

    assert len(out) == len(bank) == 457_776
    coverage = out.groupby("model").agg(rows=("case_id", "size"), cases=("case_id", "nunique"), scenarios=("scenario_id", "nunique"))
    assert coverage.rows.eq(114_444).all()
    assert coverage.cases.eq(3_468).all()
    assert coverage.scenarios.eq(33).all()
    assert not out.duplicated(["model"] + KEYS).any()

    original_other = bank.loc[~bank.model.isin(replacements)].set_index(["model"] + KEYS).sort_index()
    output_other = out.loc[~out.model.isin(replacements)].set_index(["model"] + KEYS).sort_index()
    assert np.array_equal(original_other.margin.to_numpy(), output_other.margin.to_numpy())

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.dst, index=False)

    print(f"written: {args.dst}")
    print(coverage.to_string())
    for model in replacements:
        base = out.loc[out.model.eq(model) & out.scenario_id.eq("BASE")]
        print(model, "BASE fold cases", base.groupby("fold").case_id.nunique().astype(int).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
