# -*- coding: utf-8 -*-
"""Phase 1. 원본 서울 AV SP 데이터 정리.

출력
  data/processed/trc_respondent_profiles.parquet
  data/processed/trc_baseline_cases.parquet
  reports/data_audit/raw_panel_summary.json
  reports/data_audit/missingness.csv
  reports/data_audit/degenerate_columns.csv
"""

from __future__ import annotations

import pandas as pd

from common import (
    PROCESSED,
    REPORTS,
    check,
    ensure_dirs,
    human_label,
    load_panel,
    transition_state,
    write_json,
)

RESPONDENT_LEVEL = [
    "respondent_id", "partition", "split", "block_ID", "BR_label",
    "SQ1_1", "SQ1_2", "SQ2", "SQ3", "gender_label", "age", "age_group",
    "QQ1", "QQ2_1", "QQ2_2", "QQ3", "QQ4", "QQ5", "QQ6", "QQ7", "main_mode_label",
    "D1", "D2", "D3", "D4_1", "D4_2", "D5", "D6", "D7", "D8", "D9",
    "car_ownership_label", "license_label", "car_access_label",
    "A1_1", "A1_2", "A1_3", "A1_4", "A1_5", "A1_6", "A1_7", "A1_8", "A1_9", "A1_10",
    "T1", "T4", "S1", "S2",
    "S3_1", "S3_2", "S3_3", "S3_4", "S3_5", "S3_6", "S3_7",
]

CASE_LEVEL = [
    "case_id", "respondent_id", "task_id", "partition",
    "distance_code", "distance_band", "distance_label", "block_ID", "BR_label",
    "current_mode_label", "future_mode_label",
    "av_cost_1000won", "av_travel_time_min", "av_wait_time_min",
    "av_transfer_count", "av_crowding_raw",
    "av_framing_percent", "av_framing_attribute",
    "choice_set_size", "available_modes",
    "pt_travel_time_min", "pt_wait_time_min", "pt_cost_1000won",
    "pt_transfer_count", "pt_crowding_raw",
    "car_travel_time_min", "car_wait_time_min", "car_cost_1000won",
    "car_transfer_count", "car_crowding_raw",
    "pm_travel_time_min", "pm_wait_time_min", "pm_cost_1000won",
    "pm_transfer_count", "pm_crowding_raw",
    "walk_travel_time_min", "walk_wait_time_min", "walk_cost_1000won",
    "walk_transfer_count", "walk_crowding_raw",
]


def run() -> dict:
    print("\n=== Phase 1. 원본 데이터 정리 ===")
    df = load_panel()
    ensure_dirs(PROCESSED, REPORTS)

    df["human_label"] = human_label(df)
    df["transition_state"] = transition_state(df)

    # --- 무결성 --------------------------------------------------------
    check("case_id 유일", int(df["case_id"].duplicated().sum()), 0, gate="Phase 1")
    check("응답자 수", int(df["respondent_id"].nunique()), 2178, gate="Phase 1")
    check("전체 cases", len(df), 13068, gate="Phase 1")

    per = df.groupby("respondent_id").size()
    check("응답자당 task 수(최소=최대=6)",
          (int(per.min()), int(per.max())), (6, 6), gate="Phase 1")

    dev, test = df[df.partition == "development"], df[df.partition == "test"]
    check("development 응답자", int(dev.respondent_id.nunique()), 1524, gate="Phase 1")
    check("development cases", len(dev), 9144, gate="Phase 1")
    check("test 응답자", int(test.respondent_id.nunique()), 654, gate="Phase 1")
    check("test cases", len(test), 3924, gate="Phase 1")

    overlap = set(dev.respondent_id) & set(test.respondent_id)
    check("분할 간 응답자 중복", len(overlap), 0, gate="Phase 1")

    # --- 단위 ----------------------------------------------------------
    fare = df["av_cost_1000won"]
    check("요금 최소 (kKRW)", round(float(fare.min()), 2), 4.0, gate="Phase 1")
    check("요금 최대 (kKRW)", round(float(fare.max()), 2), 28.5, gate="Phase 1")
    check("요금 중앙값 (kKRW)", round(float(fare.median()), 2), 6.0, gate="Phase 1")
    print("  [info] ride %.0f-%.0f 분, wait %.0f-%.0f 분"
          % (df.av_travel_time_min.min(), df.av_travel_time_min.max(),
             df.av_wait_time_min.min(), df.av_wait_time_min.max()))

    # --- 퇴화 컬럼 (상수 / 전량 결측) -----------------------------------
    deg = []
    for c in df.columns:
        nn = int(df[c].notna().sum())
        nu = int(df[c].nunique(dropna=True))
        if nn == 0:
            deg.append({"column": c, "issue": "all_null", "n_unique": 0})
        elif nu == 1:
            deg.append({"column": c, "issue": "constant",
                        "n_unique": 1, "value": str(df[c].dropna().iloc[0])})
    deg_df = pd.DataFrame(deg)
    deg_df.to_csv(REPORTS / "degenerate_columns.csv", index=False, encoding="utf-8-sig")
    print("  [info] 퇴화 컬럼 %d 개 -> reports/data_audit/degenerate_columns.csv"
          % len(deg_df))
    for r in deg:
        print("         %-24s %s" % (r["column"], r["issue"]))

    # --- 결측 ----------------------------------------------------------
    miss = (df.isna().mean().rename("missing_rate").reset_index()
            .rename(columns={"index": "column"}))
    miss = miss[miss.missing_rate > 0].sort_values("missing_rate", ascending=False)
    miss.to_csv(REPORTS / "missingness.csv", index=False, encoding="utf-8-sig")
    print("  [info] 결측 있는 컬럼 %d 개" % len(miss))

    # --- 저장 ----------------------------------------------------------
    resp_cols = [c for c in RESPONDENT_LEVEL if c in df.columns]
    resp = df[resp_cols].drop_duplicates("respondent_id").reset_index(drop=True)
    if len(resp) != 2178:
        raise SystemExit("응답자 수준 변수 중 task 마다 달라지는 것이 있다: %d 행" % len(resp))
    resp.to_parquet(PROCESSED / "trc_respondent_profiles.parquet", index=False)

    case_cols = [c for c in CASE_LEVEL if c in df.columns] + [
        "human_label", "transition_state", "scenario_id"]
    cases = df[case_cols].reset_index(drop=True)
    cases.to_parquet(PROCESSED / "trc_baseline_cases.parquet", index=False)
    print("  [ok] trc_respondent_profiles.parquet  %d 행 × %d 열" % resp.shape)
    print("  [ok] trc_baseline_cases.parquet       %d 행 × %d 열" % cases.shape)

    summary = {
        "respondents": int(df.respondent_id.nunique()),
        "cases": len(df),
        "development": {"respondents": int(dev.respondent_id.nunique()), "cases": len(dev)},
        "test": {"respondents": int(test.respondent_id.nunique()), "cases": len(test)},
        "split_source": "parquet split column (validation + test -> test)",
        "fare_unit": "kKRW",
        "fare": {"min": float(fare.min()), "max": float(fare.max()),
                 "median": float(fare.median())},
        "transition_state_counts": df.transition_state.value_counts().to_dict(),
        "degenerate_columns": deg,
    }
    write_json(REPORTS / "raw_panel_summary.json", summary)
    return summary


if __name__ == "__main__":
    run()
