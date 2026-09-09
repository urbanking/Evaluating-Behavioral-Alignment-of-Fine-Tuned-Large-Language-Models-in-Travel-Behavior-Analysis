# -*- coding: utf-8 -*-
"""Phase 3. pro / retro 정보조건 view 생성 — Gate 2.

두 view 는 라벨과 task 값이 완전히 같고 태도 field 유무만 다르다.
DGP 진실 생성에는 pro 스키마만 쓴다(태도를 넣으면 행동세계가 정보조건별로 갈라진다).

출력
  data/views/{pro,retro}/baseline_cases.parquet
  data/views/{pro,retro}/prompt_records.jsonl
  reports/data_audit/information_conditions_gate2.json
"""

from __future__ import annotations

import json

import pandas as pd

from common import (
    DATA,
    REPORTS,
    VIEWS,
    check,
    ensure_dirs,
    load_config,
    load_panel,
    load_schemas,
    restricted_mask,
    write_json,
)

ID_COLS = ["case_id", "respondent_id", "task_id", "partition"]
LABEL_COL = "human_label"


def flatten(node) -> list[str]:
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "include":
                continue
            out += flatten(v)
    elif isinstance(node, list):
        for v in node:
            out += flatten(v)
    elif isinstance(node, str):
        out.append(node)
    return out


def build_view(df: pd.DataFrame, cols: list[str], binary_cols: list[str],
               suffix: str) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns and c not in binary_cols]
    view = df[ID_COLS + [LABEL_COL] + keep].copy()
    for c in binary_cols:
        if c in df.columns and c in cols:
            view[c + suffix] = df[c].notna().astype(int)
    return view


def run() -> dict:
    print("\n=== Phase 3. 정보조건 생성 (Gate 2) ===")
    cfg, sch = load_config(), load_schemas()
    df = load_panel()
    # 통행 O/D 와 거주지 공간속성 (p01b). 패널이 거주지만 들고 있어 통행 도착지가 빠져
    # 있었다 — 도착지는 거주지와 23.7% 만 같아 실질적인 새 정보다.
    od = DATA / "external" / "spatial_od.parquet"
    if od.exists():
        sp = pd.read_parquet(od)
        before = len(df)
        df = df.merge(sp, on="respondent_id", how="left")
        assert len(df) == before, "OD join 이 행을 늘렸다 — respondent_id 중복 확인"
        print("  OD·공간속성 %d개 열 결합" % (len(sp.columns) - 1))
    else:
        print("  [warn] %s 없음 — OD·공간속성 없이 진행" % od.name)
    df["human_label"] = (df["future_mode_label"].astype(str) == "AV").astype(int)
    df = df[restricted_mask(df)].reset_index(drop=True)
    print("  제한 도메인 %d cases" % len(df))

    tf = sch["transforms"]["multi_select_to_binary"]
    bin_cols, suffix = tf["columns"], tf["suffix"]

    pro_cols = flatten(sch["schemas"]["pro"])
    retro_cols = pro_cols + flatten(sch["schemas"]["retro"]["additional_attitudes"])

    missing = sorted({c for c in retro_cols if c not in df.columns})
    if missing:
        print("  [warn] 패널에 없는 스키마 변수: %s" % missing)

    views = {}
    for name, cols in (("pro", pro_cols), ("retro", retro_cols)):
        v = build_view(df, cols, bin_cols, suffix)
        d = VIEWS / name
        ensure_dirs(d)
        v.to_parquet(d / "baseline_cases.parquet", index=False)
        views[name] = v
        print("  [ok] views/%-5s baseline_cases.parquet  %d 행 × %d 열"
              % (name, *v.shape))

    pro_v, retro_v = views["pro"], views["retro"]

    # ---- Gate 2 -------------------------------------------------------
    check("행 수 일치", len(pro_v), len(retro_v), gate="Gate 2")
    check("case_id 순서 일치",
          bool((pro_v.case_id.values == retro_v.case_id.values).all()), True, gate="Gate 2")
    check("라벨 일치",
          bool((pro_v[LABEL_COL].values == retro_v[LABEL_COL].values).all()),
          True, gate="Gate 2")
    for c in ("av_cost_1000won", "av_travel_time_min", "av_wait_time_min"):
        check("task 값 일치: %s" % c,
              bool((pro_v[c].values == retro_v[c].values).all()), True, gate="Gate 2")

    extra = [c for c in retro_v.columns if c not in pro_v.columns]
    expected_extra = sorted(
        [c for c in flatten(sch["schemas"]["retro"]["additional_attitudes"])
         if c not in bin_cols and c in df.columns]
        + [c + suffix for c in bin_cols if c in df.columns])
    check("retro 추가 열 개수", len(extra), len(expected_extra), gate="Gate 2")
    check("retro 추가 열이 태도 변수뿐", sorted(extra) == expected_extra, True, gate="Gate 2")
    print("     추가 열: %s" % sorted(extra))

    # ---- prompt records ------------------------------------------------
    for name, v in views.items():
        path = VIEWS / name / "prompt_records.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in v.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print("  [ok] views/%-5s prompt_records.jsonl" % name)

    print("  [GATE 2 PASS] 라벨·task 동일, 차이는 태도 열뿐")
    result = {
        "gate": "Gate 2", "status": "PASS",
        "cases": len(pro_v),
        "pro_features": int(pro_v.shape[1] - len(ID_COLS) - 1),
        "retro_features": int(retro_v.shape[1] - len(ID_COLS) - 1),
        "attitude_columns_added": sorted(extra),
        "schema_vars_absent_from_panel": missing,
    }
    write_json(REPORTS / "information_conditions_gate2.json", result)
    return result


if __name__ == "__main__":
    run()
