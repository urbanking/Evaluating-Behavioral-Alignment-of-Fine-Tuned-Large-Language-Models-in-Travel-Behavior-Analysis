# -*- coding: utf-8 -*-
"""Phase 2. 제한 AV-vs-Retain 표본 확정 — Gate 1.

탐색이 아니라 검증이다. 필터 규칙은
methods/fine_tuning/prepare_av_pairwise_dataset.py 의 build_rows 와 동일하며,
동결된 EXPECTED 값과 정확히 일치해야 한다.

출력
  data/manifests/restricted_{dev,test}_cases.csv
  data/manifests/restricted_{dev,test}_respondents.csv
  reports/data_audit/restricted_domain_gate1.json
"""

from __future__ import annotations

import pandas as pd

from common import (
    MANIFESTS,
    REPORTS,
    check,
    ensure_dirs,
    load_panel,
    restricted_mask,
    transition_state,
    write_json,
)

EXPECTED = {
    "development": {"respondents": 1518, "cases": 8116, "retain": 6588, "AV": 1528,
                    "other_non_AV": 1028},
    # test 의 other_non_AV = 456 (= validation 234 + test 222, 동결 EXPECTED 합).
    # 3,924 - 3,468 = 456 으로 산술도 맞는다. 이전 문서의 420 은 split70_seed42 기준
    # 계산을 잘못 옮긴 값이었고 Gate 1 이 이를 잡아냈다.
    "test":        {"respondents": 651,  "cases": 3468, "retain": 2858, "AV": 610,
                    "other_non_AV": 456},
}


def run() -> dict:
    print("\n=== Phase 2. 제한 도메인 확정 (Gate 1) ===")
    df = load_panel()
    df["transition_state"] = transition_state(df)
    df["in_domain"] = restricted_mask(df)
    df["human_label"] = (df["future_mode_label"].astype(str) == "AV").astype(int)
    df["exclusion_reason"] = df.apply(
        lambda r: "" if r.in_domain else "switch_to_other_non_AV", axis=1)
    df["inclusion_reason"] = df["transition_state"].where(df.in_domain, "")

    ensure_dirs(MANIFESTS, REPORTS)
    out = {}

    for part in ("development", "test"):
        sub = df[df.partition == part]
        dom = sub[sub.in_domain]
        counts = sub.transition_state.value_counts().to_dict()
        exp = EXPECTED[part]

        print("  -- %s" % part)
        check("%s 제한 응답자" % part, int(dom.respondent_id.nunique()),
              exp["respondents"], gate="Gate 1")
        check("%s 제한 cases" % part, len(dom), exp["cases"], gate="Gate 1")
        check("%s retain" % part, int(counts.get("retain", 0)), exp["retain"], gate="Gate 1")
        check("%s AV" % part, int(counts.get("AV", 0)), exp["AV"], gate="Gate 1")
        check("%s 제외(other_non_AV)" % part, int(counts.get("other_non_AV", 0)),
              exp["other_non_AV"], gate="Gate 1")

        cm = dom[["case_id", "respondent_id", "task_id", "partition", "human_label",
                  "transition_state", "inclusion_reason"]].copy()
        cm.to_csv(MANIFESTS / ("restricted_%s_cases.csv" % part),
                  index=False, encoding="utf-8-sig")

        rm = (dom.groupby("respondent_id")
                 .agg(n_cases=("case_id", "size"), n_av=("human_label", "sum"))
                 .reset_index())
        rm["partition"] = part
        rm.to_csv(MANIFESTS / ("restricted_%s_respondents.csv" % part),
                  index=False, encoding="utf-8-sig")

        out[part] = {
            "respondents": int(dom.respondent_id.nunique()),
            "cases": len(dom),
            "counts": {k: int(v) for k, v in counts.items()},
            "av_share_case": round(float(dom.human_label.mean()), 4),
            "respondents_with_zero_eligible_case":
                int(sub.respondent_id.nunique() - dom.respondent_id.nunique()),
        }
        print("     AV 비율(case 기준) %.4f, 유효 case 0개 응답자 %d 명"
              % (out[part]["av_share_case"],
                 out[part]["respondents_with_zero_eligible_case"]))

    dev_ids = set(pd.read_csv(MANIFESTS / "restricted_development_respondents.csv")
                  ["respondent_id"])
    test_ids = set(pd.read_csv(MANIFESTS / "restricted_test_respondents.csv")
                   ["respondent_id"])
    check("제한 표본 간 응답자 중복", len(dev_ids & test_ids), 0, gate="Gate 1")

    print("  [GATE 1 PASS] 동결값과 완전 일치")
    write_json(REPORTS / "restricted_domain_gate1.json",
               {"gate": "Gate 1", "status": "PASS", "expected": EXPECTED, "observed": out})
    return out


if __name__ == "__main__":
    run()
