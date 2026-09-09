# -*- coding: utf-8 -*-
"""CV5 전용 33조건 격자 - **전체 11,584 사례**(development + test) 를 덮는다.

**기존 산출물과 섞지 않는다.** p09 가 만드는 data/scenarios/test_scenario_grid.parquet
(test 3,468건, 114,444행)은 손대지 않고, 여기서는 cv5_scenario_grid.parquet 을 따로 쓴다.
주 실험(고정 분할)은 계속 옛 격자를 읽고, CV5 실험만 이 격자를 읽는다.

p09 와 다른 점은 **파티션 필터 하나뿐**이다:
    p09   raw[restricted_mask(raw) & (raw.partition == "test")]   / build_design("test")
    여기  raw[restricted_mask(raw)]                               / build_design(None)
조건 정의(사다리·신뢰도·번들)와 support 분류는 한 글자도 바꾸지 않는다 - 바꾸면 CV5 결과가
주 실험과 다른 조건을 재게 된다.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from common import (CONFIG, DATA, PROCESSED, REPORTS, ensure_dirs, load_config,
                    load_panel, restricted_mask, write_json)
from design import build_design
from p04_support_map import ATTRS, EXTRAPOLATION, INVALID, PRIMARY, SupportClassifier

SCEN = DATA / "scenarios"
ATTR_KEY = {"fare": "fare", "ride": "ride", "wait": "wait"}


def load_ladders():
    with open(CONFIG / "trc_band_ladders.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run() -> dict:
    print("\n=== Phase 9. 33조건 시나리오 격자 ===")
    cfg = load_config()
    lad = load_ladders()
    sm = pd.read_parquet(PROCESSED / "trc_support_map.parquet")
    clf = SupportClassifier(sm)
    ensure_dirs(SCEN, REPORTS)

    d = build_design(None)
    si = d.service_idx
    # 격자는 **원 단위**로 만든다. 설계행렬은 거리대 내 중심화가 적용돼 있어 그 값을 쓰면
    # 해석 가능한 요금·분이 아니고, 중심화 이전에 만든 격자와 섞이면 조인이 조용히 어긋난다.
    # 진실 계산 쪽(p10)이 band_center 를 빼서 설계 척도로 변환한다.
    raw = load_panel()
    raw = raw[restricted_mask(raw)]
    raw = raw.sort_values(["respondent_id", "task_id"]).reset_index(drop=True)
    assert list(raw.case_id) == list(d.case_id), "격자와 설계행렬의 case 순서가 다르다"
    base = {k: raw[ATTRS[k]].to_numpy(float) for k in ATTRS}
    # AV 신뢰도(정상 완료 가능성 %). 서비스 속성 셋과 달리 거리대 내 중심화를 하지 않고
    # 표준화된 통제변수로 들어가므로, 격자에는 원 단위(99.99~100)로 담고 p10 이 환산한다.
    base["rel"] = raw["av_service_reliability_pct"].to_numpy(float)
    n = d.n
    rows = []

    def add(sid, stype, f, r, w, attr="", amount=np.nan, rel=None):
        rows.append(pd.DataFrame({
            "case_id": d.case_id, "respondent_id": d.respondent_ids[d.respondent],
            "distance_band": d.band, "scenario_id": sid, "scenario_type": stype,
            "edit_attribute": attr, "edit_amount": amount,
            "fare_base": base["fare"], "ride_base": base["ride"], "wait_base": base["wait"],
            "fare_cf": f, "ride_cf": r, "wait_cf": w,
            "rel_base": base["rel"], "rel_cf": (base["rel"] if rel is None else rel),
        }))

    add("BASE", "baseline", base["fare"].copy(), base["ride"].copy(), base["wait"].copy())

    # ---- 단일 속성 편집 (거리대별 사다리) --------------------------------
    for attr in ATTRS:
        steps = np.array([lad["ladders"][attr][b] for b in d.band])   # (n, 6)
        for j in range(steps.shape[1]):
            delta = steps[:, j]
            f, r, w = (base["fare"].copy(), base["ride"].copy(), base["wait"].copy())
            {"fare": f, "ride": r, "wait": w}[attr][:] += delta
            sign = "M" if delta[0] < 0 else "P"       # 사다리는 -3h..-h, +h..+3h 순
            sid = "%s_%s%d" % (attr[0].upper(), sign, [3, 2, 1, 1, 2, 3][j])
            add(sid, "single_edit", f, r, w, attr, np.nan)
            rows[-1]["edit_amount"] = delta

    # ---- AV 신뢰도 사다리 (A_M3 .. A_P3) ---------------------------------
    #
    # 요금·시간과 달리 거리대와 무관하게 설계 범위가 같다(99.99~100). 스텝 하나를 쓴다.
    # 100 을 넘거나 99.99 아래로 내려가는 칸이 생기는데, 그건 설문이 물어본 적 없는
    # 구간이므로 아래 support 분류가 INVALID/EXTRAPOLATION 으로 잡는다.
    rel_step = float(lad["steps"]["rel"]["D1"])          # 거리대 무관, 아무 값이나 같다
    for j, mult in enumerate((-3, -2, -1, 1, 2, 3)):
        v = base["rel"] + mult * rel_step
        sid = "A_%s%d" % ("M" if mult < 0 else "P", abs(mult))
        add(sid, "single_edit", base["fare"].copy(), base["ride"].copy(),
            base["wait"].copy(), "rel", np.nan, rel=v)
        rows[-1]["edit_amount"] = mult * rel_step

    # ---- 정책 번들 ------------------------------------------------------
    for name, sc in cfg["scenarios"].items():
        if not isinstance(sc, dict):
            continue
        f, r, w = (base["fare"].copy(), base["ride"].copy(), base["wait"].copy())
        if "fare_pct" in sc:
            f *= (1.0 + float(sc["fare_pct"]) / 100.0)
        if "ride_min" in sc:
            r += float(sc["ride_min"])
        if "wait_min" in sc:
            w += float(sc["wait_min"])
        add(name, "bundle", f, r, w, "", np.nan)

    grid = pd.concat(rows, ignore_index=True)

    # ---- support / 유효성 ------------------------------------------------
    # 신뢰도는 support map 에 없다(거리대별 셀이 아니다). 범위를 설정에서 읽는다.
    #
    # **설계범위 밖과 물리적 불가능을 구분한다.** 명세 5절의 세 등급은
    #   in-support / out-of-support **feasible** / invalid(물리적으로 불가능) 이다.
    # 신뢰도 99.987% 는 설문이 물어본 적 없을 뿐 완벽히 가능한 값이므로 EXTRAPOLATION 이고,
    # 100% 를 넘는 값만 INVALID 다. 전에는 둘을 한데 묶어 INVALID 로 처리했고, 그 결과
    # 신뢰도 사다리의 31%(6,447행)가 버려지고 EXTRAPOLATION 행이 하나도 없어 외삽
    # 스트레스 테스트를 신뢰도에서만 돌릴 수 없었다. 게다가 WTP 의 중앙차분이 쓰는
    # A_M1 이 32% INVALID 여서 자율주행 고유 지표가 오염된 다리로 계산됐다.
    sup = lad["rel_support"]
    REL_LO, REL_HI = float(sup["design_min"]), float(sup["design_max"])
    PHYS_LO, PHYS_HI = float(sup["physical_min"]), float(sup["physical_max"])

    def rel_tier(v):
        if v > PHYS_HI + 1e-9 or v < PHYS_LO - 1e-9:
            return INVALID                  # 확률이 100%를 넘거나 음수 - 존재할 수 없다
        if v < REL_LO - 1e-9 or v > REL_HI + 1e-9:
            return EXTRAPOLATION            # 설계범위 밖이지만 물리적으로 가능
        return PRIMARY

    tier = np.empty(len(grid), dtype=object)
    for i, (b, f, r, w, rl) in enumerate(zip(grid.distance_band, grid.fare_cf,
                                             grid.ride_cf, grid.wait_cf, grid.rel_cf)):
        t = [clf.classify(b, "fare", f), clf.classify(b, "ride", r),
             clf.classify(b, "wait", w), rel_tier(rl)]
        tier[i] = (INVALID if INVALID in t else
                   (EXTRAPOLATION if EXTRAPOLATION in t else PRIMARY))
    grid["analysis_tier"] = tier
    grid["within_support"] = grid.analysis_tier == PRIMARY
    grid["physically_valid"] = grid.analysis_tier != INVALID

    grid.to_parquet(SCEN / "cv5_scenario_grid.parquet", index=False)

    n_cond = grid.scenario_id.nunique()
    res = {"cases": n, "conditions": int(n_cond), "rows": len(grid),
           "expected_rows": int(n * n_cond),
           "tier_counts": grid.analysis_tier.value_counts().to_dict(),
           "by_type": grid.groupby("scenario_type").size().to_dict(),
           "invalid_by_scenario": grid[~grid.physically_valid]
                                  .scenario_id.value_counts().to_dict()}
    print("  cases %d × 조건 %d = %d 행" % (n, n_cond, len(grid)))
    print("  tier: %s" % res["tier_counts"])
    assert n_cond == 33, "조건 수가 33 이 아니다: %d" % n_cond   # 27 + 신뢰도 6
    assert len(grid) == n * 33
    print("  [PASS] 33조건 × %d cases = %d 행" % (n, len(grid)))
    write_json(REPORTS / "scenario_grid.json", res)
    return res


if __name__ == "__main__":
    run()
