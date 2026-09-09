# -*- coding: utf-8 -*-
"""Phase 4b. 거리대별 사다리 스텝 산출.

고정 절대 사다리(fare ±1/±2/±3 kKRW 등)는 SP 설계 범위보다 넓어서 D1 은 in-support 편집이
0 건이었다. 대신 각 거리대의 설계 범위에서 스텝 h 를 유도한다.

    h(band, attr) = round_to_grid( (max - min) / 3 )
    ladder        = {-3h, -2h, -h, +h, +2h, +3h}

편집 개수(속성당 6개), 곡선 점수(7점), 총 조건 수(27), 시나리오 행 수(93,636)는 그대로다.
바뀌는 것은 스텝 크기와 유한차분 분모뿐이다.

    D_a p = ( p(a + h) - p(a - h) ) / (2h)

출력
  configs/trc_band_ladders.yaml
  reports/data_audit/band_ladder_coverage.json
"""

from __future__ import annotations

import pandas as pd
import yaml

from common import CONFIG, PROCESSED, REPORTS, load_panel, restricted_mask, write_json
from p04_support_map import ATTRS, SupportClassifier, PRIMARY, EXTRAPOLATION, INVALID

# 스텝을 붙일 격자. 요금은 0.25 kKRW(250원), 시간은 1분.
GRID = {"fare": 0.25, "ride": 1.0, "wait": 1.0}


def round_to_grid(x: float, grid: float) -> float:
    return max(grid, round(x / grid) * grid)


def run() -> dict:
    print("\n=== Phase 4b. 거리대별 사다리 산출 ===")
    sm = pd.read_parquet(PROCESSED / "trc_support_map.parquet")
    clf = SupportClassifier(sm)

    ladders: dict[str, dict[str, list[float]]] = {}
    steps: dict[str, dict[str, float]] = {}
    print("  %-5s %-4s %-14s %6s   %s" % ("attr", "band", "design range", "h", "ladder"))
    for attr in ATTRS:
        ladders[attr], steps[attr] = {}, {}
        for r in sm[sm.attribute == attr].sort_values("distance_band").itertuples():
            width = r.maximum - r.minimum
            h = round_to_grid(width / 3.0, GRID[attr])
            lad = [round(-3 * h, 4), round(-2 * h, 4), round(-h, 4),
                   round(h, 4), round(2 * h, 4), round(3 * h, 4)]
            ladders[attr][r.distance_band] = lad
            steps[attr][r.distance_band] = h
            print("  %-5s %-4s [%5.1f,%5.1f] %6.2f   %s"
                  % (attr, r.distance_band, r.minimum, r.maximum, h, lad))

    # ---- 커버리지 확인 -------------------------------------------------
    df = load_panel()
    df = df[restricted_mask(df) & (df.partition == "test")]
    cov, zero = [], []
    print("\n  test 편집의 계층 분포")
    for attr, col in ATTRS.items():
        tot = {PRIMARY: 0, EXTRAPOLATION: 0, INVALID: 0}
        for band, g in df.groupby("distance_band"):
            c = {PRIMARY: 0, EXTRAPOLATION: 0, INVALID: 0}
            for v in g[col].values:
                for d in ladders[attr][band]:
                    c[clf.classify(band, attr, float(v) + d)] += 1
            n = sum(c.values())
            row = {"attribute": attr, "distance_band": band,
                   "step": steps[attr][band],
                   "primary": c[PRIMARY], "extrapolation": c[EXTRAPOLATION],
                   "invalid": c[INVALID], "primary_share": round(c[PRIMARY] / n, 4)}
            cov.append(row)
            if c[PRIMARY] == 0:
                zero.append(row)
            for k in tot:
                tot[k] += c[k]
        n = sum(tot.values())
        print("  %-5s %6d건 -> PRIMARY %5d (%.1f%%) / EXTRAP %5d / INVALID %4d"
              % (attr, n, tot[PRIMARY], 100 * tot[PRIMARY] / n,
                 tot[EXTRAPOLATION], tot[INVALID]))

    if zero:
        print("  [FAIL] PRIMARY 0 인 셀이 남았다: %s"
              % [(z["attribute"], z["distance_band"]) for z in zero])
    else:
        print("  [PASS] 모든 (속성, 거리대) 셀에서 in-support 편집이 존재한다")

    by_band = pd.DataFrame(cov)
    worst = by_band.nsmallest(3, "primary_share")[
        ["attribute", "distance_band", "step", "primary_share"]]
    print("  최저 커버리지 3개:")
    for r in worst.itertuples():
        print("    %-5s %-4s h=%.2f  primary %.1f%%"
              % (r.attribute, r.distance_band, r.step, 100 * r.primary_share))

    # ---- 저장 ----------------------------------------------------------
    doc = {
        "note": ("거리대별 사다리. 고정 절대 사다리는 SP 설계 범위보다 넓어 D1 에서 "
                 "in-support 편집이 0 건이었다. h = round_to_grid((max-min)/3, grid) 로 "
                 "유도하고 ladder = {-3h,-2h,-h,+h,+2h,+3h} 를 쓴다."),
        "grid": GRID,
        "rule": "h = round_to_grid((design_max - design_min) / 3, grid)",
        "finite_difference": "D_a p = (p(a+h) - p(a-h)) / (2h)   # h 는 (band, attr) 별",
        "steps": {a: {b: float(v) for b, v in s.items()} for a, s in steps.items()},
        "ladders": {a: {b: [float(x) for x in l] for b, l in d.items()}
                    for a, d in ladders.items()},
    }
    with open(CONFIG / "trc_band_ladders.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    print("  [ok] configs/trc_band_ladders.yaml")

    write_json(REPORTS / "band_ladder_coverage.json",
               {"coverage_by_band": cov, "cells_with_zero_primary": zero})
    return {"steps": steps, "zero_primary_cells": zero}


if __name__ == "__main__":
    run()
