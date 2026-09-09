# -*- coding: utf-8 -*-
"""Phase 4. SP 설계 support map.

in-support 판정은 전역 min-max 가 아니라 (block, distance band) 별 설계 수준 범위로 한다.
긴 통행에서 흔한 요금이 짧은 통행에서는 제시된 적이 없을 수 있고, 전역 범위를 쓰면 그런
사례가 in-support 로 잘못 분류된다.

출력
  data/processed/trc_support_map.parquet
  reports/data_audit/support_map_summary.json
"""

from __future__ import annotations

import json

import pandas as pd

from common import PROCESSED, REPORTS, ensure_dirs, load_panel, restricted_mask, write_json

ATTRS = {
    "fare": "av_cost_1000won",
    "ride": "av_travel_time_min",
    "wait": "av_wait_time_min",
}
# support 단위는 distance_band 다. (block_ID, distance_band) 로 잡으면 셀마다 설계 수준이
# 정확히 1개뿐이라 어떤 편집도 무조건 범위를 벗어난다 — 블록 설계에서 각 블록은 거리대별로
# 프로파일 하나만 제시하기 때문이다. 블록을 통합해야 그 거리대의 실제 설계 공간이 된다.
KEYS = ["distance_band"]

PRIMARY, EXTRAPOLATION, INVALID = "PRIMARY", "EXTRAPOLATION", "INVALID"


def build_support_map(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attr, col in ATTRS.items():
        g = df.groupby(KEYS)[col]
        for key, s in g:
            levels = sorted(float(v) for v in pd.Series(s).dropna().unique())
            band = key if isinstance(key, str) else key[0]
            rows.append({
                "distance_band": band, "attribute": attr,
                "column": col, "n_levels": len(levels),
                "minimum": levels[0], "maximum": levels[-1],
                "allowed_levels": json.dumps(levels),
            })
    return pd.DataFrame(rows)


class SupportClassifier:
    def __init__(self, support: pd.DataFrame):
        self._rng = {
            (r.distance_band, r.attribute): (r.minimum, r.maximum)
            for r in support.itertuples()
        }

    def classify(self, band, attr, value) -> str:
        if attr == "fare" and value < 0:
            return INVALID
        if attr == "wait" and value < 0:
            return INVALID
        if attr == "ride" and value <= 0:
            return INVALID
        lo, hi = self._rng[(band, attr)]
        return PRIMARY if lo <= value <= hi else EXTRAPOLATION


def run() -> dict:
    print("\n=== Phase 4. SP 설계 support map ===")
    df = load_panel()
    df = df[restricted_mask(df)]
    ensure_dirs(PROCESSED, REPORTS)

    sm = build_support_map(df)
    sm.to_parquet(PROCESSED / "trc_support_map.parquet", index=False)
    print("  [ok] trc_support_map.parquet  %d 행 (band %d × attr %d)"
          % (len(sm), df.distance_band.nunique(), len(ATTRS)))

    clf = SupportClassifier(sm)

    # 전역 범위와 설계 범위가 실제로 다른지 확인한다. 같다면 이 Phase 는 무의미하다.
    diff = []
    for attr, col in ATTRS.items():
        g_lo, g_hi = float(df[col].min()), float(df[col].max())
        sub = sm[sm.attribute == attr]
        narrower = int(((sub.minimum > g_lo) | (sub.maximum < g_hi)).sum())
        diff.append({"attribute": attr, "global_min": g_lo, "global_max": g_hi,
                     "cells": len(sub), "cells_narrower_than_global": narrower})
        print("  %-5s 전역 [%.1f, %.1f] / 설계셀 %d개 중 전역보다 좁은 셀 %d개"
              % (attr, g_lo, g_hi, len(sub), narrower))

    # 사다리를 적용했을 때 in-support 비율을 미리 본다 (Phase 9 에서 실제 생성).
    cfg_edits = {"fare": [-3, -2, -1, 1, 2, 3],
                 "ride": [-15, -10, -5, 5, 10, 15],
                 "wait": [-9, -6, -3, 3, 6, 9]}
    tiers, by_band = {}, []
    test = df[df.partition == "test"]
    for attr, col in ATTRS.items():
        c = {PRIMARY: 0, EXTRAPOLATION: 0, INVALID: 0}
        for r in test.itertuples():
            base = getattr(r, col)
            for d in cfg_edits[attr]:
                c[clf.classify(r.distance_band, attr, base + d)] += 1
        for band, g in test.groupby("distance_band"):
            cb = {PRIMARY: 0, EXTRAPOLATION: 0, INVALID: 0}
            for r in g.itertuples():
                for d in cfg_edits[attr]:
                    cb[clf.classify(band, attr, getattr(r, col) + d)] += 1
            n = sum(cb.values())
            by_band.append({"attribute": attr, "distance_band": band,
                            "primary": cb[PRIMARY], "extrapolation": cb[EXTRAPOLATION],
                            "invalid": cb[INVALID],
                            "primary_share": round(cb[PRIMARY] / n, 4)})
        tiers[attr] = c
        tot = sum(c.values())
        print("  %-5s 편집 %6d건 -> PRIMARY %5d (%.1f%%) / EXTRAP %5d / INVALID %4d"
              % (attr, tot, c[PRIMARY], 100 * c[PRIMARY] / tot,
                 c[EXTRAPOLATION], c[INVALID]))

    zero = [b for b in by_band if b["primary"] == 0]
    if zero:
        print("  [WARN] PRIMARY 가 0 인 (속성, 거리대) 셀 %d 개:" % len(zero))
        for b in zero:
            print("         %-5s %-3s  (설계범위가 사다리 폭보다 좁다)"
                  % (b["attribute"], b["distance_band"]))
    result = {"support_cells": len(sm), "global_vs_design": diff,
              "test_edit_tier_preview": tiers, "by_band": by_band,
              "cells_with_zero_primary": zero}
    write_json(REPORTS / "support_map_summary.json", result)
    return result


if __name__ == "__main__":
    run()
