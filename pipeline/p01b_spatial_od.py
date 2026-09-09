# -*- coding: utf-8 -*-
"""Phase 1b. 통행 OD 와 거주지 공간 속성을 곁들임 테이블로 만든다.

**왜 필요한가.** TRC 파이프라인은 거주지(SQ1_1/SQ1_2)만 들고 있고 통행 출발지·도착지를
중간에 떨어뜨렸다. 응답자가 실제로 답한 것은 "평소 평일 오전 통행"이고 그 통행의 O/D 가
설문에 있다. 후보모형이 이걸 못 보면 모형이 실제보다 적은 정보로 판단하게 된다.

**확인한 사실 (중요).**
  - `spatial_prior_features.parquet` 의 열 이름이 `origin_*` 이지만 실제로는
    **거주지 기준**이다. 거주지 시군구와 100.0% 일치한다. 이름을 그대로 쓰면
    "통행 출발지의 혼잡도" 라고 오독하게 되므로 `residence_*` 로 바꿔 싣는다.
  - 통행 출발지(QQ1A)는 거주지와 **97.2%** 같다. 새 정보가 거의 없지만 비용이 없어 넣는다.
  - 통행 도착지(QQ1B)는 거주지와 **23.7%** 만 같다. 이쪽이 진짜 새 정보다.

읍면동(QQ1A_3/QQ1B_3)은 넣지 않는다. 고유값이 817/724개라 원핫에서 차원이 터지고
프롬프트에서도 의미 없는 고유명사가 된다. SQ1_2(77개)를 다루던 기준과 같다.

**시점.** QQ1A/QQ1B 는 설문 Step 2(평소 통행 확인)에서 묻는다. SP 선택과제(Step 3)보다
앞이므로 선택 이전 정보이고 오염이 아니다. 공간 속성은 설문 응답이 아니라 지역 통계에서
유도한 값이라 시점 문제가 없다.

출력
  data/external/spatial_od.parquet     (respondent_id 키)
  reports/data_audit/spatial_od.json
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import DATA, REPORTS, ensure_dirs, write_json

DATA_EN = DATA / "DATA_EN (1).xlsx"
SPATIAL = Path(r"C:\SP_LLM\data\processed\spatial_prior_features.parquet")
OUT = DATA / "external" / "spatial_od.parquet"

OD_COLS = {"QQ1A_1": "trip_origin_sido", "QQ1A_2": "trip_origin_sigungu",
           "QQ1B_1": "trip_dest_sido", "QQ1B_2": "trip_dest_sigungu"}

# 거주지에서 유도한 속성. 이름에서 origin 을 떼어 오독을 막는다.
SP_COLS = {"origin_region_type": "residence_region_type",
           "origin_urbanization": "residence_urbanization",
           "transit_accessibility_origin": "residence_transit_accessibility",
           "congestion_prior_origin": "residence_congestion_prior",
           "parking_pressure_origin": "residence_parking_pressure",
           "first_last_mile_burden_origin": "residence_first_last_mile_burden"}


def run() -> dict:
    print("\n=== Phase 1b. 통행 OD + 거주지 공간속성 ===")
    ensure_dirs(OUT.parent, REPORTS)
    for p in (DATA_EN, SPATIAL):
        if not p.exists():
            raise SystemExit("원본 없음: %s" % p)

    en = pd.read_excel(DATA_EN)[["IDX"] + list(OD_COLS)].rename(columns=OD_COLS)
    sp = pd.read_parquet(SPATIAL)[["IDX"] + list(SP_COLS)].rename(columns=SP_COLS)

    # 이름이 오도하지 않는지 매번 확인한다. 이 검사가 없으면 나중에 원본이 바뀌었을 때
    # "통행 출발지 속성" 이라고 잘못 쓴 채로 논문까지 간다.
    chk = pd.read_parquet(SPATIAL)[["IDX", "origin_sigungu_kr"]]
    ko = pd.read_excel(DATA / "DATA_평일 통행·업무 통행 설문조사_260624 (2).xlsx")[
        ["IDX", "SQ1_2", "QQ1A_2", "QQ1B_2"]]
    j = chk.merge(ko, on="IDX")
    same_res = float((j.origin_sigungu_kr == j.SQ1_2).mean())
    if same_res < 0.99:
        raise SystemExit("spatial_prior 가 거주지 기준이 아니다 (일치율 %.3f) — 이름 재확인" % same_res)

    d = en.merge(sp, on="IDX", how="outer").rename(columns={"IDX": "respondent_id"})
    d.to_parquet(OUT, index=False)

    res = {
        "rows": int(len(d)),
        "columns": [c for c in d.columns if c != "respondent_id"],
        "spatial_prior_is_residence_based": same_res,
        "trip_origin_equals_residence": float((j.QQ1A_2 == j.SQ1_2).mean()),
        "trip_dest_equals_residence": float((j.QQ1B_2 == j.SQ1_2).mean()),
        "missing_rate": {c: float(d[c].isna().mean()) for c in d.columns},
    }
    write_json(REPORTS / "spatial_od.json", res)
    print("  응답자 %d명, 열 %d개" % (len(d), len(res["columns"])))
    print("  공간속성이 거주지 기준임을 확인: 일치율 %.3f" % same_res)
    print("  통행 출발지 == 거주지 %.1f%%  /  통행 도착지 == 거주지 %.1f%%"
          % (100 * res["trip_origin_equals_residence"], 100 * res["trip_dest_equals_residence"]))
    mx = max(res["missing_rate"].values())
    print("  [%s] 결측 최대 %.2f%%" % ("PASS" if mx < 0.01 else "WARN", 100 * mx))
    print("  [ok] %s" % OUT.relative_to(DATA.parent))
    return res


if __name__ == "__main__":
    run()
