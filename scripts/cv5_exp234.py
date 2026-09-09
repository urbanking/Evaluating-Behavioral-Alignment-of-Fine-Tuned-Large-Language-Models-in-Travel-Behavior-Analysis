# -*- coding: utf-8 -*-
"""CV5 전용 실험 2~4 - **폴드별로** 반사실 반응·행동지표를 재산출한다.

**기존 실험과 완전히 분리한다.**
  읽는 것   predictions/cv5_tabular/   data/truth/cv5_*.parquet   data/scenarios/cv5_scenario_grid.parquet
  쓰는 것   results/cv5_exp2_metrics.csv 등 `cv5_` 접두 파일
  건드리지 않는 것  predictions/tabular/, data/truth/*.parquet(cv5_ 없는 것), results/exp*.csv

**지표 계산은 analysis.py 의 함수를 그대로 쓴다.** 폴드 위에서 다른 식을 쓰면 주 실험과
비교가 안 된다. 바뀌는 것은 입력 경로와, 폴드/레인 열이 결과에 붙는다는 점뿐이다.

LLM 은 아직 폴드 학습이 끝나지 않아 이 스크립트에는 tabular 만 들어간다. 나중에 LLM
폴드 예측이 생기면 같은 방식으로 더하면 된다.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs
import analysis as A

RESULTS = ROOT / "results"
CV5_PRED = ROOT / "predictions" / "cv5_tabular"


def _patch_grid_base():
    """analysis._grid_base 를 CV5 격자로 갈아 끼운다.

    그 함수는 test_scenario_grid.parquet(3,468 사례)만 읽는다. 폴드 평가 사례에는 옛
    development 사례가 섞여 있어 distance_band 가 NaN 이 되고, wtp_table 이
    `lad["steps"]["fare"][nan]` 에서 KeyError 로 죽는다. **원본 파일은 고치지 않는다** -
    이 스크립트가 도는 동안만 함수를 바꿔 끼우고, 주 실험은 그대로 옛 격자를 본다.
    """
    g = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")
    cols = [c for c in ("distance_band", "fare_base", "ride_base", "wait_base", "rel_base")
            if c in g.columns]
    base = g.drop_duplicates("case_id").set_index("case_id")[cols]
    A._grid_base = lambda: base
    print("  [패치] _grid_base -> CV5 격자 (%d 사례)" % len(base))


def load_cv5_truth():
    """CV5 격자·진실값. analysis.load_truth 와 같은 모양으로 돌려준다."""
    t = pd.read_parquet(DATA / "truth" / "cv5_counterfactual_truth.parquet")
    g = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")[
        ["case_id", "scenario_id", "distance_band", "analysis_tier"]]
    return t, g


def load_cv5_predictions(fold: int, lane: str) -> pd.DataFrame:
    """한 폴드·한 레인의 tabular 예측. analysis 가 기대하는 열 이름으로 맞춘다."""
    fs = sorted(CV5_PRED.glob("f%d_*_%s.parquet" % (fold, lane)))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    # analysis 는 choice(원선택)를 쓴다. tabular 의 원선택은 p>=0.5 다 (analysis.py 주석).
    d["choice"] = (d["p"] >= 0.5).astype(float)
    return d[["case_id", "scenario_id", "world", "info", "model", "seed", "p", "choice"]]


def main():
    ensure_dirs(RESULTS)
    _patch_grid_base()
    truth, grid = load_cv5_truth()
    folds = sorted(pd.read_csv(DATA / "splits" / "cv5_folds.csv").fold.unique())
    e2, e3, wtp = [], [], []
    for f in folds:
        for lane in ("unweighted", "weighted"):
            preds = load_cv5_predictions(f, lane)
            if preds.empty:
                print("  [건너뜀] fold %d / %s : 예측 없음" % (f, lane)); continue
            # 이 폴드의 평가 사례만 진실값·격자에서 자른다.
            ids = set(preds.case_id)
            t = truth[truth.case_id.isin(ids)]
            g = grid[grid.case_id.isin(ids)]
            print("  fold %d %-10s : 예측 %d행 / 진실 %d행" % (f, lane, len(preds), len(t)),
                  flush=True)
            for tier in (None, "PRIMARY", "EXTRAPOLATION"):
                x = A.exp2_table(preds, t, g, tier=tier)
                if len(x):
                    e2.append(x.assign(fold=f, lane=lane))
            x = A.exp3_table(preds, t, g)
            if len(x):
                e3.append(x.assign(fold=f, lane=lane))
            x = A.wtp_table(preds, t, g)
            if len(x):
                wtp.append(x.assign(fold=f, lane=lane))
    for name, parts in (("cv5_exp2_metrics", e2), ("cv5_exp3_metrics", e3),
                        ("cv5_exp3_wtp", wtp)):
        if parts:
            d = pd.concat(parts, ignore_index=True)
            d.to_csv(RESULTS / (name + ".csv"), index=False)
            print("[ok] results/%s.csv (%d행)" % (name, len(d)))
        else:
            print("[없음] %s" % name)


if __name__ == "__main__":
    main()
