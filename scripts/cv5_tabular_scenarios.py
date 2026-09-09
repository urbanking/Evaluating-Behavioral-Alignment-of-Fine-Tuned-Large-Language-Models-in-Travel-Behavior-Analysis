# -*- coding: utf-8 -*-
"""CV5 폴드별 tabular 33조건 예측 - 실험 2~4 를 폴드 위에서 재산출하기 위한 재료.

**기존 산출물과 섞지 않는다.** 출력은 predictions/cv5_tabular/ 아래에만 쓰고,
predictions/tabular/(고정 분할)은 건드리지 않는다. 격자도 CV5 전용
data/scenarios/cv5_scenario_grid.parquet(전체 11,584 사례)을 쓴다.

폴드 f: 나머지 4/5 로 학습 -> 그 1/5 의 33조건을 예측. 두 레인(가중/비가중) 모두.
전처리·하이퍼파라미터 탐색·모형 정의는 p13 이 쓰는 tabular.py 를 그대로 부른다.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs, load_config
from tabular import PANEL_FULL, TabularSystem, tune
from p13_tabular_full import MODELS as BASE_MODELS

MODELS = list(BASE_MODELS) + [PANEL_FULL]
WEIGHTED = {
    "panel_logit":      {"class_weight": "balanced"},
    "panel_logit_full": {"class_weight": "balanced"},
    "svm":              {"class_weight": "balanced"},
    "random_forest":    {"class_weight": "balanced_subsample"},
    "xgboost":          {"scale_pos_weight": 4.31},
    "ffnn":             {"balance": True},
    "dnn":              {"balance": True},
}
OUT = ROOT / "predictions" / "cv5_tabular"
WORLD, INFO, SEED = "HUMAN", "retro", 42


def main():
    cfg = load_config()
    ensure_dirs(OUT)
    attr = dict(cfg["dgp_specification"]["service_attributes"])
    grid = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")
    folds = pd.read_csv(DATA / "splits" / "cv5_folds.csv", dtype={"respondent_id": str})
    fold_of = dict(zip(folds.respondent_id, folds.fold))
    pool = pd.concat([pd.read_parquet(DATA / "model_inputs" / ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                      for p in ("development", "test")], ignore_index=True)
    pool["fold"] = pool.respondent_id.astype(str).map(fold_of)
    print("모집단 %d건 | 격자 %d행 (%d사례 × %d조건)"
          % (len(pool), len(grid), grid.case_id.nunique(), grid.scenario_id.nunique()))

    for f in sorted(pool.fold.unique()):
        tr = pool[pool.fold != f].reset_index(drop=True)
        te = pool[pool.fold == f].reset_index(drop=True)
        g = grid[grid.case_id.isin(set(te.case_id))]
        print("\n--- fold %d : 학습 %d / 평가 %d건, 격자 %d행 ---" % (f, len(tr), len(te), len(g)))
        for name in MODELS:
            best, _ = tune(name, tr, tr.y.to_numpy(), tr.respondent_id.to_numpy(), seed=42)
            for lane in ("unweighted", "weighted"):
                out = OUT / ("f%d_%s_%s.parquet" % (f, name, lane))
                if out.exists():
                    print("  [skip] %s" % out.name); continue
                pars = dict(best)
                if lane == "weighted":
                    pars.update(WEIGHTED[name])
                t0 = time.time()
                sysm = TabularSystem(name, SEED, pars).fit(tr, tr.y.to_numpy())
                sp = sysm.predict_scenarios(te, g, attr)
                sp["fold"], sp["model"], sp["lane"], sp["seed"] = f, name, lane, SEED
                sp["world"], sp["info"] = WORLD, INFO
                sp.to_parquet(out, index=False)
                print("  [ok] %-16s %-10s %7d행 (%.0fs)"
                      % (name, lane, len(sp), time.time() - t0), flush=True)
    print("\n[완료] %s" % OUT)


if __name__ == "__main__":
    main()
