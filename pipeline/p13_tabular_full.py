# -*- coding: utf-8 -*-
"""Phase 13. Tabular 전체 학습·추론.

데이터 조건 6개 = 합성 4세계 × pro + HUMAN × {pro, retro}
모델 4종 × seed 3개. tabular 은 저렴하므로 전 세계에서 seed 를 다 돌려
LLM 이 못 얻는 seed 분산을 확보한다.

resume: predictions/tabular/{world}_{info}_{model}_seed{n}.parquet 있으면 건너뛴다.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from common import DATA, ROOT, ensure_dirs, load_config, write_json, REPORTS
from tabular import TabularSystem

PRED = ROOT / "predictions" / "tabular"
MODEL_IN, SCEN = DATA / "model_inputs", DATA / "scenarios"
from tabular import MODEL_NAMES, tune
MODELS = MODEL_NAMES          # MNL(=이항로짓) / SVM / RF / XGBoost / NN / DNN


def conditions(cfg):
    out = []
    g = cfg["grid"]
    for w in g["synthetic"]["worlds"]:
        for i in g["synthetic"]["information_conditions"]:
            out.append((w, i))
    for w in g["human"]["worlds"]:
        for i in g["human"]["information_conditions"]:
            out.append((w, i))
    return out


def run(force=False) -> dict:
    print("\n=== Phase 13. Tabular 전체 학습 ===")
    cfg = load_config()
    ensure_dirs(PRED, REPORTS)
    grid = pd.read_parquet(SCEN / "test_scenario_grid.parquet")
    attr = dict(cfg["dgp_specification"]["service_attributes"])
    seeds = cfg["grid"]["tabular_seeds"]
    conds = conditions(cfg)
    print("  조건 %d × 모델 %d × seed %d = %d fits"
          % (len(conds), len(MODELS), len(seeds), len(conds) * len(MODELS) * len(seeds)))

    done, made, timing = 0, [], []
    tuned, tune_log = {}, []
    for world, info in conds:
        tr = pd.read_parquet(MODEL_IN / ("%s_%s_development.parquet" % (world, info)))
        te = pd.read_parquet(MODEL_IN / ("%s_%s_test.parquet" % (world, info)))
        for name in MODELS:
            for seed in seeds:
                f = PRED / ("%s_%s_%s_seed%d.parquet" % (world, info, name, seed))
                if f.exists() and not force:
                    done += 1
                    continue
                # 학습·추론 시간을 따로 잰다. LLM 과의 계산비용 대비가 이 논문의 논점 중
                # 하나이고(초 대 일), 예전에는 화면에만 찍혀 나중에 표로 못 썼다.
                # 하이퍼파라미터는 **조건마다 한 번만** 고르고 seed 간에 재사용한다.
                # seed 마다 다시 고르면 seed 분산에 탐색 잡음이 섞여, "모형 차이가 seed
                # 변동보다 큰가" 라는 판단 기준 자체가 흔들린다.
                key = (world, info, name)
                if key not in tuned:
                    t0 = time.time()
                    best, trace = tune(name, tr, tr.y.to_numpy(),
                                       tr.respondent_id.to_numpy(), seed=42)
                    tuned[key] = best
                    tune_log.append({"world": world, "info": info, "model": name,
                                     "best": best, "seconds": round(time.time()-t0, 1),
                                     "grid": trace})
                    print("  [tune] %-4s %-5s %-14s %s (%.0fs)"
                          % (world, info, name, best, time.time()-t0), flush=True)
                t0 = time.time()
                sysm = TabularSystem(name, seed, tuned[key]).fit(tr, tr.y.to_numpy())
                t_fit = time.time() - t0
                t1 = time.time()
                sp = sysm.predict_scenarios(te, grid, attr)
                t_pred = time.time() - t1
                sp["world"], sp["info"], sp["model"], sp["seed"] = world, info, name, seed
                sp.to_parquet(f, index=False)
                made.append(f.name)
                timing.append({"world": world, "info": info, "model": name, "seed": seed,
                               "n_train": int(len(tr)), "n_predict": int(len(sp)),
                               "fit_seconds": round(t_fit, 3),
                               "predict_seconds": round(t_pred, 3),
                               "hardware": "CPU"})
                print("  [ok] %-4s %-5s %-14s seed%-4d %6d행 (학습 %.1fs + 추론 %.1fs)"
                      % (world, info, name, seed, len(sp), t_fit, t_pred))

    if timing:
        t = pd.DataFrame(timing)
        f = ROOT / "results" / "computation_cost.csv"
        if f.exists():          # LLM 쪽 기록과 섞이지 않게 model 키로 갱신한다
            old = pd.read_csv(f)
            old = old[~old.model.isin(t.model.unique())]
            t = pd.concat([old, t], ignore_index=True)
        ensure_dirs(f.parent)
        t.to_csv(f, index=False)
        print("  [ok] results/computation_cost.csv (%d행)" % len(t))

    if tune_log:
        write_json(REPORTS / "tabular_tuning.json", {"runs": tune_log})
        print("  [ok] 하이퍼파라미터 탐색 기록 %d건 -> reports/data_audit/tabular_tuning.json"
              % len(tune_log))
    res = {"conditions": [list(c) for c in conds], "models": list(MODELS),
           "tuned": {"%s|%s|%s" % k: v for k, v in tuned.items()},
           "seeds": list(seeds), "skipped_existing": done, "written": len(made)}
    write_json(REPORTS / "tabular_full.json", res)
    print("  [ok] 신규 %d개 / 기존 %d개" % (len(made), done))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
