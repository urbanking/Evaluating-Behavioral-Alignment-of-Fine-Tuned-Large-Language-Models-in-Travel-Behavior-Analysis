# -*- coding: utf-8 -*-
"""tabular(ML/DL/DCM) 을 LLM 과 **같은 5-fold** 로 학습·평가한다 (안정성 검정).

주 결과는 고정 분할 그대로다(CLAUDE.md). 이건 "성능 차이가 분할 운에 얼마나 좌우되는가"
를 재는 보조 분석이고, LLM 쪽 CV5 와 **같은 응답자 배정**(data/splits/cv5_folds.csv)을 쓴다.
그래야 두 갈래의 산포를 같은 축에서 비교할 수 있다.

p13 과 다른 점은 데이터 출처뿐이다. 전처리·하이퍼파라미터 탐색·모형 정의는 p13 이 쓰는
tabular.py 를 그대로 부른다 - 조건이 하나라도 다르면 비교가 성립하지 않는다.

BASE 조건만 평가한다. 반사실 격자는 실험 2~4 용이고 안정성 검정에는 필요 없다.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, load_config
from tabular import PANEL_FULL, TabularSystem, tune
from p13_tabular_full import MODELS as BASE_MODELS
# panel_logit_full(태도 포함 DCM)은 p13 의 MODELS 에 없다 - D250 이후 별도 레인이라
# 여기서 명시적으로 더한다. 실험1 표에 그 행이 있으므로 안정성도 같이 재야 한다.
MODELS = list(BASE_MODELS) + [PANEL_FULL]
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, roc_auc_score)

WORLD, INFO = "HUMAN", "retro"
SEEDS = [42]      # 안정성 검정에서는 seed 를 1개로 둔다. 여기서 재려는 산포는 **분할** 이다.

# **두 레인을 모두 잰다 (D250).** 실험1 은 class-weighted lane 을 보고하고 실험2~4 는
# unweighted lane 을 쓴다. 안정성도 각 레인에서 따로 재야 그 표와 같은 기준이 된다.
# 가중 인자는 모형마다 이름이 다르다 (tabular.make_model 참조):
#   panel_logit / panel_logit_full / svm / random_forest : class_weight
#   xgboost                                              : scale_pos_weight (1-p)/p = 4.31
#   ffnn / dnn                                           : BalancedMLP 의 balance (소수 복제)
WEIGHTED = {
    "panel_logit":      {"class_weight": "balanced"},
    "panel_logit_full": {"class_weight": "balanced"},
    "svm":              {"class_weight": "balanced"},
    "random_forest":    {"class_weight": "balanced_subsample"},
    "xgboost":          {"scale_pos_weight": 4.31},
    "ffnn":             {"balance": True},
    "dnn":              {"balance": True},
}


def main():
    cfg = load_config()
    folds = pd.read_csv(DATA / "splits" / "cv5_folds.csv", dtype={"respondent_id": str})
    fold_of = dict(zip(folds.respondent_id, folds.fold))
    pool = pd.concat([pd.read_parquet(DATA / "model_inputs" / ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                      for p in ("development", "test")], ignore_index=True)
    pool["fold"] = pool.respondent_id.astype(str).map(fold_of)
    if pool.fold.isna().any():
        raise SystemExit("폴드 미배정 %d건" % int(pool.fold.isna().sum()))
    print("모집단 %d건 / %d명" % (len(pool), pool.respondent_id.nunique()))

    rows = []
    for f in sorted(pool.fold.unique()):
        tr = pool[pool.fold != f].reset_index(drop=True)
        te = pool[pool.fold == f].reset_index(drop=True)
        print("\n--- fold %d : 학습 %d건 / 평가 %d건 ---" % (f, len(tr), len(te)))
        for name in MODELS:
            t0 = time.time()
            best, _ = tune(name, tr, tr.y.to_numpy(), tr.respondent_id.to_numpy(), seed=42)
            for lane in ("unweighted", "weighted"):
                # unweighted = 탐색이 고른 그대로(격자에 가중이 있어도 탐색이 끄면 꺼진다).
                # weighted   = 그 위에 D250 의 균형화 인자를 **덮어씌운다**.
                pars = dict(best)
                if lane == "weighted":
                    pars.update(WEIGHTED[name])
                sysm = TabularSystem(name, SEEDS[0], pars).fit(tr, tr.y.to_numpy())
                p = sysm.predict_proba(te)
                hard = (p > 0.5).astype(int)
                y = te.y.to_numpy().astype(int)
                rows.append({"fold": int(f), "model": name, "lane": lane, "n": len(te),
                             "roc_auc": roc_auc_score(y, p),
                             "pr_auc": average_precision_score(y, p),
                             "av_f1": f1_score(y, hard),
                             "macro_f1": f1_score(y, hard, average="macro"),
                             "accuracy": accuracy_score(y, hard),
                             "log_loss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)),
                             "pred_av_rate": float(hard.mean()),
                             "seconds": round(time.time() - t0, 1)})
                print("  %-16s %-10s AUC %.4f  AV-F1 %.4f  Macro-F1 %.4f  AV율 %.3f"
                      % (name, lane, rows[-1]["roc_auc"], rows[-1]["av_f1"],
                         rows[-1]["macro_f1"], rows[-1]["pred_av_rate"]))
    d = pd.DataFrame(rows)
    out = ROOT / "results" / "cv5_tabular.csv"
    d.to_csv(out, index=False)
    print("\n[ok] %s (%d행)" % (out, len(d)))
    g = d.groupby(["lane", "model"])[["roc_auc", "pr_auc", "av_f1", "macro_f1", "accuracy", "log_loss"]]
    summ = g.agg(["mean", "std"]).round(4)
    print("\n=== 폴드 간 평균 ± 표준편차 ===")
    print(summ.to_string())
    summ.to_csv(ROOT / "results" / "cv5_tabular_summary.csv")


if __name__ == "__main__":
    main()
