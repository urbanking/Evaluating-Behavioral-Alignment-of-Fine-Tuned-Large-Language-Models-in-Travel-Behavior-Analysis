# -*- coding: utf-8 -*-
"""CV5 LLM 폴드 채점 결과를 tabular 5-fold 와 **같은 열 구조**로 모은다.

왜 필요한가. results/cv5_tabular.csv 는 fold,model,lane,n,roc_auc,... 형식이다.
LLM 쪽을 같은 형식으로 내놓아야 "SFT 가 tabular 를 이겼다" 를 단일 분할이 아니라
같은 폴드 위에서 비교할 수 있다.

lane 이름. tabular 의 lane 은 클래스 가중(unweighted/weighted)이다. LLM SFT 는 둘 중
어느 쪽도 아니고 aaai_composite 목적함수(pos_weight=1.0, wbce .65/focal .25/brier .10)를
쓴다. 그래서 lane 을 'sft_aaai_composite' 로 적는다. weighted 로 적으면 안 된다.

확률. D321 이 등록한 쌍 정규화 확률 sigmoid(q_AV - q_Keep) 을 그대로 쓴다.
사후 보정은 하지 않는다. log_loss 는 그 확률 위에서 계산한 값이다.

하드라벨. D239 규칙: margin > 0, 동점(|margin| < 1e-9)은 free_argmax.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions" / "cv5_llm"
OUT = ROOT / "results"
FOLDS = (1, 2, 3, 4, 5)
LANE = "sft_aaai_composite"
METRIC_COLS = ["roc_auc", "pr_auc", "av_f1", "macro_f1", "accuracy",
               "log_loss", "pred_av_rate"]


def fold_metrics(df):
    m = df["margin"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    tie = np.abs(m) < 1e-9
    hard = np.where(tie, df["free_argmax"].to_numpy() == "AV", m > 0).astype(int)
    p = 1.0 / (1.0 + np.exp(-m))                      # sigmoid(q_AV - q_Keep)
    return {
        "n": len(df),
        "roc_auc": roc_auc_score(y, m),
        "pr_auc": average_precision_score(y, m),
        "av_f1": f1_score(y, hard),
        "macro_f1": f1_score(y, hard, average="macro"),
        "accuracy": accuracy_score(y, hard),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "pred_av_rate": float(hard.mean()),
        "true_av_rate": float(y.mean()),
        "candidate_mass": float(df["candidate_mass"].mean()),
        "tie_count": int(tie.sum()),
    }


def paired_vs_tabular(per):
    """같은 폴드 위에서 짝지은 AV-F1 차이를 본다.

    왜 평균+/-표준편차 비교로 부족한가. tabular 와 LLM 은 **같은 5 폴드**를 쓴다
    (fold 1 은 양쪽 다 2,324건). 두 개의 평균+/-표준편차를 눈으로 비교하면 폴드가
    공유된다는 정보를 버리게 되고, 폴드 간 변동이 차이를 가린다.

    검정은 하지 않는다. CV 폴드는 학습자료를 공유해서 독립이 아니고, k-fold
    차이에 대한 paired t-test 는 1종 오류가 부풀려지는 것으로 알려져 있다.
    그래서 폴드별 차이, 평균, 표준편차, 부호 개수만 기술적으로 적는다.
    """
    tab_f = OUT / "cv5_tabular.csv"
    if not tab_f.exists():
        print("\n  results/cv5_tabular.csv 가 없다. 짝지은 비교를 건너뛴다.")
        return None

    tab = pd.read_csv(tab_f)
    llm = per.set_index("fold")
    have = set(llm.index)
    rows = []
    print("\n--- 같은 폴드 위 짝지은 AV-F1 차이 (LLM - tabular) ---")
    pairs = [("weighted", "xgboost"), ("weighted", "random_forest"),
             ("weighted", "panel_logit"), ("weighted", "svm"),
             ("weighted", "ffnn"), ("weighted", "dnn")]
    for lane, model in pairs:
        sub = tab[(tab.lane == lane) & (tab.model == model)].set_index("fold")
        common = sorted(have & set(sub.index))
        if not common:
            continue
        a = llm.loc[common, "av_f1"].to_numpy(dtype=float)
        b = sub.loc[common, "av_f1"].to_numpy(dtype=float)
        d = a - b
        # 폴드마다 사례 수가 같아야 짝짓기가 성립한다.
        n_llm = llm.loc[common, "n"].to_numpy()
        n_tab = sub.loc[common, "n"].to_numpy()
        same_n = bool((n_llm == n_tab).all())
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
        warn = "" if same_n else "   [경고: 폴드 사례수 불일치 - 짝짓기 무효]"
        print("  vs %-14s (%s): 차이 평균 %+.4f +/- %.4f | LLM 이 높은 폴드 %d/%d%s"
              % (model, lane, float(d.mean()), sd, int((d > 0).sum()), len(d), warn))
        print("     폴드별 " + "  ".join("f%d %+.4f" % (f, x)
                                        for f, x in zip(common, d)))
        rows.append({"tabular_model": model, "tabular_lane": lane,
                     "n_folds": len(d), "llm_av_f1_mean": float(a.mean()),
                     "tabular_av_f1_mean": float(b.mean()),
                     "diff_mean": float(d.mean()), "diff_sd": sd,
                     "folds_llm_higher": int((d > 0).sum()),
                     "paired_n_match": same_n})
    print("  검정통계량은 내지 않는다: CV 폴드는 학습자료를 공유하므로 독립이 아니다.")
    return pd.DataFrame(rows)


def main(family="qwen_mid"):
    files = {f: PRED / ("%s_f%d_base.parquet" % (family, f)) for f in FOLDS}
    have = {f: p for f, p in files.items() if p.exists()}
    missing = [f for f in FOLDS if f not in have]
    print("%s: %d/5 폴드 존재" % (family, len(have)))
    if missing:
        print("  없는 폴드: %s  -> 아래 평균/표준편차는 %d 폴드만의 값이다."
              % (missing, len(have)))
    if not have:
        print("  채점 결과가 하나도 없다. 먼저 scripts/cv5_score_fold.py 를 돌려야 한다.")
        return 1

    rows = []
    for f, p in sorted(have.items()):
        d = pd.read_parquet(p)
        r = {"fold": f, "model": family, "lane": LANE}
        r.update(fold_metrics(d))
        rows.append(r)
    per = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    per_f = OUT / ("cv5_llm_%s.csv" % family)
    per.to_csv(per_f, index=False)

    agg = per[METRIC_COLS].agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]
    agg.insert(0, "n_folds", len(per))
    agg.insert(0, "lane", LANE)
    agg.insert(0, "model", family)
    sum_f = OUT / ("cv5_llm_%s_summary.csv" % family)
    agg.to_csv(sum_f)

    pd.set_option("display.width", 220)
    print("\n--- 폴드별 ---")
    cols = ["fold", "n", "true_av_rate", "candidate_mass", "tie_count"] + METRIC_COLS
    print(per[cols].to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\n--- %d 폴드 평균 +/- 표준편차 ---" % len(per))
    for k in METRIC_COLS:
        print("  %-13s %.4f +/- %.4f" % (k, agg.loc[k, "mean"], agg.loc[k, "std"]))

    tab_sum = OUT / "cv5_tabular_summary.csv"
    if tab_sum.exists():
        t = pd.read_csv(tab_sum, header=[0, 1], index_col=[0, 1])
        av = t[("av_f1", "mean")]
        best_lane, best_model = av.idxmax()
        print("\n--- 같은 5 폴드 위의 tabular 최고 AV-F1 ---")
        print("  %s / %s: %.4f +/- %.4f"
              % (best_lane, best_model, av.max(),
                 t[("av_f1", "std")].loc[(best_lane, best_model)]))
        print("  %s: %.4f +/- %.4f"
              % (family, agg.loc["av_f1", "mean"], agg.loc["av_f1", "std"]))
        print("  차이(LLM - tabular): %+.4f"
              % (agg.loc["av_f1", "mean"] - av.max()))
        print("  주의: lane 이 다르다. tabular 는 클래스 가중, LLM 은 %s 이다." % LANE)

    paired = paired_vs_tabular(per)
    if paired is not None and len(paired):
        pair_f = OUT / ("cv5_llm_%s_paired_av_f1.csv" % family)
        paired.to_csv(pair_f, index=False)
        print("\n저장: %s" % pair_f)

    print("저장: %s" % per_f)
    print("저장: %s" % sum_f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "qwen_mid"))
