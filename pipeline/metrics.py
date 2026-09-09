# -*- coding: utf-8 -*-
"""Exp 1-3 지표.

모든 집계는 respondent-first 다 (응답자 내 평균 -> 응답자 간 평균).
응답자마다 유효 task 수가 다르므로 case 단순평균을 쓰면 task 가 많은 응답자에 가중이 실린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def respondent_first(values, respondent_id):
    s = pd.Series(values).groupby(pd.Series(respondent_id)).mean()
    return float(s.mean())


# ------------------------------------------------------------------- Exp 1

def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def macro_f1(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    f1s = []
    for c in (0, 1):
        tp = np.sum((yh == c) & (y == c))
        fp = np.sum((yh == c) & (y != c))
        fn = np.sum((yh != c) & (y == c))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def gmpca(y, p):
    """Geometric Mean Probability of Correct Assignment.

    선택된 대안에 부여한 확률의 기하평균이다. `exp(-log_loss)` 와 정확히 같으므로 **새 정보를
    담지 않는다** - 순위도 로그손실과 항상 일치한다. 그래도 함께 싣는 이유는 교통 선택모형
    문헌(예: Martin-Baos et al. 2023, TR-C)이 이 단위로 보고해서 우리 수치를 그쪽 표와
    바로 견줄 수 있기 때문이다. 값이 0.72 면 "실제 고른 쪽에 평균 0.72 의 확률을 줬다" 로
    읽히므로 로그손실 0.33 보다 해석이 쉽다.
    """
    return float(np.exp(-log_loss(y, p)))


def pr_auc(y, p):
    """Precision-Recall AUC (average precision). **불균형 자료에서는 이쪽이 맞는 지표다.**

    양성이 17.6% 뿐인 우리 자료에서 ROC-AUC 는 낙관적으로 보인다 - 다수인 음성을 잘
    맞히기만 해도 올라가기 때문이다. PR-AUC 의 기저선은 **양성 비율 그 자체**이므로,
    0.31 이라는 값은 "무작위(0.176)보다 1.76배" 로 바로 읽힌다.

    선행 프로젝트도 같은 결론에 도달했다 (DECISIONS.md D-709):
    "Report AV-candidate metrics separately ... ROC-AUC, PR-AUC, top-k precision/recall/lift,
    and train-tuned threshold F1."
    """
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(p)))


def topk_lift(y, p, frac=0.10):
    """점수 상위 frac 만 양성으로 뽑았을 때의 (정밀도, 재현율, 향상배수).

    **정책에서 실제로 쓰는 형태다.** "AV 로 갈아탈 만한 상위 5% 에게 안내를 보낸다" 같은
    질문에 답한다. 향상배수 = 정밀도 / 기저율. 1.0 이면 무작위와 같다.
    """
    y = np.asarray(y); p = np.asarray(p)
    k = max(1, int(round(len(p) * frac)))
    idx = np.argsort(-p)[:k]
    base = float(np.mean(y))
    prec = float(np.mean(y[idx]))
    rec = float(y[idx].sum() / y.sum()) if y.sum() else float("nan")
    return prec, rec, (prec / base if base > 0 else float("nan"))


def tune_threshold(score, y, groups=None, n_splits=5, objective="macro_f1"):
    """**학습자료에서** 결정 임계값을 고른다. test 를 보고 고르면 낙관적이다.

    0.5 로 자르면 소수 클래스를 거의 못 잡는다 - 선행 프로젝트에서 실측으로 확인됐고
    (0.5 에서 "almost never flagged actual AV transitioners"), 임계값을 0.2048 로 내리자
    F1 이 0.338 -> 0.430 으로 올랐다.

    groups 를 주면 응답자 단위 교차검증으로 고른다. 같은 사람의 여러 문항이 학습·평가에
    나뉘어 들어가면 임계값이 그 사람에게 맞춰져 낙관적으로 나온다.
    """
    score = np.asarray(score, float); y = np.asarray(y)
    cand = np.quantile(score, np.linspace(0.01, 0.99, 199))

    def obj(th, sc, yy):
        pred = (sc > th).astype(int)
        return macro_f1(yy, pred.astype(float), thr=0.5) if objective == "macro_f1" else                f1_binary(yy, pred)

    if groups is None:
        scores = [(obj(t, score, y), t) for t in cand]
        best = max(scores)
        return {"threshold": float(best[1]), "objective": objective,
                "value_in_sample": float(best[0]), "cv": False}
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    picks, oof = [], np.zeros(len(y))
    for tr, te in gkf.split(score.reshape(-1, 1), y, groups):
        t = max((obj(t, score[tr], y[tr]), t) for t in cand)[1]
        picks.append(t)
        oof[te] = (score[te] > t).astype(int)
    th = float(np.median(picks))
    return {"threshold": th, "objective": objective,
            "value_oof": float(macro_f1(y, oof.astype(float), thr=0.5)),
            "fold_thresholds": [float(x) for x in picks], "cv": True}


def f1_binary(y, pred):
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def bayes_accuracy_ceiling(p_true):
    return float(np.mean(np.maximum(p_true, 1 - p_true)))


def true_distribution_ce(p_true, p_hat):
    p_hat = np.clip(p_hat, EPS, 1 - EPS)
    return float(-np.mean(p_true * np.log(p_hat) + (1 - p_true) * np.log(1 - p_hat)))


def exp1(y, p_hat, p_true_marg=None, respondent_id=None, choice=None):
    # choice = 모델이 실제로 고른 답 (0/1). LLM 은 margin>0 (두 후보 중 점수 1등)이고,
    # 이걸 주면 분류 지표(accuracy, F1)는 여기서 나온다. 안 주면 p>=0.5 (tabular 의
    # 원선택과 같다). 보정확률을 0.5 에서 자르면 Platt 이 작동점을 옮겨 분류 지표가
    # 모델의 실제 선택과 다른 답을 채점하게 된다. 확률 지표는 계속 p_hat 을 쓴다.
    hard = (np.asarray(choice, float) if choice is not None
            else (np.asarray(p_hat) >= 0.5).astype(float))
    m = {"log_loss": log_loss(y, p_hat), "gmpca": gmpca(y, p_hat),
         "brier": brier(y, p_hat), "macro_f1": macro_f1(y, hard),
         "accuracy": float(np.mean(hard.astype(int) == y)),
         "pr_auc": pr_auc(y, p_hat), "positive_rate": float(np.mean(y)),
         "av_f1": f1_binary(np.asarray(y), hard.astype(int))}
    from sklearn.metrics import roc_auc_score as _ras
    m["roc_auc"] = (float(_ras(y, p_hat)) if len(np.unique(y)) > 1 else float("nan"))
    m["pr_auc_lift"] = (m["pr_auc"] / m["positive_rate"]
                        if m["positive_rate"] > 0 else float("nan"))
    for fr in (0.05, 0.10, 0.25):
        pr, rc, lf = topk_lift(y, p_hat, fr)
        m["top%02d_precision" % int(fr*100)] = pr
        m["top%02d_recall" % int(fr*100)] = rc
        m["top%02d_lift" % int(fr*100)] = lf
    if p_true_marg is not None:
        m["pmae"] = float(np.mean(np.abs(p_hat - p_true_marg)))
        m["prmse"] = float(np.sqrt(np.mean((p_hat - p_true_marg) ** 2)))
        # **확률 MAE 만 보면 안 되는 이유.** MAE 는 평균 쪽으로 오므린 예측기에게 유리하다.
        # 모두에게 기저율만 주면 큰 오차가 안 생기므로 MAE 가 작아지는데, 실제로는 아무도
        # 구분하지 못한 것이다. 실측: LC 세계에서 random_forest 가 MAE 0.056 으로 1위인데
        # 산포비가 0.71 이고, MAE 0.067 인 panel_logit 과 참값 상관은 사실상 같았다
        # (0.573 대 0.572). 순위를 만든 것은 정확도가 아니라 오므림이었다.
        # 그래서 셋을 함께 싣는다 - 상관은 순서를 맞혔는지, 산포비는 폭을 맞혔는지 본다.
        sd_t = float(np.std(p_true_marg))
        m["p_corr"] = (float(np.corrcoef(p_hat, p_true_marg)[0, 1])
                       if np.std(p_hat) > 0 and sd_t > 0 else float("nan"))
        m["p_spread_ratio"] = float(np.std(p_hat) / sd_t) if sd_t > 0 else float("nan")
        m["true_dist_ce"] = true_distribution_ce(p_true_marg, p_hat)
        m["bayes_accuracy_ceiling"] = bayes_accuracy_ceiling(p_true_marg)
        m["accuracy_gap_to_ceiling"] = m["bayes_accuracy_ceiling"] - m["accuracy"]
    if respondent_id is not None:
        # **명세 10절: 모든 집계는 respondent-first (응답자 내 평균 -> 응답자 간 평균).**
        # 행 단위 평균은 사례가 많은 응답자를 더 무겁게 센다. test 는 균형이 아니다 -
        # 6건 409명, 5건 126, 4건 66, 3건 27, 2건 16, 1건 7. 원고 주 표는 이 열을 쓴다.
        #
        # ROC/PR-AUC 는 여기 넣지 않는다. 응답자 안에 한 클래스만 있는 경우가 대부분이라
        # 응답자별 AUC 가 정의되지 않는다. 행 단위 값을 그대로 두고 표 각주에 명시한다.
        y_a, p_a = np.asarray(y, float), np.asarray(p_hat, float)
        eps = 1e-15
        pc = np.clip(p_a, eps, 1 - eps)
        per_row = {
            "log_loss_resp": -(y_a * np.log(pc) + (1 - y_a) * np.log(1 - pc)),
            "brier_resp": (p_a - y_a) ** 2,
            "accuracy_resp": (hard.astype(int) == y_a).astype(float),
        }
        for k, v in per_row.items():
            m[k] = respondent_first(v, respondent_id)
        m["av_propensity_model"] = respondent_first(p_hat, respondent_id)
        m["av_propensity_observed"] = respondent_first(y, respondent_id)
        m["propensity_error"] = abs(m["av_propensity_model"] - m["av_propensity_observed"])
    return m


# ------------------------------------------------------------------- Exp 2

LADDER_ORDER = ["M3", "M2", "M1", "P1", "P2", "P3"]


def exp2(dp_model, dp_true, tau_a):
    """단일 편집 반응 지표. dp_* 는 같은 순서의 1차원 배열.

    informative_share 는 참 반응이 tau_a 를 넘는 비율이다. 이 값이 0 이면 그 세계에서
    해당 속성은 **참 반응이 평평**하다는 뜻이고, direction accuracy 는 정의되지 않으며
    amplitude ratio 는 0 으로 나눈 값이라 의미가 없다. 지표를 억지로 만들지 말고
    not-applicable 로 보고해야 한다.
    """
    d = np.abs(dp_true) > tau_a
    out = {"n": int(len(dp_true)), "n_above_tau": int(d.sum()),
           "informative_share": float(d.mean()) if len(d) else 0.0,
           "true_response_flat": bool(d.sum() == 0),
           "rme": float(np.mean(np.abs(dp_model - dp_true)))}
    out["direction_accuracy"] = (
        float(np.mean(np.sign(dp_model[d]) == np.sign(dp_true[d]))) if d.any() else np.nan)
    rng_t = float(np.max(dp_true) - np.min(dp_true)) if len(dp_true) else np.nan
    rng_m = float(np.max(dp_model) - np.min(dp_model)) if len(dp_model) else np.nan
    # 참 반응이 평평하면 비율이 발산한다. 보고하지 않는다.
    out["amplitude_ratio"] = (float(rng_m / rng_t)
                              if (rng_t and not out["true_response_flat"]) else np.nan)
    return out


def monotonicity_violation_rate(curve, tau_mono):
    """curve: (n, 7) baseline 포함 사다리 확률. 서비스 악화 방향으로 정렬돼 있어야 한다."""
    d = np.diff(curve, axis=1)
    return float(np.mean(d > tau_mono))


def ladder_curve_error(curve_model, curve_true):
    return float(np.mean(np.abs(curve_model - curve_true)))


# ------------------------------------------------------------------- Exp 3

def scenario_share(p, respondent_id):
    return respondent_first(p, respondent_id)


def share_change_error(ms_model, ms_true, base_model, base_true):
    return float(abs((ms_model - base_model) - (ms_true - base_true)))


def local_slopes(p_plus, p_minus, h):
    return (p_plus - p_minus) / (2.0 * h)


def elasticity(slope, x_base, p_base, eps=1e-9):
    """탄력성 = (dP/dx) * (x/P).

    **한계효과와 왜 따로 보나.** 한계효과는 "1kKRW 당 몇 %p" 라 단위가 붙어 있어서 거리대마다
    비교가 안 된다 - D1 은 요금이 4,500원이고 D6 은 20,000원대다. 탄력성은 "1% 올리면 몇 %"
    라 단위가 없어서 거리대·집단·모형 사이에 그대로 견줄 수 있다. 교통 문헌의 표준 보고
    항목이고, 참고 논문(Martin-Baos et al. 2023)도 Table 1 의 지표 목록에 넣는다.
    """
    x = np.asarray(x_base, float)
    p = np.asarray(p_base, float)
    ok = (np.abs(p) > eps)
    e = np.full(len(x), np.nan)
    e[ok] = np.asarray(slope, float)[ok] * x[ok] / p[ok]
    return e


def wtp_from_slopes(d_num, d_fare, kappa_f, scale=1.0):
    """WTP = scale * (dP/dx) / (dP/dfare).

    분자 속성 한 단위를 얻으려고 요금을 얼마나 더 낼 용의가 있나. 부호 규약은 VOT 와 같다 -
    **요금 기울기가 음수여야** 하고(비쌀수록 덜 고른다) 절댓값이 kappa_f 이상이어야 한다.
    분모가 0 에 가까우면 값이 발산하므로 거른다.

      차내시간·대기시간   scale=60, d_num<0 을 요구 (오래 걸릴수록 덜 고른다)
      신뢰도             scale=1,  d_num>0 을 요구 (믿을수록 더 고른다)

    돌려주는 것: (값, 유효 마스크)
    """
    d_num = np.asarray(d_num, float); d_fare = np.asarray(d_fare, float)
    ok = (d_fare < 0) & (np.abs(d_fare) >= kappa_f) & np.isfinite(d_num)
    v = np.full(len(d_fare), np.nan)
    v[ok] = scale * d_num[ok] / d_fare[ok]
    return v, ok


def share_jsd(p_model, p_true, eps=1e-12):
    """점유율 분포의 Jensen-Shannon divergence (0 = 같음, 1 = 완전히 다름).

    이항 선택이므로 (AV, Keep) 두 칸짜리 분포로 본다. L1 차이보다 분포 비교에 표준적이고,
    **소프트(확률 평균)와 하드(0.5 로 자른 라벨 비율) 두 가지**로 낼 수 있다. 둘이 갈리면
    "확률은 맞는데 자르는 지점이 틀렸다" 는 신호다 - 이 프로젝트에서 실제로 본 현상이다.
    """
    a = np.clip([p_model, 1 - p_model], eps, 1)
    b = np.clip([p_true, 1 - p_true], eps, 1)
    m = 0.5 * (np.asarray(a) + np.asarray(b))
    def kl(x, y):
        return float(np.sum(np.asarray(x) * np.log(np.asarray(x) / y)))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def vot_from_slopes(d_fare, d_ride, kappa_f):
    ok = (d_fare < 0) & (d_ride < 0) & (np.abs(d_fare) >= kappa_f)
    v = np.full(len(d_fare), np.nan)
    v[ok] = 60.0 * d_ride[ok] / d_fare[ok]
    return v, ok


def heterogeneity_compression(group_resp_model, group_resp_true):
    """group_resp_*: (K,) 집단별 평균 반응."""
    sm = float(np.std(group_resp_model, ddof=0))
    st = float(np.std(group_resp_true, ddof=0))
    return float(sm / st) if st > 0 else np.nan


def class_contrast(group_resp):
    K = len(group_resp)
    return {"C%d-C%d" % (k, l): float(group_resp[k] - group_resp[l])
            for k in range(K) for l in range(k + 1, K)}
