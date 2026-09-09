# -*- coding: utf-8 -*-
"""Tabular 후보모형 공통 인터페이스.

pooled panel logit / XGBoost / Random Forest / Feed-forward NN.
전처리는 development 에서만 적합하고 test 에 그대로 적용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLS = ("case_id", "respondent_id", "task_id", "partition", "human_label", "y")
CARDINALITY_CAP = 12


def high_cardinality_allowlist():
    """고유값이 많아도 반드시 범주형으로 넣을 열. configs 에서 읽는다."""
    from common import load_schemas
    return list(load_schemas().get("high_cardinality_allowlist", []))


def ladder_numeric_columns():
    """반사실 사다리에서 값을 바꾸는 열. 고유값이 적어도 반드시 **수치형**으로 보낸다.

    **왜 예외가 필요한가.** 아래 CARDINALITY_CAP 규칙은 고유값이 12개 이하인 수치열을
    범주형으로 돌린다. 예측만 할 거면 합리적이다 - 설계가 제시한 값이 몇 개뿐이면 원핫이
    더 유연하다. 그런데 우리는 이 열들을 미분한다. **원핫된 변수에는 기울기가 없다.**

    실측 (2026-08-06): av_wait_time_min 은 고유값 4개, av_service_reliability_pct 도 4개라
    둘 다 원핫으로 갔다. 사다리가 만드는 값의 26.9% / 17.0% 는 학습에서 본 적 없는 값이라
    그 지점의 원핫은 전부 0 이 된다. 모형이 받는 신호는 '대기시간이 더 길다' 가 아니라
    '대기시간 범주가 없다' 이고, 그래서 참 반응이 0.000001 인 HOM 에서도 여섯 모형 모두
    0.002~0.010 의 없는 반응을 만들어 냈다. WTP_wait 과 WTP_reliability 가 통째로
    무의미해지는 자리다.

    목록은 설정에서 읽는다 - 사다리를 늘리면 여기도 자동으로 따라온다.
    """
    from common import load_config
    spec = load_config()["dgp_specification"]
    out = set(spec["service_attributes"].values())
    out |= set((spec.get("controls") or {}).get("numeric") or [])
    return out


def excluded_columns():
    """설정의 excluded 에 적힌 열. 대부분 상류에서 이미 빠지지만 여기서 한 번 더 막는다.

    상류 필터를 통과해 model_inputs 까지 살아남은 것이 실제로 있었다(block_ID).
    설정 파일을 유일한 근거로 삼으려면 소비 지점에서도 같은 목록을 봐야 한다.
    """
    from common import load_schemas
    ex = load_schemas().get("excluded", {})
    out = set()
    for v in ex.values():
        for n in (v if isinstance(v, list) else [v]):
            out.add(n.get("name") if isinstance(n, dict) else n)
    return {c for c in out if isinstance(c, str) and "*" not in c}


def feature_columns(df: pd.DataFrame):
    """수치형 / 범주형 / 제외로 나눈다.

    고유값이 많은 **문자열** 변수는 기본적으로 제외한다. 수치형으로 넘기면 median
    imputer 가 'Ansan-si Sangrok-gu' 를 float 로 바꾸려다 실패하고, 원핫으로 넘기면
    차원이 폭발하기 때문이다.

    예외를 두고 싶으면 configs 의 high_cardinality_allowlist 에 적는다. 지금은 비어 있다 -
    LLM 프롬프트가 읽는 시군구를 표 모형에도 세 방식으로 주어 봤지만 held-out test AUC 가
    오르지 않았기 때문이다(측정값은 그 설정 파일의 주석에 있다). 응답자 2,178명에 시군구
    76개면 구당 29명이라, 개발집합 응답자의 구별 기저율을 외울 뿐 옮겨가지 않는다.
    """
    allow = set(high_cardinality_allowlist())
    force_num = ladder_numeric_columns()
    ex = excluded_columns()
    cols = [c for c in df.columns if c not in ID_COLS and c not in ex]
    num, cat, dropped = [], [], []
    for c in cols:
        s = df[c]
        nu = s.nunique(dropna=True)
        if c in force_num and pd.api.types.is_numeric_dtype(s):
            num.append(c)                 # 미분할 열은 고유값 개수와 무관하게 연속으로
        elif c in allow:
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            (num if nu > CARDINALITY_CAP else cat).append(c)
        elif nu <= CARDINALITY_CAP:
            cat.append(c)
        else:
            dropped.append((c, nu))
    feature_columns.last_dropped = dropped
    return num, cat


feature_columns.last_dropped = []


def make_preprocessor(num, cat, drop_first=False):
    # drop_first 는 **무정규화 MLE(panel_logit)에만 켠다.** 전체 원핫은 각 범주형의
    # 더미 합이 1이라 절편과 완전공선이고, L2 가 있으면 그 중 최소노름 해를 골라 주지만
    # 무정규화면 해가 비유일해 lbfgs 가 임의의 해에 앉는다. 실측(2026-08-16): 같은
    # 자료에서 기준범주를 떨어뜨리면 VOT_ride +7.2, 안 떨어뜨리면 -2.95 kKRW/h.
    # 트리·SVM·NN 은 공선성이 무해하고 handle_unknown 흡수가 이로우므로 기본값 유지.
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=10,
                                               drop="first" if drop_first else None))]),
         cat),
    ], remainder="drop")


# 후보모형 여섯. Martin-Baos et al. (2023, TR-C) 의 모형군과 맞춘다 —
# MNL / SVM / RF / XGBoost / NN / DNN. 우리는 선택지가 둘이라 MNL 이 이항 로짓이다.
#
# **DNN 을 NN 과 따로 두는 이유.** 그쪽 논문이 둘을 나눈 것을 따른다. NN 은 얕고(64-32)
# DNN 은 깊다(256-128-64-32). 깊이가 행동지표 회복에 영향을 주는지가 이 계열의 질문 중
# 하나이고, 우리 경우에는 LLM(수십 층)까지 있으므로 깊이 축이 더 의미가 있다.
MODEL_NAMES = ("panel_logit", "svm", "random_forest", "xgboost", "ffnn", "dnn")

# panel_logit 과 **같은 추정기**이되 선택 후 심리문항 21개를 빼지 않은 판.
# 왜 따로 두는가: DCM 은 두 가지 역할을 겸하고 있고 두 역할이 서로 다른 특징집합을 요구한다.
#   행동 참조모형  VOT·WTP 를 회복해야 한다  -> 선택 후에 잰 변수를 조건으로 넣으면 안 된다 (D247)
#   예측 기준선    LLM 과 AUC·F1 을 겨룬다   -> LLM 이 보는 것과 같은 X 를 봐야 공정하다
# 하나로 합치면 둘 중 하나가 반드시 틀린다. 그래서 이름을 나눠 둘 다 낸다.
# (D250 이 class_weight 를 실험1/실험2~4 로 나눈 것과 같은 모양의 해법이다.)
PANEL_FULL = "panel_logit_full"

# 탐색 격자. 값 개수를 작게 잡았다 — 조건 6 x 모형 6 이라 격자가 크면 몇 시간이 된다.
# 이 격자는 configs 가 아니라 여기 둔다. 모형 구현과 붙어 있어야 의미가 있고,
# 실험 계약(configs)은 "튜닝을 한다" 는 사실만 담는다.
class BalancedMLP(MLPClassifier):
    """MLPClassifier 는 class_weight 도 sample_weight 도 받지 않는다. 소수 클래스를
    복제해 같은 효과를 낸다.

    **Pipeline 의 마지막 단계라서 안전하다.** cross_val_score 는 학습 fold 만 fit 에
    넘기므로, 여기서 복제해도 검증 fold 로 새지 않는다. 전처리 앞단에서 복제하면
    누수가 생긴다.

    복제 배수는 다수/소수 비율이다 (AV 18.8% 이면 약 4.3배). class_weight="balanced"
    가 하는 일과 같다.

    **__init__ 인자를 명시적으로 적는다.** **kw 로 받으면 sklearn 의 get_params 가
    balance 를 못 찾아 clone 이 깨지고, 교차검증이 조용히 균형 없이 돈다.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800,
                 early_stopping=True, n_iter_no_change=20, random_state=None,
                 balance=False):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                         max_iter=max_iter, early_stopping=early_stopping,
                         n_iter_no_change=n_iter_no_change, random_state=random_state)
        self.balance = balance

    def fit(self, X, y):
        if self.balance:
            y = np.asarray(y)
            pos = np.flatnonzero(y == 1)
            neg = np.flatnonzero(y == 0)
            if len(pos) and len(neg) and len(neg) > len(pos):
                k = max(1, int(round(len(neg) / len(pos))))
                idx = np.concatenate([neg, np.tile(pos, k)])
                rng = np.random.default_rng(self.random_state or 0)
                rng.shuffle(idx)
                X = X[idx]
                y = y[idx]
        return super().fit(X, y)


PARAM_GRID = {
    # 정규화 강도. 전에는 C=1e6 (사실상 무벌점) 고정이었는데, 시군구를 넣었을 때
    # dev .722 -> test .609 로 벌어진 것이 그 탓이다.
    # MNL 은 튜닝하지 않는다 - 무정규화 MLE 하나뿐이다 (make_model 의 주석 참조).
    "panel_logit": [{}],
    "svm": [{"C": c, "gamma": g, "class_weight": w}
            for c in (0.5, 2.0) for g in ("scale", 0.01) for w in (None, "balanced")],
    "random_forest": [{"min_samples_leaf": m, "max_features": f, "class_weight": w}
                      for m in (1, 5, 20) for f in ("sqrt", 0.3)
                      for w in (None, "balanced_subsample")],
    "xgboost": [{"max_depth": d, "learning_rate": lr, "reg_lambda": rl,
                 "scale_pos_weight": spw}
                for d in (3, 6) for lr in (0.05, 0.15) for rl in (1.0, 10.0)
                for spw in (1.0, 4.31)],
    "ffnn": [{"alpha": a, "hidden_layer_sizes": h, "balance": b}
             for a in (1e-4, 1e-2) for h in ((64, 32), (128, 64)) for b in (False, True)],
    "dnn": [{"alpha": a, "hidden_layer_sizes": h, "balance": b}
            for a in (1e-4, 1e-2) for h in ((256, 128, 64, 32), (128, 128, 64))
            for b in (False, True)],
}
# 불균형 처리를 **격자에 넣고 탐색으로 고른다.** 고정으로 켜면 "가중치를 켜서 이겼다" 가
# 아니라 "켠 것이 이 자료에서 나은지" 를 못 보여준다. 끄는 쪽이 나으면 탐색이 끈다.
#   panel_logit/svm/random_forest  class_weight
#   xgboost                        scale_pos_weight (1.0 대 4.31 = 0.812/0.188)
#   ffnn/dnn                       소수 클래스 복제 (BalancedMLP)


def make_model(name: str, seed: int, params: dict | None = None):
    p = dict(params or {})
    if name in ("panel_logit", PANEL_FULL):
        # 이항 로짓 = 선택지 둘일 때의 MNL. **무정규화 MLE, 균형화 없음.**
        # 2026-08-16 이전에는 C·class_weight 를 f1_macro 튜닝 격자에 넣었는데, 그 결과
        # L2 수축이 작은 시간 계수를 짓누르고 균형화가 부호까지 뒤집어 HUMAN VOT 가
        # 0·음수로 나왔다. 무정규화로 돌리면 계수가 fare -0.1006, ride -0.0121,
        # wait -0.0224 (원단위) 로 전부 음수이고 VOT_ride +7.2 kKRW/h 로 정상이다.
        # DCM 은 MLE 로 추정하는 참조 기술이지 튜닝 대상이 아니다 (참조 논문 §4.2).
        #
        # **class_weight 는 기본이 None 이고, 명시적으로 줄 때만 켜진다.** 파이프라인은
        # 아무 데서도 주지 않으므로 위 서술이 그대로 유지된다. 켤 수 있게 둔 이유는
        # D250 이 실험 1 을 균형화 lane 으로 보고하기 때문이다 - 그 lane 의 숫자는
        # 2026-08-09 에 따로 저장해 둔 refit(predictions/_superseded_balanced_tabular/)에서
        # 오는데, 특징집합을 바꿔 가며 그 lane 을 다시 재려면 여기서 켤 수 있어야 한다.
        return LogisticRegression(max_iter=5000, solver="lbfgs", penalty=None,
                                  class_weight=p.get("class_weight"))
    if name == "svm":
        # SVM 은 확률을 직접 내지 않는다. probability=True 가 내부 5-fold Platt 을 돌려
        # 확률을 만든다 - 참고 논문도 같은 처리를 한다. 그만큼 느리다.
        from sklearn.svm import SVC
        return SVC(C=p.get("C", 1.0), gamma=p.get("gamma", "scale"),
                   class_weight=p.get("class_weight"),
                   kernel="rbf", probability=True, random_state=seed, cache_size=512)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500,
                                      min_samples_leaf=p.get("min_samples_leaf", 5),
                                      max_features=p.get("max_features", "sqrt"),
                                      class_weight=p.get("class_weight"),
                                      random_state=seed, n_jobs=4)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=p.get("max_depth", 4),
                             learning_rate=p.get("learning_rate", 0.05),
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=p.get("reg_lambda", 1.0),
                             scale_pos_weight=p.get("scale_pos_weight", 1.0),
                             eval_metric="logloss", random_state=seed, n_jobs=4,
                             tree_method="hist")
    if name in ("ffnn", "dnn"):
        default = (64, 32) if name == "ffnn" else (256, 128, 64, 32)
        return BalancedMLP(hidden_layer_sizes=p.get("hidden_layer_sizes", default),
                           alpha=p.get("alpha", 1e-3), max_iter=800,
                           early_stopping=True, n_iter_no_change=20,
                           random_state=seed, balance=bool(p.get("balance", False)))
    raise ValueError(name)


def tune(name, df_train, y, groups, seed=42, n_splits=3, scoring="neg_log_loss"):
    # 튜닝 기준을 f1_macro 에서 **교차엔트로피(neg_log_loss)** 로 바꿨다 (2026-08-16).
    # f1_macro 는 확률을 0.5 에서 자른 뒤의 점수라, 균형화로 작동점을 옮긴 후보가 이긴다.
    # 그렇게 뽑힌 균형화 적합은 기저 점유율을 0.40 으로 부풀리고 (관측 0.176) 행동량
    # (점유율·VOT)을 전부 오염시켰다. 참조 논문 §4.3 도 교차엔트로피로 튜닝한다.
    # 확률로 채점하면 균형화가 이길 이유가 사라지고, 분류 작동점 문제는 D239 (원선택
    # 채점) 가 이미 해결한다.
    """하이퍼파라미터를 **응답자 단위** 교차검증으로 고른다.

    **응답자 단위여야 하는 이유.** 한 사람이 문항 여섯 개에 답했다. 행 단위로 쪼개면 같은
    사람이 학습 쪽과 검증 쪽에 동시에 들어가서, 그 사람의 버릇을 외운 값이 검증 점수를
    올린다. 그러면 "외우는 쪽" 하이퍼파라미터가 뽑히는데, 정작 test 는 처음 보는 응답자다.
    실측으로도 그 격차가 컸다 - random_forest 가 dev .985 / test .684 였다.

    **왜 튜닝이 필요한가.** 전에는 손으로 넣은 고정값을 썼고 panel_logit 은 C=1e6 으로
    정규화가 사실상 꺼져 있었다. 그 상태로 LLM 과 비교하면 "베이스라인을 불리하게 놓았다"
    는 지적을 받는다. 참고 논문(Martin-Baos et al. 2023)도 5x2 CV 로 튜닝한다.

    **탐색 기준이 f1_macro 다.** 전에는 neg_log_loss 였는데, 그건 다수 클래스에 확률을
    잘 맞히면 좋은 점수가 나온다. 그래서 소수 클래스를 잡는 설정이 선택되지 않았고,
    실제로 random_forest 가 AUC 0.7664(2위)인데 AV F1 0.1100(꼴찌)으로 나왔다.
    이 연구가 보는 것은 "전환과 유지를 둘 다 맞히는가" 이므로 macro F1 이 맞는 기준이다.

    돌려주는 것: (최적 파라미터, 격자 전체의 점수표)
    """
    from sklearn.model_selection import GroupKFold, cross_val_score
    grid = PARAM_GRID.get(name) or [{}]
    if len(grid) == 1:
        return grid[0], []
    num, cat = feature_columns(df_train)
    X = df_train[num + cat]
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    rows, best, best_s = [], grid[0], -np.inf
    for p in grid:
        pipe = Pipeline([("prep", make_preprocessor(num, cat)),
                         ("clf", make_model(name, seed, p))])
        try:
            s = float(np.mean(cross_val_score(pipe, X, y, groups=groups, cv=gkf,
                                              scoring=scoring, n_jobs=1)))
        except Exception as e:                    # 격자 한 점이 죽어도 탐색은 계속한다
            rows.append({"params": p, "score": np.nan, "error": str(e)[:80]})
            continue
        rows.append({"params": p, "score": s})
        if s > best_s:
            best, best_s = p, s
    return best, rows


class TabularSystem:
    """fit / predict_proba / 반사실 예측을 한 곳에서."""

    def __init__(self, name: str, seed: int, params: dict | None = None):
        self.name, self.seed = name, seed
        self.params = params or {}
        self.pipe = None
        self.num, self.cat = [], []

    def fit(self, df_train: pd.DataFrame, y: np.ndarray):
        self.num, self.cat = feature_columns(df_train)
        if self.name == "panel_logit":
            # DCM 효용식은 선택 후 심리 문항(태도 A1_*, 이유 S3_*, 의향 T1·T4·S1·S2)을
            # 받지 않는다. 목록·근거는 trc_feature_schemas.yaml 의 dcm_postchoice_excluded.
            # panel_logit_full 은 **일부러 빼지 않는다** - 위 PANEL_FULL 주석 참조.
            from common import load_schemas
            post = set(load_schemas().get("dcm_postchoice_excluded", []))
            self.num = [c for c in self.num if c not in post]
            self.cat = [c for c in self.cat if c not in post]
        prep = make_preprocessor(self.num, self.cat,
                                 drop_first=(self.name in ("panel_logit", PANEL_FULL)))
        self.pipe = Pipeline([("prep", prep),
                              ("clf", make_model(self.name, self.seed, self.params))])
        self.pipe.fit(df_train[self.num + self.cat], y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(df[self.num + self.cat])[:, 1]

    def predict_scenarios(self, base: pd.DataFrame, grid: pd.DataFrame,
                          attr_cols: dict) -> pd.DataFrame:
        """grid 의 fare_cf/ride_cf/wait_cf 로 baseline 행을 덮어써 예측한다."""
        idx = {c: i for i, c in enumerate(base.case_id)}
        out = []
        for sid, g in grid.groupby("scenario_id", sort=False):
            rows = base.iloc[[idx[c] for c in g.case_id]].copy()
            rows[attr_cols["fare"]] = g.fare_cf.to_numpy()
            rows[attr_cols["ride"]] = g.ride_cf.to_numpy()
            rows[attr_cols["wait"]] = g.wait_cf.to_numpy()
            # AV 신뢰도. 격자에 rel_cf 가 있으면 반영한다 - 없으면(옛 격자) 건너뛴다.
            # **원 단위(%)로 덮어쓴다.** 표준화는 파이프라인 안에서 일어나므로 여기서
            # 되돌릴 필요가 없다. 표준화된 벡터를 직접 건드리면 sigma 를 곱해 되돌려야 하는데,
            # 그 경로를 쓰지 않는 이유가 이것이다.
            if "rel_cf" in g.columns and "av_service_reliability_pct" in rows.columns:
                rows["av_service_reliability_pct"] = g.rel_cf.to_numpy()
            out.append(pd.DataFrame({"case_id": g.case_id.to_numpy(), "scenario_id": sid,
                                     "p": self.predict_proba(rows)}))
        return pd.concat(out, ignore_index=True)
