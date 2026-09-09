# -*- coding: utf-8 -*-
"""TRC 파이프라인 공통 유틸.

설계 근거: Research Design/설계_확정사항.md
구현 계약: Research Design/implementation_spec.md
실행 순서: Research Design/IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

# 경로를 박아두면 다른 장비(예: GV100 서버)에서 코드가 통째로 안 돈다.
# pipeline/ 의 부모를 프로젝트 루트로 잡고, 필요하면 TRC_ROOT 로 덮어쓴다.
import os
ROOT = Path(os.environ.get("TRC_ROOT") or Path(__file__).resolve().parent.parent)
CONFIG = ROOT / "configs"
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"
VIEWS = DATA / "views"
REPORTS = ROOT / "reports" / "data_audit"


# utf-8-sig 로 읽는다. BOM 이 없으면 utf-8 과 똑같이 동작하고, 있으면 떼어낸다.
# GV100 서버 사본은 PowerShell 로 dtype 한 줄을 바꾸는데 PS 5.1 의 Set-Content -Encoding utf8
# 은 **BOM 을 붙인다.** 그러면 첫 키가 '﻿experiment' 가 되어 파이프라인이 한참 뒤에서
# KeyError 로 죽고, 원인이 인코딩이라는 걸 알아내기 어렵다.
CONFIG_ENCODING = "utf-8-sig"


# 기계마다 달라야 하는 값. 설정 파일을 그대로 동기화해도 깨지지 않게 환경변수가 이긴다.
#
#   TRC_MODEL_ROOT  가중치가 놓인 자리. 이 워크스테이션은 C:/SP_LLM/models, GV100 은
#                   C:/Users/dell/models 다.
#   TRC_DTYPE       연산 dtype. GV100 은 Volta(sm_70) 라 bfloat16 을 못 쓴다. 실제로
#                   거기서 성공한 9B 학습은 float16 이었다(run_manifest 확인).
#
# 2026-08-10 에 설정 파일을 GV100 으로 동기화하면서 이 두 값까지 워크스테이션 값으로
# 덮어썼다. model_root 는 그 자리에서 `Repo id must use alphanumeric chars` 로 죽어
# 바로 드러났지만, dtype 은 조용히 통과해서 **float16 으로 학습한 9B 를 bfloat16 으로
# 채점**했다. 죽지 않는 종류의 어긋남이라 더 위험하다.
ENV_OVERRIDES = (("TRC_MODEL_ROOT", ("llm", "model_root"), str),
                 ("TRC_DTYPE", ("llm", "dtype"), str),
                 # 채점 배치. 9B 는 배치 8 x 길이 1280 에서 마지막 선형층이 어휘 248k 로
                 # 로짓을 만들며 4.1GB 를 한 번에 요구하고, 2026-08-18 새벽 대량 추론이
                 # 그 자리에서 CUDA OOM 으로 죽었다(HUMAN 98/229, HOM 82/229 에서 중단).
                 # 카드·모델마다 한계가 다르므로 설정 파일이 아니라 환경변수로 낮춘다.
                 ("TRC_SCORE_BATCH", ("llm", "score_batch_size"), int),
                 # ICL(예시를 프롬프트에 넣는 조건) 전용. 프롬프트 1개가 1,089토큰이라
                 # k=4 면 약 4,166토큰이 되는데 기본값 1280 으로는 잘린다. 왼쪽 자르기라
                 # system 과 페르소나가 날아가므로 조용히 다른 실험이 된다.
                 ("TRC_MAX_SEQ_LEN", ("llm", "max_seq_len"), int))


def load_config() -> dict:
    with open(CONFIG / "trc_experiment.yaml", encoding=CONFIG_ENCODING) as f:
        cfg = yaml.safe_load(f)
    for var, path, cast in ENV_OVERRIDES:
        v = os.environ.get(var)
        if not v:
            continue
        d = cfg
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = cast(v)
    return cfg


def load_schemas() -> dict:
    with open(CONFIG / "trc_feature_schemas.yaml", encoding=CONFIG_ENCODING) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 최종 산출물에 **들어갈 예정인** 시스템 격자.
#
# 왜 필요한가. 표와 그림을 "지금 있는 예측" 으로만 만들면, LLM 이 아직 없는 지금은
# 표 모형 4종만 나오고 완성본이 어떤 모양인지 알 수 없다. 나중에 LLM 이 붙었을 때
# 자리 배치·정렬·범례가 다 바뀌어 다시 검토해야 한다.
#
# 그래서 여기서 **최종 격자를 미리 선언**하고, 표·그림이 이 격자에 맞춰 빈칸을 잡는다.
# 예측이 들어오면 그 칸이 채워질 뿐 구조는 그대로다.
# ---------------------------------------------------------------------------
# 표 모형 여섯. Martín-Baos (2023) 의 모형군과 맞춘 목록이다. svm 과 dnn 은 나중에
# 추가됐는데 이 튜플을 안 고쳐서, 예측이 있는데도 T2/T10 표에서 두 모형이 통째로
# 빠진 채 조판된 적이 있다 (2026-08-16 발견). 표시 순서: DCM 먼저, 그다음 ML.
TABULAR_MODELS = ("panel_logit", "svm", "random_forest", "xgboost", "ffnn", "dnn")
LLM_STATES = ("zeroshot", "sft")

# **이 논문은 세 계열의 비교다.** 표와 그림에서 그 구분이 보여야 한다.
#
#   DCM  이산선택모형. 효용식을 사람이 선언하고 계수를 추정한다. 그래서 계수에서
#        VOT·탄력성을 해석적으로 뽑을 수 있다. 여기서는 pooled panel logit 하나.
#   ML   기계학습. 함수형을 선언하지 않고 자료에서 맞춘다. 예측은 잘하는 편이지만
#        행동지표를 뽑으려면 예측을 수치미분해야 한다.
#   LLM  글을 읽고 답하는 모형. 입력이 벡터가 아니라 문장이다.
#
# panel_logit 을 ML 과 한 덩어리('tabular')로 묶으면 이 논문의 축이 안 보인다.
# 참고 논문(Martin-Baos et al. 2023, TR-C)도 RUM 대 ML 로 나눠 보고한다.
SYSTEM_CLASS = {"panel_logit": "DCM",
                "xgboost": "ML", "random_forest": "ML", "ffnn": "ML"}


def system_class(model: str) -> str:
    """시스템 이름 -> DCM / ML / LLM."""
    return SYSTEM_CLASS.get(model, "LLM")

# 모델군별로 어느 stage 를 도는가. 규모 사다리(0.8B/2B)는 인간자료만 한다 —
# 합성 세계까지 돌리면 3080 이 며칠 더 걸리는데 규모 축의 질문은 인간자료로 답할 수 있다.
# qwen_mid(2B)는 원래 HUMAN 파일럿이었는데 2026-08-14 부터 세 자료 전부의 본 실험 팔이
# 됐다. human 전용으로 남겨 두면 합성자료 표에서 2B 행이 예측이 있어도 떨어진다.
FAMILY_STAGES = {"qwen": ("human", "synthetic"),
                 "llama": ("human", "synthetic"),
                 "qwen_small": ("human",),
                 "qwen_mid": ("human", "synthetic")}


def expected_grid(cfg=None):
    """(world, info, model, kind, pending) 목록. pending 은 아직 예측이 없다는 뜻이 아니라
    '이 칸은 최종본에 존재한다' 는 선언이다. 실제 유무는 호출부가 예측과 대조해 정한다.

    kind: tabular / llm
    """
    import pandas as pd
    cfg = cfg or load_config()
    g = cfg["grid"]
    rows = []
    # 표 모형: 6조건 전부
    conds = [(w, i) for w in g["synthetic"]["worlds"]
             for i in g["synthetic"]["information_conditions"]]
    conds += [(w, i) for w in g["human"]["worlds"]
              for i in g["human"]["information_conditions"]]
    for w, i in conds:
        for m in TABULAR_MODELS:
            rows.append({"world": w, "info": i, "model": m, "kind": "tabular"})
    # LLM: 모델군 x stage x 상태
    for fam, v in cfg["llm"]["families"].items():
        if not v.get("research", True):
            continue                      # smoke 는 배관 검증 전용이라 보고하지 않는다
        for stage in FAMILY_STAGES.get(fam, ("human",)):
            gg = cfg["grid"][stage]
            for w in gg["worlds"]:
                for i in gg["information_conditions"]:
                    for st in LLM_STATES:
                        rows.append({"world": w, "info": i,
                                     "model": "%s_%s" % (fam, st), "kind": "llm",
                                     "available": bool(v.get("available"))})
    d = pd.DataFrame(rows)
    if "available" not in d.columns:
        d["available"] = True
    d["available"] = d["available"].astype("object").where(d["available"].notna(), True)
    d["available"] = d["available"].astype(bool)
    return d.drop_duplicates(subset=["world", "info", "model"]).reset_index(drop=True)


def expected_models(cfg=None, include_unavailable=False):
    """최종본에 들어갈 시스템 이름을 보고 순서대로.

    미확보 모델군(Llama)은 기본으로 뺀다. 확보 여부가 불확실한 채로 빈 행을 두면 표가
    영영 비어 있고 "왜 이 칸만 비었나" 를 묻게 된다. configs 의 available 을 true 로
    바꾸면 자리가 생긴다.
    """
    d = expected_grid(cfg)
    if not include_unavailable:
        d = d[d.available]
    order = list(TABULAR_MODELS)
    llm = sorted(set(d.loc[d.kind == "llm", "model"]))
    # 큰 모델부터. qwen(9B) -> qwen_mid(2B) -> qwen_small(0.8B) -> llama
    rank = {"qwen": 0, "llama": 1, "qwen_mid": 2, "qwen_small": 3}
    llm.sort(key=lambda m: (rank.get(m.rsplit("_", 1)[0], 9), m.endswith("sft"), m))
    return order + llm


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    ensure_dirs(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


class GateFailure(RuntimeError):
    """게이트 통과 실패. 다음 Phase 로 넘어가면 안 된다."""


def check(label: str, got, expected, gate: str | None = None) -> bool:
    ok = got == expected
    mark = "PASS" if ok else "FAIL"
    print("  [%s] %-46s got=%-9s expected=%s" % (mark, label, got, expected))
    if not ok and gate:
        raise GateFailure("%s 실패: %s (got=%s, expected=%s)" % (gate, label, got, expected))
    return ok


def load_panel() -> pd.DataFrame:
    """원본 패널을 읽고 공식 70/30 분할을 적용한다.

    split 컬럼은 train / validation / test 세 층이며, 두 held-out 층을 합쳐 test 로 쓴다.
    data/splits/respondent_holdout/split70_seed42.csv 는 응답자 42% 가 다르게 배정된
    별개 분할이므로 사용하지 않는다.
    """
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["data"]["panel"])

    dev_levels = set(cfg["split"]["development"])
    test_levels = set(cfg["split"]["test"])
    known = dev_levels | test_levels
    unknown = set(df["split"].unique()) - known
    if unknown:
        raise GateFailure("split 컬럼에 예상치 못한 값: %s" % sorted(unknown))

    df["partition"] = df["split"].map(
        lambda s: "development" if s in dev_levels else "test"
    )
    df["case_id"] = df["scenario_id"].astype(str)
    df["respondent_id"] = df["IDX"].astype(int)
    df["task_id"] = df["distance_code"].astype(str)
    df = add_service_reliability(df)
    return df


def add_service_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """AV 신뢰도를 **하나의 공통 척도**로 만든다.

    **왜 필요한가.** 설문은 같은 잠재량을 응답자마다 다른 방향으로 물었다. 코드북 정의가
    그렇다 — "충돌 확률이 아니다. 배차 지연·경로 변경·원격 개입·시스템 점검 같은 큰 중단
    없이 통행을 정상 완료할 가능성" 이다.

        서비스 이상 가능성    0 / 0.001 / 0.005 / 0.01 %
        작동 안정성          100 / 99.999 / 99.995 / 99.99 %

    그런데 `av_framing_percent` 는 이 둘을 **숫자 그대로 한 열에 담고 있었다.** 결과가 셋이다.
      (1) 표준화 sigma 가 49.997 이 된다 (두 덩어리 사이 거리가 지배). 0.001%p 변화가
          Δz = 0.00002 로 눌려 신뢰도 효과가 사실상 지워진다. 공통 척도면 Δz = 0.2556 이다.
      (2) 프레이밍 더미(`av_framing_attribute`)와 **상관이 정확히 1.0000** 이다. 즉 이 열은
          신뢰도를 하나도 담지 않고 더미를 중복해서 넣은 것이고, 두 계수가 따로 식별되지 않는다.
      (3) 그래서 신뢰도 WTP 를 낼 수 없다. 미분하면 "척도를 바꾼 효과" 가 나온다.

    공통 척도로 바꾸면 상관이 0.0095 로 떨어지고, 두 프레이밍 집단이 같은 네 수준
    (99.99 / 99.995 / 99.999 / 100)을 균형 있게 돈다. 그제서야 프레이밍 더미가 "표현 방식의
    효과", 이 열이 "신뢰도 수준의 효과" 를 각각 잡는다.

    **LLM 프롬프트는 이 열을 쓰지 않는다.** 응답자가 실제로 읽은 문장을 그대로 보여줘야 하므로
    원래 프레이밍과 원래 숫자를 쓴다 (`Operation stability 99.995%` / `Service abnormality 0.005%`).
    이 열은 DGP 와 표 모형처럼 숫자를 직접 먹는 쪽에서만 쓴다.
    """
    if "av_framing_percent" not in df.columns:
        return df
    p = pd.to_numeric(df["av_framing_percent"], errors="coerce")
    # 50 을 기준으로 어느 방향의 표현인지 가른다. 두 척도가 0~0.01 과 99.99~100 이라
    # 그 사이에 값이 없다 (확인함).
    df["av_service_reliability_pct"] = (p.where(p >= 50, 100.0 - p)).round(6)
    return df


def restricted_mask(df: pd.DataFrame) -> pd.Series:
    """AV 채택 대 정확한 현재수단 유지 도메인.

    methods/fine_tuning/prepare_av_pairwise_dataset.py 의 build_rows 와 동일한 규칙.
    """
    adopt = df["future_mode_label"].astype(str) == "AV"
    retain = (~adopt) & (
        df["future_mode_label"].astype(str) == df["current_mode_label"].astype(str)
    )
    return adopt | retain


def human_label(df: pd.DataFrame) -> pd.Series:
    return (df["future_mode_label"].astype(str) == "AV").astype(int)


def transition_state(df: pd.DataFrame) -> pd.Series:
    adopt = df["future_mode_label"].astype(str) == "AV"
    retain = (~adopt) & (
        df["future_mode_label"].astype(str) == df["current_mode_label"].astype(str)
    )
    return pd.Series(
        ["AV" if a else ("retain" if r else "other_non_AV") for a, r in zip(adopt, retain)],
        index=df.index,
    )
