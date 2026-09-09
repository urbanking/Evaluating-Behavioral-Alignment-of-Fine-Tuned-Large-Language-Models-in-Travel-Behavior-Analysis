# -*- coding: utf-8 -*-
"""Phase 14. LLM 프롬프트 렌더링과 학습 JSONL.

기존 methods/prompts/templates.py 의 AAAI 렌더러를 쓰지 않고 TRC 전용 렌더러를 둔다.
이유는 두 가지다.
  1) pro / retro 가 **태도 블록 유무만** 다르다는 것을 코드 수준에서 보장해야 한다.
  2) 편집 프롬프트가 해당 속성 표현만 바뀐다는 것을 diff 로 검증해야 한다.
두 요구 모두 원고의 주장(정보조건 대비가 정보량 차이지 표현 차이가 아니다)에 직결된다.

**2026-08-05 전면 개정.** 이전 렌더러는 설문 상황을 거의 담지 못했다. 문제가 세 가지였다.

  (a) 코드값을 그대로 찍었다. "Household income band 3", "2 car(s)" 는 사람이 못 알아듣고,
      D6 은 대수가 아니라 구간 코드라서 코드 2 를 "2 car(s)" 로 쓰면 **사실이 틀린다**
      (코드 2 = 1대, 코드 1 = 0대). 이제 configs/trc_codebook_labels.yaml 을 통해
      공식 영어 코드북의 보기 문구로 푼다.

  (b) 선택 상황이 없었다. 이 설문은 같은 통행을 두 번 보여준다. 2026 조건에서 기존 수단들의
      LOS 를 보고 하나를 고르고, 2030 조건에서 기존 수단은 그대로 둔 채 자율주행 호출
      서비스가 추가된 상태에서 다시 고른다. 이전 프롬프트에는 2026 대안들의 LOS 도,
      2026 에 무엇을 골랐는지도 없었다. 응답자가 실제로 본 표의 대부분이 빠져 있었다.

  (c) DGP 가 쓰는 변수가 빠져 있었다. av_framing_percent 와 av_framing_attribute 는
      `dgp_specification.controls` 에 들어가 진실을 만드는 데 쓰이고 tabular 후보모형도
      그 열을 보는데, 프롬프트에는 한 글자도 없었다. D8 도 마찬가지다. 즉 LLM 만
      설명변수가 빠진 채 경쟁하고 있었다.

표시 규칙
  - 비용은 설문 표시본과 같이 KRW 정수로 쓴다 (데이터는 kKRW 이므로 x1000).
  - 파생값(총 시간, km당 비용)은 넣지 않는다. 중복인 데다, 요금·시간을 편집하면 파생값도
    같이 바뀌어 "한 줄만 바뀐다" 는 불변식이 깨진다.
  - 혼잡도는 대중교통만 쓴다. 설문 표에서 나머지 수단은 항상 Light 로 고정이라 정보가 없다.
  - AV 속성은 반드시 **한 줄**에 둔다. 편집 diff 불변식을 지키기 위해서다.

후보 시퀀스는 항상 "AV" / "Keep" 고정이다. Option A/B 처럼 순서가 바뀌면 안 된다.
둘 다 1토큰이어야 학습 손실(토큰 평균)이 대칭이 된다. "Retain" 은 Ret+ain 2조각이라 바꿨다.

출력
  data/llm/{world}_{info}_{partition}.jsonl
  reports/data_audit/llm_prompts.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from common import CONFIG, DATA, REPORTS, ROOT, ensure_dirs, load_config, write_json

LLM_DIR = DATA / "llm"
MODEL_IN, SCEN = DATA / "model_inputs", DATA / "scenarios"
CANDIDATES = ("AV", "Keep")

_LAB = None


def labels() -> dict:
    global _LAB
    if _LAB is None:
        p = CONFIG / "trc_codebook_labels.yaml"
        if not p.exists():
            raise SystemExit("%s 가 없다. 먼저 p14b_codebook_labels.py 를 돌릴 것" % p.name)
        _LAB = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _LAB


ATTITUDE_LABELS = {
    "A1_1": "openness to new mobility services", "A1_2": "time over cost",
    "A1_3": "cost over time", "A1_4": "avoids transfer, waiting and crowding",
    "A1_5": "believes AV is as safe as a conventional car",
    "A1_6": "uneasy riding a driverless AV alone",
    "A1_7": "values in-vehicle activity in an AV",
    "A1_8": "would use transit less if AV hailing existed",
    "A1_9": "would feel less need to own a car", "A1_10": "concerned about location data",
    "T1": "would shift away from conventional taxi",
    "T4": "would still use AV despite industry conflict",
    "S1": "willing to use shared AV", "S2": "accepts extra time for a discount",
}


# 답변 형식 지시. 내용이 아니라 **양식**이라 여기 모아 둔다.
ANSWER_INSTRUCTION = {
    # 학습 목표 형식. 파인튜닝된 모형은 한 토큰만 내면 된다.
    "sft": "Answer with 'AV' or 'Keep' and nothing else.",
    # zero-shot 용. 계산할 자리를 준다. 저울질 방법도 채택률도 알려주지 않는다.
    "zeroshot": ("Work through this traveler's situation first, then give the decision. "
                 "The decision is one of two: 'AV' if they switch to the driverless AV "
                 "ride-hailing service, 'Keep' if they stay with the mode they used in 2026."),
    # 최종 중립판. 두 대안을 나란히 보여주고 그 둘 중 하나를 고르라고만 한다.
    "pairwise": ("Consider the traveler information and the two options shown, then give the "
                 "decision. The decision is one of two: 'AV' or 'Keep'."),
    # noatt = zeroshot 과 **같은 지시문**. 다른 점은 선택 후 태도 블록이 빠지는 것뿐이라,
    # 두 팔의 차이가 변수 집합 하나로만 설명되게 한다.
    # ab = 대칭 선택집합. 두 대안을 같은 자격으로 놓고 하나를 고르게 한다.
    "ab": ("Work through this traveler's situation first, then give the decision. "
           "Two options are available for this trip; the decision is which one this "
           "traveler takes."),
    "noatt": ("Work through this traveler's situation first, then give the decision. "
              "The decision is one of two: 'AV' if they switch to the driverless AV "
              "ride-hailing service, 'Keep' if they stay with the mode they used in 2026."),
    # explained = 끝맺음에서 두 라벨의 **뜻을 풀어 설명**한다 (2026-08-23, 사용자 가설:
    # llama 의 전-AV 는 맨몸 토큰 'Keep' 이 "현재 수단을 계속 쓴다"로 접지되지 않아서일
    # 수 있다). 두 선택지를 같은 자격으로 한 줄씩 서술할 뿐, 저울질 지침은 없다.
    "explained": ("Work through this traveler's situation first, then give the decision. "
                  "Two options for this trip are described at the end; the decision is "
                  "which one this traveler actually takes."),
    # abswap = ab 와 지시문까지 동일, 선택지 **제시 순서만** 뒤집는다 (위치 편향 검정).
    "abswap": ("Work through this traveler's situation first, then give the decision. "
               "Two options are available for this trip; the decision is which one this "
               "traveler takes."),
    # ── 2026-08-24. 여기부터 두 팔은 **문구 변형이 아니라 과제 자체를 바꾼다.**
    #
    # 진단(D273/D274): llama 는 라벨 뜻을 알고, 선택지 위치에 무관하며, 이동시간 하나로
    # 결정한다. 즉 이 과제를 "속성이 더 나은 대안 고르기"로 풀고 있다. 그런데 그렇게 풀도록
    # **우리 시스템 프롬프트가 유도하고 있었다** - "Weigh the whole situation - travel time,
    # waiting time, cost, crowding ... - and answer as this traveler would realistically
    # behave." 이 문장은 속성을 열거하고 저울질을 지시한다. 이 파일 466행의 원칙("프롬프트는
    # 상황을 서술하되 판단 방법을 지시하지 않는다")과 어긋난다.
    #
    # respondent = 과제를 **기록된 응답의 예측**으로 되돌린다. 정답 라벨은 실제로 이 사람이
    # 설문에서 기록한 답이므로 사실 서술이다. 방향을 암시하는 말(채택률·VOT·저울질 지침)은
    # 넣지 않는다.
    "respondent": ("Work through this respondent's situation first, then state the recorded "
                   "response. The recorded response is one of two: 'AV' if this respondent "
                   "recorded switching to the driverless AV ride-hailing service, 'Keep' if "
                   "this respondent recorded staying with the mode they used in 2026."),
    # plain = 저울질 지시문을 **덜어내기만** 한다. 추가하는 문장이 없으므로 어느 방향으로도
    # 밀 수 없는 팔이고, respondent 가 움직였을 때 그 원인이 과제 재정의인지 지시문 제거인지
    # 가르는 대조군이 된다.
    "plain": "Answer 'AV' or 'Keep', then give one or two sentences of reasoning.",
    # fresh = 백지 신작. system 은 FRESH_SYSTEM 전용이라 이 표의 값은 쓰이지 않지만,
    # STYLE 검증을 위해 자리는 둔다.
    "fresh": "",
    # named = 2026-08-25. **답 라벨에서 'Keep' 이라는 추상어를 없앤다.**
    #
    # 왜. 지금까지 답은 'AV' 대 'Keep' 이었다. 'AV' 는 수단 이름이고 'Keep' 은 행동을
    # 가리키는 말이라 두 선택지가 같은 종류가 아니다. 게다가 프롬프트 본문에 'AV' 라는
    # 글자가 수십 번 나오는 반면 'Keep' 은 답 안내에서만 나온다 - 두 후보의 사전확률이
    # 애초에 기울어 있다(문헌의 surface form competition).
    #
    # 그래서 두 답을 **둘 다 수단 이름**으로 만든다. 답 칸은 { } 로 비워 두고, 그 안에
    # 두 이름 중 하나를 그대로 옮겨 적게 한다. 어느 쪽으로도 미는 말은 없다.
    "named": ("Read the record, then name the mode this respondent uses for the trip in "
              "2030. Copy one of the two mode names exactly as it is written."),
    # namedswap = named 와 문구가 한 글자도 다르지 않고, 두 수단 이름의 **제시 순서만**
    # 뒤집는다 (named 는 AV 가 먼저). 3B 가 A/B 판에서 400건 전부 두 번째 항목을 골랐으므로
    # 동결 양식에서도 위치에 반응하는지 본다. abswap(D274) 과 같은 종류의 통제다.
    "namedswap": ("Read the record, then name the mode this respondent uses for the trip in "
                  "2030. Copy one of the two mode names exactly as it is written."),
    # neutral / neutralswap = 2026-08-25. **답 토큰을 의미 없는 글자 A/B 로 둔다.**
    #
    # 왜. named 로 'Keep' 은 없앴지만 남은 비대칭이 실측됐다: 한 사례의 프롬프트에서
    # 'AV' 는 19회, 현재 수단 이름은 2회 나온다(태도 문항 9회가 가장 큰 몫인데 그건
    # 자료라 뺄 수 없다). 자주 본 문자열이 확률을 더 받는 문제(surface form competition)가
    # 그대로 남는다. A/B 는 둘 다 프롬프트에 거의 없고 의미도 없어 이 비대칭이 사라진다.
    #
    # 두 판을 **쌍으로** 돌린다: neutral 은 A=AV, neutralswap 은 A=현재수단. 글자 자체의
    # 사전확률(모형이 A 를 선호하는 성향)은 두 판의 P(AV) 를 평균해 상쇄한다.
    # 어느 쪽으로 답하라는 말은 여전히 없다.
    # shortnamed = fresh 의 짧은 본문 + named 의 수단이름 답 (system 은 전용이라 미사용).
    "shortnamed": "",
    "neutral": ("Read the record, then give the letter of the mode this respondent uses "
                "for the trip in 2030."),
    "neutralswap": ("Read the record, then give the letter of the mode this respondent "
                    "uses for the trip in 2030."),
    # card = 골격을 새로 쓴 양식(아래 card_user 참조). 과제 서술은 respondent 와 같다.
    "card": ("Work through this respondent's record first, then state the answer they "
             "recorded. The recorded answer is one of two: 'AV' if this respondent recorded "
             "starting to use the driverless AV ride-hailing service, 'Keep' if this "
             "respondent recorded continuing with the mode they used in 2026."),
}
CLOSING_QUESTION = {
    "sft": "Answer 'AV' to switch, or 'Keep' to stay with %s.",
    "zeroshot": "Decide as this traveler would: 'AV' to switch, or 'Keep' to stay with %s.",
    "pairwise": None,     # pairwise 는 아래 decision_block 이 대신한다
    "noatt": "Decide as this traveler would: 'AV' to switch, or 'Keep' to stay with %s.",
    "ab": None,     # ab 는 choice_set_block 이 대신한다
    "explained": None,   # explained 는 future_2030_block 의 전용 분기가 대신한다
    "abswap": None,      # abswap 은 choice_set_block(swap=True) 이 대신한다
    "respondent": None,  # 아래 future_2030_block 의 전용 분기가 대신한다
    "plain": None,       # 〃
    "card": None,        # card 는 card_user 가 통째로 만든다
    "fresh": None,       # fresh 는 FRESH_SYSTEM + fresh_user 가 통째로 만든다
    "named": None,       # named 는 card_user 의 전용 마무리를 쓴다
    "namedswap": None,   # 〃 (두 이름의 순서만 반전)
    "shortnamed": None,  # fresh 본문 + named 답 (길이 가설 검정)
    "neutral": None,     # 〃 (A=AV)
    "neutralswap": None, # 〃 (A=현재수단)
}

# 현재 수단 코드 -> (이동시간, 대기시간, 요금) 열. baseline_2026_block 의 spec 과 같은 열이다.
CUR_COLS = {"PT": ("pt_travel_time_min", "pt_wait_time_min", "pt_cost_1000won"),
            "Car": ("car_travel_time_min", "car_wait_time_min", "car_cost_1000won"),
            "PM": ("pm_travel_time_min", "pm_wait_time_min", "pm_cost_1000won"),
            "Walk": ("walk_travel_time_min", None, None)}


def decision_block(r, av_line) -> str:
    """비교 대상 **둘만** 나란히 놓는다.

    **새 정보가 하나도 없다.** KEEP 줄은 2026 블록을 만든 `_mode_line` 을 이 응답자의 실제
    현재 수단에 다시 불러 만든 것이라, 문구도 숫자도 위에 이미 있던 것과 같다. 위치만 옮긴다.

    **왜 옮기는가.** 기존 프롬프트는 2030 블록에 AV 한 줄만 두고 "keep using Walking" 이라고만
    한다. 비교 상대의 숫자는 위쪽 2026 블록에 다른 3~4개 수단과 섞여 있어 모델이 스스로 찾아야
    했고, 2B 는 그 검색에 41.3% 실패해 엉뚱한 수단과 비교했다(실측).

    **넣지 않는 것.** 시간·비용 저울질 지침, 빨라지면 비싸도 된다는 말, 사람의 AV 채택률,
    VOT, 탄력성, AV 를 권하거나 현상유지를 깎는 표현, AV 비율을 올리려고 고른 어떤 문구.
    표현의 중립성과 측정 성립 여부로만 판단한다. AV 비율은 목표가 아니라 진단으로만 본다.
    """
    code = str(r.get("current_mode_label"))
    cols = CUR_COLS.get(code)
    keep = (_mode_line(r, code, *cols) if cols else None) or _mode(code)
    return "\n".join([
        "[Decision to make]",
        "The traveler must choose between exactly these two options for the same trip.",
        "",
        "KEEP - continue using %s, unchanged from 2026:" % _mode(code),
        "  %s" % keep,
        "",
        "AV - switch to %s:" % _mode("AV"),
        "  %s" % av_line,
    ])


def choice_set_block(r, av_line, swap: bool = False) -> str:
    """두 대안을 **대칭으로** 놓는다. 어느 쪽도 기본값으로 틀 짓지 않는다.

    **왜 바꾸는가 (실측 근거).** 기존 문구는 "switch to AV, or keep using X **exactly as in
    2026**" 이라 현상유지 쪽이 언어적 기본값이 된다. 9B 는 AV 가 **더 싼** 100건에서도 99건을
    Keep 했고(사람은 60% 가 AV), 이유를 보면 대안과 비교하지 않고 AV 요금을 절대 기준으로
    판단한다 - "the cost of 16,500 KRW is still too high" (같은 사례의 현재 수단은 20,100원).
    즉 비교 자체가 일어나지 않았다.

    그래서 (a) 두 대안을 같은 서식·같은 자격으로 나란히 놓고, (b) switch/stay/exactly as in
    2026 같은 비대칭 표현을 뺀다. 실제 SP 설문이 대안을 대칭으로 제시하는 것과 같은 방식이다.

    **넣지 않는 것.** 시간·비용 저울질 지침, 차이값 계산, 채택률, VOT, 어느 쪽을 고르라는
    어떤 암시. 정보는 하나도 추가하지 않는다 - Keep 줄은 2026 블록과 같은 `_mode_line` 이다.
    """
    code = str(r.get("current_mode_label"))
    cols = CUR_COLS.get(code)
    keep = (_mode_line(r, code, *cols) if cols else None) or _mode(code)
    # swap=True 는 **제시 순서만** 뒤집는다 (abswap 양식, 2026-08-24). 문구는 한 글자도 다르지
    # 않다. ab 에서 qwen 9B 는 첫 옵션(Keep)으로, llama 는 둘째 옵션(AV)으로 쏠렸으므로,
    # 순서를 뒤집었을 때 라벨이 따라가면 붕괴가 내용이 아니라 위치 편향이라는 증거가 된다.
    opts = ['Option "Keep" - %s' % keep, 'Option "AV" - %s' % av_line]
    if swap:
        opts = opts[::-1]
    return "\n".join([
        "[Decision to make]",
        "Under the 2030 conditions this traveler makes one trip. Two options are available:",
        "",
        opts[0],
        "",
        opts[1],
        "",
        "Which option does this traveler take?",
    ])


def system_prompt(style: str = "sft") -> str:
    L = labels()
    return (
        "You are answering a stated-preference travel survey as the traveler described below. "
        + L["survey_note"] + "\n\n"
        "The same trip situation is shown twice. Under 2026 conditions the traveler chose "
        "among the existing modes. Under 2030 conditions the existing modes keep exactly the "
        "same conditions and a driverless AV ride-hailing service is added.\n\n"
        # AV 가 무엇인지 한 줄로 정의한다. 설문지 p11 의 mode definition 이고, AAAI 렌더러도
        # 같은 문장을 갖고 있었다. 이 문장이 없으면 모델이 "AV" 를 자가용 자율주행차로
        # 읽을 수 있는데, 여기서는 앱으로 부르는 무인 호출 서비스다.
        "Driverless AV ride-hailing means an app-requested, driverless on-demand vehicle "
        "service - used like app-based ride-hailing but with no human driver. The fare shown "
        "is the one-way amount the traveler personally pays. Car travel (non-AV) means "
        "driving oneself or riding with family or friends; it excludes taxi and ride-hailing."
        "\n\n"
        "\"Operation stability\" and \"service abnormality\" describe the AV service, not a "
        "crash risk. " + L["framing_note"] + "\n\n"
        + _judgement_note(style)
        + ANSWER_INSTRUCTION[style]
    )


def _judgement_note(style: str) -> str:
    """판단 지시문. **respondent/plain 은 이 문장을 쓰지 않는다** (2026-08-24).

    기존 문장은 속성을 열거하고 "weigh" 하라고 지시한다. D273/D274 로 llama 의 붕괴가
    속성 하나에 고정된 결정 규칙이라는 것이 밝혀졌는데, 그 규칙을 우리 지시문이 부추기고
    있었다. 두 새 팔은 지시문을 **덜어내거나**(plain), 과제를 기록된 응답의 예측으로
    **되돌린다**(respondent). 어느 쪽으로 답하라는 암시는 넣지 않는다.
    """
    if style == "plain":
        return ""
    if style in ("respondent", "card", "named", "namedswap", "neutral", "neutralswap"):
        return (
            "The answer below is the response this particular respondent recorded in the "
            "survey. Respondents facing the same numbers recorded different answers, so the "
            "answer is not determined by the numbers alone. State what this individual "
            "recorded, not what a typical or ideal traveler would do.\n\n"
        )
    return (
        "There is no correct answer. Weigh the whole situation - travel time, waiting time, "
        "cost, crowding, how well the current mode already serves this trip, and the "
        "conditions offered by the AV service - and answer as this traveler would "
        "realistically behave.\n\n"
    )


# ------------------------------------------------------------------ helpers

def _num(v):
    return "unknown" if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)


def _lab(var, v, default="not reported"):
    """코드값을 코드북 문구로 바꾼다."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return labels()["values"].get(var, {}).get(int(v), default)


def _krw(v_kkrw) -> str:
    if v_kkrw is None or (isinstance(v_kkrw, float) and np.isnan(v_kkrw)):
        return "not applicable"
    return "{:,} KRW".format(int(round(float(v_kkrw) * 1000)))


def _age(r) -> str:
    for k in ("SQ3", "age"):
        v = r.get(k)
        if v is not None and pd.notna(v):
            return str(int(v))
    return "unknown"


def _mode(code) -> str:
    return labels()["modes"].get(str(code), str(code))


# ------------------------------------------------------------------ blocks

def persona_block(r) -> str:
    # SQ1_1/SQ1_2 는 한글 지명이다. 영어 프롬프트에 한글이 섞이면 토크나이저가 낯선
    # 조각으로 쪼개고 모델이 지역을 못 읽는다. p14b 가 DATA_EN 에서 뽑아둔 로마자 표기를 쓴다.
    reg = labels().get("regions", {})
    where = [reg.get(str(r.get(k)), str(r.get(k))) for k in ("SQ1_2", "SQ1_1")
             if r.get(k) is not None and pd.notna(r.get(k))]
    n_hh = _num(r.get("D4_1"))
    kids = _num(r.get("D4_2"))
    hh = ("Lives alone" if n_hh == 1 else "Household of %s people" % n_hh)
    hh += ("" if kids == "unknown" else
           (" with no children under 18" if kids == 0 else
            " with %d child%s under 18" % (kids, "" if kids == 1 else "ren")))
    return "\n".join([
        "[Traveler]",
        "Lives in %s. Age %s, %s. Occupation: %s. Education: %s. Marital status: %s."
        % (", ".join(where) if where else "the Seoul Capital Area", _age(r),
           _lab("SQ2", r.get("SQ2"), "unknown"), _lab("D1", r.get("D1"), "not reported"),
           _lab("D2", r.get("D2"), "not reported"), _lab("D3", r.get("D3"), "not reported")),
        "%s; monthly household income %s." % (hh, _lab("D5", r.get("D5"))),
        "Household vehicles: %s. Driving licence: %s. Car access when needed: %s. "
        "When using a conventional non-AV car: %s."
        % (_lab("D6", r.get("D6")), _lab("D7", r.get("D7"), "not reported"),
           _lab("D8", r.get("D8")), _lab("D9", r.get("D9"))),
    ])


def region_block(r) -> str:
    """통행 O/D 와 거주지 공간 특성.

    거주지만 주면 모델이 통행이 어디로 가는지 모른다. 도착지는 거주지와 23.7% 만 같아
    실질적인 새 정보다. 공간 특성(지역 유형·대중교통 접근성·혼잡·주차 압력)은 지역 통계에서
    유도한 값이라 선택 이후 정보가 아니다. 지명만 주면 한국 도시지리를 모르는 모델이
    해석할 수 없으므로 이 속성들을 함께 준다.
    """
    reg = labels().get("regions", {})
    def g(k):
        v = r.get(k)
        return None if v is None or pd.isna(v) else reg.get(str(v), str(v))
    o1, o2, d1, d2 = g("trip_origin_sido"), g("trip_origin_sigungu"), \
                     g("trip_dest_sido"), g("trip_dest_sigungu")
    lines = ["[Trip location context]"]
    if o2 or d2:
        lines.append("Usual weekday-morning trip runs from %s to %s."
                     % (", ".join(x for x in (o2, o1) if x) or "an unreported origin",
                        ", ".join(x for x in (d2, d1) if x) or "an unreported destination"))
    attrs = [("residence_region_type", "area type"),
             ("residence_urbanization", "urbanisation"),
             ("residence_transit_accessibility", "transit accessibility"),
             ("residence_congestion_prior", "road congestion"),
             ("residence_parking_pressure", "parking pressure"),
             ("residence_first_last_mile_burden", "first/last-mile burden")]
    got = ["%s %s" % (lab, str(r.get(k)).replace("_", " "))
           for k, lab in attrs if r.get(k) is not None and pd.notna(r.get(k))]
    if got:
        lines.append("Residence-side context: " + "; ".join(got) + ".")
    return "\n".join(lines) if len(lines) > 1 else ""


def usual_trip_block(r) -> str:
    dep, arr = r.get("QQ2_1"), r.get("QQ2_2")
    when = ("departs around %02d:00 and arrives around %02d:00"
            % (int(dep), int(arr))) if (pd.notna(dep) and pd.notna(arr)) else "usual timing not reported"
    return "\n".join([
        "[Usual weekday-morning trip]",
        "Purpose: %s. Typically %s." % (_lab("QQ1", r.get("QQ1"), "not reported"), when),
        "Usual total duration %s; usual out-of-pocket cost %s."
        % (_lab("QQ3", r.get("QQ3")), _lab("QQ4", r.get("QQ4"))),
        "Most often travels by %s, mainly because of %s, usually %s."
        % (_lab("QQ5", r.get("QQ5"), "an unreported mode"),
           _lab("QQ6", r.get("QQ6"), "an unreported reason"),
           _lab("QQ7", r.get("QQ7"), "not reported")),
    ])


def _mode_line(r, code, t_col, w_col, c_col) -> str | None:
    t = r.get(t_col)
    if t is None or pd.isna(t):
        return None
    bits = ["travel time %.0f min" % float(t)]
    w = r.get(w_col) if w_col else None
    if w is not None and pd.notna(w):
        bits.append("out-of-vehicle + waiting time %.0f min" % float(w))
    c = r.get(c_col) if c_col else None
    bits.append("total cost %s" % (_krw(c) if c is not None and pd.notna(c) else "0 KRW"))
    if code == "PT":
        n = r.get("pt_transfer_count")
        if pd.notna(n):
            bits.append("%d transfer%s" % (int(n), "" if int(n) == 1 else "s"))
        cw = labels()["crowding"].get(str(r.get("pt_crowding_raw")))
        if cw:
            bits.append("crowding %s" % cw)
    return "%s: %s" % (_mode(code), ", ".join(bits))


_REASON_CACHE = {}
# 2026 선택 이유를 사람이 읽는 말로. 원값은 cost / time / wait_access_transfer /
# familiar_mode / crowding_comfort / driving_parking_fatigue / other 다.
_REASON_TEXT = {
    "cost": "cost", "time": "travel time",
    "wait_access_transfer": "waiting, access or transfer burden",
    "familiar_mode": "being used to that mode",
    "crowding_comfort": "crowding and comfort",
    "driving_parking_fatigue": "driving and parking fatigue",
    "other": "another reason",
}


def reason_2026(case_id):
    """그 거리대에서 **2026 수단을 고른 이유** 를 최대 3개까지 순서대로 돌려준다.

    출처는 원본 패널의 current_rating_1/2/3_2026_label 이다. 거리대마다 따로 물은
    문항이라 같은 응답자도 거리대별로 답이 다르다 (2,178명 중 1,705명).

    **이 변수는 2026-08-05 에 exclude 로 판정돼 있었다** (trc_feature_schemas.yaml 의
    resolved). 근거는 "SP 블록 안에서 방금 고른 2026 수단에 조건부로 물었으므로
    sp_contaminated" 였다. 2026-08-09 에 그 판정을 뒤집는다 - 조건이 되는 2026 선택을
    프롬프트가 이미 명시적으로 알려주고 있고("Under these 2026 conditions the traveler
    chose: PT"), 우리가 예측하는 것은 2030 선택이다. 2030 이후 정보(av_reason_*,
    AV_reason_2030_*)는 그대로 excluded.post_choice 로 둔다.

    선행 AAAI 프로젝트도 같은 항목을 프롬프트에 넣었다
    ("Stated 2026 reason(s): cost, time.").

    **표 모형에는 아직 이 변수가 없다.** LLM 에만 주면 비교가 기울어지므로,
    표 모형 쪽도 넣을지는 따로 판단해야 한다.
    """
    if not _REASON_CACHE:
        cfg = load_config()
        src = ROOT / cfg["data"]["panel"]
        cols = ["case_id"] + ["current_rating_%d_2026_label" % i for i in (1, 2, 3)]
        d = pd.read_parquet(src)
        if "case_id" not in d.columns:            # 패널은 respondent_id + distance_code 로 키를 만든다
            d = d.assign(case_id=d.respondent_id.astype(str) + "_" + d.distance_code.astype(str))
        d = d[[c for c in cols if c in d.columns]]
        for _, row in d.iterrows():
            vals = [row.get("current_rating_%d_2026_label" % i) for i in (1, 2, 3)]
            vals = [_REASON_TEXT.get(str(v), str(v)) for v in vals if pd.notna(v)]
            _REASON_CACHE[str(row["case_id"])] = ", ".join(vals)
    return _REASON_CACHE.get(str(case_id), "")


def baseline_2026_block(r) -> str:
    avail = str(r.get("available_modes") or "").split("+")
    spec = [("PT", "pt_travel_time_min", "pt_wait_time_min", "pt_cost_1000won"),
            ("Car", "car_travel_time_min", "car_wait_time_min", "car_cost_1000won"),
            ("PM", "pm_travel_time_min", "pm_wait_time_min", "pm_cost_1000won"),
            ("Walk", "walk_travel_time_min", None, None)]
    L = labels()
    band = str(r.get("distance_band"))
    km = L.get("band_distance_km", {}).get(band)
    sit = L.get("band_situation", {}).get(band, "")
    head = ("[2026 conditions - one one-way trip of about %s km]" % km) if km else \
           ("[2026 conditions - one one-way trip of about %s]"
            % (r.get("distance_label") or band))
    lines = [head]
    # 거리대 상황설명("Imagine a trip from home to a nearby local destination.")은 뺀다.
    # 대표거리가 머리말에 이미 있어 상황을 특정하는 정보가 중복이고, 매 건 반복되는
    # 토큰 비용만 남는다. 설정(band_situation)에는 남겨두어 필요하면 되살릴 수 있게 한다.
    _ = sit
    for code, t, w, c in spec:
        if code not in avail:
            continue
        ln = _mode_line(r, code, t, w, c)
        if ln:
            lines.append(ln)
    lines.append("Under these 2026 conditions the traveler chose: %s."
                 % _mode(r.get("current_mode_label")))
    why = reason_2026(r.get("case_id"))
    if why:
        lines.append("Reasons given for that 2026 choice: %s." % why)
    return "\n".join(lines)


def future_2030_block(r, fare, ride, wait, rel=None, style: str = "sft") -> str:
    """rel: 신뢰도(%) 덮어쓰기. 반사실 편집 전용이다.

    Phase 9 격자의 A_* 시나리오는 요금·시간을 기준값에 두고 신뢰도만 ±0.1~0.3%p 움직이는데,
    이 함수가 신뢰도를 r(av_framing_percent)에서만 읽으면 그 편집이 프롬프트에 들어올 길이
    없다 - A_* 여섯 조건이 전부 BASE 와 같은 문자열이 되고, 오류도 안 난다. 실제로
    2026-08-11 p18 점검에서 그렇게 될 뻔했다. 기본값 None 이면 기존과 바이트 단위로
    동일하므로 학습 프롬프트는 변하지 않는다.
    """
    L = labels()
    fr = L["framing_attribute"].get(str(r.get("av_framing_attribute")), "Operation stability")
    pct = r.get("av_framing_percent")
    if rel is not None and pd.notna(rel):
        # 격자의 rel 은 **항상 안정성(%) 척도**다 (Phase 9 확인: 이상률 프레이밍 1,710건에서
        # rel_base = 100 - av_framing_percent, 오차 0). 이 사례의 프레이밍이 이상률이면
        # 100-rel 로 되돌려 넣어야 한다. 안 하면 "Service abnormality 0.005%" 사례가
        # "Service abnormality 99.995%" 로 렌더된다 - 실제로 첫 검증에서 BASE 200건 중
        # 90건이 그렇게 바뀌었다.
        pct = float(rel) if "abnormality" not in fr.lower() else 100.0 - float(rel)
    fr_txt = ("%s %s%%" % (fr, ("%.3f" % float(pct)).rstrip("0").rstrip("."))
              if pd.notna(pct) else fr + " not reported")
    cur = _mode(r.get("current_mode_label"))
    # AV 속성은 한 줄이어야 한다 (편집 diff 불변식).
    av_line = ("%s: travel time %.0f min, out-of-vehicle + waiting time %.0f min, "
               "total cost %s, %s"
               % (_mode("AV"), float(ride), float(wait), _krw(fare), fr_txt))
    if style in ("ab", "abswap"):
        return "\n".join([
            "[2030 conditions - same trip, existing modes unchanged, AV added]",
            av_line,
            "",
            choice_set_block(r, av_line, swap=(style == "abswap")),
        ])
    if style == "pairwise":
        return "\n".join([
            "[2030 conditions - same trip, existing modes unchanged, AV added]",
            av_line,
            "",
            decision_block(r, av_line),
        ])
    if style in ("respondent", "plain"):
        code = str(r.get("current_mode_label"))
        cols = CUR_COLS.get(code)
        keep = (_mode_line(r, code, *cols) if cols else None) or _mode(code)
        if style == "plain":
            return "\n".join([
                "[2030 conditions - same trip, existing modes unchanged, AV added]",
                av_line,
                "",
                "Which does this traveler use for this trip in 2030?",
                'Answer "AV" for the driverless AV ride-hailing service above, '
                'or "Keep" for %s.' % cur,
            ])
        return "\n".join([
            "[2030 conditions - same trip, existing modes unchanged, AV added]",
            av_line,
            "",
            "[Response to state]",
            "This respondent recorded one of two answers for the 2030 scenario:",
            "",
            '"AV" - recorded using the driverless AV ride-hailing service: %s' % av_line,
            "",
            '"Keep" - recorded using %s, the same mode as in 2026: %s' % (cur, keep),
            "",
            "Which answer did this respondent record?",
        ])
    if style == "explained":
        # 라벨 뜻을 풀어 쓴 끝맺음. 두 선택지는 같은 문형 한 줄씩 - 비대칭 문구를 넣으면
        # 그때부터 프롬프트가 답을 미는 것이 된다.
        return "\n".join([
            "[2030 conditions - same trip, existing modes unchanged, AV added]",
            av_line,
            "",
            "Which option does this traveler actually take for this trip in 2030?",
            "- Switch to the driverless AV ride-hailing service shown above. "
            "If so, answer 'AV'.",
            "- Keep traveling by %s, the same mode as in 2026. If so, answer 'Keep'." % cur,
        ])
    return "\n".join([
        "[2030 conditions - same trip, existing modes unchanged, AV added]",
        av_line,
        "",
        "Under the 2030 conditions, does this traveler switch to %s, or keep using %s "
        "exactly as in 2026?" % (_mode("AV"), cur),
        CLOSING_QUESTION[style] % cur,
    ])


# 태도 블록에 "이건 결정적이지 않다" 는 판단 지침은 **넣지 않는다**.
# pro/retro 대비가 재려는 것이 바로 "모형이 태도에 얼마나 기대는가" 인데, 태도를 주면서
# 덜 쓰라고 지시하면 그 대비가 모형의 성질이 아니라 우리 지시의 결과가 된다.
# 프롬프트는 상황을 서술하되 판단 방법을 지시하지 않는다는 원칙이다.
# 머리말의 "recorded after the choice tasks" 는 사실 서술이므로 남긴다.
def attitude_block(r) -> str:
    items = []
    for k, lab in ATTITUDE_LABELS.items():
        v = r.get(k)
        if v is not None and pd.notna(v):
            items.append("%s: %s/5" % (lab, int(v)))
    sel = [k for k in r.index if k.endswith("_sel") and r[k] == 1]
    if sel:
        items.append("shared-AV concerns reported: %d" % len(sel))
    return ("[Stated attitudes - recorded after the choice tasks]\n"
            + "; ".join(items) + ".")


# ═══════════════════════════════════════════════════════════════════════════
# fresh - 2026-08-24 저녁. **백지에서 다시 쓴** 양식. 기존 골격의 문장을 하나도 가져오지
# 않는다 (사용자 지시: "아예 새롭게"). 설계 원칙:
#   - 과제를 처음부터 **예측**으로 명시한다: "실제 답은 숨겨져 있다. 예측하라."
#     ("따져 본 뒤 답하라" 류의 사고-먼저 지시를 없애 answer-first suffix 와의 긴장 제거)
#   - 전체를 절반 이하로 압축한다 (4,600자 -> 약 2,000자). 변수는 하나도 버리지 않는다 -
#     같은 정보를 짧은 문장으로 다시 쓸 뿐이다.
#   - 끝은 두 대안(Keep=본인의 2026 수단, AV)만 숫자와 함께 대칭으로 세운다.
#   - 저울질 지침·채택률·VOT·방향 암시는 없다 (D256).
FRESH_SYSTEM = (
    "You are predicting one hidden answer from a South Korean travel survey "
    "(Seoul Capital Area, one ordinary weekday-morning trip; all figures are for a "
    "single one-way trip).\n\n"
    "The respondent answered this question: when a driverless AV ride-hailing service "
    "is added in 2030 to a trip they described, do they start using it, or continue "
    "exactly as in 2026? Their actual answer is hidden. Your task is to predict it. "
    "Respondents facing the same numbers gave different answers.\n\n"
    "Driverless AV ride-hailing = an on-demand vehicle called by app, with no human "
    "driver; the fare shown is what the respondent personally pays one way. "
    "'Operation stability' is the chance the dispatched service completes the trip "
    "without a service disruption - it is not a crash risk. Car travel (non-AV) means "
    "driving oneself or riding with family or friends, not taxi.\n\n"
    "Reply with one word on the first line: AV (they start using the driverless "
    "service) or Keep (they continue with their 2026 mode unchanged). Then give one "
    "or two sentences of reasoning."
)


# shortnamed = 2026-08-25. **fresh 의 짧은 본문 + named 의 수단이름 답.**
# 길이 가설을 검정하려고 만든다: fresh(3,227자)는 옛 라벨(AV/Keep)이라 3B 가 200/200 AV
# 였고, named(4,602자)는 새 라벨이라 132/68 이었다. 둘을 합치면 "길이가 원인인가"만 남는다.
# 변수는 하나도 빼지 않는다 - 말만 줄인 fresh 본문을 그대로 쓴다.
FRESH_SYSTEM_NAMED = (
    FRESH_SYSTEM.rsplit("\n\n", 1)[0] + "\n\n"
    "Reply on the first line with the name of the mode this respondent uses in 2030, "
    "copied exactly from the two names given at the end. Then give one or two sentences "
    "of reasoning."
)


def fresh_user(r, info: str, fare, ride, wait, rel=None, style: str = "fresh") -> str:
    L = labels()
    reg = L.get("regions", {})

    def g(k):
        v = r.get(k)
        return None if v is None or pd.isna(v) else reg.get(str(v), str(v))

    # 사람 - 한 문단
    n_hh, kids = _num(r.get("D4_1")), _num(r.get("D4_2"))
    hh = "lives alone" if n_hh == 1 else "household of %s" % n_hh
    if kids != "unknown" and kids != 0:
        hh += " incl. %s under 18" % kids
    who = ("%s, age %s. %s; education: %s; marital status: %s. %s in %s; monthly "
           "household income %s. Vehicles at home: %s; driving licence: %s; car "
           "availability: %s; when using a non-AV car: %s."
           % (_lab("SQ2", r.get("SQ2"), "sex not reported"), _age(r),
              _lab("D1", r.get("D1"), "occupation not reported"),
              _lab("D2", r.get("D2"), "not reported"),
              _lab("D3", r.get("D3"), "not reported"),
              hh[0].upper() + hh[1:],
              ", ".join(x for x in (g("SQ1_2"), g("SQ1_1")) if x) or "the Seoul Capital Area",
              _lab("D5", r.get("D5")), _lab("D6", r.get("D6")),
              _lab("D7", r.get("D7"), "not reported"), _lab("D8", r.get("D8")),
              _lab("D9", r.get("D9"))))

    # 평소 통행 + 거주지 맥락 - 한 문단
    dep, arr = r.get("QQ2_1"), r.get("QQ2_2")
    when = ("around %02d:00-%02d:00" % (int(dep), int(arr))
            if pd.notna(dep) and pd.notna(arr) else "timing not reported")
    od = ""
    o2, d2 = g("trip_origin_sigungu"), g("trip_dest_sigungu")
    if o2 or d2:
        od = " from %s to %s" % (o2 or "an unreported origin", d2 or "an unreported destination")
    ctx = [("residence_region_type", "area type"),
           ("residence_urbanization", "urbanisation"),
           ("residence_transit_accessibility", "transit access"),
           ("residence_congestion_prior", "congestion"),
           ("residence_parking_pressure", "parking pressure"),
           ("residence_first_last_mile_burden", "first/last-mile burden")]
    ctx_txt = "; ".join("%s %s" % (lab, str(r.get(k)).replace("_", " "))
                        for k, lab in ctx if r.get(k) is not None and pd.notna(r.get(k)))
    trip = ("Usual weekday-morning trip%s: %s, %s; total duration %s; out-of-pocket "
            "cost %s; most often by %s (main reason: %s; travels %s). Residence area: %s."
            % (od, _lab("QQ1", r.get("QQ1"), "purpose not reported"), when,
               _lab("QQ3", r.get("QQ3")), _lab("QQ4", r.get("QQ4")),
               _lab("QQ5", r.get("QQ5"), "an unreported mode"),
               _lab("QQ6", r.get("QQ6"), "an unreported reason"),
               _lab("QQ7", r.get("QQ7"), "not reported"), ctx_txt or "not reported"))

    # 태도 - 한 문단 (retro 만)
    att = ""
    if info == "retro":
        items = ["%s %d" % (lab, int(r.get(k))) for k, lab in ATTITUDE_LABELS.items()
                 if r.get(k) is not None and pd.notna(r.get(k))]
        sel = [k for k in r.index if k.endswith("_sel") and r[k] == 1]
        if sel:
            items.append("shared-AV concerns %d" % len(sel))
        att = ("Self-ratings, 1 disagree - 5 agree (asked after the choice tasks): "
               + "; ".join(items) + ".")

    # 문제의 통행 - 2026 조건 한 문단
    band = str(r.get("distance_band"))
    km = L.get("band_distance_km", {}).get(band)
    head = "about %s km one way" % km if km else str(r.get("distance_label") or band)
    avail = str(r.get("available_modes") or "").split("+")
    parts = []
    for code, t, w, c in _CARD_SPEC:
        if code not in avail:
            continue
        cells = _card_cells(r, code, t, w, c)
        if cells:
            name, tt, ww, cc, note = cells
            parts.append("%s %s + %s wait, %s%s"
                         % (name, tt, ww if ww != "-" else "no", cc,
                            (" (%s)" % note) if note else ""))
    cur = _mode(r.get("current_mode_label"))
    why = reason_2026(r.get("case_id"))
    base = ("The trip in question (%s). Options and conditions in 2026: %s. "
            "They chose %s%s." % (head, "; ".join(parts), cur,
                                  (" - stated reasons: %s" % why) if why else ""))

    # 2030 질문 - 두 대안만 대칭으로
    code = str(r.get("current_mode_label"))
    cols = CUR_COLS.get(code)
    keep_line = (_mode_line(r, code, *cols) if cols else None) or cur
    av_line = ("%s: travel time %.0f min, out-of-vehicle + waiting time %.0f min, "
               "total cost %s, %s"
               % (_mode("AV"), float(ride), float(wait), _krw(fare), _av_frame_text(r, rel)))
    if style == "shortnamed":
        q = "\n".join([
            "In 2030 every 2026 option stays exactly the same, and one option is added.",
            "The two possible modes for this trip:",
            "  %s" % av_line,
            "  %s" % keep_line,
            "",
            "Which of the two does this respondent use? Write that mode name on the first "
            "line, copied exactly, in the braces:",
            "  { }",
        ])
    else:
        q = "\n".join([
            "In 2030 every 2026 option stays exactly the same, and one option is added.",
            "The two possible answers:",
            "  Keep - continue with %s" % keep_line,
            "  AV   - %s" % av_line,
            "",
            "Predict this respondent's hidden answer. First line: AV or Keep.",
        ])

    blocks = [who, trip]
    if att:
        blocks.append(att)
    blocks += [base, q]
    return "\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════════════
# card - 2026-08-24 에 **골격부터** 새로 쓴 양식. 지금까지 12팔은 같은 골격에 문구만
# 바꾼 것이었고 전부 전-AV 로 붕괴했다(D271/D273/D274). 원문을 통째로 읽고 나서 드러난
# 구조적 결함 세 가지를 고친다.
#
#   (1) 2030 블록에 **AV 한 줄만** 있었다. 비교 상대인 현재 수단 줄은 훨씬 위 2026
#       블록에 있어서, 두 대안이 같은 자리에서 마주 본 적이 없다. ab/explained 는 두
#       줄을 되풀이해 붙였을 뿐 나머지 수단은 여전히 멀리 있었다.
#       -> 2030 시점 **선택집합 전체를 하나의 표**로 놓는다. 열이 같으니 시간도 요금도
#          같은 자격으로 나란히 선다. 어느 한 속성을 부각하는 것이 아니다.
#   (2) 태도 15문항이 한 문단에 세미콜론으로 이어져 있었다 -> 주제별 줄로 나눈다.
#       시간과 비용은 **같은 줄**에 둬서 묶음 자체가 답을 가리키지 않게 한다.
#   (3) ab/explained 는 AV 줄을 두 번 냈다 -> 표에 한 번만 나온다.
#
# **정보는 하나도 더하지 않고 하나도 빼지 않는다.** 모든 변수가 그대로 있고, 저울질
# 지침·채택률·VOT 는 없다(_judgement_note 참조). 바뀐 것은 배치와 과제 서술뿐이다.
_CARD_SPEC = [("PT", "pt_travel_time_min", "pt_wait_time_min", "pt_cost_1000won"),
              ("Car", "car_travel_time_min", "car_wait_time_min", "car_cost_1000won"),
              ("PM", "pm_travel_time_min", "pm_wait_time_min", "pm_cost_1000won"),
              ("Walk", "walk_travel_time_min", None, None)]
_CARD_FMT = "  %-27s %10s %12s %12s   %s"
_CARD_ATT_GROUPS = [
    ("On new mobility and AV", ["A1_1", "A1_5", "A1_6", "A1_7", "S1"]),
    ("On time and cost", ["A1_2", "A1_3", "S2", "A1_4"]),
    ("On what they would change if AV hailing existed", ["A1_8", "A1_9", "T1", "T4"]),
    ("On data privacy", ["A1_10"]),
]


def _av_frame_text(r, rel=None) -> str:
    """AV 신뢰도 문구. future_2030_block 의 계산과 같은 식이다 - 그쪽은 건드리지 않는다
    (동결된 ab/zeroshot 프롬프트가 한 바이트도 바뀌면 안 되므로 공유하지 않고 따로 둔다)."""
    L = labels()
    fr = L["framing_attribute"].get(str(r.get("av_framing_attribute")), "Operation stability")
    pct = r.get("av_framing_percent")
    if rel is not None and pd.notna(rel):
        pct = float(rel) if "abnormality" not in fr.lower() else 100.0 - float(rel)
    return ("%s %s%%" % (fr, ("%.3f" % float(pct)).rstrip("0").rstrip("."))
            if pd.notna(pct) else fr + " not reported")


def _card_cells(r, code, t_col, w_col, c_col):
    t = r.get(t_col)
    if t is None or pd.isna(t):
        return None
    w = r.get(w_col) if w_col else None
    c = r.get(c_col) if c_col else None
    note = ""
    if code == "PT":
        bits = []
        n = r.get("pt_transfer_count")
        if pd.notna(n):
            bits.append("%d transfer%s" % (int(n), "" if int(n) == 1 else "s"))
        cw = labels()["crowding"].get(str(r.get("pt_crowding_raw")))
        if cw:
            bits.append("crowding %s" % cw)
        note = ", ".join(bits)
    return (_mode(code), "%.0f min" % float(t),
            ("%.0f min" % float(w)) if w is not None and pd.notna(w) else "-",
            _krw(c) if c is not None and pd.notna(c) else "0 KRW", note)


def card_choice_table(r, fare, ride, wait, rel=None) -> str:
    avail = str(r.get("available_modes") or "").split("+")
    cur = str(r.get("current_mode_label"))
    rows = [_CARD_FMT % ("mode", "in-vehicle", "wait/access", "cost", "")]
    for code, t, w, c in _CARD_SPEC:
        if code not in avail:
            continue
        cells = _card_cells(r, code, t, w, c)
        if not cells:
            continue
        name, tt, ww, cc, note = cells
        if code == cur:
            note = "this respondent's 2026 choice" + ("; " + note if note else "")
        rows.append(_CARD_FMT % (name, tt, ww, cc, note))
    rows.append(_CARD_FMT % (_mode("AV"), "%.0f min" % float(ride),
                             "%.0f min" % float(wait), _krw(fare),
                             "added in 2030; " + _av_frame_text(r, rel)))
    return "\n".join(rows)


def card_attitudes(r) -> str:
    lines = ["[What this respondent said about themselves - 1 (disagree) to 5 (agree), "
             "recorded after the choice tasks]"]
    for title, keys in _CARD_ATT_GROUPS:
        items = ["%s %d" % (ATTITUDE_LABELS[k], int(r.get(k)))
                 for k in keys if r.get(k) is not None and pd.notna(r.get(k))]
        if items:
            lines.append("  %s: %s" % (title, "; ".join(items)))
    sel = [k for k in r.index if k.endswith("_sel") and r[k] == 1]
    if sel:
        lines.append("  Shared-AV concerns reported: %d" % len(sel))
    return "\n".join(lines)


def card_user(r, info: str, fare, ride, wait, rel=None, style: str = "card") -> str:
    L = labels()
    band = str(r.get("distance_band"))
    km = L.get("band_distance_km", {}).get(band)
    head = (("[The scenario this respondent answered - one one-way trip of about %s km]" % km)
            if km else ("[The scenario this respondent answered - one one-way trip of about %s]"
                        % (r.get("distance_label") or band)))
    cur = _mode(r.get("current_mode_label"))
    blocks = [b for b in (persona_block(r), region_block(r), usual_trip_block(r)) if b]
    if info == "retro":
        blocks.append(card_attitudes(r))
    body = [head,
            "The modes available in 2026 keep exactly the same conditions in 2030; the "
            "driverless AV ride-hailing service is the only addition.",
            "",
            card_choice_table(r, fare, ride, wait, rel)]
    why = reason_2026(r.get("case_id"))
    if why:
        body += ["", "Reasons this respondent gave for the 2026 choice: %s." % why]
    blocks.append("\n".join(body))
    if style in ("neutral", "neutralswap"):
        # 답 토큰을 의미 없는 글자로 둔다. 두 줄의 문형은 완전히 같고, 어느 글자가 어느
        # 수단인지만 판마다 뒤바뀐다.
        av_first = (style == "neutral")
        pairs = ([("A", _mode("AV")), ("B", cur)] if av_first
                 else [("A", cur), ("B", _mode("AV"))])
        blocks.append("\n".join([
            "[The answer]",
            "Two modes are possible for this trip in 2030:",
            "  %s - %s" % pairs[0],
            "  %s - %s" % pairs[1],
            "",
            "Which one does this respondent use? Write that letter on the first line, "
            "in the braces:",
            "  { }",
        ]))
    elif style in ("named", "namedswap"):
        # **답을 둘 다 수단 이름으로 준다.** 'Keep' 이라는 추상어를 쓰지 않는다.
        # 답 칸은 { } 로 비워 두고 두 이름 중 하나를 그대로 옮겨 적게 한다.
        blocks.append("\n".join([
            "[The answer]",
            "Name the mode this respondent uses for this trip in 2030. It is one of these "
            "two, and only these two:",
            *(["  %s" % _mode("AV"), "  %s" % cur] if style == "named"
              else ["  %s" % cur, "  %s" % _mode("AV")]),
            "",
            "Write it on the first line, copied exactly, in the braces:",
            "  { }",
        ]))
    else:
        blocks.append("\n".join([
            "[The recorded answer]",
            "This respondent recorded one of two answers for the 2030 scenario:",
            '  "AV"   - started using the driverless AV ride-hailing service',
            '  "Keep" - continued with %s, unchanged from 2026' % cur,
            "",
            "Which answer did this respondent record?",
        ]))
    return "\n\n".join(blocks)


def render(r, info: str, fare, ride, wait, rel=None,
           style: str = "sft") -> list[dict]:
    """style 은 **답변 형식 지시**만 고른다. 변수 블록은 어느 쪽이든 동일하다.
    sft      한 단어로만 답하라 (학습 목표 형식)
    zeroshot 먼저 따져 보고 답하라 (계산할 자리를 준다)
    """
    # fresh 는 system 까지 전용이다 (백지 신작).
    if style in ("fresh", "shortnamed"):
        return [{"role": "system",
                 "content": FRESH_SYSTEM if style == "fresh" else FRESH_SYSTEM_NAMED},
                {"role": "user",
                 "content": fresh_user(r, info, fare, ride, wait, rel, style=style)}]
    # card / named 는 블록 구성 자체가 다르다(표 형식 선택집합). 전용 렌더러로 넘긴다.
    # 둘의 차이는 **마지막 답 블록뿐**이라 라벨 방식의 효과만 분리해서 볼 수 있다.
    if style in ("card", "named", "namedswap", "neutral", "neutralswap"):
        return [{"role": "system", "content": system_prompt(style)},
                {"role": "user",
                 "content": card_user(r, info, fare, ride, wait, rel, style=style)}]
    blocks = [b for b in (persona_block(r), region_block(r), usual_trip_block(r),
                          baseline_2026_block(r)) if b]
    # **noatt 는 선택 후 심리문항 블록을 뺀다.**
    #
    # 이건 표현을 손보는 것이 아니라 **변수 집합**을 바꾸는 것이고, 근거는 D247 이다.
    # A1_*, S3_*_sel, T1/T4/S1/S2 스물한 개는 선택 과제를 **끝낸 뒤에** 측정한 것이라
    # DCM 효용식에서 이미 빼고 있다 - 선택 뒤에 잰 값을 선택의 원인으로 조건에 넣으면
    # 속성 계수가 편향되기 때문이다. 그런데 LLM 프롬프트에는 그대로 들어가 있어 기준이
    # 어긋나 있었다 (DCM 51열 대 LLM 72열 상당).
    #
    # 9B 가 거절할 때 인용하는 두 문항이 하필 이 블록 안에 있다: "cost over time"(응답자의
    # 68.5% 가 4점 이상)과 "uneasy riding a driverless AV alone"(53.3%). 두 문항의 실제
    # 선택과의 상관은 -.076, -.064 로 거의 0 이다. 강해 보이지만 예측력이 없는 신호다.
    #
    # AV 를 더 고르게 하려고 넣은 문구가 아니라, DCM 과 같은 기준을 적용하고 그 결과를
    # 보는 것이다. 결과가 안 바뀌어도 그것대로 기록한다.
    if info == "retro" and style != "noatt":
        blocks.append(attitude_block(r))
    blocks.append(future_2030_block(r, fare, ride, wait, rel, style))
    return [{"role": "system", "content": system_prompt(style)},
            {"role": "user", "content": "\n\n".join(blocks)}]


# ------------------------------------------------------------------ run

def run(force=False) -> dict:
    print("\n=== Phase 14. LLM 프롬프트 ===")
    cfg = load_config()
    ensure_dirs(LLM_DIR, REPORTS)
    attr = cfg["dgp_specification"]["service_attributes"]
    g = cfg["grid"]
    cells = ([(w, i) for w in g["synthetic"]["worlds"]
              for i in g["synthetic"]["information_conditions"]]
             + [(w, i) for w in g["human"]["worlds"]
                for i in g["human"]["information_conditions"]])

    made = []
    for world, info in cells:
        for part in ("development", "test"):
            f = LLM_DIR / ("%s_%s_%s.jsonl" % (world, info, part))
            if f.exists() and not force:
                continue
            df = pd.read_parquet(MODEL_IN / ("%s_%s_%s.parquet" % (world, info, part)))
            with open(f, "w", encoding="utf-8") as out:
                for _, r in df.iterrows():
                    msg = render(r, info, r[attr["fare"]], r[attr["ride"]], r[attr["wait"]])
                    msg.append({"role": "assistant",
                                "content": CANDIDATES[0] if r["y"] == 1 else CANDIDATES[1]})
                    out.write(json.dumps({"case_id": r["case_id"],
                                          "respondent_id": int(r["respondent_id"]),
                                          "world": world, "information_condition": info,
                                          "partition": part, "messages": msg,
                                          "label": int(r["y"])},
                                         ensure_ascii=False) + "\n")
            made.append(f.name)
            print("  [ok] %-28s %d 행" % (f.name, len(df)))

    # ---- QA: pro/retro 는 태도 블록만 다르다 ---------------------------
    # **같은 행에서 두 프롬프트를 렌더링한다.** 예전에는 pro view 와 retro view 를 각각
    # 읽어 비교했는데, 정보조건이 retro 하나가 되면서 pro 파일이 없다. retro 행은 pro 열을
    # 전부 포함하므로 여기서 pro 를 렌더링할 수 있고, 자료가 같으니 검사가 더 엄밀해진다.
    a = pd.read_parquet(MODEL_IN / "HUMAN_retro_test.parquet").iloc[0]
    b = pd.read_parquet(MODEL_IN / "HUMAN_retro_test.parquet").iloc[0]
    ma = render(a, "pro", a[attr["fare"]], a[attr["ride"]], a[attr["wait"]])[1]["content"]
    mb = render(b, "retro", b[attr["fare"]], b[attr["ride"]], b[attr["wait"]])[1]["content"]
    only_attitude = mb.replace(attitude_block(b) + "\n\n", "") == ma
    print("  [%s] pro/retro 차이가 태도 블록뿐" % ("PASS" if only_attitude else "FAIL"))

    # ---- QA: 편집 프롬프트는 해당 속성 표현만 바뀐다 ---------------------
    grid = pd.read_parquet(SCEN / "test_scenario_grid.parquet")
    row = pd.read_parquet(MODEL_IN / "HUMAN_retro_test.parquet").iloc[0]
    gb = grid[grid.case_id == row.case_id].set_index("scenario_id")
    base = render(row, "pro", *gb.loc["BASE", ["fare_cf", "ride_cf", "wait_cf"]])[1]["content"]
    diffs = {}
    for sid in ("F_P1", "R_M1", "W_P1"):
        if sid not in gb.index:
            continue
        e = render(row, "pro", *gb.loc[sid, ["fare_cf", "ride_cf", "wait_cf"]])[1]["content"]
        changed = [i for i, (x, y) in enumerate(zip(base.split("\n"), e.split("\n"))) if x != y]
        diffs[sid] = changed
    ok_diff = all(len(v) == 1 for v in diffs.values())
    print("  [%s] 편집 프롬프트가 한 줄만 변경 %s"
          % ("PASS" if ok_diff else "FAIL", diffs))

    # ---- QA: 코드값이 그대로 새어나가지 않았는가 -------------------------
    # 열 이름이 어긋나면 예외가 아니라 조용한 "unknown" 홍수로 나타난다. 실제로 나이를
    # `age` 로 읽어 8,116행 전부가 "Age unknown" 이었다 (SQ3 가 맞는 열이다).
    # DGP 가 쓰는 변수는 프롬프트에 반드시 있어야 한다. 없으면 LLM 만 불리해진다.
    need = ["Operation stability", "Service abnormality"]
    tot = age_unknown = framing_missing = 0
    with open(LLM_DIR / "HUMAN_retro_development.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)["messages"][1]["content"]; tot += 1
            if "Age unknown" in u:
                age_unknown += 1
            if not any(k in u for k in need):
                framing_missing += 1
    r_age = age_unknown / max(1, tot)
    r_fr = framing_missing / max(1, tot)
    ok_persona = r_age <= 0.05 and r_fr <= 0.001
    print("  [%s] 나이 결측 %.1f%% (임계 5%%) / AV 프레이밍 누락 %.1f%% (임계 0.1%%)"
          % ("PASS" if ok_persona else "FAIL", 100 * r_age, 100 * r_fr))

    res = {"files": made, "cells": [list(c) for c in cells],
           "pro_retro_differs_only_by_attitudes": bool(only_attitude),
           "edit_changes_single_line": bool(ok_diff), "edit_diff_lines": diffs,
           "persona_age_unknown_rate": r_age, "av_framing_missing_rate": r_fr,
           "candidates": list(CANDIDATES),
           "example_prompt_pro": ma, "example_prompt_retro": mb}
    write_json(REPORTS / "llm_prompts.json", res)
    if not (only_attitude and ok_diff):
        raise SystemExit("프롬프트 QA 실패 — 정보조건/편집 대비가 오염된다")
    if not ok_persona:
        raise SystemExit("프롬프트에 DGP 변수가 빠졌다 — 나이 %.1f%%, 프레이밍 %.1f%% 결측"
                         % (100 * r_age, 100 * r_fr))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
