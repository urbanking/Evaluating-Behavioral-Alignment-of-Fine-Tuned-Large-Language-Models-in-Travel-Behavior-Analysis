# -*- coding: utf-8 -*-
"""LLM 런타임 — 로딩, candidate sequence scoring, QLoRA 학습.

확률은 first-token softmax 가 아니라 **candidate sequence score** 로 만든다.
  q_j = 프롬프트에 이어지는 후보 시퀀스 j 의 로그우도 합
  m   = q_AV - q_Keep
  p   = sigmoid(a*m + b)   (a>0, development 에서만 적합)

sampling generation 은 쓰지 않는다. 모든 값이 결정적이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from common import load_config

# 후보는 둘 다 **1토큰**이어야 한다. "Retain" 은 Qwen 토크나이저에서 Ret+ain 2조각이라
# 학습 손실이 토큰 평균이므로 Retain 사례가 AV 사례보다 1.5배 무게를 갖는다.
# 데이터가 이미 유지 81 : 채택 19 로 기울어 있어 그 편향이 곱해진다. "Keep" 은 1토큰이다.
CANDIDATES = ("AV", "Keep")


def model_dir(family: str) -> Path:
    """가중치가 놓인 자리. **기계마다 다른 유일한 경로라서 환경변수로 덮을 수 있다.**

    여기 두 대에서 돌린다. 이 워크스테이션은 C:/SP_LLM/models, GV100 서버는
    C:/Users/dell/models 다. 2026-08-10 에 config 를 서버로 동기화하면서 이 값까지
    같이 덮어써 채점이 죽었다 - transformers 가 없는 경로를 Hub 모델 이름으로 보고
    `Repo id must use alphanumeric chars` 를 냈다. 경로만 TRC_MODEL_ROOT 로 빼두면
    config 는 그대로 동기화해도 된다. 다른 상수는 여전히 config 에서만 읽는다.
    """
    import os
    cfg = load_config()
    root = Path(os.environ.get("TRC_MODEL_ROOT") or cfg["llm"]["model_root"])
    return root / cfg["llm"]["families"][family]["path"]


def cfg_llm() -> dict:
    return load_config()["llm"]


def load_base(family: str, four_bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    d = str(model_dir(family))
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # 학습은 build_sft_dataset 에서 `ids[-max_len:]` 로 **앞을** 버린다. 토크나이저 기본값은
    # 반대(뒤를 버림)라, 길이를 넘기는 순간 학습은 질문을 남기고 채점은 질문을 잘라
    # 서로 다른 텍스트를 보게 된다. 방향을 맞춘다.
    tok.truncation_side = "left"
    dt = getattr(torch, str(cfg_llm().get("dtype", "bfloat16")))
    kw = {"local_files_only": True, "dtype": dt, "device_map": "auto"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
    model = AutoModelForCausalLM.from_pretrained(d, **kw)
    model.config.use_cache = False
    return tok, model


def check_single_token(tok, candidates=CANDIDATES) -> list[int]:
    """후보가 이 토크나이저에서 **각각 1토큰**인지 확인하고 토큰 id 를 돌려준다.

    설계가 1토큰을 전제한다.
      - 학습 손실이 토큰 평균이라, 후보 길이가 다르면 긴 쪽이 더 큰 무게를 갖는다.
        (Qwen 에서 "Retain" 은 Ret+ain 2조각이라 유지 쪽이 1.5배 무거웠다. "Keep" 으로 바꿨다.)
      - candidate_mass 가 답변 자리 한 위치의 분포를 보므로 1토큰이어야 정의된다.

    **토큰화는 모델계열마다 다르다.** Qwen 에서 1토큰이어도 Llama 에서 2토큰일 수 있다.
    조용히 첫 조각만 쓰면 결과가 틀린 채로 며칠이 지나간다. 여기서 멈춘다.
    """
    ids = {c: tok(c, add_special_tokens=False).input_ids for c in candidates}
    bad = {c: v for c, v in ids.items() if len(v) != 1}
    if bad:
        raise SystemExit(
            "후보가 1토큰이 아니다: %s\n"
            "  이 모델에서는 다른 단어를 써야 한다. 1토큰 후보를 찾아 configs 와 p14 를 함께 고칠 것."
            % {c: [tok.decode([x]) for x in v] for c, v in bad.items()})
    return [ids[c][0] for c in candidates]


def render_chat(tok, messages, enable_thinking=None) -> str:
    """assistant 응답 직전까지의 프롬프트 문자열.

    **enable_thinking.** Qwen3.5 계열이 공식으로 제공하는 스위치다. 9B 는 기본 서식이
    `<|im_start|>assistant\\n<think>\\n` 으로 끝나 사고 블록을 열어 놓는데, 그러면 우리가
    로짓을 읽는 자리가 답변 자리가 아니라 추론 시작 자리가 된다(D251).

    전에는 `NO_THINK` 문자열을 이어 붙여 같은 결과를 냈고 두 문자열이 **바이트 단위로 같음**을
    확인했지만, 모델별 문자열을 우리가 조립하는 것보다 공식 인자를 쓰는 편이 깨끗하다 -
    체크포인트가 서식을 바꾸면 우리 문자열만 낡는다.

    2B 는 기본 서식이 이미 `<think>\\n\\n</think>\\n\\n` 으로 끝나므로 이 인자를 줘도 결과가
    같다(실측). 그래서 두 모형에 같은 값을 줘도 안전하다.
    """
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, **kw)
    except Exception as e:
        # **Gemma 계열은 system 역할이 없다** - 서식이 "System role not supported" 를 던진다.
        # 관례(모델 카드 안내)대로 system 내용을 첫 user 메시지 앞에 접어 넣는다.
        # 프롬프트 내용은 한 글자도 안 바뀌고 배치만 바뀐다. 다른 예외는 그대로 올린다.
        if "System role not supported" not in str(e):
            raise
        merged, sys_txt = [], None
        for m in messages:
            if m["role"] == "system" and sys_txt is None:
                sys_txt = m["content"]
            elif m["role"] == "user" and sys_txt is not None:
                merged.append({"role": "user",
                               "content": sys_txt + chr(10) * 2 + m["content"]})
                sys_txt = None
            else:
                merged.append(m)
        return tok.apply_chat_template(merged, tokenize=False,
                                       add_generation_prompt=True, **kw)


# 사고 블록을 열자마자 닫는 접두사. 아래 resolve_answer_prefix 의 설명을 볼 것.
NO_THINK = "\n</think>\n\n"

# 프롬프트 양식 -> 파일·폴더 꼬리표. 양식이 다르면 margin 이 다른 값이므로 산출물 이름을
# 가른다. p17(보정기)·p18(채점)·p18d(생성)가 같은 규칙을 써야 셀 이름이 서로 맞는다.
NAMED_STYLES = ("named", "shortnamed", "namedswap")

STYLE_TAG = {"sft": "", "zeroshot": "_zsp", "pairwise": "_pw", "noatt": "_na", "ab": "_ab",
             "explained": "_ex", "abswap": "_abr",
             "respondent": "_rs", "plain": "_pl", "card": "_cd", "fresh": "_fr", "named": "_nm", "neutral": "_nt", "neutralswap": "_nts", "shortnamed": "_sn", "namedswap": "_nmr"}

# named/shortnamed 채점용. 모델이 쓰는 형식이 `{ 이름 }` 이라 답 자리에 여는 중괄호를
# 미리 놓아 둔다. 두 후보에 공통이라 비교에 영향이 없다.
#
# **공백을 접두사에 넣지 않는다.** 2026-08-26 실측: 모델이 실제로 쓴 토큰열은
# ['{', ' Public', ' transit', ' }'] 로, 공백이 **다음 토큰에 붙어 있다**. 접두사를 "{ "
# 로 두면 토큰이 ['{', ' '] 가 되어 그 자리에서 찾아야 할 것이 앞공백 없는 'Personal' 이
# 되는데, 그건 모델이 쓰는 형태가 아니다. 그 상태로 21조각을 돌린 결과 candidate_mass 가
# 0.0023 이었고 어휘 1등이 ' }' 였다 - 빈 템플릿을 닫으려 한 것이다. 후보 쪽에 앞공백을
# 붙이는 것으로 경계를 모델과 맞춘다 (이어붙임 토큰열이 일치함을 확인).
BRACE = "{"
CAND_LEAD = " "   # 후보 이름 앞에 붙이는 공백

ANSWER_PREFIXES = {
    "nothink": NO_THINK,
    "brace": BRACE,
    "aaai": '{"pairwise_choice":"',      # p17 이 쓰던 이름. 같은 표에 모아 둔다
}


def resolve_answer_prefix(name):
    """접두사를 **이름으로** 받는다. 이름이 아니면 준 문자열을 그대로 쓴다.

    **왜 이름으로 받는가.** 필요한 문자열이 `"\\n</think>\\n\\n"` 인데, 개행과 꺾쇠를
    PowerShell -> ssh -> argparse 로 그대로 통과시키기가 어렵다. p17 이 이미 같은 이유로
    `aaai` 라는 이름을 쓰고 있었다.

    **`nothink` 이 무엇을 고치는가.** 9B(qwen) 의 채팅 서식은 add_generation_prompt=True 일 때
    `<|im_start|>assistant\\n<think>\\n` 으로 끝난다 - 즉 **사고 블록을 열어 놓고** 끝난다.
    candidate_scores 는 그 다음 자리의 로짓을 읽으므로, 우리가 재는 자리는 답을 쓰는 자리가
    아니라 사고를 시작하는 자리다. 실측(2026-08-19, HUMAN test BASE 200건):

        접두사 없음        candidate_mass 0.0000   어휘 1등 'Thinking' x200
        'Answer: '        candidate_mass 0.0040   어휘 1등 '<think>' x197

    두 번째가 더 나빠진 이유도 같다. `<think>\\n` 뒤에 'Answer: ' 를 붙인 것이라 여전히
    사고 블록 안이다. 그 상태의 margin 비교는 확률이 0 인 두 토큰의 대소를 재는 것이라
    잡음이고, 실제로 9B zero-shot 의 AV-F1 이 0.0000 으로 나왔다.

    `NO_THINK` 을 붙이면 문자열이 `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n` 이 되어,
    토크나이저가 `enable_thinking=False` 로 만드는 것과 **글자 그대로 같다**(그것도 실측).
    프롬프트 내용을 바꾸는 게 아니라 모델이 문서화한 비사고 모드로 두는 것이다.

    2B(qwen_mid) 는 이 문제가 없다 - 서식이 사고 블록을 열지 않아 candidate_mass 가
    0.9797 이다. 그래서 이 접두사는 9B 에만 쓴다.

    SFT 는 9B 라도 붙이지 않는다. 학습이 같은 서식(사고 블록이 열린 상태)에서 이뤄져
    그 자리에 답을 쓰도록 배웠고, 실측 candidate_mass 가 0.9971 이다.
    """
    if not name:
        return None
    return ANSWER_PREFIXES.get(name, name)


def prefix_tag(answer_prefix) -> str:
    """접두사를 파일·폴더 이름에 붙일 짧은 꼬리표로 바꾼다.

    접두사가 다르면 margin 이 **다른 값**이다. 꼬리표로 이름을 갈라 두지 않으면
    접두사 없이 만든 조각·뱅크를 그대로 재사용해서, 다시 돌린 의미가 조용히 사라진다.
    보정기와 추론이 서로 다른 접두사로 만들어지면 보정이 엉뚱한 척도에 걸리는데,
    오류가 나지 않아 숫자만 틀린다.
    """
    p = resolve_answer_prefix(answer_prefix)
    if not p:
        return ""
    if p == NO_THINK:
        return "_nothink"
    if p == ANSWER_PREFIXES["aaai"]:
        return "_aaai"
    if p == BRACE:
        return "_brace"
    return "_pfx"


def stop_token_ids(tok, model=None) -> list[int]:
    """`generate()` 에 넘길 종료 토큰 목록.

    **왜 명시해야 하는가.** 9B(qwen) 를 실측하면 (2026-08-18):
        tok.eos_token                     '<|im_end|>'   id 248046
        model.generation_config.eos_token '<|endoftext|>' id 248044
    채팅 서식은 assistant 턴을 `<|im_end|>` 로 닫는데, generate() 는 아무 것도 안 주면
    generation_config 의 `<|endoftext|>` 만 종료로 본다. 채팅 튜닝된 모델은 그 토큰을
    내지 않으므로 **영원히 안 멈춘다.** 실제로 9B SFT 생성 3,468건이 답을 맞게 쓰고도
    'Keep9ingKeep7KeepKeepKeep...' 처럼 상한까지 반복했다(후보 일치율 0.00%).
    2B(qwen_mid) 는 이 문제가 없었지만, 계열마다 다르므로 양쪽 다 명시한다.
    """
    ids = set()
    for x in (getattr(tok, "eos_token_id", None),
              getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if isinstance(x, int):
            ids.add(x)
        elif isinstance(x, (list, tuple)):
            ids.update(int(i) for i in x)
    for t in ("<|im_end|>", "<|endoftext|>"):
        try:
            i = tok.convert_tokens_to_ids(t)
        except Exception:
            i = None
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return sorted(ids)


_LOGIT_KW = None        # 어떤 인자 이름이 먹는지 한 번만 알아내 재사용한다


def last_position_logits(model, enc):
    """**마지막 위치의 logits 만** 계산한다.

    기본 forward 는 모든 위치의 logits 를 만든다: 배치 4 x 길이 1280 x 어휘 248,046 x
    2바이트 = 2.5GB. 우리는 `[:, -1, :]` 한 줄만 쓰고 나머지를 버린다. 그 낭비가
    2026-08-18 의 9B 대량 추론을 두 번 죽였다 -
    `CUDA out of memory. Tried to allocate 2.09 GiB` (카드에 22GB 가 남아 있는데도;
    Windows WDDM 은 GPU 할당에 시스템 RAM 뒷받침이 필요한데 이 기계는 남의 프로세스가
    937GB 를 잡고 있어 여유가 26GB 뿐이었다).

    transformers 는 이 계산을 줄이는 인자를 준다 - 최신은 `logits_to_keep`,
    예전은 `num_logits_to_keep`. 버전에 따라 이름이 다르므로 한 번 시도해 보고 고른다.
    1 을 주면 마지막 한 위치만 LM head 를 통과하므로 할당이 2.5GB -> 2MB 로 줄고,
    그만큼 계산도 준다(긴 프롬프트에서는 LM head 가 forward 비용의 큰 몫이다).
    """
    global _LOGIT_KW
    if _LOGIT_KW is None:
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(**enc, **{kw: 1})
                _LOGIT_KW = kw
                return out.logits[:, -1, :]
            except TypeError:
                continue
        _LOGIT_KW = ""      # 둘 다 안 먹으면 예전 방식으로 간다
    if _LOGIT_KW:
        return model(**enc, **{_LOGIT_KW: 1}).logits[:, -1, :]
    return model(**enc).logits[:, -1, :]


@torch.no_grad()
def candidate_scores(model, tok, prompts: list[str], candidates=CANDIDATES,
                     batch_size: int = 8, max_len: int = 1024,
                     return_extra: bool = False):
    """(n, len(candidates)) 후보 시퀀스 로그우도 합.

    각 후보의 토큰 로그확률을 더한다. 길이 정규화는 하지 않는다 — 후보가 고정 문자열이라
    길이 차이가 모든 사례에 동일하게 걸리고, 뒤따르는 Platt 보정이 절편으로 흡수한다.

    **후보가 모두 1토큰이면 forward 를 한 번만 한다.** 후보를 이어붙이기 전 위치의 분포
    하나에 두 후보의 확률이 모두 들어 있기 때문이다. 후보별로 따로 돌릴 이유가 없다.
    실측으로 2회 방식과 **완전히 동일**함을 확인했다 (최대 절대차이 0.00e+00).
    Phase 18 의 749,088건에서 추론량이 절반이 된다.

    return_extra=True 면 (scores, mass) 를 돌려준다. mass 는 답변 자리에서 두 후보가
    차지하는 확률 질량으로, "후보를 둘로 제한해도 되는가" 의 근거이자 진단값이다.
    """
    model.eval()
    ids1 = [tok(c, add_special_tokens=False).input_ids for c in candidates]
    if all(len(v) == 1 for v in ids1):
        cid = [v[0] for v in ids1]
        out = np.zeros((len(prompts), len(candidates)), dtype=np.float64)
        mass = np.zeros(len(prompts), dtype=np.float64)
        top = np.zeros(len(prompts), dtype=np.int64)
        for s0 in range(0, len(prompts), batch_size):
            chunk = prompts[s0:s0 + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(model.device)
            lg = last_position_logits(model, enc).float()
            lp = torch.log_softmax(lg, dim=-1)
            out[s0:s0 + len(chunk)] = lp[:, cid].cpu().numpy()
            mass[s0:s0 + len(chunk)] = torch.softmax(lg, dim=-1)[:, cid].sum(1).cpu().numpy()
            # 모델이 그냥 놔뒀으면 실제로 썼을 토큰 = 어휘 전체의 1등.
            # 후보 둘 중 1등(raw_argmax)과 다를 수 있다 — 'Keeping', 'The' 같은 게 1등이면
            # 프롬프트가 모델을 두 선택지로 몰지 못했다는 뜻이다. 이미 logits 가 있으므로 공짜다.
            top[s0:s0 + len(chunk)] = lg.argmax(dim=-1).cpu().numpy()
            del enc, lg, lp
            torch.cuda.empty_cache()
        if return_extra:
            free = np.array([tok.decode([int(t)]) for t in top], dtype=object)
            return out, mass, free
        return out
    if return_extra:
        raise SystemExit("후보가 1토큰이 아니면 mass 를 정의할 수 없다: %s" % (candidates,))
    out = np.zeros((len(prompts), len(candidates)), dtype=np.float64)
    cand_ids = [tok(c, add_special_tokens=False).input_ids for c in candidates]

    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        for ci, cid in enumerate(cand_ids):
            L = len(cid)
            texts = [p + candidates[ci] for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(model.device)
            logits = model(**enc).logits
            # 후보 토큰을 예측하는 위치만 잘라낸다. 전체 [B, T, V] 를 float32 로 올리면
            # vocab 15만 x seq 1024 에서 수 GB 가 되어 12GB 카드가 바로 터진다.
            lg = logits[:, -(L + 1):-1, :].float()
            tgt = enc.input_ids[:, -L:]
            logp = torch.log_softmax(lg, dim=-1)
            tokw = torch.gather(logp, 2, tgt.unsqueeze(-1)).squeeze(-1)
            out[s:s + len(chunk), ci] = tokw.sum(dim=1).cpu().numpy()
            del enc, logits, lg, logp, tokw
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def candidate_scores_percase(model, tok, prompts: list[str], cand_pairs,
                             batch_size: int = 8, max_len: int = 1024,
                             return_extra: bool = False):
    """사례마다 후보가 다른 채점. named/shortnamed 양식 전용.

    답이 'AV'/'Keep' 한 토큰이 아니라 **수단 이름**이고, 현재 수단이 사례마다 달라 후보
    문자열도 사례마다 다르다. 프롬프트 끝에는 여는 중괄호 `{`(BRACE)가 붙어 있고 후보에는
    앞공백이 붙는다 - 모델이 실제로 쓰는 토큰 경계가 ['{', ' Public', ...] 이기 때문이다.

    **두 후보의 첫 토큰만 비교한다.** 이름들의 첫 토큰이 서로 다르므로(' Driver' 대
    ' Public'/' Car'/' Personal'/' Walking') 그 자리 한 위치의 분포로 비교가 성립하고,
    기존 1토큰 후보 채점과 **같은 종류의 양**이 된다.

    전체 이름을 이어붙여 채점하는 방식(토큰당 평균 로그확률)도 만들어 재 봤으나 버렸다.
    2026-08-26 development 200건 실측:

        토큰당 평균   llama 3B  AUC 0.3449   <- 0.5 미만, 즉 순서가 뒤집힘
        첫 토큰만     llama 3B  AUC 0.4904   llama 8B  AUC 0.7196, mass 0.9638

    이름마다 길이·어휘 빈도가 달라 평균 로그확률이 "어느 쪽을 고르는가" 가 아니라 "그 이름을
    쓰기 쉬운가" 를 재고 있었다. 첫 토큰 비교는 그 오염이 없고 forward 도 한 번뿐이다.

    mass 는 그 자리 분포에서 두 후보 첫 토큰이 차지하는 확률 합이다.
    """
    model.eval()
    n = len(prompts)
    out = np.zeros((n, 2), dtype=np.float64)
    mass = np.zeros(n, dtype=np.float64)
    top = np.zeros(n, dtype=np.int64)
    cand_pairs = [tuple(CAND_LEAD + c for c in pair) for pair in cand_pairs]
    fid = [[tok(c, add_special_tokens=False).input_ids[0] for c in pair]
           for pair in cand_pairs]
    for s0 in range(0, n, batch_size):
        chunk = prompts[s0:s0 + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        lg = last_position_logits(model, enc).float()
        lp = torch.log_softmax(lg, dim=-1)
        pr = torch.softmax(lg, dim=-1)
        for bi in range(len(chunk)):
            a, b = fid[s0 + bi]
            out[s0 + bi] = [float(lp[bi, a]), float(lp[bi, b])]
            mass[s0 + bi] = float(pr[bi, a] + pr[bi, b])
        top[s0:s0 + len(chunk)] = lg.argmax(dim=-1).cpu().numpy()
        del enc, lg, lp, pr
        torch.cuda.empty_cache()
    if return_extra:
        free = np.array([tok.decode([int(t)]) for t in top], dtype=object)
        return out, mass, free
    return out


@torch.no_grad()
def candidate_mass(model, tok, prompts: list[str], candidates=CANDIDATES,
                   batch_size: int = 8, max_len: int = 1280) -> np.ndarray:
    """답변 자리에서 후보 두 개가 차지하는 확률 질량.

    **왜 재는가.** 학습은 어휘 전체(25만)에 대한 교차엔트로피이고, 추론은 후보 둘의
    로그우도 차이만 쓴다. 두 목적함수가 어긋나 보이지만, 답변 자리 확률이 두 후보에
    거의 다 몰려 있으면 실질적으로 같은 문제가 된다. 그 정도를 숫자로 확인한다.

    이 값은 원고에도 필요하다. "후보를 둘로 제한해도 되는가" 의 근거가 된다.
    값이 낮으면 모델이 엉뚱한 토큰에 질량을 남기고 있다는 뜻이고, 그때는 마진 기반
    학습으로 바꾸는 것을 검토해야 한다.
    """
    model.eval()
    cid = [check_single_token(tok, candidates)[i] for i in range(len(candidates))]
    out = np.zeros(len(prompts), dtype=np.float64)
    for s0 in range(0, len(prompts), batch_size):
        chunk = prompts[s0:s0 + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        lg = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(lg, dim=-1)
        out[s0:s0 + len(chunk)] = p[:, cid].sum(dim=1).cpu().numpy()
        del enc, lg, p
    torch.cuda.empty_cache()
    return out


def margins(scores: np.ndarray) -> np.ndarray:
    return scores[:, 0] - scores[:, 1]


def read_jsonl(path: Path, limit: int | None = None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def build_sft_dataset(tok, rows, max_len: int = 1024):
    """assistant 토큰에만 손실을 거는 학습 데이터."""
    from datasets import Dataset
    recs = []
    for r in rows:
        msgs = r["messages"]
        prompt = render_chat(tok, msgs[:-1])
        answer = msgs[-1]["content"]
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p_ids + a_ids)[-max_len:]
        labels = ([-100] * len(p_ids) + a_ids)[-max_len:]
        recs.append({"input_ids": ids, "labels": labels,
                     "attention_mask": [1] * len(ids)})
    return Dataset.from_list(recs)


def lora_config():
    """LoRA 설정. 대상 모듈은 코드가 아니라 configs 에서 읽는다 — 모델계열마다 다르다.

    Qwen3.5 는 32층 중 8층만 표준 어텐션이고 24층은 선형 어텐션이라 이름이 다르다.
    표준 이름만 적으면 어텐션의 3/4 에 LoRA 가 안 붙는다. 자세한 근거는 configs 주석 참조.
    """
    from peft import LoraConfig
    c = load_config()["sft"]
    tm = c.get("target_modules")
    if isinstance(tm, str):
        raise SystemExit("target_modules 는 목록이어야 한다 (모델계열마다 다름): %r" % tm)
    kw = dict(r=int(c["lora_rank"]), lora_alpha=int(c["lora_alpha"]),
              lora_dropout=float(c["lora_dropout"]), bias="none",
              task_type="CAUSAL_LM", target_modules=list(tm))
    ex = c.get("exclude_modules")
    if ex:
        try:
            return LoraConfig(exclude_modules=ex, **kw)
        except TypeError:
            print("  [warn] 이 peft 버전은 exclude_modules 를 지원하지 않는다 — mtp 도 학습된다")
    return LoraConfig(**kw)

def select_families(cfg, family=None):
    """이번 실행에서 돌 모델군. family 를 주면 그것만.

    **왜 필요한가.** 설정에 규모 사다리(qwen_small 0.8B, qwen_mid 2B)를 넣으면서
    available 인 모델군이 셋이 됐다. 이 함수 없이 available 을 전부 돌면, 10GB 카드에서
    보정을 돌릴 때 9B 까지 불러들여 메모리가 넘친다. 카드마다 담당 모델군이 다르므로
    실행할 때 명시적으로 고른다.

    쉼표로 여러 개를 줄 수 있다: --family qwen_small,qwen_mid
    """
    fams = [f for f, v in cfg["llm"]["families"].items() if v.get("available")]
    if not family:
        return fams
    want = [x.strip() for x in str(family).split(",")]
    unknown = [x for x in want if x not in cfg["llm"]["families"]]
    if unknown:
        raise SystemExit("모르는 모델군 %s - 설정의 families 에 없다" % unknown)
    sel = [f for f in fams if f in want]
    if not sel:
        raise SystemExit("%s 는 available: false 다" % want)
    return sel
