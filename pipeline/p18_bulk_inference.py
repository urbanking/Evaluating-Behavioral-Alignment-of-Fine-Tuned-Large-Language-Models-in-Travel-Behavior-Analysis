# -*- coding: utf-8 -*-
"""Phase 18. Test 27조건 bulk inference — Gate 9.

편집은 속성공간에서 정의하고 프롬프트는 그 뒤에 렌더링한다(p14 와 동일 렌더러).
resume: shard 단위로 저장하고 이미 있는 shard 는 건너뛴다.

zero-shot 재사용
  data/model_inputs/{world}_{info}_test.parquet 는 세계 간 `y` 열만 다르고 나머지가 모두
  같다(확인함). 프롬프트에는 y 가 들어가지 않으므로 zero-shot margin 은 세계와 무관하다.
  따라서 margin 은 (family, info) 당 한 번만 계산해 _raw 디렉터리에 두고, 세계별 셀은
  그 margin 에 각 세계의 보정기를 씌워 만든다. 세계마다 다시 추론하면 같은 계산을
  네 번 하게 된다.

출력
  predictions/llm/_raw_*/shard_*.parquet          (보정 전 margin)
  predictions/llm/{state}_{family}_{world}_{info}_{seed}/shard_*.parquet
  predictions/llm_scores.parquet                  (병합)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from common import DATA, REPORTS, ROOT, ensure_dirs, load_config, write_json
# 어댑터 경로와 보정기 이름을 p16/p17 과 **같은 규칙으로** 만든다. 여기서 seed 만으로
# 조립하면 목적함수·프로파일 접미사가 붙은 폴더를 못 찾는다 (2026-08-08 에 겪었다).
from p16_sft_runs import adapter_dir
from p17_calibration import apply_variant
from llm_runtime import (NAMED_STYLES, STYLE_TAG, candidate_scores,
                         candidate_scores_percase, load_base, margins,
                         prefix_tag, render_chat, resolve_answer_prefix,
                         select_families)
import p14_llm_prompts as P14

PRED = ROOT / "predictions" / "llm"
CAL = ROOT / "artifacts" / "calibration"
ADP = ROOT / "artifacts" / "models" / "llm"
SCEN = DATA / "scenarios"
SHARD = 500



def development_prompts(world, info, cfg, style="sft", limit=None):
    """**development** 사례의 BASE 프롬프트. 프롬프트 감사 전용이다.

    test 를 프롬프트 개발에 쓰면 그 test 는 더 이상 held-out 이 아니다. 이미 test 100건으로
    프롬프트를 고르는 판단을 한 적이 있어(2026-08-20), 그 100건은 prompt-development 표본으로
    기록하고 앞으로의 프롬프트 작업은 여기서만 한다.

    development 에는 반사실 격자가 없다. BASE 는 편집하지 않은 원조건이므로 행의 av_* 값을
    그대로 쓰면 test 격자의 BASE 와 같은 정의가 된다.
    """
    mi = pd.read_parquet(DATA / "model_inputs" / ("%s_%s_development.parquet" % (world, info)))
    if limit:
        mi = mi.head(limit)
    recs = []
    for _, r in mi.iterrows():
        msg = P14.render(r, info, r["av_cost_1000won"], r["av_travel_time_min"],
                         r["av_wait_time_min"], r.get("av_service_reliability_pct"),
                         style=style)
        recs.append((r["case_id"], "BASE", msg))
    return recs


def scenario_prompts(world, info, cfg, tok=None, style="sft"):
    """27조건 프롬프트를 (case_id, scenario_id, prompt) 로 만든다.

    **tok 을 주면 채팅 서식을 적용해 문자열로 돌려준다.** `P14.render` 는 메시지 목록
    (`[{role, content}, ...]`)을 돌려주므로 그대로 토크나이저에 넣으면
    "text input must be of type str" 로 죽는다. p17 은 `render_chat` 을 거치는데 여기만
    빠져 있었다 - 9B 를 GV100 에서 돌렸어도 똑같이 죽었을 자리다.
    """
    mi = pd.read_parquet(DATA / "model_inputs" / ("%s_%s_test.parquet" % (world, info)))
    grid = pd.read_parquet(SCEN / "test_scenario_grid.parquet")
    # drop=False: case_id 를 색인으로 쓰되 **열로도 남긴다.** 지우면 렌더러 안의
    # reason_2026(r.get("case_id")) 이 None 을 받아 2026년 선택이유 문장이 통째로
    # 빠진다. 오류가 안 나서 첫 검증에서야 잡혔다 (학습 jsonl 과 50/50 불일치).
    mi = mi.set_index("case_id", drop=False)
    recs = []
    for sid, g in grid.groupby("scenario_id", sort=False):
        sub = mi.loc[g.case_id]
        # rel_cf 까지 넘긴다. 신뢰도 사다리(A_*) 여섯 조건은 요금·시간을 기준값에 두고
        # 신뢰도만 편집하므로, 이걸 빼먹으면 그 20,808행이 전부 BASE 와 같은 프롬프트가
        # 되고 오류는 나지 않는다. BASE 등 나머지 조건은 rel_cf == rel_base 라 동일하다.
        for (cid, r), f, rd, w, rl in zip(sub.iterrows(), g.fare_cf, g.ride_cf,
                                          g.wait_cf, g.rel_cf):
            msg = P14.render(r, info, f, rd, w, rl, style=style)
            recs.append((cid, sid, render_chat(tok, msg) if tok is not None else msg))
    return recs


def load_calibrator(state, family, world, info, tag=None, answer_prefix=None,
                    style="sft"):
    """보정기를 읽는다. **접두사와 프롬프트 양식이 이름에 들어간다.**

    보정기는 margin 위에 적합된 것이고, 접두사·양식이 다르면 margin 이 다른 값이다.
    이름을 안 나누면 다른 조건으로 만든 보정기가 조용히 씌워진다 - 오류가 나지 않고
    숫자만 틀린다.
    """
    pt = prefix_tag(answer_prefix) + STYLE_TAG[style]
    f = (CAL / ("sft_%s_%s_%s_%s%s.json" % (family, world, info, tag, pt)) if state == "sft"
         else CAL / ("zero_shot_%s_%s_%s%s.json" % (family, info, world, pt)))
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def raw_dir(family, world, info, state, tag, answer_prefix=None, style="sft"):
    """보정 전 margin 을 두는 곳. zero-shot 은 세계 무관이므로 world 를 키에서 뺀다.

    **접두사를 쓰면 이름을 나눈다.** 같은 폴더를 쓰면 접두사 없이 만든 옛 조각을
    그대로 건너뛰어, 다시 돌린 의미가 사라진다(9B zero-shot 은 접두사 없이는
    candidate_mass 가 0.0000 이라 값 자체가 무의미하다).
    """
    # 접두사마다 이름을 다르게 둔다. 하나로 묶으면 접두사를 바꿔 다시 돌렸을 때 옛 조각을
    # 조용히 건너뛴다 - 이 프로젝트에서 반복해서 당한 종류의 버그다.
    suf = prefix_tag(answer_prefix) + STYLE_TAG[style]
    name = ("_raw_zeroshot_%s_%s%s" % (family, info, suf) if state == "zeroshot"
            else "_raw_sft_%s_%s_%s_%s%s" % (family, world, info, tag, suf))
    return PRED / name


def as_text(recs, tok, answer_prefix=None):
    """메시지 목록으로 만들어 둔 프롬프트를 채팅 서식 문자열로 바꾼다.

    **answer_prefix.** assistant 턴을 이 문구로 시작해 둔다. 그러면 다음 토큰이 곧 답이라
    후보 두 개가 그 자리에서 확률을 갖는다.

    이게 왜 필요한가. 9B zero-shot 은 답변 자리에서 `Thinking` 을 1등으로 내고 AV·Keep 에
    사실상 확률을 주지 않는다 - 실측 candidate_mass 가 **0.0000**(HUMAN 3,468건 전부)이고
    어휘 1등이 3,468건 모두 `Thinking` 이었다. 그 상태로 margin 을 비교하면 확률이 0 인
    두 토큰의 대소를 재는 것이라 결과가 잡음이다(AV-F1 0.0000 이 그렇게 나왔다).

    **원인은 채팅 서식이다.** 9B 서식은 add_generation_prompt=True 일 때
    `<|im_start|>assistant\\n<think>\\n` 으로 끝난다 - 사고 블록을 **열어 놓고** 끝난다.
    우리가 로짓을 읽는 자리는 답을 쓰는 자리가 아니라 사고를 시작하는 자리였다.
    쓸 접두사는 `nothink`(= `resolve_answer_prefix` 의 NO_THINK)다.

    **"Answer: " 는 여기서 듣지 않는다.** 생성 실험(p18d)에서 형식 준수가 0% -> 100% 로
    올라간 것은 사실이지만, 그 프롬프트에는 답부터 쓰라는 지시문이 함께 붙어 있었다.
    채점 프롬프트에 같은 접두사만 붙여 재보니 candidate_mass 0.0040, 어휘 1등이
    `<think>` 197/200 으로 **오히려 나빠졌다** (2026-08-19 실측). 사고 블록 안에 문구를
    하나 더 넣은 것이기 때문이다. 생성 경로의 근거를 채점 경로로 옮겨 쓰면 안 된다.

    SFT 는 접두사 없이도 candidate_mass 가 0.997 이므로 그대로 둔다 - 학습 때 본 서식이
    그것(사고 블록이 열린 상태)이고 그 자리에 답을 쓰도록 배웠다. 접두사를 붙이면
    오히려 학습 분포에서 벗어난다.
    """
    return [(c, s, (render_chat(tok, p) if not isinstance(p, str) else p)
             + (answer_prefix or ""))
            for c, s, p in recs]


def score_raw(family, world, info, state, seed, recs, cfg, ad_tag="-",
              answer_prefix=None, style="sft"):
    """margin shard 를 만든다. 이미 있는 shard 는 건너뛴다."""
    out = raw_dir(family, world, info, state, ad_tag, answer_prefix, style)
    ensure_dirs(out)
    lc = cfg["llm"]
    # named 계열은 사례마다 후보(AV 이름, 현재수단 이름)가 다르다. case_id -> 쌍 표.
    keep_map = {}
    if style in NAMED_STYLES:
        mi_ = pd.read_parquet(DATA / "model_inputs" / ("%s_%s_test.parquet" % (world, info)),
                              columns=["case_id", "current_mode_label"])
        av_ = P14._mode("AV")
        keep_map = {str(c): (av_, P14._mode(m))
                    for c, m in zip(mi_.case_id, mi_.current_mode_label)}
    n_sh = (len(recs) + SHARD - 1) // SHARD
    tok = model = None
    for k in range(n_sh):
        f = out / ("shard_%04d.parquet" % k)
        if f.exists():
            continue
        if model is None:
            tok, base = load_base(family, four_bit=lc["four_bit"])
            model = base
            if state == "sft":
                from peft import PeftModel
                # **어댑터 경로는 p16 과 같은 함수로.** "seed%d" 를 직접 조립하면
                # 목적함수·프로파일 접미사가 붙은 폴더를 못 찾는다.
                model = PeftModel.from_pretrained(
                    base, str(adapter_dir(family, world, info, seed, cfg)))
            recs = as_text(recs, tok, answer_prefix)   # 메시지 목록 -> 채팅 서식 문자열
        chunk = recs[k * SHARD:(k + 1) * SHARD]
        t = time.time()
        if style in NAMED_STYLES:
            # 후보가 사례마다 다른 **수단 이름**이다. 길이 정규화 채점을 쓴다
            # (llm_runtime.candidate_scores_percase 의 설명 참조).
            pairs = [keep_map[str(c[0])] for c in chunk]
            sc, mass, free = candidate_scores_percase(
                model, tok, [c[2] for c in chunk], pairs,
                batch_size=lc["score_batch_size"],
                max_len=lc["max_seq_len"], return_extra=True)
        else:
            sc, mass, free = candidate_scores(model, tok, [c[2] for c in chunk],
                                    batch_size=lc["score_batch_size"],
                                    max_len=lc["max_seq_len"], return_extra=True)
        m = margins(sc)
        # raw 값을 그대로 남긴다. q_* 는 로그확률, p_*_raw 는 그 지수(보정 전 확률),
        # candidate_mass 는 두 후보가 답변 자리에서 차지하는 비중이다.
        #
        # **예측은 calibrated_probability 하나뿐이다.** 아래 raw_argmax 는 예측이 아니라
        # 감사용 기록으로, "이 모델을 보통 챗봇처럼 글자를 뽑게 했다면 어느 쪽을 썼을까" 다.
        # 파이프라인은 이 값을 어디에도 쓰지 않는다. 보정 전 모델이 어느 쪽으로 기울어
        # 있었는지를 남겨두어, 보정이 무엇을 바로잡았는지 추적할 수 있게 한다.
        # (이름을 model_choice 로 두면 최종 선택으로 오독된다 — 실제 분류는 보정 확률 0.5 기준.)
        #
        # 나중에 보정을 다시 적합해도 이 열들만 있으면 모델을 다시 돌릴 필요가 없다.
        pd.DataFrame({"case_id": [c[0] for c in chunk],
                      "scenario_id": [c[1] for c in chunk],
                      "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": m,
                      "p_av_raw": np.exp(sc[:, 0]), "p_keep_raw": np.exp(sc[:, 1]),
                      "candidate_mass": mass,
                      "raw_argmax": np.where(m > 0, "AV", "Keep"),   # 후보 둘 중 1등
                      "free_argmax": free,      # 어휘 전체 1등 = 모델이 실제로 썼을 글자
                      }).to_parquet(f, index=False)
        # **후보질량을 여기서 본다.** 이 값은 예전부터 조각에 기록만 되고 아무도 안 봤다.
        # 그래서 9B zero-shot 이 114,444행 **전부** 질량 0.000000(최대값조차 0.0000)인 채로
        # 33조건을 다 돌고, 실험 1 표에 AV-F1 0.0000 으로 실려 나갔다. 오류는 한 번도 안 났다.
        # 답을 쓰지 않는 자리에서 두 후보의 대소를 비교한 값이라 숫자 자체가 무의미하다.
        mm = float(np.mean(mass))
        if mm < 0.5:
            print("    [!!] shard %d: candidate_mass 평균 %.6f — 모델이 이 자리에서 답을 "
                  "쓰고 있지 않다. margin 비교가 무의미하므로 이 결과를 표에 넣지 말 것. "
                  "어휘 1등: %s" % (k, mm, pd.Series(free).value_counts().head(2).to_dict()),
                  flush=True)
        print("    shard %d/%d (%.0fs, 후보질량 %.4f)"
              % (k + 1, n_sh, time.time() - t, mm), flush=True)
    if model is not None:
        del model
        import torch; torch.cuda.empty_cache()
    return out, n_sh


def run_cell(family, world, info, state, seed, cfg, limit=None, recalibrate=False,
             answer_prefix=None, style="sft"):
    """한 셀의 test 27조건을 채점한다.

    **비싼 부분(모델 forward)과 싼 부분(확률 변환)이 분리돼 있다.**
    margin 은 `_raw_*` 에 따로 저장하고, 보정 확률은 그 위에 씌운다. 그래서 보정기가
    아직 없을 때 먼저 돌려 두고, 나중에 보정기가 생기면 `--recalibrate` 로 확률만 다시
    붙일 수 있다 - GPU 를 다시 쓰지 않는다. 두 GPU 를 동시에 굴리려면 이게 필요하다
    (한쪽에서 development 를 채점하는 동안 다른 쪽에서 test 를 채점한다).
    """
    ad_tag = adapter_dir(family, world, info, seed, cfg).name if seed is not None else "-"
    tag = "%s_%s_%s_%s_%s%s" % (state, family, world, info, ad_tag,
                                prefix_tag(answer_prefix) + STYLE_TAG[style])
    out = PRED / tag
    ensure_dirs(out)
    if recalibrate:
        # 보정 확률만 다시 만든다. raw margin 은 건드리지 않는다.
        import shutil
        for f in out.glob("shard_*.parquet"):
            f.unlink()
    recs = scenario_prompts(world, info, cfg, style=style)
    # **비-sft 양식은 answer-first suffix 를 붙인다.** 동결 프롬프트는 "먼저 따져 보고
    # 답하라" 고 지시하므로, suffix 없이는 첫 토큰 자리에 답이 아니라 추론 산문이 온다 -
    # 실측(2026-08-21 첫 조각): 9B 질량 0.0000(1등 '###'), 2B 0.0261(1등 '**').
    # 생성 경로(p18d)와 development 감사는 전부 이 suffix 를 붙인 상태로 측정했으므로,
    # 채점도 같은 조건이어야 margin 이 같은 분포를 잰다. suffix 가 "첫 줄: AV 또는 Keep"
    # 을 요구하므로 첫 토큰이 답이 되고 candidate 채점이 성립한다.
    # **named 계열은 안내 문구가 다르다.** 답이 'AV'/'Keep' 한 단어가 아니라 중괄호 안의
    # 수단 이름이므로 `answer_first_suffix_named` 를 붙여야 한다. 2026-08-26 에 이 분기가
    # 없어서 채점 프롬프트만 "첫 줄: AV 또는 Keep" 을 달고 돌았고, 그 결과 답 자리의 어휘
    # 1등이 공백 없는 'AV'(937/2,000)가 되어 우리가 재려던 ' Driver'/' Car' 와 어긋났다
    # (질량 0.1392). 생성·감사·보정기는 모두 named 문구를 쓰고 있었으므로 채점만 다른
    # 프롬프트를 보고 있던 것이다.
    if style != "sft":
        skey = ("answer_first_suffix_named" if style in NAMED_STYLES
                else "answer_first_suffix_zeroshot")
        sfx = str(cfg["llm"].get(skey, cfg["llm"]["answer_first_suffix"])).strip()
        recs = [(c, sid, m[:-1] + [{"role": m[-1]["role"],
                                    "content": m[-1]["content"] + "\n\n" + sfx}])
                for c, sid, m in recs]
    if limit:
        recs = recs[:limit]
    cal = load_calibrator(state, family, world, info, ad_tag, answer_prefix, style)
    if cal is None:
        print("    [warn] 보정기 없음 — calibrated_probability 를 NaN 으로 둔다", flush=True)
    src, n_sh = score_raw(family, world, info, state, seed, recs, cfg, ad_tag,
                          answer_prefix, style)
    for k in range(n_sh):
        f = out / ("shard_%04d.parquet" % k)
        if f.exists():
            continue
        r = pd.read_parquet(src / ("shard_%04d.parquet" % k))
        m = r.margin.to_numpy()
        p = (1.0 / (1.0 + np.exp(-(cal["a"] * m + cal["b"])))) if cal else np.full(len(m), np.nan)
        # 이 사람이 무엇을 골랐다고 보는가 — 최종 예측 라벨.
        # tabular 후보모형도 확률을 0.5 로 잘라 라벨을 만든다. 같은 기준을 쓴다.
        pred = np.where(p >= 0.5, "AV", "Keep")
        r = r.assign(world=world, information_condition=info, model_family=family,
                     model_state=state, model="%s_%s" % (family, state),
                     seed=seed or 0, calibrated_probability=p, pred_label=pred)
        r.to_parquet(f, index=False)
    return out


def merge():
    frames = [pd.read_parquet(f) for d in sorted(PRED.glob("*"))
              if d.is_dir() and not d.name.startswith("_")
              for f in sorted(d.glob("shard_*.parquet"))]
    if not frames:
        return None
    all_ = pd.concat(frames, ignore_index=True)
    all_.to_parquet(ROOT / "predictions" / "llm_scores.parquet", index=False)
    return all_


def run(limit=None, stages=None, family=None, recalibrate=False, state=None,
        profile=None, objective=None, tau=None, run_tag=None, answer_prefix=None,
        world=None, style="sft"):
    print("\n=== Phase 18. LLM bulk inference ===")
    # run_tag 를 빼먹으면 어댑터 폴더 이름에서 접미사가 빠져, 이름이 비슷한 **다른**
    # 어댑터를 조용히 채점하거나 [skip] 으로 건너뛴다. p16b 에서 이미 겪은 함정이다.
    cfg = apply_variant(load_config(), profile, objective, run_tag, tau)
    # 'nothink' 같은 이름을 실제 문자열로 바꾼다. 개행·꺾쇠를 명령줄로 못 넘겨서 이름을 쓴다.
    answer_prefix = resolve_answer_prefix(answer_prefix)
    if answer_prefix:
        print("  [답변접두사] %r" % answer_prefix, flush=True)
    fams = select_families(cfg, family)   # --family 로 카드별 담당을 고른다
    order = stages or cfg["grid"].get("execution_order", ["human", "synthetic"])
    only = set(world.split(",")) if world else None
    for fam in fams:
        for stage in order:
            g = cfg["grid"][stage]
            worlds = [w for w in g["worlds"] if only is None or w in only]
            if only and not worlds:
                continue
            for world in worlds:
                for info in g["information_conditions"]:
                    # **state 로 무엇을 먼저/만 돌릴지 고른다.**
                    # 기본 순서는 제로샷 -> SFT 인데, 파인튜닝 결과가 급할 때 제로샷을
                    # 몇 시간 기다리게 된다. 둘은 서로 독립이라 순서를 바꿔도 무방하다.
                    if state in (None, "zeroshot"):
                        print("  zero-shot %s/%s/%s" % (fam, world, info), flush=True)
                        run_cell(fam, world, info, "zeroshot", None, cfg, limit,
                                 recalibrate, answer_prefix, style)
                    if state == "zeroshot":
                        continue
                    for seed in g["llm_seeds"]:
                        _ad = adapter_dir(fam, world, info, seed, cfg)
                        if not (_ad / "adapter_config.json").exists():
                            print("  [skip] adapter 없음 %s/%s/%s/%s"
                                  % (fam, world, info, _ad.name), flush=True)
                            continue
                        print("  SFT %s/%s/%s/%s" % (fam, world, info, _ad.name), flush=True)
                        # SFT 에는 접두사를 붙이지 않는다 - 학습 때 본 형식이 아니다.
                        run_cell(fam, world, info, "sft", seed, cfg, limit, recalibrate)
    a = merge()
    n = 0 if a is None else len(a)
    write_json(REPORTS / "llm_bulk_inference.json", {"rows": n})
    print("  [ok] predictions/llm_scores.parquet  %d 행" % n)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage", default=None, choices=["human", "synthetic"])
    ap.add_argument("--world", default=None,
                    help="이 세계만 (예: HOM). --stage synthetic 은 HOM 과 CD 를 한 GPU 에서 "
                         "차례로 돌기 때문에, 카드가 두 장 남을 때 나눠 걸려면 이게 필요하다")
    ap.add_argument("--state", default=None, choices=["zeroshot", "sft"],
                    help="이것만 돌린다. 생략하면 제로샷 -> SFT 둘 다")
    ap.add_argument("--recalibrate", action="store_true",
                    help="raw margin 은 그대로 두고 보정 확률만 다시 붙인다 (GPU 불필요)")
    ap.add_argument("--family", default=None,
                    help="이 모델군만 (예: qwen / qwen_small,qwen_mid). 생략하면 available 전부")
    ap.add_argument("--profile", default=None, help="학습 때 쓴 프로파일")
    ap.add_argument("--tau", type=float, default=None,
                    help="학습 때 쓴 logit_adjust_tau. 폴더 이름이 여기에 달려 있다")
    ap.add_argument("--objective", default=None,
                    choices=["token_ce", "margin_bce", "weighted_token_ce", "aaai_composite"],
                    help="학습 때 쓴 목적함수. 폴더 이름이 달라지므로 맞춰야 한다")
    ap.add_argument("--answer-prefix", default=None, dest="answer_prefix",
                    help="assistant 턴을 이 문구로 시작한다. **9B zero-shot 은 nothink 를 "
                         "쓴다** - 9B 채팅 서식이 <think> 를 열어 놓고 끝나서 채점 자리가 "
                         "답변 자리가 아니다 (candidate_mass 0.0000). 이름: nothink, aaai. "
                         "이름이 아니면 준 문자열을 그대로 붙인다")
    ap.add_argument("--prompt-style", default="sft", dest="style",
                    choices=["sft", "zeroshot", "pairwise", "noatt", "ab", "explained", "abswap", "respondent", "plain", "card", "fresh", "named", "neutral", "neutralswap", "shortnamed", "namedswap"],
                    help="zero-shot 프롬프트 양식 (D257 동결: 2B=zeroshot, 9B=ab). SFT 셀에는 "
                         "적용되지 않는다 - 학습 형식이 sft 양식이다. 폴더에 _zsp/_ab 등이 붙는다")
    ap.add_argument("--run-tag", default=None, dest="run_tag",
                    help="학습 때 쓴 --run-tag. 어댑터 폴더 이름의 접미사라서 빠뜨리면 "
                         "다른 어댑터를 집거나 [skip] 된다")
    a = ap.parse_args()
    run(limit=a.limit, stages=[a.stage] if a.stage else None, family=a.family,
        recalibrate=a.recalibrate, state=a.state, profile=a.profile, objective=a.objective,
        tau=a.tau, run_tag=a.run_tag, answer_prefix=a.answer_prefix, world=a.world,
        style=a.style)
