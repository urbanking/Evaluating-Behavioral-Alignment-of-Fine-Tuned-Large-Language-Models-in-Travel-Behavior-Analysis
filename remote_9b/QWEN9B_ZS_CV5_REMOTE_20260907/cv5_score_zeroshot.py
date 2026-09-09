# -*- coding: utf-8 -*-
"""9B **zero-shot** 으로 CV5 폴드 held-out 을 채점한다 (BASE 조건).

**어댑터가 없으므로 폴드마다 모델이 다르지 않다.** zero-shot margin 은 폴드와 무관하다.
그래서 사례를 **한 번만** 채점하고, 지표를 낼 때 폴드별로 나눈다. 폴드마다 다시 돌리면
같은 계산을 다섯 번 하는 것이다. 다섯 폴드의 held-out 합집합이 곧 전체 11,584건이다.

**9B zero-shot 은 접두사와 양식을 반드시 맞춰야 한다.** 그냥 돌리면 답변 자리가 사고
블록을 연 상태(`<|im_start|>assistant\\n<think>\\n`)라, 우리가 읽는 로짓이 답을 쓰는
자리가 아니라 사고를 시작하는 자리다. 실제로 기존 산출물
`predictions/llm/_raw_zeroshot_qwen_retro` 는 **후보질량 0.000000, 어휘 1등이 전부
`Thinking`** 이다 - 확률이 0 인 두 토큰의 대소를 잰 값이라 숫자 자체가 무의미하다.

  접두사  nothink (= NO_THINK, `\\n</think>\\n\\n`)   비사고 모드로 닫는다
  양식    ab                                        D257 로 동결된 9B zero-shot 양식
  꼬리말  answer_first_suffix_zeroshot              user 턴 끝에 붙인다

**꼬리말을 빠뜨리면 안 된다.** `ab` 양식의 system 은 "Work through this traveler's
situation first" 라고 먼저 추론하라고 시킨다. 그래서 답변 자리가 답을 쓰는 자리가 아니게
되고, 꼬리말 없이 돌리면 후보질량이 **0.000000**(어휘 1등 `###`)이 된다 - 실측했다.
본실험이 쓴 프롬프트(`data/llm/HUMAN_retro_development_ab.jsonl`)에는 이 문단이 들어
있고, 그 조합의 score_bank 는 후보질량 0.9958(최소 0.9889)이다.

이 조합은 기존 보정기 이름 `zero_shot_qwen_retro_HUMAN_nothink_ab.json` 과
score_bank `qwen_zeroshot_retro_development_nothink_ab.parquet` 이 증거다.

출력
  predictions/cv5_llm_zs/{family}_base_p{part}/shard_*.parquet
  predictions/cv5_llm_zs/{family}_base_p{part}.parquet
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs, load_config
from llm_runtime import (candidate_scores, load_base, margins, render_chat,
                         resolve_answer_prefix)
import p14_llm_prompts as P14

OUT = ROOT / "predictions" / "cv5_llm_zs"
WORLD, INFO = "HUMAN", "retro"
STYLE = "ab"
PREFIX = "nothink"
SHARD = 500


def all_eval_cases():
    """다섯 폴드 held-out 의 합집합. 폴드 번호도 함께 들고 온다."""
    rec = {}
    for f in range(1, 6):
        for l in (DATA / "llm" / ("CV5_f%d_eval.jsonl" % f)).open(encoding="utf-8"):
            r = json.loads(l)
            rec[r["case_id"]] = (f, int(r["label"]))
    ids = sorted(rec)
    return ids, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen")
    ap.add_argument("--part", type=int, default=1, help="1 또는 2 — 카드 두 장에 나눠 돌린다")
    ap.add_argument("--nparts", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="연기시험용")
    a = ap.parse_args()

    cfg = load_config()
    lc = cfg["llm"]
    prefix = resolve_answer_prefix(PREFIX)
    out = OUT / ("%s_base_p%d" % (a.family, a.part))
    ensure_dirs(out)

    ids, rec = all_eval_cases()
    pool = pd.concat([pd.read_parquet(DATA / "model_inputs" / ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                      for p in ("development", "test")], ignore_index=True)
    pool = pool[pool.case_id.isin(set(ids))].set_index("case_id", drop=False).loc[ids]
    # 카드 수만큼 잘라 이 몫만 돈다.
    pool = pool.iloc[a.part - 1::a.nparts]
    if a.limit:
        pool = pool.head(a.limit)

    tok, model = load_base(a.family, four_bit=lc["four_bit"])
    model.eval()

    suffix = str(lc["answer_first_suffix_zeroshot"]).strip()
    prompts = []
    for _, r in pool.iterrows():
        msg = list(P14.render(r, INFO, r["av_cost_1000won"], r["av_travel_time_min"],
                              r["av_wait_time_min"], r.get("av_service_reliability_pct"),
                              style=STYLE))
        # 꼬리말은 마지막 user 턴 끝에 빈 줄 하나를 두고 붙인다 (본실험 jsonl 과 동일).
        msg[-1] = dict(msg[-1], content=msg[-1]["content"] + "\n\n" + suffix)
        prompts.append(render_chat(tok, msg) + prefix)

    n_sh = (len(prompts) + SHARD - 1) // SHARD
    print("zero-shot %s  몫 %d/%d  %d건  (조각 %d)  양식=%s 접두사=%r"
          % (a.family, a.part, a.nparts, len(prompts), n_sh, STYLE, prefix), flush=True)

    t0 = time.time()
    for k in range(n_sh):
        f = out / ("shard_%04d.parquet" % k)
        if f.exists():
            continue
        lo, hi = k * SHARD, min((k + 1) * SHARD, len(prompts))
        t = time.time()
        sc, mass, free = candidate_scores(model, tok, prompts[lo:hi],
                                          batch_size=lc["score_batch_size"],
                                          max_len=lc["max_seq_len"], return_extra=True)
        m = margins(sc)
        cids = list(pool.case_id)[lo:hi]
        pd.DataFrame({"case_id": cids, "scenario_id": "BASE",
                      "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": m,
                      "candidate_mass": mass, "free_argmax": free,
                      "y": [rec[c][1] for c in cids],
                      "fold": [rec[c][0] for c in cids],
                      "family": a.family, "state": "zeroshot",
                      "world": WORLD, "info": INFO,
                      }).to_parquet(f, index=False)
        mm = float(np.mean(mass))
        if mm < 0.5:
            print("    [!!] 조각 %d: 후보질량 %.6f — 모델이 답을 쓰고 있지 않다. "
                  "이 결과를 표에 넣지 말 것. 어휘 1등: %s"
                  % (k, mm, pd.Series(free).value_counts().head(2).to_dict()), flush=True)
        el = time.time() - t0
        print("    조각 %d/%d (%.0fs, 후보질량 %.4f, 남은 %.0f분)"
              % (k + 1, n_sh, time.time() - t, mm, el / 60 / (k + 1) * (n_sh - k - 1)), flush=True)

    parts = [pd.read_parquet(p) for p in sorted(out.glob("shard_*.parquet"))]
    all_ = pd.concat(parts, ignore_index=True)
    merged = OUT / ("%s_base_p%d.parquet" % (a.family, a.part))
    all_.to_parquet(merged, index=False)
    print("  [ok] %s  %d행  질량 %.4f  (%.0f분)"
          % (merged.name, len(all_), all_.candidate_mass.mean(), (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
