# -*- coding: utf-8 -*-
"""9B **zero-shot** 으로 33조건 반사실 격자를 채점한다 (SFT 33조건의 대조군).

**폴드로 나누지 않는다.** zero-shot 은 어댑터가 없어 모델이 하나뿐이므로 margin 이
폴드와 무관하다. 격자 전체(3,468사례 x 33조건 = 114,444행)를 **한 번만** 채점하고,
비교할 때 SFT 쪽 폴드 구분을 그대로 얹으면 된다. 폴드마다 돌리면 같은 계산을 다섯 번 한다.

**프롬프트 규칙은 cv5_score_zeroshot.py 와 같아야 한다.** 9B zero-shot 은 셋을 다 맞춰야
답 자리에서 답을 쓴다 - 하나라도 빠지면 후보질량이 0 이 되고 숫자가 무의미해진다.

  양식    ab                            D257 동결
  꼬리말  answer_first_suffix_zeroshot  마지막 user 턴 끝에 빈 줄 하나 두고
  접두사  nothink (`\\n</think>\\n\\n`)   채팅 서식이 연 사고 블록을 닫는다

**편집값은 p18 과 같은 자리에서 온다.** 격자의 fare_cf/ride_cf/wait_cf/rel_cf 를 렌더러에
넘긴다. `rel_cf` 를 빠뜨리면 신뢰도 사다리(A_*) 여섯 조건 20,808행이 전부 BASE 와 같은
프롬프트가 되고 오류는 나지 않는다.

출력
  predictions/cv5_llm_zs_grid/{family}_p{part}/shard_*.parquet
  predictions/cv5_llm_zs_grid/{family}_p{part}.parquet
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs, load_config
from llm_runtime import (candidate_scores, load_base, margins, render_chat,
                         resolve_answer_prefix)
import p14_llm_prompts as P14

OUT = ROOT / "predictions" / "cv5_llm_zs_grid"
SCEN = DATA / "scenarios"
WORLD, INFO = "HUMAN", "retro"
STYLE, PREFIX = "ab", "nothink"
SHARD = 500


def build_prompts(tok, suffix, prefix):
    """격자 전체의 (case_id, scenario_id, 프롬프트). p18.scenario_prompts 와 같은 순서."""
    mi = pd.read_parquet(DATA / "model_inputs" / ("%s_%s_test.parquet" % (WORLD, INFO)))
    mi = mi.set_index("case_id", drop=False)          # drop=False 를 지우지 말 것
    grid = pd.read_parquet(SCEN / "test_scenario_grid.parquet")
    recs = []
    for sid, g in grid.groupby("scenario_id", sort=False):
        sub = mi.loc[g.case_id]
        for (cid, r), f, rd, w, rl in zip(sub.iterrows(), g.fare_cf, g.ride_cf,
                                          g.wait_cf, g.rel_cf):
            msg = list(P14.render(r, INFO, f, rd, w, rl, style=STYLE))
            msg[-1] = dict(msg[-1], content=msg[-1]["content"] + "\n\n" + suffix)
            recs.append((cid, sid, render_chat(tok, msg) + prefix))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen")
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--nparts", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="연기시험용")
    a = ap.parse_args()

    lc = load_config()["llm"]
    prefix = resolve_answer_prefix(PREFIX)
    suffix = str(lc["answer_first_suffix_zeroshot"]).strip()
    out = OUT / ("%s_p%d" % (a.family, a.part))
    ensure_dirs(out)

    tok, model = load_base(a.family, four_bit=lc["four_bit"])
    model.eval()

    recs = build_prompts(tok, suffix, prefix)
    recs = recs[a.part - 1::a.nparts]                 # 카드 수만큼 갈라 이 몫만
    if a.limit:
        recs = recs[:a.limit]
    n_sh = (len(recs) + SHARD - 1) // SHARD
    print("zero-shot 33조건  몫 %d/%d  %d행  (조각 %d)  양식=%s 접두사=%r"
          % (a.part, a.nparts, len(recs), n_sh, STYLE, prefix), flush=True)

    t0 = time.time()
    for k in range(n_sh):
        f = out / ("shard_%04d.parquet" % k)
        if f.exists():
            continue
        chunk = recs[k * SHARD:(k + 1) * SHARD]
        t = time.time()
        sc, mass, free = candidate_scores(model, tok, [c[2] for c in chunk],
                                          batch_size=lc["score_batch_size"],
                                          max_len=lc["max_seq_len"], return_extra=True)
        m = margins(sc)
        pd.DataFrame({"case_id": [c[0] for c in chunk],
                      "scenario_id": [c[1] for c in chunk],
                      "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": m,
                      "p_av_raw": np.exp(sc[:, 0]), "p_keep_raw": np.exp(sc[:, 1]),
                      "candidate_mass": mass,
                      "raw_argmax": np.where(m > 0, "AV", "Keep"),
                      "free_argmax": free,
                      "family": a.family, "state": "zeroshot",
                      "world": WORLD, "info": INFO,
                      }).to_parquet(f, index=False)
        mm = float(np.mean(mass))
        if mm < 0.5:
            print("    [!!] 조각 %d: 후보질량 %.6f — 모델이 답을 쓰고 있지 않다. 이 결과를 "
                  "표에 넣지 말 것. 어휘 1등: %s"
                  % (k, mm, pd.Series(free).value_counts().head(2).to_dict()), flush=True)
        el = time.time() - t0
        print("    조각 %d/%d (%.0fs, 후보질량 %.4f, 경과 %.0f분, 남은 %.0f분)"
              % (k + 1, n_sh, time.time() - t, mm, el / 60,
                 el / 60 / (k + 1) * (n_sh - k - 1)), flush=True)

    parts = [pd.read_parquet(p) for p in sorted(out.glob("shard_*.parquet"))]
    all_ = pd.concat(parts, ignore_index=True)
    merged = OUT / ("%s_p%d.parquet" % (a.family, a.part))
    all_.to_parquet(merged, index=False)
    print("  [ok] %s  %d행  질량 %.4f  (%.0f분)"
          % (merged.name, len(all_), all_.candidate_mass.mean(), (time.time() - t0) / 60),
          flush=True)


if __name__ == "__main__":
    main()
