# -*- coding: utf-8 -*-
"""CV5 폴드 어댑터로 그 폴드의 **held-out 33조건 격자**를 채점한다.

**왜 이 스크립트가 따로 필요한가.** `cv5_score_fold.py` 는 폴드 held-out 전체를 BASE
한 조건으로만 채점한다. 본실험(p18)의 33조건은 `data/scenarios/test_scenario_grid.parquet`
에만 있고 그 격자는 **test 3,468건에만** 붙어 있다(development 8,116건에는 반사실 격자가
없다). 그래서 폴드 held-out 중 격자를 가진 사례만 33조건으로 채점한다.

**폴드 다섯 개를 합치면 정확히 격자 전체(114,444행)가 된다.** test 3,468건이 폴드에
겹침 없이 나뉘어 있기 때문이다. 즉 이 스크립트를 다섯 번 돌리면 본실험과 같은 사례·조건을
덮으면서, 모든 사례가 **자기를 학습에 쓰지 않은 어댑터**로 채점된다. 본실험의 33조건은
단일 분할이라 이 성질이 없었다.

**프롬프트 구성은 p18 과 한 글자도 다르지 않아야 한다.** 다르면 본실험과 비교가 성립하지
않는다. 특히 아래 둘은 오류가 나지 않고 숫자만 조용히 틀어지는 자리다.
  - `rel_cf` 를 렌더러에 넘긴다. 빼면 신뢰도 사다리(A_*) 여섯 조건이 전부 BASE 와 같은
    프롬프트가 된다.
  - `mi.set_index("case_id", drop=False)`. drop=True 로 두면 렌더러 안의
    reason_2026(case_id) 이 None 을 받아 2026년 선택이유 문장이 통째로 빠진다.

출력
  predictions/cv5_llm_grid/{family}_f{fold}/shard_*.parquet   (조각, 재개 단위)
  predictions/cv5_llm_grid/{family}_f{fold}.parquet           (병합본)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs, load_config
from llm_runtime import candidate_scores, load_base, margins, render_chat
from p16_sft_runs import adapter_dir
from p17_calibration import apply_variant
import p14_llm_prompts as P14

OUT = ROOT / "predictions" / "cv5_llm_grid"
SCEN = DATA / "scenarios"
WORLD, INFO, SEED = "HUMAN", "retro", 42
SHARD = 500


def fold_grid_prompts(fold, tok, style="sft"):
    """이 폴드 held-out 중 격자를 가진 사례의 33조건 프롬프트.

    p18.scenario_prompts 와 같은 방식으로 만들되 대상 사례만 이 폴드의 것으로 줄인다.
    """
    ev = [json.loads(l) for l in
          (DATA / "llm" / ("CV5_f%d_eval.jsonl" % fold)).open(encoding="utf-8")]
    ids = set(r["case_id"] for r in ev)

    mi = pd.read_parquet(DATA / "model_inputs" / ("%s_%s_test.parquet" % (WORLD, INFO)))
    # drop=False 를 지우지 말 것 — 위 docstring 참조.
    mi = mi.set_index("case_id", drop=False)

    grid = pd.read_parquet(SCEN / "test_scenario_grid.parquet")
    grid = grid[grid.case_id.isin(ids)]
    if grid.empty:
        raise SystemExit("fold %d: 격자에 걸리는 사례가 없다" % fold)

    recs = []
    for sid, g in grid.groupby("scenario_id", sort=False):
        sub = mi.loc[g.case_id]
        # rel_cf 까지 넘긴다 — 빼면 A_* 여섯 조건이 BASE 와 같아진다.
        for (cid, r), f, rd, w, rl in zip(sub.iterrows(), g.fare_cf, g.ride_cf,
                                          g.wait_cf, g.rel_cf):
            msg = P14.render(r, INFO, f, rd, w, rl, style=style)
            recs.append((cid, sid, render_chat(tok, msg)))
    return recs, sorted(grid.case_id.unique()), sorted(grid.scenario_id.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()

    cfg = apply_variant(load_config(), "aaai", "aaai_composite", "reason_cv%d" % a.fold, 1.0)
    lc = cfg["llm"]
    out = OUT / ("%s_f%d" % (a.family, a.fold))
    ensure_dirs(out)

    ad = adapter_dir(a.family, WORLD, INFO, SEED, cfg)
    if not (ad / "adapter_config.json").exists():
        raise SystemExit("어댑터가 없다: %s" % ad)

    tok, base = load_base(a.family, four_bit=lc["four_bit"])
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(ad))

    recs, cases, sids = fold_grid_prompts(a.fold, tok)
    n_sh = (len(recs) + SHARD - 1) // SHARD
    print("fold %d  사례 %d건 x 조건 %d개 = %d행  (조각 %d개)"
          % (a.fold, len(cases), len(sids), len(recs), n_sh), flush=True)
    print("  어댑터 %s" % ad.name, flush=True)

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
                      "fold": a.fold, "family": a.family,
                      "world": WORLD, "info": INFO,
                      }).to_parquet(f, index=False)
        mm = float(np.mean(mass))
        if mm < 0.5:
            print("    [!!] 조각 %d: 후보질량 평균 %.6f — 모델이 이 자리에서 답을 쓰고 있지 "
                  "않다. 이 결과를 표에 넣지 말 것. 어휘 1등: %s"
                  % (k, mm, pd.Series(free).value_counts().head(2).to_dict()), flush=True)
        el = time.time() - t0
        print("    조각 %d/%d (%.0fs, 후보질량 %.4f, 경과 %.0f분, 남은 %.0f분)"
              % (k + 1, n_sh, time.time() - t, mm, el / 60,
                 el / 60 / max(k + 1, 1) * (n_sh - k - 1)), flush=True)

    parts = [pd.read_parquet(p) for p in sorted(out.glob("shard_*.parquet"))]
    all_ = pd.concat(parts, ignore_index=True)
    merged = OUT / ("%s_f%d.parquet" % (a.family, a.fold))
    all_.to_parquet(merged, index=False)
    print("  [ok] %s  %d행  질량 %.4f  (%.0f분)"
          % (merged.name, len(all_), all_.candidate_mass.mean(), (time.time() - t0) / 60),
          flush=True)


if __name__ == "__main__":
    main()
