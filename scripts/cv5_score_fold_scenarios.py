# -*- coding: utf-8 -*-
"""CV5 폴드 어댑터로 그 폴드의 held-out 사례 x **33조건** 을 채점한다 (실험 2~4 재료).

왜 따로 있는가. cv5_score_fold.py 는 BASE 한 조건만 채점한다(실험 1). p18 은 고정 분할의
test 3,468건과 test_scenario_grid 만 읽는다. 폴드의 평가집합은 옛 development 와 test 가
섞여 있어 p18 로는 못 돌린다. 그래서 p18 의 프롬프트 규칙을 그대로 가져와 격자만 CV5 것
(data/scenarios/cv5_scenario_grid.parquet, 전체 11,584 사례)으로 바꾼다.

**주 실험과 같은 것.** 프롬프트 렌더러(P14.render, style='sft'), 채팅 서식(render_chat),
후보 채점(candidate_scores, AV/Keep 1토큰), margin = q_AV - q_Keep, shard 500, 조각 재개.
p18 의 scenario_prompts 와 같이 fare_cf/ride_cf/wait_cf/**rel_cf** 를 전부 넘긴다 - rel_cf 를
빼먹으면 A_* 여섯 조건이 BASE 와 같은 프롬프트가 되고 오류는 나지 않는다.

**주 실험과 다른 것.** 보정 확률을 붙이지 않는다. D321 이 등록한 Qwen 확률은 보정 없는
sigmoid(q_AV - q_Keep) 이고, 폴드별 보정기를 새로 적합하는 것은 별도 설계 판단이다.
그래서 이 파일은 raw margin 까지만 저장한다.

**기존 산출물과 섞지 않는다.** 출력은 predictions/cv5_llm/scenarios/ 와
predictions/cv5_llm/{family}_f{N}_scenarios.parquet 에만 쓴다.

자체 검증. 병합 후 (1) 행 수 == 격자 행 수, (2) (case_id, scenario_id) 중복 0,
(3) BASE 조건의 margin 이 앞서 cv5_score_fold.py 가 만든 BASE parquet 과 일치하는지 본다.
(3) 이 맞으면 어댑터·프롬프트·채점 경로가 실험 1 과 동일하다는 뜻이다.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs, load_config
from llm_runtime import candidate_scores, load_base, margins, render_chat
from p16_sft_runs import adapter_dir
from p17_calibration import apply_variant
import p14_llm_prompts as P14

OUT = ROOT / "predictions" / "cv5_llm"
WORLD, INFO, SEED = "HUMAN", "retro", 42
SHARD = 500


def build_prompts(pool, grid):
    """(case_id, scenario_id, messages) 를 조건-우선 순서로 만든다 (p18 과 같은 순서)."""
    recs = []
    for sid, g in grid.groupby("scenario_id", sort=False):
        sub = pool.loc[g.case_id]
        for (cid, r), f, rd, w, rl in zip(sub.iterrows(), g.fare_cf, g.ride_cf,
                                          g.wait_cf, g.rel_cf):
            recs.append((cid, sid, P14.render(r, INFO, f, rd, w, rl, style="sft")))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test 용. 앞 N 개 프롬프트만 (조건-우선 순서라 BASE 부터)")
    ap.add_argument("--scenarios", default=None,
                    help="smoke test 용. 쉼표로 조건을 제한 (예: BASE,F_P3,A_M3)")
    a = ap.parse_args()

    cfg = apply_variant(load_config(), "aaai", "aaai_composite", "reason_cv%d" % a.fold, 1.0)
    lc = cfg["llm"]
    sdir = OUT / "scenarios" / ("%s_f%d" % (a.family, a.fold))
    ensure_dirs(sdir)

    # 평가 사례는 BASE 채점과 같은 출처(eval jsonl)에서 읽는다 - 사례 집합이 같아야 한다.
    ev = [pd.read_json(l, typ="series") if False else __import__("json").loads(l)
          for l in (DATA / "llm" / ("CV5_f%d_eval.jsonl" % a.fold)).open(encoding="utf-8")]
    ids = [r["case_id"] for r in ev]
    y = {r["case_id"]: int(r["label"]) for r in ev}
    pool = pd.concat([pd.read_parquet(DATA / "model_inputs" / ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                      for p in ("development", "test")], ignore_index=True)
    pool = pool[pool.case_id.isin(set(ids))]
    # drop=False: case_id 를 열로도 남긴다. 지우면 렌더러의 reason_2026(r.get("case_id")) 이
    # None 을 받아 2026년 선택이유 문장이 통째로 빠진다 (p18 주석과 같은 함정).
    pool = pool.set_index("case_id", drop=False)

    grid = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")
    grid = grid[grid.case_id.isin(set(ids))].reset_index(drop=True)
    if a.scenarios:
        keep = [s.strip() for s in a.scenarios.split(",")]
        grid = grid[grid.scenario_id.isin(keep)].reset_index(drop=True)
    n_cases, n_scen = grid.case_id.nunique(), grid.scenario_id.nunique()
    print("fold %d: 평가 %d건 x %d조건 = 격자 %d행" % (a.fold, n_cases, n_scen, len(grid)), flush=True)
    assert n_cases == len(ids), "격자 사례 수(%d) != eval 사례 수(%d)" % (n_cases, len(ids))

    ad = adapter_dir(a.family, WORLD, INFO, SEED, cfg)
    if not (ad / "adapter_config.json").exists():
        raise SystemExit("어댑터가 없다: %s" % ad)
    print("  어댑터 %s" % ad.name, flush=True)

    recs = build_prompts(pool, grid)
    if a.limit:
        recs = recs[:a.limit]
    n_sh = (len(recs) + SHARD - 1) // SHARD
    todo = [k for k in range(n_sh) if not (sdir / ("shard_%04d.parquet" % k)).exists()]
    print("  프롬프트 %d개 -> shard %d개 (남은 %d)" % (len(recs), n_sh, len(todo)), flush=True)

    if todo:
        tok, base = load_base(a.family, four_bit=lc["four_bit"])
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(ad))
        texts = [render_chat(tok, m) for _, _, m in recs]
        t_all = time.time()
        for i, k in enumerate(todo):
            f = sdir / ("shard_%04d.parquet" % k)
            lo, hi = k * SHARD, min((k + 1) * SHARD, len(recs))
            t = time.time()
            sc, mass, free = candidate_scores(model, tok, texts[lo:hi],
                                              batch_size=lc["score_batch_size"],
                                              max_len=lc["max_seq_len"], return_extra=True)
            m = margins(sc)
            pd.DataFrame({
                "case_id": [c for c, _, _ in recs[lo:hi]],
                "scenario_id": [s for _, s, _ in recs[lo:hi]],
                "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": m,
                "p_av_raw": np.exp(sc[:, 0]), "p_keep_raw": np.exp(sc[:, 1]),
                "candidate_mass": mass,
                "raw_argmax": np.where(m > 0, "AV", "Keep"),
                "free_argmax": free,
                "fold": a.fold, "family": a.family, "world": WORLD, "info": INFO,
                "seed": SEED, "adapter": ad.name,
            }).to_parquet(f, index=False)
            mm = float(np.mean(mass))
            if mm < 0.5:
                print("    [!!] shard %d: candidate_mass 평균 %.4f - 답을 쓰지 않는 자리다. "
                      "이 결과를 표에 넣지 말 것. 어휘 1등: %s"
                      % (k, mm, pd.Series(free).value_counts().head(2).to_dict()), flush=True)
            done = i + 1
            rate = (time.time() - t_all) / done
            print("    shard %d/%d (%.0fs, 질량 %.4f)  남은 예상 %.1f분"
                  % (k + 1, n_sh, time.time() - t, mm, rate * (len(todo) - done) / 60), flush=True)
        del model
        import torch; torch.cuda.empty_cache()

    if a.limit or a.scenarios:
        print("  [smoke] 부분 실행이라 병합/검증은 건너뛴다.", flush=True)
        # smoke 검증: BASE margin 이 실험 1 BASE parquet 과 같은가
        bp = OUT / ("%s_f%d_base.parquet" % (a.family, a.fold))
        parts = [pd.read_parquet(p) for p in sorted(sdir.glob("shard_*.parquet"))]
        if parts and bp.exists():
            s = pd.concat(parts, ignore_index=True)
            b = pd.read_parquet(bp)[["case_id", "margin"]].rename(columns={"margin": "margin_exp1"})
            chk = s[s.scenario_id == "BASE"].merge(b, on="case_id")
            if len(chk):
                d = (chk.margin - chk.margin_exp1).abs()
                print("  [smoke] BASE margin vs 실험1 BASE: n=%d  max|Δ|=%.2e  mean|Δ|=%.2e"
                      % (len(chk), d.max(), d.mean()), flush=True)
            for sid in [x for x in s.scenario_id.unique() if x != "BASE"]:
                x = s[s.scenario_id == sid].merge(s[s.scenario_id == "BASE"][["case_id", "margin"]],
                                                on="case_id", suffixes=("", "_base"))
                if len(x):
                    print("  [smoke] %-5s vs BASE: n=%d  mean Δmargin=%+.4f  |Δ|>1e-6 비율=%.2f"
                          % (sid, len(x), (x.margin - x.margin_base).mean(),
                             ((x.margin - x.margin_base).abs() > 1e-6).mean()), flush=True)
        return

    # ---- 병합 + 검증 -------------------------------------------------------------
    parts = [pd.read_parquet(p) for p in sorted(sdir.glob("shard_*.parquet"))]
    s = pd.concat(parts, ignore_index=True)
    assert len(s) == len(grid), "병합 행 수 %d != 격자 %d" % (len(s), len(grid))
    dup = s.duplicated(["case_id", "scenario_id"]).sum()
    assert dup == 0, "(case_id, scenario_id) 중복 %d" % dup
    s = s.merge(grid[["case_id", "scenario_id", "distance_band", "scenario_type",
                      "edit_attribute", "edit_amount", "fare_cf", "ride_cf", "wait_cf",
                      "rel_cf", "analysis_tier", "within_support", "physically_valid"]],
                on=["case_id", "scenario_id"], how="left")
    s["y"] = s.case_id.map(y)
    out = OUT / ("%s_f%d_scenarios.parquet" % (a.family, a.fold))
    s.to_parquet(out, index=False)
    print("  [ok] %s  %d행  질량 %.4f" % (out.name, len(s), s.candidate_mass.mean()), flush=True)

    bp = OUT / ("%s_f%d_base.parquet" % (a.family, a.fold))
    if bp.exists():
        b = pd.read_parquet(bp)[["case_id", "margin"]].rename(columns={"margin": "margin_exp1"})
        chk = s[s.scenario_id == "BASE"].merge(b, on="case_id")
        d = (chk.margin - chk.margin_exp1).abs()
        print("  [검증] BASE margin vs 실험1 BASE parquet: n=%d  max|Δ|=%.2e" % (len(chk), d.max()),
              flush=True)
    # 방향 요약(실험 2 미리보기): 요금·시간 +/- 에서 평균 Δp 부호
    p = 1 / (1 + np.exp(-s.margin))
    base = s[s.scenario_id == "BASE"].set_index("case_id")
    pb = 1 / (1 + np.exp(-base.margin))
    for sid in ("F_M1", "F_P1", "R_M1", "R_P1", "W_M1", "W_P1"):
        x = s[s.scenario_id == sid].set_index("case_id")
        dp = (1 / (1 + np.exp(-x.margin))) - pb.reindex(x.index)
        print("  [미리보기] %-4s mean ΔP(AV)=%+.4f" % (sid, dp.mean()), flush=True)


if __name__ == "__main__":
    main()
