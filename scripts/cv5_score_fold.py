# -*- coding: utf-8 -*-
"""CV5 폴드 어댑터로 그 폴드의 **held-out 20% BASE** 를 채점한다.

**왜 기존 p18 을 못 쓰는가.** p18 은 HUMAN_retro_test.parquet(3,468건)만 읽는다. 폴드의
평가집합은 옛 development 와 test 가 섞여 있어(fold1 기준 1,572 + 752) 그 파일로는
사례의 3분의 2를 못 찾는다.

**기존 산출물과 섞지 않는다.** 출력은 predictions/cv5_llm/ 에만 쓴다.

프롬프트·후보·채점 방식은 SFT 본실험과 **완전히 동일**하다 (style='sft', 후보 AV/Keep
1토큰, margin = q_AV - q_Keep). 조건이 다르면 본실험과 비교가 성립하지 않는다.
"""
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

OUT = ROOT / "predictions" / "cv5_llm"
WORLD, INFO, SEED = "HUMAN", "retro", 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()

    cfg = apply_variant(load_config(), "aaai", "aaai_composite", "reason_cv%d" % a.fold, 1.0)
    lc = cfg["llm"]
    ensure_dirs(OUT)

    ev = [json.loads(l) for l in
          (DATA / "llm" / ("CV5_f%d_eval.jsonl" % a.fold)).open(encoding="utf-8")]
    ids = [r["case_id"] for r in ev]
    y = {r["case_id"]: int(r["label"]) for r in ev}
    pool = pd.concat([pd.read_parquet(DATA / "model_inputs" / ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                      for p in ("development", "test")], ignore_index=True)
    pool = pool[pool.case_id.isin(set(ids))].set_index("case_id", drop=False)
    pool = pool.loc[ids]                      # eval jsonl 순서를 유지한다
    print("fold %d 평가 %d건 (AV율 %.4f)" % (a.fold, len(pool), np.mean(list(y.values()))))

    ad = adapter_dir(a.family, WORLD, INFO, SEED, cfg)
    if not (ad / "adapter_config.json").exists():
        raise SystemExit("어댑터가 없다: %s" % ad)
    print("  어댑터 %s" % ad.name)

    tok, base = load_base(a.family, four_bit=lc["four_bit"])
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(ad))

    prompts = []
    for _, r in pool.iterrows():
        msg = P14.render(r, INFO, r["av_cost_1000won"], r["av_travel_time_min"],
                         r["av_wait_time_min"], r.get("av_service_reliability_pct"),
                         style="sft")
        prompts.append(render_chat(tok, msg))
    t0 = time.time()
    sc, mass, free = candidate_scores(model, tok, prompts,
                                      batch_size=lc["score_batch_size"],
                                      max_len=lc["max_seq_len"], return_extra=True)
    m = margins(sc)
    out = pd.DataFrame({"case_id": list(pool.case_id), "scenario_id": "BASE",
                        "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": m,
                        "candidate_mass": mass, "free_argmax": free,
                        "y": [y[c] for c in pool.case_id],
                        "fold": a.fold, "family": a.family, "world": WORLD, "info": INFO})
    f = OUT / ("%s_f%d_base.parquet" % (a.family, a.fold))
    out.to_parquet(f, index=False)
    print("  [ok] %s  %d행  질량 %.4f  (%.0f분)"
          % (f.name, len(out), mass.mean(), (time.time() - t0) / 60))

    # 하드라벨 지표 (D239: margin>0, 동점은 free_argmax)
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
    tie = np.abs(m) < 1e-9
    hard = np.where(tie, free == "AV", m > 0).astype(int)
    t = out.y.to_numpy()
    print("  AUC %.4f | PR-AUC %.4f | AV-F1 %.4f | Macro-F1 %.4f | 정확도 %.4f | 예측AV율 %.4f"
          % (roc_auc_score(t, m), average_precision_score(t, m), f1_score(t, hard),
             f1_score(t, hard, average="macro"), accuracy_score(t, hard), hard.mean()))


if __name__ == "__main__":
    main()
