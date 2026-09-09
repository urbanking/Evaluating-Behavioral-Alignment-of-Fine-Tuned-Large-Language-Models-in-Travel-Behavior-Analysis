# -*- coding: utf-8 -*-
"""CV5 zero-shot 33-condition scoring. Sibling of scripts/cv5_score_fold_scenarios.py.

Produces the EXP2-4 material for the zero-shot cell that the SFT cell already has.

SAME AS THE SFT SCENARIO LANE
-----------------------------
Grid data/scenarios/cv5_scenario_grid.parquet restricted to each fold's eval cases,
scenario-major prompt order, fare_cf/ride_cf/wait_cf/rel_cf all passed to the renderer
(dropping rel_cf would silently make the six A_* conditions identical to BASE),
candidate_scores on the AV/Keep single tokens, margin = q_AV - q_Keep, shard 500 with
shard-level resume, raw margins only with no calibration attached (D321).

DIFFERENT, AND ONLY THESE
-------------------------
1. No adapter. Zero-shot is the base model.
2. Prompt style from the D257/D258 freeze: qwen_mid = "zeroshot", qwen = "ab".
   The SFT lane uses "sft"; these are different prompts.
3. ZSP_TAIL is appended to the user message for the "zeroshot" style. The renderer
   drifted after the registered run and now omits the closing format paragraph.
   Without it the 2B writes "**" at the scored position and candidate_mass collapses
   to 0.026 against the 0.9898 on record. Verified: render(style="zeroshot") + ZSP_TAIL
   reproduces the frozen data/llm/HUMAN_retro_development_zsp.jsonl user message byte
   for byte on all 8,116 development cases, system message unchanged, zero mismatches.

All five folds run in one process so the base model is loaded once instead of five times.
Output: predictions/cv5_llm/zs_scenarios/{family}_f{N}/shard_*.parquet, merged into
predictions/cv5_llm/{family}_zs_f{N}_scenarios.parquet.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, ensure_dirs, load_config
from llm_runtime import candidate_scores, load_base, margins, render_chat
from p17_calibration import apply_variant
import p14_llm_prompts as P14

OUT = ROOT / "predictions" / "cv5_llm"
WORLD, INFO = "HUMAN", "retro"
SHARD = 500

ZS_STYLE = {"qwen_mid": "zeroshot", "qwen": "ab"}
ZSP_TAIL = ("\n\nReply in this format. First line: AV or Keep (one word only, no "
            "punctuation). Second line: one or two sentences explaining why, referring "
            "to the specific numbers shown for the two options.")


def build_prompts(pool, grid, style):
    """(case_id, scenario_id, messages), scenario-major order, same as p18."""
    recs = []
    for sid, g in grid.groupby("scenario_id", sort=False):
        sub = pool.loc[g.case_id]
        for (cid, r), f, rd, w, rl in zip(sub.iterrows(), g.fare_cf, g.ride_cf,
                                          g.wait_cf, g.rel_cf):
            msg = P14.render(r, INFO, f, rd, w, rl, style=style)
            if style == "zeroshot":
                msg = [dict(m) for m in msg]
                for m in msg:
                    if m["role"] == "user":
                        m["content"] = m["content"] + ZSP_TAIL
            recs.append((cid, sid, msg))
    return recs


def run_fold(fold, family, style, tok, model, lc, pool_all, grid_all):
    ev = [json.loads(l) for l in
          (DATA / "llm" / ("CV5_f%d_eval.jsonl" % fold)).open(encoding="utf-8")]
    ids = [r["case_id"] for r in ev]
    y = {r["case_id"]: int(r["label"]) for r in ev}
    pool = pool_all[pool_all.case_id.isin(set(ids))].set_index("case_id", drop=False)
    grid = grid_all[grid_all.case_id.isin(set(ids))].reset_index(drop=True)
    n_cases, n_scen = grid.case_id.nunique(), grid.scenario_id.nunique()
    print("\n=== fold %d: %d cases x %d conditions = %d grid rows ==="
          % (fold, n_cases, n_scen, len(grid)), flush=True)
    assert n_cases == len(ids), "grid cases %d != eval cases %d" % (n_cases, len(ids))

    sdir = OUT / "zs_scenarios" / ("%s_f%d" % (family, fold))
    ensure_dirs(sdir)
    recs = build_prompts(pool, grid, style)
    n_sh = (len(recs) + SHARD - 1) // SHARD
    todo = [k for k in range(n_sh) if not (sdir / ("shard_%04d.parquet" % k)).exists()]
    print("  %d prompts -> %d shards (%d remaining)" % (len(recs), n_sh, len(todo)),
          flush=True)

    if todo:
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
                "fold": fold, "family": family, "state": "zeroshot",
                "world": WORLD, "info": INFO,
            }).to_parquet(f, index=False)
            mm = float(np.mean(mass))
            if mm < 0.5:
                print("    [!!] shard %d candidate_mass %.4f - the model is NOT answering "
                      "at the scored position. Do not report these numbers. top: %s"
                      % (k, mm, pd.Series(free).value_counts().head(2).to_dict()),
                      flush=True)
            done = i + 1
            rate = (time.time() - t_all) / done
            print("    shard %d/%d (%.0fs, mass %.4f) est %.1f min left in this fold"
                  % (k + 1, n_sh, time.time() - t, mm, rate * (len(todo) - done) / 60),
                  flush=True)

    parts = [pd.read_parquet(p) for p in sorted(sdir.glob("shard_*.parquet"))]
    s = pd.concat(parts, ignore_index=True)
    assert len(s) == len(grid), "merged %d != grid %d" % (len(s), len(grid))
    dup = int(s.duplicated(["case_id", "scenario_id"]).sum())
    assert dup == 0, "duplicate (case_id, scenario_id): %d" % dup
    s = s.merge(grid[["case_id", "scenario_id", "distance_band", "scenario_type",
                      "edit_attribute", "edit_amount", "fare_cf", "ride_cf", "wait_cf",
                      "rel_cf", "analysis_tier", "within_support", "physically_valid"]],
                on=["case_id", "scenario_id"], how="left")
    s["y"] = s.case_id.map(y)
    out = OUT / ("%s_zs_f%d_scenarios.parquet" % (family, fold))
    s.to_parquet(out, index=False)
    print("  [ok] %s  %d rows  mass %.4f" % (out.name, len(s), s.candidate_mass.mean()),
          flush=True)

    # same-path check: BASE margins must equal the EXP1 zero-shot BASE parquet
    bp = OUT / ("%s_zs_f%d_base.parquet" % (family, fold))
    if bp.exists():
        b = pd.read_parquet(bp)[["case_id", "margin"]].rename(
            columns={"margin": "margin_base_run"})
        chk = s[s.scenario_id == "BASE"].merge(b, on="case_id")
        d = (chk.margin - chk.margin_base_run).abs()
        print("  [check] BASE margin vs the EXP1 zero-shot BASE parquet: n=%d "
              "max|d|=%.2e" % (len(chk), d.max()), flush=True)
    for sid in ("F_M1", "F_P1", "R_M1", "R_P1", "W_M1", "W_P1"):
        x = s[s.scenario_id == sid].merge(
            s[s.scenario_id == "BASE"][["case_id", "margin"]], on="case_id",
            suffixes=("", "_base"))
        if len(x):
            dp = (1 / (1 + np.exp(-x.margin))) - (1 / (1 + np.exp(-x.margin_base)))
            print("  [preview] %-5s mean dP(AV)=%+.4f" % (sid, dp.mean()), flush=True)
    return len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen_mid")
    ap.add_argument("--folds", default="1,2,3,4,5")
    a = ap.parse_args()

    style = ZS_STYLE.get(a.family)
    if style is None:
        raise SystemExit("no frozen zero-shot style registered for family %r" % a.family)
    folds = [int(x) for x in a.folds.split(",")]

    cfg = apply_variant(load_config(), "aaai", "aaai_composite", "reason_cv1", 1.0)
    lc = cfg["llm"]
    ensure_dirs(OUT)
    print("family %s | zero-shot style %r (D257/D258 frozen) | folds %s"
          % (a.family, style, folds), flush=True)

    pool_all = pd.concat([pd.read_parquet(DATA / "model_inputs" /
                                          ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                          for p in ("development", "test")], ignore_index=True)
    grid_all = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")

    tok, model = load_base(a.family, four_bit=lc["four_bit"])
    print("  base model loaded once, no adapter. four_bit=%s" % lc["four_bit"], flush=True)

    t0, total = time.time(), 0
    for n in folds:
        total += run_fold(n, a.family, style, tok, model, lc, pool_all, grid_all)
        print("  elapsed %.2f h, %d rows written so far"
              % ((time.time() - t0) / 3600, total), flush=True)
    print("\nDONE  %d rows in %.2f h" % (total, (time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    main()
