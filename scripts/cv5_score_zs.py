# -*- coding: utf-8 -*-
"""CV5 zero-shot BASE scoring. Sibling of scripts/cv5_score_fold.py.

WHY THIS IS ONE PASS, NOT FIVE
------------------------------
Zero-shot has no training, so there is no per-fold adapter and the prompt does
not depend on the fold (no few-shot examples are drawn from a training split).
The only thing the fold decides is which cases are evaluated. Scoring the union
of the five eval sets once and then partitioning by fold gives numerically
identical per-fold results at one fifth of the cost. The fold membership still
comes from the same data/llm/CV5_f{N}_eval.jsonl files the SFT lane used, so the
case sets and labels match exactly.

PROMPT STYLE IS FROZEN BY D257
------------------------------
qwen_mid (2B) = "zeroshot", qwen (9B) = "ab" (+ nothink, D258). This is NOT the
"sft" style that cv5_score_fold.py uses. D257 froze it on a development-only
audit and forbids changing it on test evidence, so it is set from a table here
rather than left to a default.

Everything else matches the SFT fold lane: same candidate scoring, same margin
rule (q_AV - q_Keep), raw margins only with no calibrated probability attached.

Output: predictions/cv5_llm/{family}_zs_f{N}_base.parquet
Shards under predictions/cv5_llm/zs_shards/{family}/ so the run is resumable.
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

# D257 / D258. Do not change these from test evidence.
ZS_STYLE = {"qwen_mid": "zeroshot", "qwen": "ab"}

# The renderer drifted after the registered run. p14_llm_prompts.render(style="zeroshot")
# today stops at the "Decide as this traveler would: ..." line and omits the closing
# format paragraph that the registered cell used. Without it the 2B writes "**" at the
# answer position: measured candidate_mass 0.0257 versus the 0.9898 on record for
# qwen_mid_zeroshot in predictions/llm_scores.parquet, and free_argmax is "**" on 48/48
# smoke cases. The instruction "one word only, no punctuation" is what puts a bare AV or
# Keep token at the scored position.
#
# The text below is recovered from data/llm/HUMAN_retro_development_zsp.jsonl, the frozen
# artefact of the registered run (D257, built 2026-08-25). VERIFIED: for all 8,116
# development cases, render(style="zeroshot") + ZSP_TAIL reproduces the stored user
# message byte for byte, and the system message matches unchanged - 8,116/8,116, zero
# mismatches. This is a reconstruction of a frozen prompt, not a new prompt.
ZSP_TAIL = ("\n\nReply in this format. First line: AV or Keep (one word only, no "
            "punctuation). Second line: one or two sentences explaining why, referring "
            "to the specific numbers shown for the two options.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen_mid")
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: score only N cases")
    a = ap.parse_args()

    style = ZS_STYLE.get(a.family)
    if style is None:
        raise SystemExit("no frozen zero-shot style registered for family %r" % a.family)

    cfg = apply_variant(load_config(), "aaai", "aaai_composite", "reason_cv1", 1.0)
    lc = cfg["llm"]
    ensure_dirs(OUT)
    shard_dir = OUT / "zs_shards" / a.family
    ensure_dirs(shard_dir)

    # fold membership and labels from the same files the SFT lane used
    fold_of, y_of, order = {}, {}, []
    for n in range(1, 6):
        for line in (DATA / "llm" / ("CV5_f%d_eval.jsonl" % n)).open(encoding="utf-8"):
            r = json.loads(line)
            c = r["case_id"]
            if c in fold_of:
                raise SystemExit("case %s appears in two folds" % c)
            fold_of[c], y_of[c] = n, int(r["label"])
            order.append(c)
    print("family %s | zero-shot style %r (D257 frozen) | %d cases across 5 folds"
          % (a.family, style, len(order)), flush=True)

    pool = pd.concat([pd.read_parquet(DATA / "model_inputs" /
                                      ("%s_%s_%s.parquet" % (WORLD, INFO, p)))
                      for p in ("development", "test")], ignore_index=True)
    pool = pool[pool.case_id.isin(set(order))].set_index("case_id", drop=False).loc[order]
    if a.limit:
        pool = pool.iloc[:a.limit]
        order = order[:a.limit]
        print("  SMOKE: first %d cases only" % len(pool), flush=True)

    tok, model = load_base(a.family, four_bit=lc["four_bit"])
    print("  base model loaded, no adapter (zero-shot). four_bit=%s" % lc["four_bit"],
          flush=True)

    prompts = []
    for _, r in pool.iterrows():
        msg = P14.render(r, INFO, r["av_cost_1000won"], r["av_travel_time_min"],
                         r["av_wait_time_min"], r.get("av_service_reliability_pct"),
                         style=style)
        if style == "zeroshot":
            msg = [dict(m) for m in msg]
            for m in msg:
                if m["role"] == "user":
                    m["content"] = m["content"] + ZSP_TAIL
        prompts.append(render_chat(tok, msg))

    n_sh = (len(prompts) + a.shard - 1) // a.shard
    parts, t0 = [], time.time()
    for i in range(n_sh):
        f = shard_dir / ("shard_%04d.parquet" % i)
        lo, hi = i * a.shard, min((i + 1) * a.shard, len(prompts))
        if f.exists() and not a.limit:
            parts.append(pd.read_parquet(f))
            continue
        ts = time.time()
        sc, mass, free = candidate_scores(model, tok, prompts[lo:hi],
                                          batch_size=lc["score_batch_size"],
                                          max_len=lc["max_seq_len"], return_extra=True)
        ids = list(pool.case_id)[lo:hi]
        part = pd.DataFrame({"case_id": ids, "scenario_id": "BASE",
                             "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": margins(sc),
                             "candidate_mass": mass, "free_argmax": free,
                             "y": [y_of[c] for c in ids],
                             "fold": [fold_of[c] for c in ids],
                             "family": a.family, "state": "zeroshot",
                             "world": WORLD, "info": INFO})
        if not a.limit:
            part.to_parquet(f, index=False)
        parts.append(part)
        # Guard, same rule as p18_bulk_inference.score_raw. A low mass means the model is
        # not writing the answer at the position we read, so the margin comparison is
        # between two near-zero probabilities and the numbers are noise. This exact
        # failure once put a 9B zero-shot AV-F1 of 0.0000 into an Experiment 1 table
        # across 114,444 rows without raising an error.
        mm = float(mass.mean())
        if mm < 0.5:
            print("    [!!] shard %d candidate_mass %.6f - the model is NOT answering at "
                  "the scored position. Do not report these numbers. free_argmax top: %s"
                  % (i, mm, pd.Series(free).value_counts().head(2).to_dict()), flush=True)
        done = hi
        rate = (time.time() - t0) / done
        print("    shard %d/%d (%.0fs, mass %.4f) est %.1f min left"
              % (i + 1, n_sh, time.time() - ts, mm,
                 rate * (len(prompts) - done) / 60), flush=True)

    out = pd.concat(parts, ignore_index=True)
    print("\n  scored %d cases, candidate mass %.4f, %.1f min"
          % (len(out), out.candidate_mass.mean(), (time.time() - t0) / 60), flush=True)

    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 log_loss, roc_auc_score)
    rows = []
    for n, g in out.groupby("fold"):
        m = g.margin.to_numpy()
        p = 1 / (1 + np.exp(-m))
        t = g.y.to_numpy()
        hard = np.where(np.abs(m) < 1e-9, g.free_argmax.to_numpy() == "AV", m > 0).astype(int)
        rows.append({"fold": int(n), "n": len(g), "true_AV": round(t.mean(), 4),
                     "AUC": round(roc_auc_score(t, m), 4),
                     "PR_AUC": round(average_precision_score(t, m), 4),
                     "AV_F1": round(f1_score(t, hard, zero_division=0), 4),
                     "Macro_F1": round(f1_score(t, hard, average="macro", zero_division=0), 4),
                     "Acc": round(accuracy_score(t, hard), 4),
                     "pred_AV": round(hard.mean(), 4),
                     "log_loss": round(log_loss(t, p, labels=[0, 1]), 4)})
        if not a.limit:
            g.to_parquet(OUT / ("%s_zs_f%d_base.parquet" % (a.family, n)), index=False)
    d = pd.DataFrame(rows)
    print(d.to_string(index=False), flush=True)
    print("\nmean +/- SD across folds:")
    for c in ("AUC", "PR_AUC", "AV_F1", "Macro_F1", "Acc", "pred_AV", "log_loss"):
        print("  %-9s %.4f +/- %.4f" % (c, d[c].mean(), d[c].std()), flush=True)
    if not a.limit:
        d.to_csv(ROOT / "results" / ("cv5_llm_%s_zs.csv" % a.family), index=False)
        print("\n[ok] results/cv5_llm_%s_zs.csv" % a.family, flush=True)


if __name__ == "__main__":
    main()
