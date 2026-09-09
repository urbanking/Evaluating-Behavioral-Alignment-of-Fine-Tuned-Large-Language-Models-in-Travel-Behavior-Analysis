# -*- coding: utf-8 -*-
"""Lane 7. FFNN/DNN with the decision threshold treated as part of the model.

ADD-ONLY. Writes only under Real_exp/NN_GPU_THR_20260907/.

WHY THIS LANE
-------------
Lanes 4 and 6 searched AV-oriented losses (weighted BCE at 1.0/4.31/8.0, focal at
four gamma/alpha settings, a soft-F1 surrogate) but selected on PR-AUC and never
carried a decision threshold. F1 was then read at a fixed p>=0.5, which at a 17.6
percent AV rate is the wrong cut for a well-calibrated model: the lane-6 DNN never
crosses 0.5 at all, so its AV-F1 reads 0.0000 while the same predictions reach
0.457 at a cut of 0.225.

FFNN and DNN are systems we configure, so the cut is ours to set. Here it becomes
a fitted part of the model rather than an assumption.

HOW THE THRESHOLD IS CHOSEN, AND WHY IT IS HONEST
-------------------------------------------------
GroupKFold(3) on respondent_id gives every development case exactly one
out-of-fold prediction. Those 8,116 out-of-fold probabilities - never seen by the
model that produced them - are pooled, and the threshold that maximises AV-F1 on
them is recorded. That threshold is then FROZEN and applied to the held-out test
unchanged.

The test set is scored once, with the frozen threshold. The
"test-optimal threshold" also printed is a DIAGNOSTIC showing how much the frozen
choice left on the table; it is not a reportable model number, because choosing it
would mean fitting on the test labels.

SELECTION
---------
Primary criterion is CV AV-F1 at the fold-chosen threshold, which is the stated
objective. CV PR-AUC is computed for every candidate as well, and the candidate
that PR-AUC would have selected is reported alongside, so the two criteria can be
compared instead of one silently replacing the other.

Preprocessing is fit inside each fold on training rows only - the lane-6 fix,
retained. Grid, losses, seeds, respondent-grouped splitting and test set are
unchanged from lanes 4 and 6.
"""
from __future__ import annotations

import sys, time, json, itertools, warnings
from pathlib import Path

ROOT = Path(r"C:\SP_LLM\TRC")
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "Real_exp" / "NN_GPU_PRAUC_20260906"))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from tabular import feature_columns, make_preprocessor
from gpu_nn_prauc import (LOSSES, train_one, predict, to_gpu, densify, lossname,
                          SIZES, DROPOUT, WD, LR, INNER_VAL_FRAC)

OUT = Path(__file__).resolve().parent
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
MI = ROOT / "data" / "model_inputs"
THR_GRID = np.linspace(0.02, 0.98, 193)


def grid(name):
    return [{"hidden": h, "dropout": d, "weight_decay": w, "loss": L, "lr": LR}
            for h, d, w, L in itertools.product(SIZES[name], DROPOUT, WD, LOSSES)]


def best_threshold(y, p, average="binary"):
    """Threshold maximising F1 on the given predictions.

    average must stay "binary" for AV-F1. sklearn's average=None returns the
    per-class array, not a scalar, and comparing that to a float raises
    "truth value of an array with more than one element is ambiguous" - which is
    exactly how the first run of this lane died, on all 16 candidates of the
    first block.
    """
    best, bt = -1.0, 0.5
    for t in THR_GRID:
        s = float(f1_score(y, (p >= t).astype(int), average=average, zero_division=0))
        if s > best:
            best, bt = s, float(t)
    return bt, best


def oof_predictions(par, Xraw, yd, groups, feats, seed=42, n_splits=3):
    """One out-of-fold probability per development case. Preprocessing nested."""
    oof = np.full(len(yd), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(np.arange(len(yd)), yd, groups):
        prep = make_preprocessor(*feats).fit(Xraw.iloc[tr_i])
        Xa = densify(prep.transform(Xraw.iloc[tr_i])).astype(np.float32)
        Xb = densify(prep.transform(Xraw.iloc[te_i])).astype(np.float32)
        gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed)
        a, b = next(gi.split(Xa, yd[tr_i], groups[tr_i]))
        model, _, _ = train_one(to_gpu(Xa[a]), to_gpu(yd[tr_i][a]), to_gpu(Xa[b]),
                                yd[tr_i][b], Xa.shape[1], par, seed)
        oof[te_i] = predict(model, to_gpu(Xb))
        del model
    assert not np.isnan(oof).any()
    return oof


def main():
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    print("device:", torch.cuda.get_device_name(0), flush=True)

    tr = pd.read_parquet(MI / "HUMAN_retro_development.parquet")
    te = pd.read_parquet(MI / "HUMAN_retro_test.parquet")
    feats = feature_columns(tr)
    num, cat = feats
    Xraw = tr[num + cat]
    yd = tr.y.to_numpy().astype(np.float32)
    ydi = yd.astype(int)
    yt = te.y.to_numpy().astype(int)
    groups = tr.respondent_id.to_numpy()

    p0 = float(yt.mean())
    all_keep_acc = float((yt == 0).mean())
    all_keep_macro = float(f1_score(yt, np.zeros_like(yt), average="macro",
                                    zero_division=0))
    print("dev %d / %d respondents | test %d / %d respondents"
          % (len(tr), tr.respondent_id.nunique(), len(te), te.respondent_id.nunique()),
          flush=True)
    print("test AV rate %.4f | ALL-KEEP: accuracy %.4f, AV-F1 0.0000, Macro-F1 %.4f | "
          "random-ranker PR-AUC %.4f" % (p0, all_keep_acc, all_keep_macro, p0), flush=True)

    trace_path = OUT / "tuning_trace.csv"
    if trace_path.exists():
        trace_path.unlink()
    hdr, chosen = False, {}
    for name in ("ffnn", "dnn"):
        cands = grid(name)
        print("\n=== %s: %d candidates x 3 folds, threshold fitted on out-of-fold ==="
              % (name, len(cands)), flush=True)
        t0, done, buf = time.time(), [], []
        for i, par in enumerate(cands, 1):
            t1 = time.time()
            try:
                oof = oof_predictions(par, Xraw, yd, groups, feats)
                thr, cv_avf1 = best_threshold(ydi, oof)
                h = (oof >= thr).astype(int)
                row = {"model": name, "hidden": str(par["hidden"]),
                       "loss": lossname(par["loss"]), "dropout": par["dropout"],
                       "weight_decay": par["weight_decay"],
                       "cv_threshold": round(thr, 4),
                       "cv_av_f1": round(cv_avf1, 4),
                       "cv_macro_f1": round(f1_score(ydi, h, average="macro",
                                                     zero_division=0), 4),
                       "cv_accuracy": round(float((h == ydi).mean()), 4),
                       "cv_pr_auc": round(float(average_precision_score(ydi, oof)), 4),
                       "cv_pred_av_share": round(float(h.mean()), 4),
                       "seconds": round(time.time() - t1, 2), "error": ""}
            except Exception as e:
                row = {"model": name, "hidden": str(par["hidden"]),
                       "loss": lossname(par["loss"]), "dropout": par["dropout"],
                       "weight_decay": par["weight_decay"], "cv_threshold": np.nan,
                       "cv_av_f1": np.nan, "cv_macro_f1": np.nan, "cv_accuracy": np.nan,
                       "cv_pr_auc": np.nan, "cv_pred_av_share": np.nan,
                       "seconds": round(time.time() - t1, 2), "error": str(e)[:140]}
            done.append(row); buf.append(row)
            if i % 16 == 0 or i == len(cands):
                pd.DataFrame(buf).to_csv(trace_path, mode="a", index=False,
                                         header=not hdr)
                hdr, buf = True, []
                fin = [r for r in done if r["cv_av_f1"] == r["cv_av_f1"]]
                b = max(fin, key=lambda r: r["cv_av_f1"])
                print("  %3d/%d  %5.0fs  best CV AV-F1 %.4f @thr %.3f  %s"
                      % (i, len(cands), time.time() - t0, b["cv_av_f1"],
                         b["cv_threshold"],
                         (b["hidden"], b["loss"], b["dropout"], b["weight_decay"])),
                      flush=True)
        fin = [r for r in done if r["cv_av_f1"] == r["cv_av_f1"]]
        chosen[name] = {"by_av_f1": max(fin, key=lambda r: r["cv_av_f1"]),
                        "by_pr_auc": max(fin, key=lambda r: r["cv_pr_auc"])}
        a, b = chosen[name]["by_av_f1"], chosen[name]["by_pr_auc"]
        print("  selected by CV AV-F1 : %s %s dr%s wd%s  AV-F1 %.4f  PR-AUC %.4f"
              % (a["hidden"], a["loss"], a["dropout"], a["weight_decay"],
                 a["cv_av_f1"], a["cv_pr_auc"]), flush=True)
        print("  (PR-AUC would pick)  : %s %s dr%s wd%s  AV-F1 %.4f  PR-AUC %.4f"
              % (b["hidden"], b["loss"], b["dropout"], b["weight_decay"],
                 b["cv_av_f1"], b["cv_pr_auc"]), flush=True)

    # ---- refit on full development, apply the FROZEN threshold to test ------
    print("\n=== refit and score held-out test with the frozen CV threshold ===",
          flush=True)
    prep_full = make_preprocessor(num, cat).fit(Xraw)
    Xd = densify(prep_full.transform(Xraw)).astype(np.float32)
    Xt = densify(prep_full.transform(te[num + cat])).astype(np.float32)
    rows = []
    for name, sel in chosen.items():
        b = sel["by_av_f1"]
        L = next(l for l in LOSSES if lossname(l) == b["loss"])
        par = {"hidden": eval(b["hidden"]), "dropout": float(b["dropout"]),
               "weight_decay": float(b["weight_decay"]), "loss": L, "lr": LR}
        thr = float(b["cv_threshold"])
        preds = []
        for seed in (42, 123, 999):
            gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                   random_state=seed)
            a, v = next(gi.split(Xd, yd, groups))
            model, _, eps = train_one(to_gpu(Xd[a]), to_gpu(yd[a]), to_gpu(Xd[v]),
                                      yd[v], Xd.shape[1], par, seed)
            p = predict(model, to_gpu(Xt))
            preds.append(p)
            rows.append(score_row(name, seed, "single", p, yt, thr, par, b,
                                  all_keep_acc, all_keep_macro, eps))
            pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": p,
                          "world": "HUMAN", "info": "retro", "model": name,
                          "seed": seed}).to_parquet(
                OUT / "predictions" / ("HUMAN_retro_%s_seed%d.parquet" % (name, seed)),
                index=False)
            torch.save(model.state_dict(), OUT / ("model_%s_seed%d.pt" % (name, seed)))
            r = rows[-1]
            print("  %-5s seed%-4d thr %.3f | AV-F1 %.4f  Macro-F1 %.4f  acc %.4f | "
                  "PR-AUC %.4f  xent %.4f"
                  % (name, seed, thr, r["AV_F1_frozen_thr"], r["Macro_F1_frozen_thr"],
                     r["accuracy_frozen_thr"], r["pr_auc"], r["xent"]), flush=True)
        pe = np.mean(preds, 0)
        rows.append(score_row(name, -1, "ensemble_3", pe, yt, thr, par, b,
                              all_keep_acc, all_keep_macro, None))
        pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": pe,
                      "world": "HUMAN", "info": "retro", "model": name,
                      "seed": -1}).to_parquet(
            OUT / "predictions" / ("HUMAN_retro_%s_ensemble.parquet" % name), index=False)
        r = rows[-1]
        print("  %-5s ENSEMBLE  thr %.3f | AV-F1 %.4f  Macro-F1 %.4f  acc %.4f"
              % (name, thr, r["AV_F1_frozen_thr"], r["Macro_F1_frozen_thr"],
                 r["accuracy_frozen_thr"]), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    json.dump({"threshold": "fitted on pooled out-of-fold development predictions, "
                            "frozen before touching test",
               "selection": "CV AV-F1 at the fold-chosen threshold; the PR-AUC choice "
                            "is recorded alongside",
               "preprocessing": "nested, fit per fold on training rows only (lane-6 fix)",
               "all_keep": {"accuracy": all_keep_acc, "av_f1": 0.0,
                            "macro_f1": all_keep_macro},
               "test_av_rate": p0, "chosen": chosen},
              open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\nDONE", flush=True)


def score_row(name, seed, kind, p, yt, thr, par, b, ak_acc, ak_macro, eps):
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    h = (p >= thr).astype(int)
    t_opt, f_opt = best_threshold(yt, p)
    return {"model": name, "seed": seed, "kind": kind,
            "frozen_threshold": round(thr, 4),
            "AV_F1_frozen_thr": round(f1_score(yt, h, zero_division=0), 4),
            "Macro_F1_frozen_thr": round(f1_score(yt, h, average="macro",
                                                  zero_division=0), 4),
            "accuracy_frozen_thr": round(float((h == yt).mean()), 4),
            "predicted_AV_share": round(float(h.mean()), 4),
            "TP": int(((h == 1) & (yt == 1)).sum()), "FP": int(((h == 1) & (yt == 0)).sum()),
            "FN": int(((h == 0) & (yt == 1)).sum()), "TN": int(((h == 0) & (yt == 0)).sum()),
            "pr_auc": round(float(average_precision_score(yt, p)), 4),
            "roc_auc": round(float(roc_auc_score(yt, p)), 4),
            "xent": round(float(-(yt * np.log(pc) + (1 - yt) * np.log(1 - pc)).mean()), 4),
            "all_keep_accuracy": round(ak_acc, 4), "all_keep_macro_f1": round(ak_macro, 4),
            "DIAG_test_optimal_threshold": round(t_opt, 4),
            "DIAG_AV_F1_at_test_optimum": round(f_opt, 4),
            "p_mean": round(float(p.mean()), 4), "p_max": round(float(p.max()), 4),
            "hidden": b["hidden"], "loss": b["loss"], "dropout": b["dropout"],
            "weight_decay": b["weight_decay"], "cv_av_f1": b["cv_av_f1"],
            "epochs": eps}


if __name__ == "__main__":
    main()
