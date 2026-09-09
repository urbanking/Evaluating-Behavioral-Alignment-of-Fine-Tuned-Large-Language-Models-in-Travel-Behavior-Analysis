# -*- coding: utf-8 -*-
"""Lane 6. Lane 4 re-run with the preprocessing leak fixed. ADD-ONLY.

THE BUG THIS FIXES
------------------
Lanes 3, 4 and 5 fit the ColumnTransformer once on the whole development set and
then split the transformed matrix for cross-validation:

    prep = make_preprocessor(num, cat).fit(tr[num + cat])     # all 8,116 rows
    Xd   = prep.transform(tr[num + cat])
    ...  GroupKFold split of Xd

So each validation fold contributed to the imputer medians, the scaler mean and
standard deviation, and the one-hot vocabulary (min_frequency=10). Selection was
therefore made on inflated scores. Measured on the lane-4 winners:

    model   leaky CV PR-AUC   nested CV PR-AUC   difference
    ffnn         0.4350            0.4297          -0.0054
    dnn          0.4278            0.4030          -0.0248

The 0.0248 on the DNN is larger than the gaps that separated the top candidates,
so the lane-4 DNN selection is not trustworthy. Lane 2 (sklearn) never had this
problem because make_preprocessor sat inside the Pipeline and cross_val_score
refit it per fold.

The held-out test was never affected: the preprocessor was always fit on
development only, so reported test numbers stay valid. The leak corrupted
selection, not evaluation.

WHAT IS DIFFERENT FROM LANE 4
-----------------------------
Exactly one thing: the preprocessor is now fit inside each outer CV fold, on that
fold's training rows only, and applied to the held-out rows. The grid, the losses,
the selection criterion (PR-AUC), the seeds, the respondent-grouped splitting and
the test set are all unchanged, so the difference between lane 4 and lane 6 is
attributable to the fix and nothing else.

ACCURACY IS REPORTED, NOT SELECTED ON
-------------------------------------
Accuracy is added to the test output because it was asked for. Read it against
the trivial floor: at a 17.6 percent AV rate, predicting Keep for every case
scores 0.8241. Across every lane built so far the best accuracy at any threshold
is 0.8391, so the entire usable range of this metric on this problem is about
1.5 percentage points. Selection stays on PR-AUC, which has a random-ranker floor
of 0.1759 and where the redesign moved 0.383 to 0.43.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from tabular import feature_columns, make_preprocessor
# reuse lane 4's model, losses and training loop unchanged
from gpu_nn_prauc import (LOSSES, MLP, train_one, predict, to_gpu, densify,
                          lossname, SIZES, DROPOUT, WD, LR, INNER_VAL_FRAC)

OUT = Path(__file__).resolve().parent
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
MI = ROOT / "data" / "model_inputs"


def grid(name):
    return [{"hidden": h, "dropout": d, "weight_decay": w, "loss": L, "lr": LR}
            for h, d, w, L in itertools.product(SIZES[name], DROPOUT, WD, LOSSES)]


def cv_score_nested(par, Xraw, yd, groups, seed=42, n_splits=3):
    """Preprocessor is fit on the training part of each fold only."""
    aps, lls = [], []
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(np.arange(len(yd)), yd, groups):
        prep = make_preprocessor(*FEATS).fit(Xraw.iloc[tr_i])
        Xa = densify(prep.transform(Xraw.iloc[tr_i])).astype(np.float32)
        Xb = densify(prep.transform(Xraw.iloc[te_i])).astype(np.float32)
        gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed)
        a, b = next(gi.split(Xa, yd[tr_i], groups[tr_i]))
        model, _, _ = train_one(to_gpu(Xa[a]), to_gpu(yd[tr_i][a]), to_gpu(Xa[b]),
                                yd[tr_i][b], Xa.shape[1], par, seed)
        p = np.clip(predict(model, to_gpu(Xb)), 1e-7, 1 - 1e-7)
        yt = yd[te_i]
        aps.append(float(average_precision_score(yt, p)))
        lls.append(float(-(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()))
        del model
    return float(np.mean(aps)), float(np.mean(lls))


def acc_block(y, p, base_rate):
    """Accuracy at the fixed cut, at a share-matched cut, and at its own best cut."""
    a5 = float(((p >= .5).astype(int) == y).mean())
    thr = float(np.quantile(p, 1 - base_rate))
    ash = float(((p >= thr).astype(int) == y).mean())
    grid_t = np.linspace(.02, .98, 193)
    accs = [(float(((p >= t).astype(int) == y).mean()), float(t)) for t in grid_t]
    best, best_t = max(accs)
    return {"accuracy_at_0.5": round(a5, 4),
            "accuracy_share_matched": round(ash, 4),
            "accuracy_best_threshold": round(best, 4),
            "best_threshold": round(best_t, 3),
            "share_matched_threshold": round(thr, 3)}


def main():
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    print("device:", torch.cuda.get_device_name(0), flush=True)

    tr = pd.read_parquet(MI / "HUMAN_retro_development.parquet")
    te = pd.read_parquet(MI / "HUMAN_retro_test.parquet")
    global FEATS
    FEATS = feature_columns(tr)
    num, cat = FEATS
    Xraw = tr[num + cat]
    yd = tr.y.to_numpy().astype(np.float32)
    yt = te.y.to_numpy().astype(np.float32)
    groups = tr.respondent_id.to_numpy()

    p0 = float(yt.mean())
    all_keep = float((yt == 0).mean())
    print("dev %d rows / %d respondents | test %d cases / %d respondents"
          % (len(tr), tr.respondent_id.nunique(), len(te), te.respondent_id.nunique()),
          flush=True)
    print("test AV rate %.4f | ALL-KEEP accuracy floor %.4f | random-ranker PR-AUC %.4f"
          % (p0, all_keep, p0), flush=True)
    print("lane 4 (leaky CV) test PR-AUC to compare against: ffnn .4289  dnn .4289",
          flush=True)

    trace_path = OUT / "tuning_trace.csv"
    if trace_path.exists():
        trace_path.unlink()
    hdr, best = False, {}
    for name in ("ffnn", "dnn"):
        cands = grid(name)
        print("\n=== %s: %d candidates x 3 folds, NESTED preprocessing ==="
              % (name, len(cands)), flush=True)
        t0, done, buf = time.time(), [], []
        for i, par in enumerate(cands, 1):
            t1 = time.time()
            try:
                ap, ll = cv_score_nested(par, Xraw, yd, groups)
                err = ""
            except Exception as e:
                ap, ll, err = float("nan"), float("nan"), str(e)[:140]
            row = {"model": name, "hidden": str(par["hidden"]),
                   "loss": lossname(par["loss"]), "dropout": par["dropout"],
                   "weight_decay": par["weight_decay"], "cv_pr_auc": ap,
                   "cv_xent_diag": ll, "seconds": round(time.time() - t1, 2),
                   "error": err}
            done.append(row); buf.append(row)
            if i % 16 == 0 or i == len(cands):
                pd.DataFrame(buf).to_csv(trace_path, mode="a", index=False,
                                         header=not hdr)
                hdr, buf = True, []
                fin = [r for r in done if r["cv_pr_auc"] == r["cv_pr_auc"]]
                b = max(fin, key=lambda r: r["cv_pr_auc"])
                print("  %3d/%d  %5.0fs  best CV PR-AUC %.4f  %s"
                      % (i, len(cands), time.time() - t0, b["cv_pr_auc"],
                         (b["hidden"], b["loss"], b["dropout"], b["weight_decay"])),
                      flush=True)
        best[name] = max([r for r in done if r["cv_pr_auc"] == r["cv_pr_auc"]],
                         key=lambda r: r["cv_pr_auc"])

    # ---- refit on full development, score the held-out test once ------------
    print("\n=== refit and score held-out test ===", flush=True)
    prep_full = make_preprocessor(num, cat).fit(Xraw)      # development only
    Xd = densify(prep_full.transform(Xraw)).astype(np.float32)
    Xt = densify(prep_full.transform(te[num + cat])).astype(np.float32)
    rows = []
    for name, b in best.items():
        L = next(l for l in LOSSES if lossname(l) == b["loss"])
        par = {"hidden": eval(b["hidden"]), "dropout": float(b["dropout"]),
               "weight_decay": float(b["weight_decay"]), "loss": L, "lr": LR}
        preds = []
        for seed in (42, 123, 999):
            gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                   random_state=seed)
            a, v = next(gi.split(Xd, yd, groups))
            model, vap, eps = train_one(to_gpu(Xd[a]), to_gpu(yd[a]), to_gpu(Xd[v]),
                                        yd[v], Xd.shape[1], par, seed)
            p = predict(model, to_gpu(Xt))
            preds.append(p)
            pc = np.clip(p, 1e-7, 1 - 1e-7)
            r = {"model": name, "seed": seed, "kind": "single",
                 "pr_auc": round(float(average_precision_score(yt, p)), 4),
                 "roc_auc": round(float(roc_auc_score(yt, p)), 4),
                 "xent_diag": round(float(-(yt * np.log(pc) +
                                            (1 - yt) * np.log(1 - pc)).mean()), 4),
                 **acc_block(yt.astype(int), p, p0),
                 "all_keep_accuracy": round(all_keep, 4),
                 "p_mean": round(float(p.mean()), 4), "p_max": round(float(p.max()), 3),
                 "epochs": eps, "hidden": b["hidden"], "loss": b["loss"],
                 "dropout": b["dropout"], "weight_decay": b["weight_decay"],
                 "cv_pr_auc_nested": round(b["cv_pr_auc"], 4)}
            rows.append(r)
            pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": p,
                          "world": "HUMAN", "info": "retro", "model": name,
                          "seed": seed}).to_parquet(
                OUT / "predictions" / ("HUMAN_retro_%s_seed%d.parquet" % (name, seed)),
                index=False)
            torch.save(model.state_dict(), OUT / ("model_%s_seed%d.pt" % (name, seed)))
            print("  %-5s seed%-4d PR-AUC %.4f  ROC-AUC %.4f  xent %.4f  "
                  "acc@0.5 %.4f (all-Keep %.4f)  best-acc %.4f"
                  % (name, seed, r["pr_auc"], r["roc_auc"], r["xent_diag"],
                     r["accuracy_at_0.5"], all_keep, r["accuracy_best_threshold"]),
                  flush=True)
        pe = np.mean(preds, 0)
        pc = np.clip(pe, 1e-7, 1 - 1e-7)
        rows.append({"model": name, "seed": -1, "kind": "ensemble_3",
                     "pr_auc": round(float(average_precision_score(yt, pe)), 4),
                     "roc_auc": round(float(roc_auc_score(yt, pe)), 4),
                     "xent_diag": round(float(-(yt * np.log(pc) +
                                                (1 - yt) * np.log(1 - pc)).mean()), 4),
                     **acc_block(yt.astype(int), pe, p0),
                     "all_keep_accuracy": round(all_keep, 4),
                     "p_mean": round(float(pe.mean()), 4),
                     "p_max": round(float(pe.max()), 3),
                     "hidden": b["hidden"], "loss": b["loss"],
                     "dropout": b["dropout"], "weight_decay": b["weight_decay"],
                     "cv_pr_auc_nested": round(b["cv_pr_auc"], 4)})
        pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": pe,
                      "world": "HUMAN", "info": "retro", "model": name,
                      "seed": -1}).to_parquet(
            OUT / "predictions" / ("HUMAN_retro_%s_ensemble.parquet" % name), index=False)
        print("  %-5s ENSEMBLE  PR-AUC %.4f  best-acc %.4f"
              % (name, rows[-1]["pr_auc"], rows[-1]["accuracy_best_threshold"]),
              flush=True)

    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    json.dump({"fix": "preprocessing fit inside each CV fold on training rows only",
               "unchanged_from_lane4": ["grid", "losses", "PR-AUC selection", "seeds",
                                        "respondent-grouped splitting", "test set"],
               "leak_measured_on_lane4_winners": {"ffnn": -0.0054, "dnn": -0.0248},
               "all_keep_accuracy_floor": all_keep, "test_av_rate": p0,
               "best": best}, open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
