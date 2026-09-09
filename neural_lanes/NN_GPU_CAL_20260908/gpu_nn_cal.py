# -*- coding: utf-8 -*-
"""Lane 8. Platt calibration on the re-tuned FFNN/DNN. ADD-ONLY.

WHY
---
The neural rows were the only ones in the prediction table carrying raw network
probabilities. Every Qwen row is already calibrated - that is why the registered
Qwen 2B ZS log loss reads 0.441 instead of its raw 0.60. Applying the same
treatment to the networks removes an inconsistency rather than adding a step.

It also dissolves the trade-off reported from lanes 6 and 7. Platt scaling is a
monotone transform, so ROC-AUC and PR-AUC cannot change; only the probability
scale moves. The threshold is re-fitted on the calibrated scale, so the
classification metrics are preserved too. Log loss is the one thing that moves.

    lane 6 (PR-AUC selection) raw:  log loss 0.405 / 0.413, Macro-F1 0.490 / 0.452
    lane 7 (AV-F1 selection)  raw:  log loss 0.509 / 0.495, Macro-F1 0.652 / 0.643
    constant-prevalence baseline:   log loss 0.465, Macro-F1 0.452

METHOD, MATCHING THE REGISTERED CONVENTION
------------------------------------------
pipeline/p17_calibration.fit_platt is reused unchanged: sigmoid(a*margin + b)
with the slope constrained to a >= 0, so a score whose ordering is inverted
collapses to the base rate instead of being silently flipped.

The margin for a network is the logit of its predicted probability. Both the
calibrator and the decision threshold are fitted on pooled out-of-fold
development predictions - respondent-grouped GroupKFold(3), preprocessing nested
inside each fold - and then frozen. The held-out sample is scored once, with the
frozen calibrator and the frozen threshold. No test label is used to fit
anything.

Both the lane-6 and the lane-7 configurations are run so the effect of
calibration can be read against the selection criterion that produced each.
"""
from __future__ import annotations

import sys, json, warnings
from pathlib import Path

ROOT = Path(r"C:\SP_LLM\TRC")
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "Real_exp" / "NN_GPU_PRAUC_20260906"))
sys.path.insert(0, str(ROOT / "Real_exp" / "NN_GPU_THR_20260907"))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

from tabular import feature_columns, make_preprocessor
from p17_calibration import fit_platt
from gpu_nn_prauc import LOSSES, train_one, predict, to_gpu, densify, lossname, LR, INNER_VAL_FRAC

OUT = Path(__file__).resolve().parent
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
MI = ROOT / "data" / "model_inputs"
THR_GRID = np.linspace(0.02, 0.98, 193)
SEEDS = (42, 123, 999)

# selected configurations, copied from each lane's summary.json
CONFIGS = {
    "lane6_prauc": {
        "ffnn": dict(hidden=(32,), dropout=0.3, weight_decay=1e-3,
                     loss=("bce", 1.0), lr=LR),
        "dnn": dict(hidden=(512, 256, 128, 64), dropout=0.0, weight_decay=1e-3,
                    loss=("focal", 1.0, 0.25), lr=LR),
    },
    "lane7_avf1": {
        "ffnn": dict(hidden=(32,), dropout=0.3, weight_decay=1e-5,
                     loss=("focal", 1.0, 0.75), lr=LR),
        "dnn": dict(hidden=(256, 128, 64, 32), dropout=0.3, weight_decay=1e-5,
                    loss=("focal", 1.0, 0.75), lr=LR),
    },
}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_threshold(y, p):
    best, bt = -1.0, 0.5
    for t in THR_GRID:
        s = float(f1_score(y, (p >= t).astype(int), zero_division=0))
        if s > best:
            best, bt = s, float(t)
    return bt, best


def oof_predictions(par, Xraw, yd, groups, feats, seed=42, n_splits=3):
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


def row(tag, name, seed, kind, y, p, thr, extra):
    h = (p >= thr).astype(int)
    return {"config": tag, "model": name, "seed": seed, "kind": kind,
            "threshold": round(thr, 4),
            "Accuracy": round(float((h == y).mean()), 4),
            "Macro_F1": round(f1_score(y, h, average="macro", zero_division=0), 4),
            "AV_F1": round(f1_score(y, h, zero_division=0), 4),
            "Log_Loss": round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15),
                                             labels=[0, 1])), 4),
            "PR_AUC": round(float(average_precision_score(y, p)), 4),
            "ROC_AUC": round(float(roc_auc_score(y, p)), 4),
            "pred_AV_share": round(float(h.mean()), 4),
            "p_mean": round(float(p.mean()), 4), "p_max": round(float(p.max()), 4),
            **extra}


def main():
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True

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
    base_ll = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    print("constant prevalence: Accuracy %.4f  Macro-F1 %.4f  Log Loss %.4f"
          % ((yt == 0).mean(), f1_score(yt, np.zeros_like(yt), average="macro",
                                        zero_division=0), base_ll), flush=True)

    prep_full = make_preprocessor(num, cat).fit(Xraw)
    Xd = densify(prep_full.transform(Xraw)).astype(np.float32)
    Xt = densify(prep_full.transform(te[num + cat])).astype(np.float32)

    rows, cal_log = [], {}
    for tag, cfg in CONFIGS.items():
        for name, par in cfg.items():
            print("\n=== %s / %s : %s %s ===" % (tag, name, par["hidden"],
                                                 lossname(par["loss"])), flush=True)
            oof = oof_predictions(par, Xraw, yd, groups, feats)
            cal = fit_platt(logit(oof), ydi, groups)
            thr_raw, _ = best_threshold(ydi, oof)
            oof_cal = 1.0 / (1.0 + np.exp(-(cal["a"] * logit(oof) + cal["b"])))
            thr_cal, cv_avf1 = best_threshold(ydi, oof_cal)
            cal_log["%s/%s" % (tag, name)] = {
                "platt_a": round(cal["a"], 5), "platt_b": round(cal["b"], 5),
                "slope_unconstrained": round(cal["slope_unconstrained"], 5),
                "orientation_flipped": cal["orientation_flipped"],
                "oof_log_loss_calibrated": round(cal["oof_log_loss"], 4),
                "oof_margin_auc": round(cal["margin_auc"], 4),
                "threshold_raw": round(thr_raw, 4), "threshold_calibrated": round(thr_cal, 4),
                "cv_av_f1_calibrated": round(cv_avf1, 4)}
            print("  Platt a=%.4f b=%.4f (unconstrained slope %.4f, flipped=%s)"
                  % (cal["a"], cal["b"], cal["slope_unconstrained"],
                     cal["orientation_flipped"]), flush=True)
            print("  OOF log loss after calibration %.4f | threshold raw %.3f -> cal %.3f"
                  % (cal["oof_log_loss"], thr_raw, thr_cal), flush=True)

            praw, pcal = [], []
            for seed in SEEDS:
                gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                       random_state=seed)
                a, v = next(gi.split(Xd, yd, groups))
                model, _, _ = train_one(to_gpu(Xd[a]), to_gpu(yd[a]), to_gpu(Xd[v]),
                                        yd[v], Xd.shape[1], par, seed)
                p = predict(model, to_gpu(Xt))
                pc = 1.0 / (1.0 + np.exp(-(cal["a"] * logit(p) + cal["b"])))
                praw.append(p); pcal.append(pc)
                ex = {"hidden": str(par["hidden"]), "loss": lossname(par["loss"])}
                rows.append(row(tag, name, seed, "raw", yt, p, thr_raw, ex))
                rows.append(row(tag, name, seed, "calibrated", yt, pc, thr_cal, ex))
                pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": pc,
                              "p_raw": p, "world": "HUMAN", "info": "retro",
                              "model": name, "seed": seed}).to_parquet(
                    OUT / "predictions" / ("%s_%s_seed%d.parquet" % (tag, name, seed)),
                    index=False)
                del model
            for kind, ps, thr in (("raw", praw, thr_raw), ("calibrated", pcal, thr_cal)):
                per = [row(tag, name, s, kind, yt, p, thr, {}) for s, p in zip(SEEDS, ps)]
                avg = {k: round(float(np.mean([r[k] for r in per])), 4)
                       for k in ("Accuracy", "Macro_F1", "AV_F1", "Log_Loss",
                                 "PR_AUC", "ROC_AUC", "pred_AV_share")}
                rows.append({"config": tag, "model": name, "seed": "mean of 3",
                             "kind": kind, "threshold": round(thr, 4), **avg,
                             "hidden": str(par["hidden"]), "loss": lossname(par["loss"])})
                print("  %-10s seed-mean  Acc %.4f  Macro-F1 %.4f  AV-F1 %.4f  "
                      "LogLoss %.4f  PR-AUC %.4f"
                      % (kind, avg["Accuracy"], avg["Macro_F1"], avg["AV_F1"],
                         avg["Log_Loss"], avg["PR_AUC"]), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "test_metrics.csv", index=False)
    json.dump({"calibration": "Platt, slope constrained a>=0, reused from "
                              "pipeline/p17_calibration.fit_platt",
               "fitted_on": "pooled out-of-fold development predictions, respondent-grouped "
                            "GroupKFold(3), preprocessing nested inside each fold",
               "threshold": "re-fitted on the calibrated out-of-fold scale, then frozen",
               "test_used_for_fitting": False,
               "constant_prevalence_log_loss": round(base_ll, 4),
               "per_config": cal_log},
              open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\n=== seed-mean summary ===", flush=True)
    m = d[d.seed == "mean of 3"][["config", "model", "kind", "threshold", "Accuracy",
                                  "Macro_F1", "AV_F1", "Log_Loss", "PR_AUC"]]
    print(m.to_string(index=False), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
