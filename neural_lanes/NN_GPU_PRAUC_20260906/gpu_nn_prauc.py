# -*- coding: utf-8 -*-
"""GPU FFNN/DNN trained and selected for AV-case detection, no hard decision rule.

ADD-ONLY. Third lane. Supersedes nothing; writes only under this folder.

WHAT CHANGED FROM Real_exp/NN_GPU_20260905 AND WHY
--------------------------------------------------
That lane trained with plain BCE and selected on cross-entropy. Two things went
wrong with that objective on a 17.6 percent positive class:

  1. cross-entropy rewarded collapse. The sklearn DNN it was compared against
     was selected at alpha=10.0 with a CV score of 0.4651 - the base-rate
     constant to four decimals - and scored ROC-AUC 0.610 with zero AV
     predictions on the test set.
  2. every threshold metric was read at a fixed p>=0.5 cut. Better-calibrated
     models never reach 0.5 at this base rate, so the GPU DNN showed AV-F1
     0.0000 while holding ROC-AUC 0.756. The number measured the threshold,
     not the model.

The stated objective is catching AV cases, and the hard decision rule is out.
So this lane changes exactly two things:

  A. THE LOSS IS SEARCHED, not fixed. Weighted BCE, focal loss, and a
     differentiable soft-F1 surrogate all enter the grid as candidates.
  B. SELECTION AND EARLY STOPPING USE PR-AUC (average precision), which is
     threshold-free and is the metric CLAUDE.md names as the meaningful one at
     this positive rate. Cross-entropy is still computed, but only as a
     calibration diagnostic - it never selects anything.

HELD IDENTICAL to the registered lane so the comparison stays about the model:
same model_inputs parquets, same feature split and preprocessing imported from
pipeline/tabular.py and fitted on development only, same outer protocol
(GroupKFold(3) on respondent_id, searched once at seed 42), same refit seeds
42/123/999, same held-out test.

PRUNED WITH EVIDENCE, not guesswork: batchnorm is fixed False. In the 288-
candidate NN_GPU_20260905 trace it lost on both mean and min for both models
(ffnn 0.4801 vs 0.4627 mean, dnn 0.5010 vs 0.4621, 72 candidates each).
pos_weight is NOT pruned even though 4.31 looked bad there - that evidence was
produced under the cross-entropy criterion, which penalises the calibration
shift a positive weight causes. Under PR-AUC it has to be re-measured.

KNOWN TENSION, stated rather than hidden: PR-AUC scores ranking, not
probability level. EXP2-EXP4 need probability responses (delta P), so a model
that ranks well can still have compressed or shifted probabilities. Test
cross-entropy is therefore reported next to PR-AUC for every fit.
"""
from __future__ import annotations

import sys, time, json, itertools, warnings
from pathlib import Path

ROOT = Path(r"C:\SP_LLM\TRC")
sys.path.insert(0, str(ROOT / "pipeline"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score

from tabular import feature_columns, make_preprocessor
import metrics as M

OUT = Path(__file__).resolve().parent
MI = ROOT / "data" / "model_inputs"
DEV = "cuda"

MAX_EPOCHS = 300
PATIENCE = 30
BATCH = 512
INNER_VAL_FRAC = 0.15      # respondent-grouped


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def make_loss(spec):
    """spec: ('bce', pos_weight) | ('focal', gamma, alpha) | ('softf1',)"""
    kind = spec[0]
    if kind == "bce":
        pw = torch.tensor([float(spec[1])], device=DEV)
        f = nn.BCEWithLogitsLoss(pos_weight=pw)
        return lambda z, y: f(z, y)
    if kind == "focal":
        gamma, alpha = float(spec[1]), float(spec[2])

        def focal(z, y):
            # numerically stable binary focal loss on logits
            logp = F.logsigmoid(z)              # log p
            log1p = F.logsigmoid(-z)            # log (1-p)
            p = torch.sigmoid(z)
            pt = torch.where(y > 0.5, p, 1 - p)
            logpt = torch.where(y > 0.5, logp, log1p)
            at = torch.where(y > 0.5, torch.full_like(y, alpha),
                             torch.full_like(y, 1 - alpha))
            return -(at * (1 - pt).pow(gamma) * logpt).mean()
        return focal
    if kind == "softf1":
        def softf1(z, y):
            p = torch.sigmoid(z)
            tp = (p * y).sum()
            fp = (p * (1 - y)).sum()
            fn = ((1 - p) * y).sum()
            return 1.0 - (2 * tp) / (2 * tp + fp + fn + 1e-8)
        return softf1
    raise ValueError(spec)


LOSSES = ([("bce", 1.0), ("bce", 4.31), ("bce", 8.0)]
          + [("focal", g, a) for g in (1.0, 2.0) for a in (0.25, 0.75)]
          + [("softf1",)])


# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout=0.0):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva_np, d_in, par, seed):
    """Early stopping MAXIMISES PR-AUC on a respondent-grouped inner split."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLP(d_in, par["hidden"], par["dropout"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=par["lr"],
                            weight_decay=par["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossf = make_loss(par["loss"])

    n = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva)).float().cpu().numpy()
        ap = float(average_precision_score(yva_np, pv)) if yva_np.sum() else 0.0
        if ap > best + 1e-5:
            best, bad = ap, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(model(X)).float().cpu().numpy().astype(np.float64)


SIZES = {
    "ffnn": [(32,), (64,), (64, 32), (128, 64)],
    "dnn":  [(128, 64, 32), (256, 128, 64), (256, 128, 64, 32), (512, 256, 128, 64)],
}
DROPOUT = [0.0, 0.3]
WD = [1e-5, 1e-3]
LR = 1e-3


def grid(name):
    return [{"hidden": h, "dropout": d, "weight_decay": w, "loss": L, "lr": LR}
            for h, d, w, L in itertools.product(SIZES[name], DROPOUT, WD, LOSSES)]


def to_gpu(a):
    return torch.tensor(np.asarray(a, dtype=np.float32), device=DEV)


def densify(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def cv_score(par, Xd, yd, groups, d_in, seed=42, n_splits=3):
    """Mean held-out PR-AUC over respondent-grouped folds. Higher is better.
    Cross-entropy on the same folds is returned alongside as a diagnostic."""
    aps, lls = [], []
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(Xd, yd, groups):
        gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed)
        a, b = next(gi.split(Xd[tr_i], yd[tr_i], groups[tr_i]))
        model, _, _ = train_one(to_gpu(Xd[tr_i][a]), to_gpu(yd[tr_i][a]),
                                to_gpu(Xd[tr_i][b]), yd[tr_i][b], d_in, par, seed)
        p = np.clip(predict(model, to_gpu(Xd[te_i])), 1e-7, 1 - 1e-7)
        yt = yd[te_i]
        aps.append(float(average_precision_score(yt, p)))
        lls.append(float(-(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()))
        del model
    return float(np.mean(aps)), float(np.mean(lls))


def lossname(L):
    return L[0] if L[0] == "softf1" else "%s(%s)" % (L[0], ",".join(str(x) for x in L[1:]))


def main(smoke=False):
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    print("device:", torch.cuda.get_device_name(0), flush=True)

    tr = pd.read_parquet(MI / "HUMAN_retro_development.parquet")
    te = pd.read_parquet(MI / "HUMAN_retro_test.parquet")
    num, cat = feature_columns(tr)
    prep = make_preprocessor(num, cat).fit(tr[num + cat])
    Xd = densify(prep.transform(tr[num + cat])).astype(np.float32)
    Xt = densify(prep.transform(te[num + cat])).astype(np.float32)
    yd = tr.y.to_numpy().astype(np.float32)
    yt = te.y.to_numpy().astype(np.float32)
    groups = tr.respondent_id.to_numpy()
    d_in = Xd.shape[1]

    p0 = float(yt.mean())
    print("dev %s -> %s   test %s -> %s" % (tr.shape, Xd.shape, te.shape, Xt.shape),
          flush=True)
    print("test AV rate %.4f   PR-AUC of a random ranker = %.4f   "
          "base-rate cross-entropy = %.4f"
          % (p0, p0, -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))), flush=True)
    print("registered sklearn PR-AUC on this test: ffnn .3841/.3959/.3694  "
          "dnn .3963/.3996/.3823", flush=True)

    if smoke:
        for name in ("ffnn", "dnn"):
            g = grid(name)
            for par in (g[0], g[len(g) // 3], g[len(g) // 2], g[-1]):
                t0 = time.time()
                ap, ll = cv_score(par, Xd, yd, groups, d_in)
                print("  smoke %-4s %-22s %-14s dr%.1f wd%.0e  PR-AUC %.4f  xent %.4f  %.1fs"
                      % (name, str(par["hidden"]), lossname(par["loss"]),
                         par["dropout"], par["weight_decay"], ap, ll, time.time() - t0),
                      flush=True)
        return

    trace_path = OUT / "tuning_trace.csv"
    if trace_path.exists():
        trace_path.unlink()
    hdr, best = False, {}
    for name in ("ffnn", "dnn"):
        cands = grid(name)
        print("\n=== %s: %d candidates x 3 folds, selecting on PR-AUC ==="
              % (name, len(cands)), flush=True)
        t0, done, buf = time.time(), [], []
        for i, par in enumerate(cands, 1):
            t1 = time.time()
            try:
                ap, ll = cv_score(par, Xd, yd, groups, d_in)
                err = ""
            except Exception as e:
                ap, ll, err = float("nan"), float("nan"), str(e)[:120]
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
                print("  %3d/%d  %5.0fs  best PR-AUC %.4f  %s"
                      % (i, len(cands), time.time() - t0, b["cv_pr_auc"],
                         (b["hidden"], b["loss"], b["dropout"], b["weight_decay"])),
                      flush=True)
        best[name] = max([r for r in done if r["cv_pr_auc"] == r["cv_pr_auc"]],
                         key=lambda r: r["cv_pr_auc"])

    print("\n=== refit and score held-out test (threshold-free) ===", flush=True)
    rows = []
    for name, b in best.items():
        L = next(l for l in LOSSES if lossname(l) == b["loss"])
        par = {"hidden": eval(b["hidden"]), "dropout": float(b["dropout"]),
               "weight_decay": float(b["weight_decay"]), "loss": L, "lr": LR}
        for seed in (42, 123, 999):
            t0 = time.time()
            gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                   random_state=seed)
            a, v = next(gi.split(Xd, yd, groups))
            model, vap, eps = train_one(to_gpu(Xd[a]), to_gpu(yd[a]),
                                        to_gpu(Xd[v]), yd[v], d_in, par, seed)
            p = predict(model, to_gpu(Xt))
            pc = np.clip(p, 1e-7, 1 - 1e-7)
            rows.append({"model": name, "seed": seed,
                         "pr_auc": round(float(average_precision_score(yt, p)), 4),
                         "roc_auc": round(float(roc_auc_score(yt, p)), 4),
                         "xent_diag": round(float(-(yt * np.log(pc) +
                                                    (1 - yt) * np.log(1 - pc)).mean()), 4),
                         "p_mean": round(float(p.mean()), 4),
                         "p_min": round(float(p.min()), 4),
                         "p_max": round(float(p.max()), 4),
                         "inner_val_pr_auc": round(vap, 4), "epochs": eps,
                         "hidden": b["hidden"], "loss": b["loss"],
                         "dropout": b["dropout"], "weight_decay": b["weight_decay"],
                         "fit_seconds": round(time.time() - t0, 2)})
            torch.save(model.state_dict(), OUT / ("model_%s_seed%d.pt" % (name, seed)))
            pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": p,
                          "world": "HUMAN", "info": "retro", "model": name,
                          "seed": seed}).to_parquet(
                OUT / "predictions" / ("HUMAN_retro_%s_seed%d.parquet" % (name, seed)),
                index=False)
            r = rows[-1]
            print("  %-5s seed%-4d PR-AUC %.4f  ROC-AUC %.4f  xent %.4f  "
                  "p in [%.3f, %.3f]  %d epochs"
                  % (name, seed, r["pr_auc"], r["roc_auc"], r["xent_diag"],
                     r["p_min"], r["p_max"], eps), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    json.dump({"selection_criterion": "PR-AUC (average precision), threshold-free",
               "test_av_rate": p0, "random_ranker_pr_auc": p0,
               "registered_pr_auc": {"ffnn": [.3841, .3959, .3694],
                                     "dnn": [.3963, .3996, .3823]},
               "best": best, "losses_searched": [lossname(l) for l in LOSSES],
               "n_candidates": {k: len(grid(k)) for k in ("ffnn", "dnn")},
               "batchnorm": "fixed False - lost on mean and min in the "
                            "288-candidate NN_GPU_20260905 trace"},
              open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
