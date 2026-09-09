# -*- coding: utf-8 -*-
"""GPU (RTX 3080) PyTorch FFNN/DNN lane for the HUMAN/retro AV-adoption task.

ADD-ONLY, AND A DIFFERENT ESTIMATOR FROM THE REGISTERED LANE.
=============================================================
The registered FFNN/DNN are sklearn MLPClassifier fits. sklearn has no GPU
path, so running on the 3080 means reimplementing the network. This file is
therefore a NEW lane, not a faster version of the registered one. Its numbers
are not a drop-in replacement for predictions/tabular/HUMAN_retro_{ffnn,dnn}_*.

What is held identical to the registered lane, so the comparison is about the
model and not about the data:
  - the same model_inputs parquets (HUMAN_retro_development / _test)
  - the same feature split and preprocessing, imported from pipeline/tabular.py
    (feature_columns, make_preprocessor), fitted on development only
  - the same outer tuning protocol: GroupKFold(3) on respondent_id, selection
    by cross-entropy, searched once at seed 42
  - the same refit seeds 42/123/999
  - the same scorer, pipeline/metrics.py exp1, on the same held-out test

What is deliberately different, and why:
  1. respondent-grouped early stopping. sklearn's early_stopping=True calls
     train_test_split(X, y, stratify=y) with no groups, so the same respondent
     lands in both the internal training and the internal stopping set. With
     six tasks per respondent that contaminates the stopping decision. Here the
     inner validation split is by respondent_id.
  2. class imbalance via pos_weight in BCEWithLogitsLoss instead of duplicating
     minority rows. Duplication changes the effective sample and inflates the
     predicted share; a loss weight does not.
  3. decoupled weight decay (AdamW) instead of sklearn's alpha-in-the-loss L2.
  4. dropout and optional batch normalisation, which MLPClassifier has neither of.
  5. cosine learning-rate schedule.

Diagnosis this is answering: on HUMAN, all 16 registered candidates and all six
registered fits scored worse than a constant base-rate predictor
(test base-rate log loss 0.4651; registered fits 0.6285 to 1.7850). The failure
looked like capacity and regularisation, so this lane gives the search real
regularisation knobs and real capacity control.

Writes only under Real_exp/NN_GPU_20260905/.
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from tabular import feature_columns, make_preprocessor
import metrics as M

OUT = Path(__file__).resolve().parent
MI = ROOT / "data" / "model_inputs"
DEV = "cuda"

MAX_EPOCHS = 300
PATIENCE = 30
BATCH = 512
INNER_VAL_FRAC = 0.15      # respondent-grouped, carved out of the training part


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout=0.0, batchnorm=False):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, d_in, par, seed):
    """Train with respondent-grouped early stopping. Returns the best state."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLP(d_in, par["hidden"], par["dropout"], par["batchnorm"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=par["lr"],
                            weight_decay=par["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    pw = torch.tensor([par["pos_weight"]], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    evalf = nn.BCEWithLogitsLoss()          # unweighted, for model selection

    n = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if idx.numel() < 2:             # batchnorm needs >1 row
                continue
            opt.zero_grad(set_to_none=True)
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(evalf(model(Xva), yva))
        if v < best - 1e-5:
            best, bad = v, 0
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
    return torch.sigmoid(model(X)).float().cpu().numpy()


# --------------------------------------------------------------------------
# search space
# --------------------------------------------------------------------------
SIZES = {
    "ffnn": [(32,), (64,), (64, 32), (128, 64)],                       # shallow
    "dnn":  [(128, 64, 32), (256, 128, 64), (256, 128, 64, 32),
             (512, 256, 128, 64)],                                     # deep
}
DROPOUT = [0.0, 0.2, 0.4]
WD = [1e-5, 1e-3, 1e-2]
BN = [False, True]
POSW = [1.0, 4.31]              # 1.0 = none; 4.31 = (1-p)/p at the 18.8% AV rate
LR = 1e-3


def grid(name):
    return [{"hidden": h, "dropout": d, "weight_decay": w, "batchnorm": b,
             "pos_weight": pw, "lr": LR}
            for h, d, w, b, pw in itertools.product(SIZES[name], DROPOUT, WD, BN, POSW)]


# --------------------------------------------------------------------------
def to_gpu(a):
    return torch.tensor(np.asarray(a, dtype=np.float32), device=DEV)


def densify(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def cv_score(par, Xd, yd, groups, d_in, seed=42, n_splits=3):
    """Mean held-out cross-entropy over respondent-grouped folds."""
    outer = GroupKFold(n_splits=n_splits)
    losses = []
    for tr_i, te_i in outer.split(Xd, yd, groups):
        gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed)
        a, b = next(gi.split(Xd[tr_i], yd[tr_i], groups[tr_i]))
        Xtr, ytr = to_gpu(Xd[tr_i][a]), to_gpu(yd[tr_i][a])
        Xva, yva = to_gpu(Xd[tr_i][b]), to_gpu(yd[tr_i][b])
        Xte, yte = to_gpu(Xd[te_i]), yd[te_i]
        model, _, _ = train_one(Xtr, ytr, Xva, yva, d_in, par, seed)
        p = np.clip(predict(model, Xte), 1e-7, 1 - 1e-7)
        losses.append(float(-(yte * np.log(p) + (1 - yte) * np.log(1 - p)).mean()))
        del Xtr, ytr, Xva, yva, Xte, model
    return float(np.mean(losses))


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
    base_ll = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    print("dev %s -> %s   test %s -> %s   base-rate test log_loss %.4f"
          % (tr.shape, Xd.shape, te.shape, Xt.shape, base_ll), flush=True)
    print("registered sklearn reference (same test): ffnn 0.6354/0.6635/1.0020  "
          "dnn 0.6285/0.9469/1.7850", flush=True)

    if smoke:
        for name in ("ffnn", "dnn"):
            g = grid(name)
            for par in (g[0], g[len(g) // 2], g[-1]):
                t0 = time.time()
                s = cv_score(par, Xd, yd, groups, d_in)
                print("  smoke %-4s %-52s cv %.4f  %.1fs"
                      % (name, str({k: par[k] for k in
                                    ("hidden", "dropout", "weight_decay",
                                     "batchnorm", "pos_weight")})[:50], s,
                         time.time() - t0), flush=True)
        return

    trace_path = OUT / "tuning_trace.csv"
    if trace_path.exists():
        trace_path.unlink()
    hdr, best = False, {}
    for name in ("ffnn", "dnn"):
        cands = grid(name)
        print("\n=== %s: %d candidates x 3 folds ===" % (name, len(cands)), flush=True)
        t0, done = time.time(), []
        for i, par in enumerate(cands, 1):
            t1 = time.time()
            try:
                s, err = cv_score(par, Xd, yd, groups, d_in), ""
            except Exception as e:
                s, err = float("nan"), str(e)[:120]
            row = {"model": name, **{k: str(v) for k, v in par.items()},
                   "cv_log_loss": s, "seconds": round(time.time() - t1, 2), "error": err}
            done.append(row)
            if i % 12 == 0 or i == len(cands):
                pd.DataFrame(done[-(i % 12 or 12):]).to_csv(
                    trace_path, mode="a", index=False, header=not hdr)
                hdr = True
                fin = [r for r in done if r["cv_log_loss"] == r["cv_log_loss"]]
                b = min(fin, key=lambda r: r["cv_log_loss"])
                print("  %3d/%d  %5.0fs  best %.4f  %s"
                      % (i, len(cands), time.time() - t0, b["cv_log_loss"],
                         (b["hidden"], b["dropout"], b["weight_decay"],
                          b["batchnorm"], b["pos_weight"])), flush=True)
        best[name] = min([r for r in done if r["cv_log_loss"] == r["cv_log_loss"]],
                         key=lambda r: r["cv_log_loss"])

    # --- refit on full development at the registered seeds ------------------
    print("\n=== refit and score held-out test ===", flush=True)
    rows = []
    for name, b in best.items():
        par = {"hidden": eval(b["hidden"]), "dropout": float(b["dropout"]),
               "weight_decay": float(b["weight_decay"]),
               "batchnorm": b["batchnorm"] == "True",
               "pos_weight": float(b["pos_weight"]), "lr": LR}
        for seed in (42, 123, 999):
            t0 = time.time()
            gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                   random_state=seed)
            a, v = next(gi.split(Xd, yd, groups))
            model, vl, eps = train_one(to_gpu(Xd[a]), to_gpu(yd[a]),
                                       to_gpu(Xd[v]), to_gpu(yd[v]), d_in, par, seed)
            p = predict(model, to_gpu(Xt)).astype(np.float64)
            m = M.exp1(yt.astype(int), p, None, te.respondent_id.to_numpy())
            m.update({"model": name, "seed": seed, "epochs": eps,
                      "inner_val_log_loss": round(vl, 4),
                      "fit_seconds": round(time.time() - t0, 2),
                      **{k: str(v2) for k, v2 in par.items()}})
            rows.append(m)
            torch.save(model.state_dict(), OUT / ("model_%s_seed%d.pt" % (name, seed)))
            pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": p,
                          "world": "HUMAN", "info": "retro", "model": name,
                          "seed": seed}).to_parquet(
                OUT / "predictions" / ("HUMAN_retro_%s_seed%d.parquet" % (name, seed)),
                index=False)
            print("  %-5s seed%-4d log_loss %.4f (base rate %.4f)  auc %.4f  "
                  "av_f1 %.4f  %d epochs  %.1fs"
                  % (name, seed, m["log_loss"], base_ll, m["roc_auc"], m["av_f1"],
                     eps, time.time() - t0), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    json.dump({"base_rate_test_log_loss": base_ll, "best": best,
               "registered_sklearn_reference": {
                   "ffnn": [0.6354, 0.6635, 1.0020],
                   "dnn": [0.6285, 0.9469, 1.7850]},
               "n_candidates": {k: len(grid(k)) for k in ("ffnn", "dnn")},
               "d_in": d_in, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               "batch": BATCH, "inner_val_frac": INNER_VAL_FRAC},
              open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
