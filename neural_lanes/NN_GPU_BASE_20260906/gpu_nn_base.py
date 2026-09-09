# -*- coding: utf-8 -*-
"""GPU FFNN/DNN pushed for held-out BASE prediction. Lane 5.

ADD-ONLY. Writes only under this folder. No counterfactual scenarios here -
this lane is only about predicting the observed BASE choice on the held-out
test respondents.

WHY THIS LANE EXISTS
--------------------
Lane 4 (NN_GPU_PRAUC_20260906) reached test PR-AUC .4289 for both models, but
its DNN winner was (512,256,128,64) - the LARGEST net in that grid. Winning at
the boundary of a search space means the space was too small. Two more axes had
also never been searched in any lane:

  - learning rate. Fixed at 1e-3 in lanes 3 and 4, and sklearn's default 1e-3
    in lanes 1 and 2. Never varied once.
  - categorical encoding. Every lane so far flattens 63 categorical columns
    into 286 one-hot columns. Entity embeddings are the standard alternative
    for tabular networks and have never been tried here.

SELECTION
---------
Unchanged from lane 4 and comparable to it: respondent-grouped GroupKFold(3) on
development, criterion PR-AUC (average precision), searched once at seed 42.
No threshold anywhere - no hard decision rule, per the stated requirement.
Cross-entropy is computed as a calibration diagnostic only and selects nothing.

The held-out test is scored ONCE, after selection, and never enters the search.

EVIDENCE-BASED PRUNES (not guesses)
-----------------------------------
  - batchnorm fixed False: lost on mean and min for both models across 288
    candidates in the NN_GPU_20260905 trace.
  - loss narrowed to focal(1.0,0.25) and bce(1.0): in the 256-candidate lane-4
    trace these were the top two for ffnn (.4350, .4284) and dnn (.4278, .4200).
    Every positive-upweighting variant - bce(4.31), bce(8.0), focal(*,0.75) -
    scored lower on PR-AUC AND worse on cross-entropy, and softf1 was last on
    both. Up-weighting the AV class does not improve AV-case detection here.

WHAT IS NEW
-----------
  1. entity embeddings as a searched alternative to one-hot. Rare levels are
     folded into an explicit "other" index using the same min_frequency=10 rule
     the registered OneHotEncoder uses, so the two encodings see the same
     information. Unseen test levels map to that same index.
  2. learning rate searched over 3e-4 / 1e-3 / 3e-3.
  3. architecture range extended upward past the lane-4 boundary winner.
  4. seed ensembling reported SEPARATELY from the single-seed numbers, so its
     effect is visible rather than bundled into the headline.

The feature SET is unchanged from every other lane - the same columns from
pipeline/tabular.py feature_columns, on the same model_inputs parquets. Only
the encoding of those columns differs, and only in embed mode.
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

MAX_EPOCHS = 400
PATIENCE = 40
BATCH = 512
INNER_VAL_FRAC = 0.15
MIN_FREQ = 10          # matches the registered OneHotEncoder(min_frequency=10)


# --------------------------------------------------------------------------
# losses (narrowed to the two that won lane 4)
# --------------------------------------------------------------------------
def make_loss(spec):
    if spec[0] == "bce":
        pw = torch.tensor([float(spec[1])], device=DEV)
        f = nn.BCEWithLogitsLoss(pos_weight=pw)
        return lambda z, y: f(z, y)
    if spec[0] == "focal":
        gamma, alpha = float(spec[1]), float(spec[2])

        def focal(z, y):
            logp, log1p = F.logsigmoid(z), F.logsigmoid(-z)
            p = torch.sigmoid(z)
            pt = torch.where(y > 0.5, p, 1 - p)
            logpt = torch.where(y > 0.5, logp, log1p)
            at = torch.where(y > 0.5, torch.full_like(y, alpha),
                             torch.full_like(y, 1 - alpha))
            return -(at * (1 - pt).pow(gamma) * logpt).mean()
        return focal
    raise ValueError(spec)


LOSSES = [("focal", 1.0, 0.25), ("bce", 1.0)]


# --------------------------------------------------------------------------
# encodings
# --------------------------------------------------------------------------
def build_onehot(tr, te, num, cat):
    prep = make_preprocessor(num, cat).fit(tr[num + cat])
    dense = lambda X: (X.toarray() if hasattr(X, "toarray") else np.asarray(X))
    return (dense(prep.transform(tr[num + cat])).astype(np.float32),
            dense(prep.transform(te[num + cat])).astype(np.float32), None)


def build_embed(tr, te, num, cat):
    """Numeric standardised; each categorical column -> integer codes.

    Index 0 is reserved for 'rare or unseen', matching min_frequency=10 in the
    registered one-hot path so both encodings carry the same information."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median").fit(tr[num])
    sc = StandardScaler().fit(imp.transform(tr[num]))
    Ntr = sc.transform(imp.transform(tr[num])).astype(np.float32)
    Nte = sc.transform(imp.transform(te[num])).astype(np.float32)

    codes_tr, codes_te, cards = [], [], []
    for c in cat:
        s = tr[c].astype("object")
        mode = s.mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "__na__"
        s = s.fillna(fill)
        vc = s.value_counts()
        keep = list(vc[vc >= MIN_FREQ].index)
        m = {v: i + 1 for i, v in enumerate(keep)}     # 0 = rare/unseen
        codes_tr.append(s.map(m).fillna(0).to_numpy(dtype=np.int64))
        t = te[c].astype("object").fillna(fill)
        codes_te.append(t.map(m).fillna(0).to_numpy(dtype=np.int64))
        cards.append(len(keep) + 1)
    Ctr = np.stack(codes_tr, 1) if cat else np.zeros((len(tr), 0), np.int64)
    Cte = np.stack(codes_te, 1) if cat else np.zeros((len(te), 0), np.int64)
    return (Ntr, Nte), (Ctr, Cte), cards


# --------------------------------------------------------------------------
class OneHotMLP(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc=None):
        return self.net(xn).squeeze(-1)


class EmbedMLP(nn.Module):
    def __init__(self, n_num, cards, hidden, dropout):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(k, min(16, (k + 1) // 2))
                                  for k in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.emb)
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.emb)]
        return self.net(torch.cat([xn] + e, 1)).squeeze(-1)


def build_model(par, shapes, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if par["enc"] == "onehot":
        return OneHotMLP(shapes["d_onehot"], par["hidden"], par["dropout"]).to(DEV)
    return EmbedMLP(shapes["n_num"], shapes["cards"], par["hidden"],
                    par["dropout"]).to(DEV)


def train_one(tr_pack, va_pack, yva_np, par, shapes, seed):
    """Early stopping MAXIMISES PR-AUC on a respondent-grouped inner split."""
    Xn, Xc, yt = tr_pack
    Vn, Vc, _ = va_pack
    model = build_model(par, shapes, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=par["lr"],
                            weight_decay=par["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossf = make_loss(par["loss"])
    n = Xn.shape[0]
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
            lossf(model(Xn[idx], Xc[idx] if Xc is not None else None),
                  yt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Vn, Vc)).float().cpu().numpy()
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
def predict(model, Xn, Xc):
    model.eval()
    return torch.sigmoid(model(Xn, Xc)).float().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
SIZES = {   # extended past the lane-4 boundary winner (512,256,128,64)
    "ffnn": [(64,), (128,), (128, 64), (256, 128)],
    "dnn":  [(256, 128, 64), (512, 256, 128), (512, 256, 128, 64),
             (1024, 512, 256, 128)],
}
DROPOUT = [0.1, 0.3]
WD = [1e-4, 1e-3]
LRS = [3e-4, 1e-3, 3e-3]        # never searched in any earlier lane
ENC = ["onehot", "embed"]       # never searched in any earlier lane


def grid(name):
    return [{"hidden": h, "dropout": d, "weight_decay": w, "lr": lr,
             "enc": e, "loss": L}
            for h, d, w, lr, e, L in itertools.product(
                SIZES[name], DROPOUT, WD, LRS, ENC, LOSSES)]


def lossname(L):
    return "%s(%s)" % (L[0], ",".join(str(x) for x in L[1:]))


def pack(enc, idx, data, dev_side=True):
    """Slice the preprocessed arrays for a row index set and push to GPU."""
    if enc == "onehot":
        X = data["oh_tr"] if dev_side else data["oh_te"]
        return (torch.tensor(X[idx], device=DEV), None,
                torch.tensor(data["y_tr"][idx], device=DEV) if dev_side else None)
    N = data["em_num_tr"] if dev_side else data["em_num_te"]
    C = data["em_cat_tr"] if dev_side else data["em_cat_te"]
    return (torch.tensor(N[idx], device=DEV), torch.tensor(C[idx], device=DEV),
            torch.tensor(data["y_tr"][idx], device=DEV) if dev_side else None)


def cv_score(par, data, groups, shapes, seed=42, n_splits=3):
    aps, lls = [], []
    yd = data["y_tr"]
    n = len(yd)
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(np.arange(n), yd, groups):
        gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                               random_state=seed)
        a, b = next(gi.split(tr_i, yd[tr_i], groups[tr_i]))
        model, _, _ = train_one(pack(par["enc"], tr_i[a], data),
                                pack(par["enc"], tr_i[b], data),
                                yd[tr_i[b]], par, shapes, seed)
        Vn, Vc, _ = pack(par["enc"], te_i, data)
        p = np.clip(predict(model, Vn, Vc), 1e-7, 1 - 1e-7)
        yt = yd[te_i]
        aps.append(float(average_precision_score(yt, p)))
        lls.append(float(-(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()))
        del model
    return float(np.mean(aps)), float(np.mean(lls))


def main(smoke=False):
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    print("device:", torch.cuda.get_device_name(0), flush=True)

    tr = pd.read_parquet(MI / "HUMAN_retro_development.parquet")
    te = pd.read_parquet(MI / "HUMAN_retro_test.parquet")
    num, cat = feature_columns(tr)
    oh_tr, oh_te, _ = build_onehot(tr, te, num, cat)
    (nt, nte), (ct, cte), cards = build_embed(tr, te, num, cat)
    data = {"oh_tr": oh_tr, "oh_te": oh_te, "em_num_tr": nt, "em_num_te": nte,
            "em_cat_tr": ct, "em_cat_te": cte,
            "y_tr": tr.y.to_numpy().astype(np.float32)}
    yt = te.y.to_numpy().astype(np.float32)
    groups = tr.respondent_id.to_numpy()
    shapes = {"d_onehot": oh_tr.shape[1], "n_num": nt.shape[1], "cards": cards}

    p0 = float(yt.mean())
    print("dev %d rows / %d respondents | onehot %d cols | embed %d numeric + "
          "%d categorical (total emb dim %d)"
          % (len(tr), tr.respondent_id.nunique(), oh_tr.shape[1], nt.shape[1],
             len(cards), sum(min(16, (k + 1) // 2) for k in cards)), flush=True)
    print("test %d cases / %d respondents | AV rate %.4f = PR-AUC of a random "
          "ranker | base-rate xent %.4f"
          % (len(te), te.respondent_id.nunique(), p0,
             -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))), flush=True)
    print("lane 4 test PR-AUC to beat: ffnn .4289  dnn .4289", flush=True)

    if smoke:
        for name in ("ffnn", "dnn"):
            g = grid(name)
            for par in (g[0], g[len(g) // 4], g[len(g) // 2], g[-1]):
                t0 = time.time()
                ap, ll = cv_score(par, data, groups, shapes)
                print("  smoke %-4s %-20s %-7s lr%.0e dr%.1f %-14s PR-AUC %.4f "
                      "xent %.4f  %.1fs"
                      % (name, str(par["hidden"]), par["enc"], par["lr"],
                         par["dropout"], lossname(par["loss"]), ap, ll,
                         time.time() - t0), flush=True)
        return

    trace_path = OUT / "tuning_trace.csv"
    if trace_path.exists():
        trace_path.unlink()
    hdr, best = False, {}
    for name in ("ffnn", "dnn"):
        cands = grid(name)
        print("\n=== %s: %d candidates x 3 folds, selecting on CV PR-AUC ==="
              % (name, len(cands)), flush=True)
        t0, done, buf = time.time(), [], []
        for i, par in enumerate(cands, 1):
            t1 = time.time()
            try:
                ap, ll = cv_score(par, data, groups, shapes)
                err = ""
            except Exception as e:
                ap, ll, err = float("nan"), float("nan"), str(e)[:140]
            row = {"model": name, "hidden": str(par["hidden"]), "enc": par["enc"],
                   "loss": lossname(par["loss"]), "lr": par["lr"],
                   "dropout": par["dropout"], "weight_decay": par["weight_decay"],
                   "cv_pr_auc": ap, "cv_xent_diag": ll,
                   "seconds": round(time.time() - t1, 2), "error": err}
            done.append(row); buf.append(row)
            if i % 24 == 0 or i == len(cands):
                pd.DataFrame(buf).to_csv(trace_path, mode="a", index=False,
                                         header=not hdr)
                hdr, buf = True, []
                fin = [r for r in done if r["cv_pr_auc"] == r["cv_pr_auc"]]
                b = max(fin, key=lambda r: r["cv_pr_auc"])
                print("  %3d/%d  %5.0fs  best CV PR-AUC %.4f  %s"
                      % (i, len(cands), time.time() - t0, b["cv_pr_auc"],
                         (b["hidden"], b["enc"], b["loss"], b["lr"], b["dropout"],
                          b["weight_decay"])), flush=True)
        best[name] = max([r for r in done if r["cv_pr_auc"] == r["cv_pr_auc"]],
                         key=lambda r: r["cv_pr_auc"])

    # --- refit at 5 seeds, score test once, then report the seed ensemble ----
    print("\n=== refit and score held-out test (threshold-free) ===", flush=True)
    SEEDS = (42, 123, 999, 2024, 7)
    rows, ens = [], {}
    yd = data["y_tr"]
    for name, b in best.items():
        L = next(l for l in LOSSES if lossname(l) == b["loss"])
        par = {"hidden": eval(b["hidden"]), "dropout": float(b["dropout"]),
               "weight_decay": float(b["weight_decay"]), "lr": float(b["lr"]),
               "enc": b["enc"], "loss": L}
        preds = []
        for seed in SEEDS:
            t0 = time.time()
            gi = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                   random_state=seed)
            a, v = next(gi.split(np.arange(len(yd)), yd, groups))
            model, vap, eps = train_one(pack(par["enc"], a, data),
                                        pack(par["enc"], v, data),
                                        yd[v], par, shapes, seed)
            Tn, Tc, _ = pack(par["enc"], np.arange(len(te)), data, dev_side=False)
            p = predict(model, Tn, Tc)
            preds.append(p)
            pc = np.clip(p, 1e-7, 1 - 1e-7)
            rows.append({"model": name, "seed": seed, "kind": "single",
                         "pr_auc": round(float(average_precision_score(yt, p)), 4),
                         "roc_auc": round(float(roc_auc_score(yt, p)), 4),
                         "xent_diag": round(float(-(yt * np.log(pc) +
                                                    (1 - yt) * np.log(1 - pc)).mean()), 4),
                         "p_mean": round(float(p.mean()), 4),
                         "p_max": round(float(p.max()), 3),
                         "inner_val_pr_auc": round(vap, 4), "epochs": eps,
                         **{k: str(par[k]) for k in
                            ("hidden", "enc", "dropout", "weight_decay", "lr")},
                         "loss": b["loss"],
                         "fit_seconds": round(time.time() - t0, 2)})
            torch.save(model.state_dict(), OUT / ("model_%s_seed%d.pt" % (name, seed)))
            pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": p,
                          "world": "HUMAN", "info": "retro", "model": name,
                          "seed": seed}).to_parquet(
                OUT / "predictions" / ("HUMAN_retro_%s_seed%d.parquet" % (name, seed)),
                index=False)
            r = rows[-1]
            print("  %-5s seed%-5d PR-AUC %.4f  ROC-AUC %.4f  xent %.4f  %d ep"
                  % (name, seed, r["pr_auc"], r["roc_auc"], r["xent_diag"], eps),
                  flush=True)
        pe = np.mean(preds, 0)
        ens[name] = pe
        pc = np.clip(pe, 1e-7, 1 - 1e-7)
        rows.append({"model": name, "seed": -1, "kind": "ensemble_%d" % len(SEEDS),
                     "pr_auc": round(float(average_precision_score(yt, pe)), 4),
                     "roc_auc": round(float(roc_auc_score(yt, pe)), 4),
                     "xent_diag": round(float(-(yt * np.log(pc) +
                                                (1 - yt) * np.log(1 - pc)).mean()), 4),
                     "p_mean": round(float(pe.mean()), 4),
                     "p_max": round(float(pe.max()), 3),
                     **{k: str(par[k]) for k in
                        ("hidden", "enc", "dropout", "weight_decay", "lr")},
                     "loss": b["loss"]})
        pd.DataFrame({"case_id": te.case_id, "scenario_id": "BASE", "p": pe,
                      "world": "HUMAN", "info": "retro", "model": name,
                      "seed": -1}).to_parquet(
            OUT / "predictions" / ("HUMAN_retro_%s_ensemble.parquet" % name),
            index=False)
        r = rows[-1]
        print("  %-5s ENSEMBLE   PR-AUC %.4f  ROC-AUC %.4f  xent %.4f"
              % (name, r["pr_auc"], r["roc_auc"], r["xent_diag"]), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    json.dump({"selection_criterion": "CV PR-AUC on development, threshold-free",
               "test_av_rate": p0, "best": best, "seeds": list(SEEDS),
               "new_axes": ["categorical encoding onehot vs entity embedding",
                            "learning rate 3e-4/1e-3/3e-3",
                            "architecture extended past the lane-4 boundary"],
               "lane4_test_pr_auc": {"ffnn": 0.4289, "dnn": 0.4289},
               "n_candidates": {k: len(grid(k)) for k in ("ffnn", "dnn")}},
              open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
