# -*- coding: utf-8 -*-
"""Section 4 results package at CASE grain with the six-adjacent-pair Direct operator (D321 lineage).

Reconstructs the generator of ``pdf/SECTION4_RESULTS_CASE`` (the paper's Section 4 source) so the same
package can be rebuilt with a different Qwen score bank (CV5 2B SFT swap, D325).

Operator definitions (verified against SECTION4_RESULTS_CASE to machine precision before this module
was written; see scripts/six_pair_operator.py, scripts/tabular_six_pair_gate.py,
scripts/human_six_pair_reference_plain.py):

* six adjacent Fare pairs (M3,M2)(M2,M1)(M1,BASE)(BASE,P1)(P1,P2)(P2,P3); same for IVT rungs;
* interval elasticity = midpoint arc elasticity; case elasticity = mean of the six interval elasticities;
* monotone_pairs = number of intervals with elasticity < 0;
* raw_DF / raw_DT = mean of the six interval slopes; case VOT = 60 * raw_DT / raw_DF when |raw_DF| >= 1e-4;
* aggregate ("population") VOT = 60 * mean(raw_DT) / mean(raw_DF) over cases with finite slopes;
* tabular systems: operator per seed (42/123/999), then averaged across seeds (interval elasticity,
  monotone count, slopes); Qwen probability = expit(margin) (pair-normalised raw, no Platt);
* Human rows of Table 2/3 and Figures 6/10/11 = plain 20-column R2 MNL (development fit, per-rung design
  substitution); Human rows of the subgroup figures and Tables 4/S05-S07 = joint-interaction MNL.

Grain: elasticity distribution tables use the interval grain (20,808 = 6 x 3,468 observations) for the
paper's Table 2 and the subgroup elasticity figure; VOT uses the case grain (3,468).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm

ROOT = Path(r"C:\SP_LLM\TRC\Real_exp")
TOOLS = ROOT / "tools"
SCRIPTS = Path(r"C:\SP_LLM\TRC\scripts")
for p in (str(TOOLS), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
import section4_results_final_support as S  # noqa: E402  (reused loaders/figures; S.OUT is re-pointed per build)
from six_pair_operator import LADDER, XCOL, six_pairs, RAW_FARE_THRESHOLD  # noqa: E402

FINAL = S.FINAL
MODEL_ORDER = S.MODEL_ORDER
QWEN_ORDER = S.QWEN_ORDER
QWEN_KEYS = S.QWEN_KEYS
GROUP_SPECS = S.GROUP_SPECS
HUMAN_E = "Human reference"
HUMAN_V = "Human same-CF reference"
FAMILY = {"Panel Logit": "DCM", "SVM": "ML", "Random Forest": "ML", "XGBoost": "ML", "FFNN": "Neural", "DNN": "Neural",
          **{m: "LLM" for m in QWEN_ORDER}, HUMAN_E: "Human", HUMAN_V: "Human"}
SEEDS = (42, 123, 999)
ML = ROOT / "data" / "07_predictions_ml"
TABULAR = {
    "Panel Logit": ML / "balanced_tabular_v2" / "HUMAN_retro_panel_logit_full_seed{seed}.parquet",
    "SVM": ML / "weighted_registered_d250" / "HUMAN_retro_svm_seed{seed}.parquet",
    "Random Forest": ML / "weighted_registered_d250" / "HUMAN_retro_random_forest_seed{seed}.parquet",
    "XGBoost": ML / "weighted_registered_d250" / "HUMAN_retro_xgboost_seed{seed}.parquet",
    "FFNN": ML / "weighted_registered_d250" / "HUMAN_retro_ffnn_seed{seed}.parquet",
    "DNN": ML / "weighted_registered_d250" / "HUMAN_retro_dnn_seed{seed}.parquet",
}
EXP4_BASE_SCRIPT = FINAL / "04_EXP4" / "Figure_07_1_overall_elasticity_AB" / "plot_exp4_overall_elasticity_ab.py"
S03_COLUMNS = ["model", "family", "case_id", "respondent_id", "distance_band", "raw_DF", "raw_DT", "n_tasks",
               "raw_finite", "raw_near_zero_fare", "raw_computable", "raw_fare_negative", "raw_ivt_negative",
               "raw_joint_conventional", "raw_vot", "raw_negative_vot", "raw_positive_vot",
               "logit_DF", "logit_DT", "logit_vot", "logit_finite", "logit_near_zero_fare", "logit_computable",
               "logit_fare_negative", "logit_ivt_negative", "logit_joint_conventional", "logit_negative_vot",
               "logit_positive_vot"]
# Paper subgroup figures (Figure 3 / Figure 5 of the manuscript): fixed display windows.
FIG3_XLIM = (-3.6, 0.9)
FIG5_XLIM = (-35.0, 65.0)
SUBGROUP_STYLE = {  # edge colour, fill colour of the "filled" (second-group) box
    "Human": ("#111111", "#9A9A9A"),
    "Conventional": ("#8C8C8C", "#D9D9D9"),
    "ZS": ("#7FB3D3", "#D6E8F3"),
    "SFT": ("#1F5F8B", "#7EA6C8"),
}


def _style_key(model: str) -> str:
    if model == HUMAN_E:
        return "Human"
    if model in QWEN_ORDER:
        return "SFT" if model.endswith("SFT") else "ZS"
    return "Conventional"


# ----------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------
def load_inputs(bank: Path) -> dict[str, object]:
    test = pd.read_parquet(S.PATH_TEST, columns=["case_id", "respondent_id", "y", "SQ2", "SQ3"])
    test["case_id"] = test.case_id.astype(str)
    assert len(test) == 3468 and test.case_id.is_unique and test.respondent_id.nunique() == 651
    master = pd.read_parquet(S.PATH_MASTER, columns=["IDX", "BR", "BR_label", "SQ2", "gender_label", "SQ3"])
    meta = (test[["respondent_id", "SQ2", "SQ3"]].drop_duplicates("respondent_id")
            .merge(master, left_on="respondent_id", right_on="IDX", how="left", validate="one_to_one",
                   suffixes=("_test", "_master")))
    assert len(meta) == 651 and meta.BR_label.notna().all()
    assert (meta.SQ2_test == meta.SQ2_master).all() and (meta.SQ3_test == meta.SQ3_master).all()
    meta["age_group"] = np.where(meta.SQ3_test.ge(50), "Age 50+", "Age <50")
    meta["sex_group"] = meta.gender_label.astype(str)
    meta["framing_group"] = meta.BR_label.astype(str)
    respondent_meta = meta[["respondent_id", "age_group", "sex_group", "framing_group"]].copy()
    expected = {"age_group": {"Age <50": 383, "Age 50+": 268}, "sex_group": {"Male": 309, "Female": 342},
                "framing_group": {"BASIC": 326, "RISK": 325}}
    for column, counts in expected.items():
        assert respondent_meta.groupby(column).size().to_dict() == counts, column

    grid_full = pd.read_parquet(S.PATH_GRID)
    grid_full["case_id"] = grid_full.case_id.astype(str)
    scenarios = list(dict.fromkeys(["BASE"] + [s for v in S.ATTRIBUTE_SCENARIOS.values() for s in v if s != "BASE"]))
    grid = grid_full.loc[grid_full.scenario_id.isin(scenarios)].copy()
    assert grid.loc[grid.scenario_id.eq("BASE")].case_id.nunique() == 3468
    assert grid.loc[grid.scenario_id.isin(LADDER["Fare"] + LADDER["IVT"])].physically_valid.all()

    scores = pd.read_parquet(bank, columns=["case_id", "scenario_id", "model", "margin", "free_argmax"])
    scores["case_id"] = scores.case_id.astype(str)
    scores = scores.loc[scores.model.isin(QWEN_KEYS) & scores.scenario_id.isin(scenarios)].copy()
    scores["system"] = scores.model.map(QWEN_KEYS)
    scores["p"] = expit(scores.margin.to_numpy(float))
    assert len(scores) == len(QWEN_ORDER) * len(scenarios) * 3468
    assert scores.groupby(["system", "scenario_id"]).size().eq(3468).all()
    return {"test": test, "meta": respondent_meta, "grid": grid, "scores": scores,
            "case_order": test.case_id.tolist(), "bank": str(bank)}


def _rung_x(grid: pd.DataFrame, attr: str, order: list[str]) -> np.ndarray:
    sids = LADDER[attr]
    return (grid.loc[grid.scenario_id.isin(sids)]
            .pivot(index="case_id", columns="scenario_id", values=XCOL[attr])[sids].loc[order].to_numpy(float))


# ----------------------------------------------------------------------------------------------------------
# six-pair case frames
# ----------------------------------------------------------------------------------------------------------
def _frames_from_operator(model: str, order: list[str], test: pd.DataFrame, E: np.ndarray, mono: np.ndarray,
                          delta_p: np.ndarray, DF: np.ndarray, DT: np.ndarray, loc_DF: np.ndarray,
                          loc_DT: np.ndarray, n_seed: int) -> dict[str, pd.DataFrame]:
    """Assemble the S02 (elasticity), interval, S03 (VOT) and S13 (local VOT) frames for one system."""
    lookup = test.set_index("case_id").respondent_id
    respondent = lookup.loc[order].to_numpy()
    band = pd.Series(order).str.rsplit("_", n=1).str[-1].to_numpy()
    family = FAMILY[model]
    elasticity = pd.DataFrame({
        "system": model, "family": family, "case_id": order, "respondent_id": respondent, "distance_band": band,
        "elasticity": E.mean(1), "delta_p": delta_p, "monotone_pairs": mono, "n_seed_values": n_seed,
    })
    interval = pd.DataFrame({
        "system": np.repeat(model, len(order) * 6), "case_id": np.repeat(order, 6),
        "respondent_id": np.repeat(respondent, 6), "interval": np.tile(np.arange(1, 7), len(order)),
        "elasticity": E.ravel(),
    })
    vot = pd.DataFrame({"model": model, "family": family, "case_id": order, "respondent_id": respondent,
                        "distance_band": band, "raw_DF": DF, "raw_DT": DT, "n_tasks": 1})
    vot["raw_finite"] = np.isfinite(vot.raw_DF) & np.isfinite(vot.raw_DT)
    vot["raw_near_zero_fare"] = vot.raw_finite & vot.raw_DF.abs().lt(RAW_FARE_THRESHOLD)
    vot["raw_computable"] = vot.raw_finite & ~vot.raw_near_zero_fare
    vot["raw_fare_negative"] = vot.raw_computable & vot.raw_DF.lt(0)
    vot["raw_ivt_negative"] = vot.raw_computable & vot.raw_DT.lt(0)
    vot["raw_joint_conventional"] = vot.raw_computable & vot.raw_DF.lt(0) & vot.raw_DT.lt(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vot["raw_vot"] = np.where(vot.raw_computable, 60 * vot.raw_DT / vot.raw_DF, np.nan)
    vot["raw_negative_vot"] = vot.raw_computable & vot.raw_vot.lt(0)
    vot["raw_positive_vot"] = vot.raw_computable & vot.raw_vot.gt(0)
    for column in S03_COLUMNS[17:]:
        vot[column] = np.nan
    vot = vot[S03_COLUMNS]
    loc_ok = np.isfinite(loc_DF) & np.isfinite(loc_DT) & (np.abs(loc_DF) >= RAW_FARE_THRESHOLD)
    with np.errstate(divide="ignore", invalid="ignore"):
        loc_vot = np.where(loc_ok, 60 * loc_DT / loc_DF, np.nan)
    local = pd.DataFrame({"model": model, "family": family, "case_id": order, "respondent_id": respondent,
                          "loc_DF": loc_DF, "loc_DT": loc_DT, "loc_vot": loc_vot})
    return {"elasticity": elasticity, "interval": interval, "vot": vot, "local": local}


def _operator_from_matrices(Pf, Xf, Pt, Xt):
    """Literal floating-point operator, as in SECTION4_RESULTS_CASE. No zero-snapping: a 1e-15 snap was tried
    and changed the published Random Forest / Qwen ZS sign shares, so exactly-zero slope sums keep whatever
    +-1e-19 residue the summation leaves (affects sign shares of quantised Qwen margins at the 0.1 pp level)."""
    Ef, Df, mono = six_pairs(Pf, Xf)
    Et, Dt, _ = six_pairs(Pt, Xt)
    return dict(E=Ef, mono=mono.sum(1), delta_p=Pf[:, 4] - Pf[:, 3], DF=Df.mean(1), DT=Dt.mean(1),
                loc_DF=(Pf[:, 4] - Pf[:, 2]) / (Xf[:, 4] - Xf[:, 2]), loc_DT=(Pt[:, 4] - Pt[:, 2]) / (Xt[:, 4] - Xt[:, 2]))


def qwen_case_frames(scores: pd.DataFrame, grid: pd.DataFrame, test: pd.DataFrame, order: list[str]) -> list[dict]:
    Xf, Xt = _rung_x(grid, "Fare", order), _rung_x(grid, "IVT", order)
    out = []
    for system in QWEN_ORDER:
        part = scores.loc[scores.system.eq(system)]
        Pf = part.loc[part.scenario_id.isin(LADDER["Fare"])].pivot(index="case_id", columns="scenario_id", values="p")[LADDER["Fare"]].loc[order].to_numpy(float)
        Pt = part.loc[part.scenario_id.isin(LADDER["IVT"])].pivot(index="case_id", columns="scenario_id", values="p")[LADDER["IVT"]].loc[order].to_numpy(float)
        op = _operator_from_matrices(Pf, Xf, Pt, Xt)
        out.append(_frames_from_operator(system, order, test, n_seed=1, **op))
    return out


def tabular_case_frames(grid: pd.DataFrame, test: pd.DataFrame, order: list[str]) -> list[dict]:
    """Operator per seed, then seed-averaged (matches SECTION4_RESULTS_CASE; n_seed_values = 3)."""
    Xf, Xt = _rung_x(grid, "Fare", order), _rung_x(grid, "IVT", order)
    out = []
    for model, template in TABULAR.items():
        acc = {k: [] for k in ("E", "mono", "delta_p", "DF", "DT", "loc_DF", "loc_DT")}
        for seed in SEEDS:
            d = pd.read_parquet(str(template).format(seed=seed), columns=["case_id", "scenario_id", "p"])
            d["case_id"] = d.case_id.astype(str)
            Pf = d.loc[d.scenario_id.isin(LADDER["Fare"])].pivot(index="case_id", columns="scenario_id", values="p")[LADDER["Fare"]].loc[order].to_numpy(float)
            Pt = d.loc[d.scenario_id.isin(LADDER["IVT"])].pivot(index="case_id", columns="scenario_id", values="p")[LADDER["IVT"]].loc[order].to_numpy(float)
            op = _operator_from_matrices(Pf, Xf, Pt, Xt)
            for k in acc:
                acc[k].append(op[k])
        averaged = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in acc.items()}
        out.append(_frames_from_operator(model, order, test, n_seed=len(SEEDS), **averaged))
    return out


def human_plain_case_frames(grid: pd.DataFrame, test: pd.DataFrame, order: list[str]) -> tuple[dict, dict]:
    """Plain 20-column R2 MNL (development fit), scored on the seven Fare and seven IVT rungs of the test grid."""
    spec = importlib.util.spec_from_file_location("exp4_elasticity_base", EXP4_BASE_SCRIPT)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(np.asarray(base.TRAIN_Y, int), base.TRAIN_X).fit(
            disp=0, maxiter=400, cov_type="cluster", cov_kwds={"groups": base.TRAIN_CASES.respondent_id.to_numpy()})
    assert bool(fit.mle_retvals.get("converged", False))
    base_order = [str(c) for c in base.TEST_ORDER]
    g_all = grid.set_index(["scenario_id", "case_id"])

    def rung_matrix(attr: str) -> np.ndarray:
        cols = []
        for sid in LADDER[attr]:
            g = g_all.loc[sid].loc[base_order]
            design = base.TEST_X.copy().reset_index(drop=True)
            design["fare_AV"] = g.fare_cf.to_numpy(float)
            design["ivt_AV"] = g.ride_cf.to_numpy(float)
            design["wait_AV"] = g.wait_cf.to_numpy(float)
            cols.append(np.asarray(fit.predict(design), float))
        return np.column_stack(cols)

    Pf = pd.DataFrame(rung_matrix("Fare"), index=base_order).loc[order].to_numpy(float)
    Pt = pd.DataFrame(rung_matrix("IVT"), index=base_order).loc[order].to_numpy(float)
    Xf, Xt = _rung_x(grid, "Fare", order), _rung_x(grid, "IVT", order)
    op = _operator_from_matrices(Pf, Xf, Pt, Xt)
    frames = _frames_from_operator(HUMAN_E, order, test, n_seed=1, **op)
    frames["vot"]["model"] = HUMAN_V
    frames["local"]["model"] = HUMAN_V
    info = {"converged": bool(fit.mle_retvals.get("converged")), "beta_fare_AV": float(fit.params["fare_AV"]),
            "beta_ivt_AV": float(fit.params["ivt_AV"]), "structural_vot": float(60 * fit.params["ivt_AV"] / fit.params["fare_AV"]),
            "n_train_cases": int(fit.nobs)}
    return frames, info


def human_joint_case_frames(grid: pd.DataFrame, test: pd.DataFrame, order: list[str]) -> tuple[dict, object, pd.DataFrame]:
    """Joint age/sex/framing interaction MNL (same specification as the 0826 support module), seven rungs."""
    import human_six_pair_reference as J  # noqa: WPS433  (scripts/; fits the joint model, returns rung probabilities)

    ind, rungs, fit = J.human_rung_probabilities()
    ids = ind.case_id.astype(str).tolist()
    Pf = pd.DataFrame(rungs["Fare"][0], index=ids).loc[order].to_numpy(float)
    Xf = pd.DataFrame(rungs["Fare"][1], index=ids).loc[order].to_numpy(float)
    Pt = pd.DataFrame(rungs["IVT"][0], index=ids).loc[order].to_numpy(float)
    Xt = pd.DataFrame(rungs["IVT"][1], index=ids).loc[order].to_numpy(float)
    op = _operator_from_matrices(Pf, Xf, Pt, Xt)
    frames = _frames_from_operator(HUMAN_E, order, test, n_seed=1, **op)
    frames["vot"]["model"] = HUMAN_V
    frames["local"]["model"] = HUMAN_V
    groups = pd.DataFrame({"respondent_id": ind.respondent_id.to_numpy(),
                           "age_group": np.where(ind.age50.eq(1), "Age 50+", "Age <50"),
                           "sex_group": np.where(ind.female.eq(1), "Female", "Male"),
                           "framing_group": np.where(ind.risk.eq(1), "RISK", "BASIC")}).drop_duplicates("respondent_id")
    return frames, fit, groups


# ----------------------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------------------
def table_02_case(elasticity: pd.DataFrame) -> pd.DataFrame:
    human_mean = float(elasticity.loc[elasticity.system.eq(HUMAN_E), "elasticity"].mean())
    rows = []
    for model in [HUMAN_E] + MODEL_ORDER:
        part = elasticity.loc[elasticity.system.eq(model)]
        values = part.elasticity.dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        rows.append({
            "Model": model, "N cases": int(len(part)), "N respondents": int(part.respondent_id.nunique()),
            "Mean elasticity": float(values.mean()), "SD": float(values.std(ddof=1)),
            "Median elasticity": float(values.median()), "IQR": float(q3 - q1), "Q1": float(q1), "Q3": float(q3),
            "Fare-response monotonicity (%)": 100 * float(part.monotone_pairs.mean() / 6.0),
            "Absolute gap from Human mean": 0.0 if model == HUMAN_E else abs(float(values.mean()) - human_mean),
        })
    return pd.DataFrame(rows)


def table_02_interval(interval: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 2: interval-grain distribution (6 adjacent Fare intervals x 3,468 cases)."""
    finite = interval.loc[np.isfinite(interval.elasticity)]
    human_mean = float(finite.loc[finite.system.eq(HUMAN_E), "elasticity"].mean())
    rows = []
    for model in [HUMAN_E] + MODEL_ORDER:
        values = finite.loc[finite.system.eq(model), "elasticity"]
        q1, q3 = values.quantile([0.25, 0.75])
        rows.append({"Model": model, "N": int(values.size), "Mean": float(values.mean()), "SD": float(values.std(ddof=1)),
                     "Median": float(values.median()), "Q1": float(q1), "Q3": float(q3),
                     "Negative (%)": 100 * float(values.lt(0).mean()),
                     "Absolute gap from Human mean": 0.0 if model == HUMAN_E else abs(float(values.mean()) - human_mean)})
    return pd.DataFrame(rows)


def table_03_case(vot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in [HUMAN_V] + MODEL_ORDER:
        all_rows = vot.loc[vot.model.eq(model)]
        valid = all_rows.loc[all_rows.raw_computable & np.isfinite(all_rows.raw_vot)]
        finite = all_rows.loc[all_rows.raw_finite]
        rows.append({
            "Model": model, "N cases": int(len(all_rows)), "N respondents": int(all_rows.respondent_id.nunique()),
            "Computable (%)": 100 * float(all_rows.raw_computable.mean()),
            "Fare-negative (%)": 100 * float(valid.raw_DF.lt(0).mean()),
            "IVT-negative (%)": 100 * float(valid.raw_DT.lt(0).mean()),
            "Joint conventional-sign (%)": 100 * float((valid.raw_DF.lt(0) & valid.raw_DT.lt(0)).mean()),
            "Population Direct VOT-eq.": float(60 * finite.raw_DT.mean() / finite.raw_DF.mean()),
            "Median": float(valid.raw_vot.median()), "Q1": float(valid.raw_vot.quantile(0.25)),
            "Q3": float(valid.raw_vot.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def table_s13(vot_summary: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = vot_summary.set_index("Model")["Population Direct VOT-eq."]
    for model in [HUMAN_V] + MODEL_ORDER:
        part = local.loc[local.model.eq(model)]
        finite = part.loc[np.isfinite(part.loc_DF) & np.isfinite(part.loc_DT)]
        loc = float(60 * finite.loc_DT.mean() / finite.loc_DF.mean())
        rows.append({"Model": model, "Main Direct VOT (six pairs)": float(main.loc[model]),
                     "Local M1/P1 Direct VOT": loc, "Absolute difference": abs(float(main.loc[model]) - loc)})
    return pd.DataFrame(rows)


def subgroup_tables(interval: pd.DataFrame, vot: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TableS_Figure04 (interval-grain elasticity) and TableS_Figure05 (case-grain VOT) by subgroup."""
    e = interval.merge(meta, on="respondent_id", validate="many_to_one")
    e = e.loc[np.isfinite(e.elasticity)]
    v = vot.merge(meta, on="respondent_id", validate="many_to_one")
    e_rows, v_rows = [], []
    for axis, spec in GROUP_SPECS.items():
        for group in spec["order"]:
            for model in [HUMAN_E] + MODEL_ORDER:
                x = e.loc[e[axis].eq(group) & e.system.eq(model), "elasticity"]
                q1, q3 = x.quantile([0.25, 0.75])
                e_rows.append({"Axis": axis, "Group": group, "Model": model, "N": int(x.size), "Mean": float(x.mean()),
                               "SD": float(x.std(ddof=1)), "Median": float(x.median()), "Q1": float(q1), "Q3": float(q3),
                               "Negative": 100 * float(x.lt(0).mean())})
                vm = HUMAN_V if model == HUMAN_E else model
                c = v.loc[v[axis].eq(group) & v.model.eq(vm)]
                comp = c.loc[c.raw_computable & np.isfinite(c.raw_vot)]
                fin = c.loc[c.raw_finite]
                v_rows.append({"Axis": axis, "Group": group, "Model": model, "N": int(len(c)),
                               "Computable": 100 * float(c.raw_computable.mean()),
                               "Aggregate": float(60 * fin.raw_DT.mean() / fin.raw_DF.mean()),
                               "Median": float(comp.raw_vot.median()), "Q1": float(comp.raw_vot.quantile(0.25)),
                               "Q3": float(comp.raw_vot.quantile(0.75)),
                               "Joint": 100 * float(comp.raw_joint_conventional.mean())})
    return pd.DataFrame(e_rows), pd.DataFrame(v_rows)


def human_joint_tables(fit, joint_elasticity: pd.DataFrame, joint_vot: pd.DataFrame, groups: pd.DataFrame, out: Path
                       ) -> pd.DataFrame:
    """Table_S05 (coefficients), Table_S07 (interaction tests), Table_S06 (respondent-grain six-pair summary)."""
    coefficients = pd.DataFrame({"term": fit.params.index, "coefficient": fit.params.to_numpy(float),
                                 "cluster_se": fit.bse.to_numpy(float), "p_value": fit.pvalues.to_numpy(float)})
    coefficients.to_csv(out / "Table_S05_human_joint_heterogeneity_mnl_coefficients.csv", index=False)
    names = list(fit.params.index)
    rows = []
    for name, dimension in [("age50", "age_group"), ("female", "sex_group"), ("risk", "framing_group")]:
        fare_term, ivt_term = f"fare_AV_x_{name}", f"ivt_AV_x_{name}"
        restriction = np.zeros((2, len(names)))
        restriction[0, names.index(fare_term)] = 1.0
        restriction[1, names.index(ivt_term)] = 1.0
        joint = fit.wald_test(restriction, scalar=True)
        rows.append({"Dimension": dimension, "Fare interaction coefficient": float(fit.params[fare_term]),
                     "Fare interaction p-value": float(fit.pvalues[fare_term]),
                     "IVT interaction coefficient": float(fit.params[ivt_term]),
                     "IVT interaction p-value": float(fit.pvalues[ivt_term]),
                     "Joint Fare+IVT chi-square": float(joint.statistic), "Joint Fare+IVT df": 2,
                     "Joint Fare+IVT p-value": float(joint.pvalue)})
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "Table_S07_human_joint_heterogeneity_tests.csv", index=False)

    resp_e = joint_elasticity.groupby("respondent_id", as_index=False).agg(elasticity=("elasticity", "mean"))
    resp_v = joint_vot.groupby("respondent_id", as_index=False).agg(raw_DF=("raw_DF", "mean"), raw_DT=("raw_DT", "mean"))
    resp_v["raw_computable"] = np.isfinite(resp_v.raw_DF) & np.isfinite(resp_v.raw_DT) & resp_v.raw_DF.abs().ge(RAW_FARE_THRESHOLD)
    resp_v["raw_vot"] = np.where(resp_v.raw_computable, 60 * resp_v.raw_DT / resp_v.raw_DF, np.nan)
    resp_v["raw_joint_conventional"] = resp_v.raw_computable & resp_v.raw_DF.lt(0) & resp_v.raw_DT.lt(0)
    resp_e = resp_e.merge(groups, on="respondent_id", validate="one_to_one")
    resp_v = resp_v.merge(groups, on="respondent_id", validate="one_to_one")
    summary_rows = []
    for dimension, spec in GROUP_SPECS.items():
        for group in spec["order"]:
            e = resp_e.loc[resp_e[dimension].eq(group), "elasticity"]
            v_all = resp_v.loc[resp_v[dimension].eq(group)]
            v_valid = v_all.loc[v_all.raw_computable]
            summary_rows.append({"Dimension": dimension, "Group": group, "N respondents": int(e.size),
                                 "Mean Direct Fare elasticity": float(e.mean()),
                                 "Median Direct Fare elasticity": float(e.median()),
                                 "Population Direct VOT-eq.": float(60 * v_all.raw_DT.mean() / v_all.raw_DF.mean()),
                                 "Median Direct VOT-eq.": float(v_valid.raw_vot.median()),
                                 "Joint conventional-sign (%)": 100 * float(v_valid.raw_joint_conventional.mean())})
    summary = pd.DataFrame(summary_rows).merge(
        tests[["Dimension", "Fare interaction p-value", "IVT interaction p-value", "Joint Fare+IVT p-value"]],
        on="Dimension", how="left", validate="many_to_one")
    summary.to_csv(out / "Table_S06_human_joint_heterogeneity_direct_summary.csv", index=False)
    return summary


def gc_lane_tables(exp4_dir: Path, fare_table: pd.DataFrame, vot_summary: pd.DataFrame, out: Path
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Table_06 / Table_07 / Table_S04 from a CF_MNL_ROBUSTNESS run directory (same mapping as
    tools/build_unified_balanced_candidate_tables.table9_exp4 + the 0826 support module)."""
    vot = pd.read_csv(exp4_dir / "Table_EXP4_CF_MNL_VOT.csv").set_index("Model")
    ela = pd.read_csv(exp4_dir / "Table_EXP4_CF_MNL_Fare_Elasticity.csv").set_index("Model")
    audit = pd.read_csv(exp4_dir / "TableS_EXP4_CF_MNL_Fit_Audit.csv")
    human_ref = json.loads((exp4_dir / "human_references.json").read_text(encoding="utf-8"))

    def zero_included(model: str) -> bool:
        lo, hi, shape = vot.loc[model, ["base_mnl_fieller_low", "base_mnl_fieller_high", "base_mnl_fieller_shape"]]
        return str(shape).lower() == "bounded" and pd.notna(lo) and pd.notna(hi) and float(lo) <= 0 <= float(hi)

    human_coef = pd.read_csv(FINAL / "03_EXP3" / "tableS_human_mnl_coefficients.csv")
    human_coef = human_coef.loc[human_coef["sample"].eq("development/train")].set_index("term")
    rows = [{"Model": "H-MNL",
             "Fare coefficient": float(human_coef.loc["fare_AV", "coefficient"]),
             "Fare cluster SE": float(human_coef.loc["fare_AV", "cluster_se"]),
             "Fare p-value": float(human_coef.loc["fare_AV", "p_value"]),
             "IVT coefficient": float(human_coef.loc["ivt_AV", "coefficient"]),
             "IVT cluster SE": float(human_coef.loc["ivt_AV", "cluster_se"]),
             "IVT p-value": float(human_coef.loc["ivt_AV", "p_value"]),
             "GC-MNL Fare elasticity": float(human_ref["human_structural_elasticity_mean"]),
             "GC-MNL VOT": float(human_ref["human_structural_vot"]),
             "Fieller status": str(human_ref["human_vot_status"])}]
    base = audit.loc[audit.projection.eq("BASE_MNL")].set_index("Model")
    for model in QWEN_ORDER:
        item = base.loc[model]
        status = "WEAKLY_IDENTIFIED" if zero_included(model) else "RESOLVED"
        if item.beta_fare_av >= 0 or item.beta_ivt_av >= 0:
            status = "SIGN_INVALID"
        rows.append({"Model": model, "Fare coefficient": float(item.beta_fare_av), "Fare cluster SE": float(item.se_fare_av),
                     "Fare p-value": float(item.p_fare_av), "IVT coefficient": float(item.beta_ivt_av),
                     "IVT cluster SE": float(item.se_ivt_av), "IVT p-value": float(item.p_ivt_av),
                     "GC-MNL Fare elasticity": float(ela.loc[model, "base_mnl_local_mean"]),
                     "GC-MNL VOT": float(vot.loc[model, "base_mnl_vot"]), "Fieller status": status})
    table_06 = pd.DataFrame(rows)
    table_06.to_csv(out / "Table_06_gc_mnl_behavioral_measures.csv", index=False)

    fare_index = fare_table.set_index("Model")
    vot_index = vot_summary.set_index("Model")
    main_rows, robust_rows = [], []
    for model in MODEL_ORDER:
        direct_e = float(fare_index.loc[model, "Mean elasticity"])
        gc_e = float(ela.loc[model, "base_mnl_local_mean"])
        direct_v = float(vot_index.loc[model, "Population Direct VOT-eq."])
        gc_v = float(vot.loc[model, "base_mnl_vot"])
        main_rows.append({"Model": model, "GC-MNL Fare elasticity": gc_e, "Direct Fare elasticity": direct_e,
                          "Absolute elasticity gap": abs(gc_e - direct_e), "GC-MNL VOT": gc_v, "Direct VOT-eq.": direct_v,
                          "Absolute VOT gap": abs(gc_v - direct_v),
                          "GC-MNL VOT status": "WEAKLY_IDENTIFIED" if zero_included(model) else "RESOLVED"})
        cf_e = float(ela.loc[model, "cf_mnl_local_mean"])
        cf_v = float(vot.loc[model, "cf_mnl_vot"])
        robust_rows.append({"Model": model, "CF-MNL Fare elasticity": cf_e, "Absolute CF-Direct elasticity gap": abs(cf_e - direct_e),
                            "CF-MNL VOT": cf_v, "Absolute CF-Direct VOT gap": abs(cf_v - direct_v)})
    table_07 = pd.DataFrame(main_rows)
    table_s04 = pd.DataFrame(robust_rows)
    table_07.to_csv(out / "Table_07_gc_mnl_direct_correspondence.csv", index=False)
    table_s04.to_csv(out / "Table_S04_cf_mnl_robustness.csv", index=False)
    return table_06, table_07, table_s04, human_ref


# ----------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------
def _save(fig: plt.Figure, out: Path, stem: str) -> tuple[Path, Path]:
    png, pdf = out / f"{stem}.png", out / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return png, pdf


def _paired_box(ax, values: np.ndarray, y: float, edge: str, fill: str, filled: bool, xlim: tuple[float, float]) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    result = ax.boxplot([values], positions=[y], vert=False, widths=0.30, whis=(10, 90), showfliers=False,
                        patch_artist=True, manage_ticks=False,
                        boxprops={"facecolor": fill if filled else "white", "edgecolor": edge, "linewidth": 1.1},
                        whiskerprops={"color": edge, "linewidth": 0.9}, capprops={"color": edge, "linewidth": 0.9},
                        medianprops={"color": edge, "linewidth": 1.7})
    for patch in result["boxes"]:
        patch.set_zorder(3)
    p10, p90 = np.percentile(values, [10, 90])
    span = xlim[1] - xlim[0]
    if p10 < xlim[0]:
        ax.plot([xlim[0] + 0.006 * span], [y], marker="<", color=edge, markersize=4.2, linestyle="none", clip_on=False, zorder=5)
    if p90 > xlim[1]:
        ax.plot([xlim[1] - 0.006 * span], [y], marker=">", color=edge, markersize=4.2, linestyle="none", clip_on=False, zorder=5)


def merged_subgroup_figure(frame: pd.DataFrame, value: str, model_col: str, human_label: str, xlabel: str,
                           xlim: tuple[float, float], out: Path, stem: str) -> tuple[Path, Path]:
    """Three-panel (age / sex / framing) paired-box figure: open box = first group, filled box = second group."""
    rows = [HUMAN_E] + MODEL_ORDER
    titles = {"age_group": ("A. Age", "open: <50     filled: 50+"),
              "sex_group": ("B. Sex", "open: Male     filled: Female"),
              "framing_group": ("C. Survey framing", "open: BASIC     filled: RISK")}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.6), sharey=True)
    for ax, (axis, spec) in zip(axes, GROUP_SPECS.items()):
        ax.grid(False)
        ax.axvline(0, color="#9A9A9A", linewidth=0.9, zorder=1)
        for y, model in enumerate(rows):
            key = _style_key(model)
            edge, fill = SUBGROUP_STYLE[key]
            label = human_label if model == HUMAN_E else model
            for offset, group in zip((-0.2, 0.2), spec["order"]):
                values = frame.loc[frame[model_col].eq(label) & frame[axis].eq(group), value].to_numpy(float)
                _paired_box(ax, values, y + offset, edge, fill, filled=offset > 0, xlim=xlim)
        for boundary in (0.5, 6.5):
            ax.axhline(boundary, color="#C8C8C8", linewidth=0.8, zorder=1)
        ax.set_xlim(*xlim)
        if xlim[1] - xlim[0] < 10:
            ax.set_xticks([t for t in range(int(np.ceil(xlim[0])), int(np.floor(xlim[1])) + 1)])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)), ["Human"] + MODEL_ORDER)
        ax.set_xlabel(xlabel)
        title, subtitle = titles[axis]
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=17)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8, color="#333333")
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.11, wspace=0.10)
    return _save(fig, out, stem)


def overall_vot_figure(vot: pd.DataFrame, summary: pd.DataFrame, out: Path, paper: bool) -> tuple[Path, Path]:
    order = [HUMAN_V] + MODEL_ORDER
    color = {HUMAN_V: "#7A3E1D", **S.COLORS}
    valid = vot.loc[vot.raw_computable & np.isfinite(vot.raw_vot)].copy()
    q02, q98 = valid.raw_vot.quantile([0.02, 0.98])
    low, high = max(-80.0, float(q02)), min(100.0, float(q98))
    low, high = min(low, -20.0), max(high, 30.0)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
    for y, model in enumerate(order):
        values = valid.loc[valid.model.eq(model), "raw_vot"].to_numpy(float)
        S._style_boxplot(axes[0], values, y, color[model])
        row = summary.loc[summary.Model.eq(model)].iloc[0]
        axes[0].scatter(row["Median"], y, s=25, color=color[model], edgecolor="white", linewidth=0.5, zorder=4)
        axes[0].scatter(row["Population Direct VOT-eq."], y, s=45, marker="D", color=color[model], edgecolor="white", linewidth=0.5, zorder=5)
    axes[0].axvline(0, color="#555555", linewidth=0.9)
    axes[0].set_xlim(low, high)
    axes[0].set_yticks(range(len(order)), ["Human reference"] + MODEL_ORDER)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Signed Direct VOT (kKRW/hour)" if paper else "Signed Direct VOT-equivalent (kKRW/hour)")
    axes[0].set_title("A. Respondent distribution and aggregate VOT" if paper else "A. Respondent distribution and population ratio")
    axes[0].grid(axis="x", color="#D7D7D7", linewidth=0.7)
    axes[0].grid(axis="y", visible=False)
    rate_columns = ["Computable (%)", "Joint conventional-sign (%)"]
    labels = ["Computable (%)", "Expected signs (%)" if paper else "Joint conventional-sign (%)"]
    for xcol, lab, marker, c in zip(rate_columns, labels, ["o", "s"], ["#2F5D7E", "#B8861B"]):
        axes[1].scatter(summary[xcol], np.arange(len(order)), marker=marker, s=35, color=c, label=lab)
    axes[1].axvline(100, color="#888888", linewidth=0.8)
    axes[1].set_xlim(0, 102)
    axes[1].set_yticks(range(len(order)), [""] * len(order))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Respondents (%)")
    axes[1].set_title("B. Computability and component-sign validity")
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].grid(axis="x", color="#D7D7D7", linewidth=0.7)
    axes[1].grid(axis="y", visible=False)
    if not paper:
        fig.suptitle("Figure 6. Overall Direct VOT-equivalent and sign validity", fontsize=14.5, y=0.995)
        fig.text(0.5, 0.025, "A positive ratio is interpreted economically only when both Fare and IVT derivatives have conventional negative signs.",
                 ha="center", fontsize=8.3, color="#444444")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.90, bottom=0.12, wspace=0.18)
    return _save(fig, out, "Figure_06_overall_direct_vot")


def correspondence_elasticity_figure(correspondence: pd.DataFrame, cf_robustness: pd.DataFrame, elasticity: pd.DataFrame,
                                     human: dict, out: Path, paper: bool) -> tuple[Path, Path]:
    direct_color = "#2F78A0"
    cf = cf_robustness.set_index("Model")
    comparison = correspondence.set_index("Model")
    human_values = elasticity.loc[elasticity.system.eq(HUMAN_E), "elasticity"].dropna()
    human_q1, human_q3 = human_values.quantile([0.25, 0.75])
    all_values = elasticity.loc[elasticity.system.isin(MODEL_ORDER), "elasticity"].dropna()
    plot_values = list(all_values.quantile([0.01, 0.99]))
    plot_values += correspondence[["GC-MNL Fare elasticity", "Direct Fare elasticity"]].to_numpy(float).ravel().tolist()
    plot_values += cf_robustness["CF-MNL Fare elasticity"].to_numpy(float).tolist()
    plot_values += [human["human_structural_elasticity_mean"], human["human_direct_elasticity_mean"]]
    low, high = min(plot_values), max(plot_values)
    pad = 0.08 * max(high - low, 1.0)
    panel_specs = [("A. Conventional prediction systems", MODEL_ORDER[:6]), ("B. Qwen systems", QWEN_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.0), sharex=True)
    for ax, (panel_title, models) in zip(axes, panel_specs):
        for yi, model in enumerate(models):
            values = elasticity.loc[elasticity.system.eq(model), "elasticity"].dropna().to_numpy(float)
            S._style_boxplot(ax, values, yi, direct_color)
            row = comparison.loc[model]
            ax.scatter(row["Direct Fare elasticity"], yi, marker="D", s=54, color=direct_color, edgecolor="white", linewidth=0.6, zorder=5)
            ax.scatter(row["GC-MNL Fare elasticity"], yi, marker="s", s=58, facecolor="white", edgecolor="#333333", linewidth=1.3, zorder=6)
            ax.scatter(cf.loc[model, "CF-MNL Fare elasticity"], yi, marker="o", s=52, facecolor="#E6A04B", edgecolor="#8C4E16", linewidth=1.0, zorder=6)
        ax.axvspan(human_q1, human_q3, color="#777777", alpha=0.08, zorder=0)
        ax.axvline(human["human_structural_elasticity_mean"], color="#9A5D35", linestyle="--", linewidth=1.35)
        ax.axvline(human["human_direct_elasticity_mean"], color="#1D8588", linestyle=":", linewidth=1.55)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        ax.set_xlim(low - pad, high + pad)
        ax.set_xlabel("AV fare elasticity" if paper else "AV Fare elasticity")
        ax.set_title(panel_title, fontsize=12.0, fontweight="semibold", pad=10)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    if not paper:
        fig.suptitle("Figure 10. Direct, GC-MNL, and CF-MNL Fare elasticity", fontsize=14.2, y=0.995)
    labels = (["Direct-response distribution (IQR; whiskers p05-p95)", "Direct-response mean", "GC-MNL", "CF-MNL",
               "H-MNL reference", "Human direct-response reference"] if paper else
              ["Direct respondent distribution (IQR; whiskers p05-p95)", "Direct population mean", "GC-MNL", "CF-MNL",
               "H-MNL GC coordinate", "Human Direct coordinate"])
    fig.legend(handles=[
        Patch(facecolor=direct_color, edgecolor=direct_color, alpha=0.23, label=labels[0]),
        Line2D([], [], marker="D", color="none", markerfacecolor=direct_color, markeredgecolor="white", label=labels[1]),
        Line2D([], [], marker="s", color="none", markerfacecolor="white", markeredgecolor="#333333", label=labels[2]),
        Line2D([], [], marker="o", color="none", markerfacecolor="#E6A04B", markeredgecolor="#8C4E16", label=labels[3]),
        Line2D([], [], color="#9A5D35", linestyle="--", label=labels[4]),
        Line2D([], [], color="#1D8588", linestyle=":", label=labels[5]),
    ], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    if not paper:
        fig.text(0.5, 0.025, "Direct boxes summarize respondent heterogeneity, not confidence intervals; GC-MNL uses original-condition generated choices and CF-MNL adds LOS-counterfactual generated choices.",
                 ha="center", fontsize=8.1, color="#444444")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.12, wspace=0.30)
    return _save(fig, out, "Figure_10_gc_mnl_direct_elasticity")


def correspondence_vot_figure(correspondence: pd.DataFrame, cf_robustness: pd.DataFrame, vot: pd.DataFrame, human: dict,
                              out: Path, paper: bool) -> tuple[Path, Path]:
    direct_color = "#2F78A0"
    cf = cf_robustness.set_index("Model")
    comparison = correspondence.set_index("Model")
    low, high = -50.0, 80.0
    panel_specs = [("A. Conventional prediction systems", MODEL_ORDER[:6]), ("B. Qwen systems", QWEN_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.0), sharex=True)
    for ax, (panel_title, models) in zip(axes, panel_specs):
        ax.axvspan(human["human_fieller_low"], human["human_fieller_high"], color="#9A5D35", alpha=0.08, zorder=0)
        ax.axvline(human["human_structural_vot"], color="#9A5D35", linestyle="--", linewidth=1.4)
        ax.axvline(0, color="#555555", linewidth=0.8)
        for yi, model in enumerate(models):
            values = vot.loc[vot.model.eq(model) & vot.raw_computable & np.isfinite(vot.raw_vot), "raw_vot"].to_numpy(float)
            S._style_boxplot(ax, values, yi, direct_color)
            row = comparison.loc[model]
            ax.scatter(row["Direct VOT-eq."], yi, marker="D", s=54, color=direct_color, edgecolor="white", linewidth=0.6, zorder=5)
            ax.scatter(row["GC-MNL VOT"], yi, marker="s", s=58, facecolor="white", edgecolor="#333333", linewidth=1.3, zorder=6)
            ax.scatter(cf.loc[model, "CF-MNL VOT"], yi, marker="o", s=52, facecolor="#E6A04B", edgecolor="#8C4E16", linewidth=1.0, zorder=6)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        ax.set_xlim(low, high)
        ax.set_xlabel("VOT / Direct VOT (kKRW/hour)" if paper else "VOT / VOT-equivalent (kKRW/hour)")
        ax.set_title(panel_title, fontsize=12.0, fontweight="semibold", pad=10)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    if not paper:
        fig.suptitle("Figure 11. Direct, GC-MNL, and CF-MNL VOT comparison", fontsize=14.2, y=0.995)
    labels = (["Direct-response distribution (IQR; whiskers p05-p95)", "Aggregate Direct VOT"] if paper else
              ["Direct respondent distribution (IQR; whiskers p05-p95)", "Direct population ratio"])
    fig.legend(handles=[
        Patch(facecolor=direct_color, edgecolor=direct_color, alpha=0.23, label=labels[0]),
        Line2D([], [], marker="D", color="none", markerfacecolor=direct_color, markeredgecolor="white", label=labels[1]),
        Line2D([], [], marker="s", color="none", markerfacecolor="white", markeredgecolor="#333333", label="GC-MNL VOT"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#E6A04B", markeredgecolor="#8C4E16", label="CF-MNL VOT"),
        Line2D([], [], color="#9A5D35", linestyle="--", label="H-MNL VOT reference"),
        Patch(facecolor="#9A5D35", alpha=0.08, label="Human Fieller set"),
    ], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    if not paper:
        fig.text(0.5, 0.025, "Direct boxes show respondent distributions; extreme p05-p95 whiskers may extend beyond the displayed range. Diamonds are ratios of population-mean slopes; Fieller shading is uncertainty context.",
                 ha="center", fontsize=8.1, color="#444444")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.12, wspace=0.30)
    return _save(fig, out, "Figure_11_gc_mnl_direct_vot")


def response_figure(response: pd.DataFrame, out: Path, paper: bool) -> tuple[Path, Path]:
    systems = [HUMAN_E] + QWEN_ORDER
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.3))
    for col, attribute in enumerate(["Fare", "IVT", "Wait"]):
        data = response.loc[response.attribute.eq(attribute) & response.system.isin(systems)].copy()
        for system in systems:
            part = data.loc[data.system.eq(system)].sort_values("rung")
            style = dict(color="black", marker="*", linestyle="-", linewidth=2.3, markersize=8)
            if system != HUMAN_E:
                style = {**S.QWEN_STYLE[system], "linewidth": 1.8, "markersize": 5.5}
            axes[0, col].plot(part.mean_physical_value, part.mean_p, label=system, **style)
            axes[1, col].plot(part.mean_physical_value, part.mean_delta_p, label=system, **style)
        base = data.loc[data.rung.eq(0), "mean_physical_value"].mean()
        for row in range(2):
            axes[row, col].axvline(base, color="#777777", linestyle=":", linewidth=1.0)
            axes[row, col].grid(color="#DADADA", linewidth=0.65)
        axes[0, col].set_title(["A. Fare", "B. IVT", "C. Wait/access"][col], fontsize=11.5)
        axes[1, col].axhline(0, color="#555555", linewidth=0.8)
        axes[1, col].set_xlabel(S.X_LABEL[attribute])
        axes[0, col].set_ylabel("Mean P(AV)" if col == 0 else "")
        axes[1, col].set_ylabel("BASE-centered Delta P(AV)" if col == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.95))
    if not paper:
        fig.suptitle("Figure 2. Direct probability responses to Fare, IVT, and Wait/access changes", fontsize=14.5, y=0.995)
        fig.text(0.5, 0.018, "Same respondents and trips are retained; only the focal AV attribute changes. Wait/access is exploratory because the Human coefficient is unresolved.",
                 ha="center", fontsize=8.2, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.10, hspace=0.24, wspace=0.22)
    return _save(fig, out, "Figure_02_probability_response_fare_ivt_wait")


def per_axis_figures(elasticity: pd.DataFrame, joint_elasticity: pd.DataFrame, vot: pd.DataFrame, joint_vot: pd.DataFrame,
                     meta: pd.DataFrame) -> None:
    """Legacy per-axis figures 03/04/05 and 07/08/09 (case grain), kept for package completeness."""
    for group_col, number in [("age_group", 3), ("sex_group", 4), ("framing_group", 5)]:
        S.subgroup_elasticity_figure(elasticity, joint_elasticity, meta, group_col, number)
    for group_col, number in [("age_group", 7), ("sex_group", 8), ("framing_group", 9)]:
        S.subgroup_vot_figure(vot, joint_vot, meta, group_col, number)


# ----------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------
def build(bank: Path, exp4_dir: Path, out: Path, label: str) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    paper_dir = out / "paper_figures"
    paper_dir.mkdir(exist_ok=True)
    S.OUT = out  # re-point the reused 0826 functions
    inputs = load_inputs(bank)
    test, meta, grid, scores, order = inputs["test"], inputs["meta"], inputs["grid"], inputs["scores"], inputs["case_order"]

    qwen = qwen_case_frames(scores, grid, test, order)
    tabular = tabular_case_frames(grid, test, order)
    human_plain, human_info = human_plain_case_frames(grid, test, order)
    human_joint, joint_fit, joint_groups = human_joint_case_frames(grid, test, order)
    assert joint_groups.merge(meta, on="respondent_id").pipe(
        lambda d: (d.age_group_x == d.age_group_y).all() and (d.sex_group_x == d.sex_group_y).all() and (d.framing_group_x == d.framing_group_y).all())

    systems = [human_plain] + tabular + qwen
    elasticity = pd.concat([f["elasticity"] for f in systems], ignore_index=True)
    interval = pd.concat([f["interval"] for f in systems], ignore_index=True)
    vot = pd.concat([f["vot"] for f in systems], ignore_index=True)
    local = pd.concat([f["local"] for f in systems], ignore_index=True)
    assert elasticity.groupby("system").size().eq(3468).all() and vot.groupby("model").size().eq(3468).all()

    elasticity.to_parquet(out / "Table_S02_direct_fare_elasticity_respondent.parquet", index=False)
    vot.to_parquet(out / "Table_S03_direct_vot_respondent.parquet", index=False)
    local.to_parquet(out / "Table_S13_local_pair_vot_case.parquet", index=False)
    interval.to_parquet(out / "Table_S02b_direct_fare_elasticity_interval.parquet", index=False)

    table_02 = table_02_case(elasticity)
    table_02.to_csv(out / "Table_02_direct_fare_response.csv", index=False)
    table_02i = table_02_interval(interval)
    table_02i.to_csv(out / "Table_02_direct_fare_response_interval.csv", index=False)
    table_03 = table_03_case(vot)
    table_03.to_csv(out / "Table_03_direct_vot_validity.csv", index=False)
    s13 = table_s13(table_03, local)
    s13.to_csv(out / "Table_S13_local_pair_vot_robustness.csv", index=False)

    # subgroup tables and merged figures use the joint-interaction Human reference
    joint_interval = human_joint["interval"]
    joint_vot = human_joint["vot"]
    sub_interval = pd.concat([joint_interval] + [f["interval"] for f in tabular + qwen], ignore_index=True)
    sub_vot = pd.concat([joint_vot] + [f["vot"] for f in tabular + qwen], ignore_index=True)
    fig04, fig05 = subgroup_tables(sub_interval, sub_vot, meta)
    fig04.to_csv(out / "TableS_Figure04_subgroup_fare_elasticity.csv", index=False)
    fig05.to_csv(out / "TableS_Figure05_subgroup_direct_vot.csv", index=False)
    e_frame = sub_interval.merge(meta, on="respondent_id", validate="many_to_one")
    v_frame = sub_vot.loc[sub_vot.raw_computable & np.isfinite(sub_vot.raw_vot)].merge(meta, on="respondent_id", validate="many_to_one")
    merged_subgroup_figure(e_frame, "elasticity", "system", HUMAN_E, "Direct-response AV fare elasticity", FIG3_XLIM, out,
                           "Figure_03_fare_elasticity_subgroups")
    merged_subgroup_figure(v_frame, "raw_vot", "model", HUMAN_V, "Signed Direct VOT (kKRW/hour)", FIG5_XLIM, out,
                           "Figure_05_direct_vot_subgroups")
    for stem in ("Figure_03_fare_elasticity_subgroups", "Figure_05_direct_vot_subgroups"):
        shutil.copy2(out / f"{stem}.pdf", paper_dir / f"{stem}.pdf")

    human_joint_tables(joint_fit, human_joint["elasticity"], joint_vot, joint_groups, out)

    # adjusted group regression (Table 4 / paper Table S14): joint Human + Qwen case frames
    S.direct_response_regression_table(elasticity, vot.rename(columns={"model": "model"}), human_joint["elasticity"],
                                       joint_vot, meta)

    # reused 0826 builders (BASE hard choices, response curves, prediction metrics)
    S.expanded_generated_choice_logit_table(scores)
    S.choice_implied_cell_vot_feasibility_audit(scores, grid)
    response = S._build_response_curves(scores, grid, test)
    response.to_csv(out / "Table_S01_probability_response_curves.csv", index=False)
    prediction = S._build_prediction_table(scores, test)
    prediction.to_csv(out / "Table_01_prediction_performance.csv", index=False)

    table_06, table_07, table_s04, human_ref = gc_lane_tables(exp4_dir, table_02, table_03, out)

    # figures (titled package copies + untitled/relabelled paper copies)
    S.prediction_figure(prediction)
    for paper, target in ((False, out), (True, paper_dir)):
        response_figure(response, target, paper)
        overall_vot_figure(vot, table_03, target, paper)
        correspondence_elasticity_figure(table_07, table_s04, elasticity, human_ref, target, paper)
        correspondence_vot_figure(table_07, table_s04, vot, human_ref, target, paper)
    per_axis_figures(elasticity, human_joint["elasticity"], vot, joint_vot, meta)

    checks = [
        {"check": "prediction rows", "passed": len(prediction) == 10, "detail": str(len(prediction))},
        {"check": "elasticity case rows", "passed": bool(elasticity.groupby("system").size().eq(3468).all()), "detail": "3468/system"},
        {"check": "elasticity respondents", "passed": bool(elasticity.groupby("system").respondent_id.nunique().eq(651).all()), "detail": "651"},
        {"check": "VOT case rows", "passed": bool(vot.groupby("model").size().eq(3468).all()), "detail": "3468/model"},
        {"check": "systems", "passed": elasticity.system.nunique() == 11, "detail": str(elasticity.system.nunique())},
        {"check": "Qwen probability", "passed": bool(np.allclose(scores.p, expit(scores.margin), atol=1e-14)), "detail": "p=expit"},
        {"check": "human subgroup rows", "passed": len(fig04) == 66 and len(fig05) == 66, "detail": f"{len(fig04)}/{len(fig05)}"},
        {"check": "figure PNG", "passed": len(list(out.glob("Figure_*.png"))) == 13, "detail": str(len(list(out.glob("Figure_*.png"))))},
        {"check": "interval rows", "passed": bool(interval.groupby("system").size().eq(20808).all()), "detail": "20808/system"},
    ]
    report = pd.DataFrame(checks)
    report.to_csv(out / "VALIDATION_CHECKS.csv", index=False)
    assert report.passed.all(), report.loc[~report.passed]

    files = sorted([p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"], key=lambda p: str(p).lower())
    manifest = {"label": label, "bank": str(bank), "exp4_dir": str(exp4_dir), "human_plain_fit": human_info,
                "operator": "six adjacent pairs; case grain; tabular seeds averaged after the operator",
                "files": [{"file": str(p.relative_to(out)), "bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"table_02": table_02, "table_02_interval": table_02i, "table_03": table_03, "table_06": table_06,
            "table_07": table_07, "table_s04": table_s04, "table_s13": s13, "prediction": prediction,
            "human_info": human_info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--exp4", required=True, help="CF_MNL_ROBUSTNESS directory with Table_EXP4_CF_MNL_*.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    result = build(Path(a.bank), Path(a.exp4), Path(a.out), a.label)
    pd.set_option("display.width", 250)
    print("human plain fit:", result["human_info"])
    print(result["table_02_interval"].round(4).to_string())
    print(result["table_03"].round(4).to_string())
