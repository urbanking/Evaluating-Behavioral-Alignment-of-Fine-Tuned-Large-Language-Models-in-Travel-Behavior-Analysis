# -*- coding: utf-8 -*-
"""Six-adjacent-pair Direct operator (D321 / SECTION4_RESULTS_CASE lineage), reimplemented.

Why this file exists. The paper's Section 4 numbers come from `Real_exp/pdf/SECTION4_RESULTS_CASE`
(case grain, 3,468 cases). Its generator is not on this machine, so the operator is rebuilt here
and must reproduce the CASE parquets from the fixed-split bank to machine precision before it is
applied to the CV5 bank. `reproduce()` is that gate.

Operator (per case, per attribute ladder [M3, M2, M1, BASE, P1, P2, P3]):
  six adjacent pairs (l, u): (M3,M2) (M2,M1) (M1,BASE) (BASE,P1) (P1,P2) (P2,P3)
  interval midpoint arc elasticity  E_k = ((p_u - p_l) / ((p_u + p_l)/2)) / ((x_u - x_l) / ((x_u + x_l)/2))
  interval slope                    D_k = (p_u - p_l) / (x_u - x_l)
  case elasticity  = mean_k E_k       (Fare)          -> CASE `elasticity`
  monotone_pairs   = #k with E_k < 0 (Fare, strict)    -> CASE `monotone_pairs` (verified)
  case raw_DF      = mean_k D_k (Fare),  raw_DT = mean_k D_k (IVT)   -> CASE `raw_DF`, `raw_DT`
  raw_vot          = 60 * raw_DT / raw_DF  when computable
  aggregate VOT    = 60 * mean(raw_DT) / mean(raw_DF) over FINITE cases (D321; verified 6.7923)
  local (M1/P1)    loc_DF = (p_P1 - p_M1)/(x_P1 - x_M1), same for IVT  -> CASE S13
Probability p = sigmoid(margin) (D321, no calibration). Grid values from the test scenario grid.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit

REAL = Path(r"C:\SP_LLM\TRC\Real_exp")
PATH_GRID = REAL / "data" / "05_scenarios" / "test_scenario_grid.parquet"
PATH_TEST = REAL / "data" / "03_model_inputs" / "HUMAN_retro_test.parquet"
LADDER = {"Fare": ["F_M3", "F_M2", "F_M1", "BASE", "F_P1", "F_P2", "F_P3"],
          "IVT":  ["R_M3", "R_M2", "R_M1", "BASE", "R_P1", "R_P2", "R_P3"]}
XCOL = {"Fare": "fare_cf", "IVT": "ride_cf"}
RAW_FARE_THRESHOLD = 1e-4
QWEN_KEYS = {"qwen_mid_zeroshot": "Qwen 2B ZS", "qwen_mid_sft": "Qwen 2B SFT",
             "qwen_zeroshot": "Qwen 9B ZS", "qwen_sft": "Qwen 9B SFT"}


def load_bank(path, models=None):
    b = pd.read_parquet(path, columns=["case_id", "scenario_id", "model", "margin", "free_argmax"])
    b["case_id"] = b.case_id.astype(str)
    if models: b = b[b.model.isin(models)]
    b["p"] = expit(b.margin.to_numpy(float))
    return b


def wide(bank, grid, attr):
    """case x rung matrices of p and x for one attribute ladder, one system."""
    sids = LADDER[attr]
    P = bank[bank.scenario_id.isin(sids)].pivot(index="case_id", columns="scenario_id", values="p")[sids]
    X = grid[grid.scenario_id.isin(sids)].pivot(index="case_id", columns="scenario_id", values=XCOL[attr])[sids]
    X = X.loc[P.index]
    return P.to_numpy(float), X.to_numpy(float), P.index.to_numpy()


def six_pairs(P, X):
    pl, pu = P[:, :-1], P[:, 1:]
    xl, xu = X[:, :-1], X[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        E = ((pu - pl) / ((pu + pl) / 2.0)) / ((xu - xl) / ((xu + xl) / 2.0))
        D = (pu - pl) / (xu - xl)
    mono = (E < 0)   # CASE definition: strictly negative interval elasticity (verified max|diff| 0)
    return E, D, mono


def case_frame(bank, grid, system_label):
    """CASE-grain frame for one system: elasticity (Fare), monotone_pairs, raw_DF, raw_DT, raw_vot, flags,
    loc_DF/loc_DT/loc_vot, plus the 6 interval elasticities (Fare) for the interval-grain statistics."""
    Pf, Xf, ids = wide(bank, grid, "Fare")
    Pt, Xt, ids_t = wide(bank, grid, "IVT")
    assert (ids == ids_t).all()
    Ef, Df, monof = six_pairs(Pf, Xf)
    Et, Dt, _ = six_pairs(Pt, Xt)
    raw_DF, raw_DT = Df.mean(axis=1), Dt.mean(axis=1)
    finite = np.isfinite(raw_DF) & np.isfinite(raw_DT)
    near0 = finite & (np.abs(raw_DF) < RAW_FARE_THRESHOLD)
    comp = finite & ~near0
    with np.errstate(divide="ignore", invalid="ignore"):
        vot = np.where(comp, 60.0 * raw_DT / raw_DF, np.nan)
        loc_DF = (Pf[:, 4] - Pf[:, 2]) / (Xf[:, 4] - Xf[:, 2])
        loc_DT = (Pt[:, 4] - Pt[:, 2]) / (Xt[:, 4] - Xt[:, 2])
        loc_vot = 60.0 * loc_DT / loc_DF
    out = pd.DataFrame({
        "system": system_label, "case_id": ids,
        "elasticity": Ef.mean(axis=1), "monotone_pairs": monof.sum(axis=1).astype(float),
        "delta_p_base_p1": Pf[:, 4] - Pf[:, 3],
        "raw_DF": raw_DF, "raw_DT": raw_DT, "raw_finite": finite, "raw_near_zero_fare": near0,
        "raw_computable": comp, "raw_fare_negative": comp & (raw_DF < 0),
        "raw_ivt_negative": comp & (raw_DT < 0), "raw_joint_conventional": comp & (raw_DF < 0) & (raw_DT < 0),
        "raw_vot": vot, "loc_DF": loc_DF, "loc_DT": loc_DT, "loc_vot": loc_vot,
    })
    for k in range(6):
        out["E_pair%d" % (k + 1)] = Ef[:, k]
    return out


def table3_summary(cf):
    c = cf[cf.raw_computable]
    return {
        "Computable (%)": 100 * cf.raw_computable.mean(),
        "Fare-negative (%)": 100 * c.raw_fare_negative.mean(),
        "IVT-negative (%)": 100 * c.raw_ivt_negative.mean(),
        "Joint conventional-sign (%)": 100 * c.raw_joint_conventional.mean(),
        # D321: ratio of FINITE case-mean slopes (all finite cases, not computable-only). Verified 6.7923.
        "Population Direct VOT-eq.": 60 * cf[cf.raw_finite].raw_DT.mean() / cf[cf.raw_finite].raw_DF.mean(),
        "Median": c.raw_vot.median(), "Q1": c.raw_vot.quantile(.25), "Q3": c.raw_vot.quantile(.75),
    }


def table2_case_summary(cf):
    e = cf.elasticity
    return {"N cases": len(cf), "Mean": e.mean(), "SD": e.std(ddof=1), "Median": e.median(),
            "Q1": e.quantile(.25), "Q3": e.quantile(.75),
            "Fare-response monotonicity (%)": 100 * cf.monotone_pairs.sum() / (6 * len(cf))}   # = interval share with E<0


def table2_interval_summary(cf):
    v = cf[["E_pair%d" % k for k in range(1, 7)]].to_numpy().ravel()
    v = v[np.isfinite(v)]
    return {"N intervals": int(v.size), "Mean": v.mean(), "SD": v.std(ddof=1), "Median": np.median(v),
            "Q1": np.quantile(v, .25), "Q3": np.quantile(v, .75), "Negative (%)": 100 * (v < 0).mean()}


def reproduce(bank_path, case_dir=REAL / "pdf" / "SECTION4_RESULTS_CASE", system="Qwen 2B SFT", key="qwen_mid_sft"):
    """Gate: does this reimplementation reproduce the CASE parquets and Table 3 from the fixed-split bank?"""
    grid = pd.read_parquet(PATH_GRID); grid["case_id"] = grid.case_id.astype(str)
    bank = load_bank(bank_path, [key])
    cf = case_frame(bank, grid, system).set_index("case_id")
    s02 = pd.read_parquet(case_dir / "Table_S02_direct_fare_elasticity_respondent.parquet")
    s02 = s02[s02.system == system].set_index("case_id").loc[cf.index]
    s03 = pd.read_parquet(case_dir / "Table_S03_direct_vot_respondent.parquet")
    s03 = s03[s03.model == system].set_index("case_id").loc[cf.index]
    s13 = pd.read_parquet(case_dir / "Table_S13_local_pair_vot_case.parquet")
    s13 = s13[s13.model == system].set_index("case_id").loc[cf.index]
    def md(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
        return float(np.nanmax(np.abs(a[m] - b[m]))) if m.any() else float("nan"), int(m.sum())
    rep = {
        "elasticity": md(cf.elasticity, s02.elasticity), "monotone_pairs": md(cf.monotone_pairs, s02.monotone_pairs),
        "raw_DF": md(cf.raw_DF, s03.raw_DF), "raw_DT": md(cf.raw_DT, s03.raw_DT), "raw_vot": md(cf.raw_vot, s03.raw_vot),
        "computable flag": (int((cf.raw_computable.to_numpy() != s03.raw_computable.to_numpy()).sum()), len(cf)),
        "loc_DF": md(cf.loc_DF, s13.loc_DF), "loc_DT": md(cf.loc_DT, s13.loc_DT), "loc_vot": md(cf.loc_vot, s13.loc_vot),
    }
    t3 = pd.read_csv(case_dir / "Table_03_direct_vot_validity.csv").set_index("Model").loc[system]
    mine = table3_summary(cf.reset_index())
    return cf, rep, {k: (float(t3[k]), mine[k]) for k in mine}, table2_case_summary(cf.reset_index()), table2_interval_summary(cf.reset_index())


if __name__ == "__main__":
    import sys
    bank = REAL / "data" / "06_predictions_llm" / "llm_scores.parquet"
    cf, rep, t3, t2c, t2i = reproduce(bank)
    print("=== reproduction gate vs SECTION4_RESULTS_CASE (Qwen 2B SFT, fixed-split bank) ===")
    for k, v in rep.items(): print("  %-16s max|diff|=%s  (n=%d)" % (k, ("%.3e" % v[0]) if k != "computable flag" else "%d mismatches" % v[0], v[1]))
    print("--- Table 3: CASE csv vs mine ---")
    for k, (a, b) in t3.items(): print("  %-30s CASE %9.4f   mine %9.4f   diff %.2e" % (k, a, b, abs(a - b)))
    print("--- Table 2 case grain (CASE csv: mean -0.5104 SD 0.3903 median -0.4233 Q1 -0.7251 Q3 -0.2313 mono 66.6042) ---")
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t2c.items()})
    print("--- Table 2 interval grain (paper tex: N 20,808 mean -0.510 SD 1.014 median -0.205 Q1 -0.882 Q3 0.000 neg 66.60) ---")
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t2i.items()})
