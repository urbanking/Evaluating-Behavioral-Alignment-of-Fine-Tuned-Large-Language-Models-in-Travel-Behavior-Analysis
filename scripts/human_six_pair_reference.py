# -*- coding: utf-8 -*-
"""Human Behavioral Reference under the six-pair operator (CASE lineage), reproduced.

Fits the same development-only joint-interaction Logit the 0826 support module fits
(`_build_human_heterogeneous_reference`), scores the 3,468 test cases on ALL seven Fare and seven
IVT rungs (logit-linear in the physical edit), then applies the six-pair operator. The gate compares
against SECTION4_RESULTS_CASE Human rows (S02 elasticity; S03 raw_DF/raw_DT/raw_vot) and the
Table 2/3 Human summaries (mean elasticity -0.8433; aggregate VOT 8.7784).

Also writes the Human 7-rung probability matrices so the CASE-grain package builder can reuse them.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.special import expit

TOOLS = Path(r"C:\SP_LLM\TRC\Real_exp\tools"); sys.path.insert(0, str(TOOLS))
sys.path.insert(0, r"C:\SP_LLM\TRC\scripts")
import section4_results_final_support as S            # noqa: E402  (imports only; OUT.mkdir side effect is harmless)
from six_pair_operator import LADDER, XCOL, six_pairs, RAW_FARE_THRESHOLD   # noqa: E402

REAL = Path(r"C:\SP_LLM\TRC\Real_exp")
CASE = REAL / "pdf" / "SECTION4_RESULTS_CASE"


def human_rung_probabilities():
    spec = json.loads(S.PATH_HUMAN_SPEC.read_text(encoding="utf-8"))["behavioral_reference_specification"]
    panel = pd.read_parquet(S.PATH_PANEL)
    panel["case_id"] = panel.respondent_id.astype(str) + "_" + panel.distance_band.astype(str)
    train_cases, train_base, train_y = S._load_human_partition(S.PATH_TRAIN, spec, panel)
    test_cases, test_base, _ = S._load_human_partition(S.PATH_TEST, spec, panel)
    assert train_cases.respondent_id.nunique() == 1518 and test_cases.respondent_id.nunique() == 651
    train_design = S._add_joint_group_interactions(train_cases, train_base)
    test_design = S._add_joint_group_interactions(test_cases, test_base)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fit = sm.Logit(train_y, train_design).fit(disp=0, maxiter=600, cov_type="cluster",
                                                  cov_kwds={"groups": train_cases.respondent_id.to_numpy()})
    assert bool(fit.mle_retvals.get("converged", False))
    base_lp = test_design.to_numpy(float) @ fit.params.to_numpy(float)
    ind = pd.DataFrame({"case_id": test_cases.case_id.astype(str).to_numpy(),
                        "respondent_id": test_cases.respondent_id.to_numpy(),
                        "distance_band": test_cases.distance_band.astype(str).to_numpy(),
                        "age50": test_cases.age.ge(50).astype(float).to_numpy(),
                        "female": test_cases.gender_label.eq("Female").astype(float).to_numpy(),
                        "risk": test_cases.BR_label.eq("RISK").astype(float).to_numpy(),
                        "base_lp": base_lp})
    bf = np.full(len(ind), float(fit.params["fare_AV"])); bt = np.full(len(ind), float(fit.params["ivt_AV"]))
    for name in ["age50", "female", "risk"]:
        bf = bf + ind[name].to_numpy() * float(fit.params[f"fare_AV_x_{name}"])
        bt = bt + ind[name].to_numpy() * float(fit.params[f"ivt_AV_x_{name}"])
    grid = pd.read_parquet(S.PATH_GRID); grid["case_id"] = grid.case_id.astype(str)
    out = {}
    for attr, beta in (("Fare", bf), ("IVT", bt)):
        col = XCOL[attr]; sids = LADDER[attr]
        X = grid[grid.scenario_id.isin(sids)].pivot(index="case_id", columns="scenario_id", values=col)[sids].loc[ind.case_id].to_numpy(float)
        base_x = X[:, 3:4]
        P = expit(ind.base_lp.to_numpy(float)[:, None] + beta[:, None] * (X - base_x))
        out[attr] = (P, X)
    return ind, out, fit


def main():
    ind, rungs, fit = human_rung_probabilities()
    Pf, Xf = rungs["Fare"]; Pt, Xt = rungs["IVT"]
    Ef, Df, monof = six_pairs(Pf, Xf); Et, Dt, _ = six_pairs(Pt, Xt)
    raw_DF, raw_DT = Df.mean(1), Dt.mean(1)
    finite = np.isfinite(raw_DF) & np.isfinite(raw_DT); comp = finite & (np.abs(raw_DF) >= RAW_FARE_THRESHOLD)
    vot = np.where(comp, 60 * raw_DT / raw_DF, np.nan)
    mine = pd.DataFrame({"case_id": ind.case_id, "elasticity": Ef.mean(1), "monotone_pairs": monof.sum(1),
                         "raw_DF": raw_DF, "raw_DT": raw_DT, "raw_vot": vot}).set_index("case_id")
    s02 = pd.read_parquet(CASE / "Table_S02_direct_fare_elasticity_respondent.parquet"); s02 = s02[s02.system == "Human reference"].set_index("case_id").loc[mine.index]
    s03 = pd.read_parquet(CASE / "Table_S03_direct_vot_respondent.parquet"); s03 = s03[s03.model == "Human same-CF reference"].set_index("case_id").loc[mine.index]
    def md(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); return float(np.abs(a[m] - b[m]).max()), int(m.sum())
    print("=== Human six-pair gate vs CASE ===")
    for k, a, b in [("elasticity", mine.elasticity, s02.elasticity), ("monotone_pairs", mine.monotone_pairs, s02.monotone_pairs),
                    ("raw_DF", mine.raw_DF, s03.raw_DF), ("raw_DT", mine.raw_DT, s03.raw_DT), ("raw_vot", mine.raw_vot, s03.raw_vot)]:
        d, n = md(a, b); print("  %-15s max|diff|=%.3e (n=%d)" % (k, d, n))
    print("  Table 2 Human: mean %.4f (CASE -0.8433)  SD %.4f (0.6278)  median %.4f (-0.5436)  mono%% %.4f (100.0)"
          % (mine.elasticity.mean(), mine.elasticity.std(ddof=1), mine.elasticity.median(), 100 * mine.monotone_pairs.sum() / (6 * len(mine))))
    c = mine[comp]
    print("  Table 3 Human: aggregate %.4f (CASE 8.7784)  median %.4f (8.7316)  computable %.2f%%  Q1 %.4f Q3 %.4f"
          % (60 * mine[finite].raw_DT.mean() / mine[finite].raw_DF.mean(), c.raw_vot.median(), 100 * comp.mean(), c.raw_vot.quantile(.25), c.raw_vot.quantile(.75)))
    # save rung probabilities for the package builder
    outdir = REAL / "pdf" / "_cv5_build_cache"; outdir.mkdir(exist_ok=True)
    for attr in ("Fare", "IVT"):
        P, X = rungs[attr]
        pd.DataFrame(P, index=ind.case_id, columns=LADDER[attr]).to_parquet(outdir / f"human_rung_p_{attr}.parquet")
        pd.DataFrame(X, index=ind.case_id, columns=LADDER[attr]).to_parquet(outdir / f"human_rung_x_{attr}.parquet")
    ind.to_parquet(outdir / "human_indicator.parquet", index=False)
    print("  saved Human rung probabilities to", outdir)


if __name__ == "__main__":
    main()
