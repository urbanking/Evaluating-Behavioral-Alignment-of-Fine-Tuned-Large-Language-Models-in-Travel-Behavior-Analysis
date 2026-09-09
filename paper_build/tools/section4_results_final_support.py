from __future__ import annotations

import hashlib
import json
import math
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
import statsmodels.api as sm
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"C:\SP_LLM\TRC\Real_exp")
FINAL = ROOT / "FINAL_CLASS_BALANCED_RAW_20260826"
OUT = ROOT / "SECTION4_RESULTS_NOTEBOOK_20260826"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data"
PATH_TEST = DATA / "03_model_inputs" / "HUMAN_retro_test.parquet"
PATH_TRAIN = DATA / "03_model_inputs" / "HUMAN_retro_development.parquet"
PATH_PANEL = DATA / "02_panel" / "paired_2026_2030_model_input.parquet"
PATH_GRID = DATA / "05_scenarios" / "test_scenario_grid.parquet"
PATH_LLM = DATA / "06_predictions_llm" / "llm_scores.parquet"
PATH_MASTER = ROOT.parent / "data" / "processed" / "respondent_master.parquet"
PATH_HUMAN_SPEC = ROOT / "tables" / "pre_exp_reference_summary.json"

QWEN_KEYS = {
    "qwen_mid_zeroshot": "Qwen 2B ZS",
    "qwen_mid_sft": "Qwen 2B SFT",
    "qwen_zeroshot": "Qwen 9B ZS",
    "qwen_sft": "Qwen 9B SFT",
}
QWEN_ORDER = list(QWEN_KEYS.values())
MODEL_ORDER = [
    "Panel Logit", "SVM", "Random Forest", "XGBoost", "FFNN", "DNN",
    "Qwen 2B ZS", "Qwen 2B SFT", "Qwen 9B ZS", "Qwen 9B SFT",
]
COLORS = {
    "Panel Logit": "#2F5D7E",
    "SVM": "#D97706",
    "Random Forest": "#3F8F4F",
    "XGBoost": "#C84C4C",
    "FFNN": "#B8861B",
    "DNN": "#76539B",
    "Qwen 2B ZS": "#DB5A91",
    "Qwen 2B SFT": "#C93F7C",
    "Qwen 9B ZS": "#168C97",
    "Qwen 9B SFT": "#007F8B",
}
FAMILY_COLOR = {"DCM/ML": "#2F5D7E", "Neural": "#B8861B", "LLM": "#A94442"}
HUMAN_COLOR = "#2B2B2B"
QWEN_STYLE = {
    "Qwen 2B ZS": dict(color="#DB5A91", marker="o", linestyle="--"),
    "Qwen 2B SFT": dict(color="#C93F7C", marker="o", linestyle="-"),
    "Qwen 9B ZS": dict(color="#168C97", marker="s", linestyle="--"),
    "Qwen 9B SFT": dict(color="#007F8B", marker="s", linestyle="-"),
}
ATTRIBUTE_SCENARIOS = {
    "Fare": ["F_M3", "F_M2", "F_M1", "BASE", "F_P1", "F_P2", "F_P3"],
    "IVT": ["R_M3", "R_M2", "R_M1", "BASE", "R_P1", "R_P2", "R_P3"],
    "Wait": ["W_M3", "W_M2", "W_M1", "BASE", "W_P1", "W_P2", "W_P3"],
}
PHYSICAL_COLUMN = {"Fare": "fare_cf", "IVT": "ride_cf", "Wait": "wait_cf"}
X_LABEL = {
    "Fare": "Mean physical AV Fare (kKRW)",
    "IVT": "Mean physical AV IVT (minutes)",
    "Wait": "Mean physical AV Wait/access time (minutes)",
}
GROUP_SPECS = {
    "age_group": {
        "title": "Age group", "order": ["Age <50", "Age 50+"],
        "note": "Between-person age groups are descriptive, not causal.",
    },
    "sex_group": {
        "title": "Sex", "order": ["Male", "Female"],
        "note": "Between-person sex groups are descriptive, not causal.",
    },
    "framing_group": {
        "title": "Survey framing", "order": ["BASIC", "RISK"],
        "note": "BASIC/RISK denotes the assigned survey information condition.",
    },
}
RAW_FARE_THRESHOLD = 1e-4
EPS = 1e-9


pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda value: f"{value:,.4g}")
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


def _save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return png, pdf


def _style_boxplot(ax, values: np.ndarray, position: int, color: str) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    result = ax.boxplot(
        [values], positions=[position], vert=False, widths=0.56,
        whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
        boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.23, "linewidth": 1.2},
        whiskerprops={"color": color, "alpha": 0.65, "linewidth": 1.0},
        capprops={"color": color, "alpha": 0.65, "linewidth": 1.0},
        medianprops={"color": color, "linewidth": 1.5},
    )
    for patch in result["boxes"]:
        patch.set_zorder(2)


def _percentile_summary(frame: pd.DataFrame, model: str, group: str, value: str) -> pd.DataFrame:
    return (
        frame.groupby([model, group], observed=True)[value]
        .agg(
            n="count", mean="mean", median="median",
            q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75),
            p05=lambda x: x.quantile(0.05), p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )


def _model_family(model: str) -> str:
    if model in {"Panel Logit", "SVM", "Random Forest", "XGBoost"}:
        return "DCM/ML"
    if model in {"FFNN", "DNN"}:
        return "Neural"
    return "LLM"


def prepare_analysis() -> dict[str, object]:
    for path in [PATH_TEST, PATH_TRAIN, PATH_PANEL, PATH_GRID, PATH_LLM, PATH_MASTER, PATH_HUMAN_SPEC]:
        assert path.exists(), path

    test = pd.read_parquet(PATH_TEST, columns=["case_id", "respondent_id", "y", "SQ2", "SQ3"])
    test["case_id"] = test.case_id.astype(str)
    assert len(test) == 3468 and test.case_id.is_unique
    assert test.respondent_id.nunique() == 651
    case_order = test.case_id.tolist()

    master = pd.read_parquet(
        PATH_MASTER, columns=["IDX", "BR", "BR_label", "SQ2", "gender_label", "SQ3"]
    )
    meta = (
        test[["respondent_id", "SQ2", "SQ3"]].drop_duplicates("respondent_id")
        .merge(master, left_on="respondent_id", right_on="IDX", how="left",
               validate="one_to_one", suffixes=("_test", "_master"))
    )
    assert len(meta) == 651 and meta.BR_label.notna().all()
    assert (meta.SQ2_test == meta.SQ2_master).all()
    assert (meta.SQ3_test == meta.SQ3_master).all()
    meta["age_group"] = np.where(meta.SQ3_test.ge(50), "Age 50+", "Age <50")
    meta["sex_group"] = meta.gender_label.astype(str)
    meta["framing_group"] = meta.BR_label.astype(str)
    respondent_meta = meta[["respondent_id", "age_group", "sex_group", "framing_group"]].copy()
    expected_counts = {
        "age_group": {"Age <50": 383, "Age 50+": 268},
        "sex_group": {"Male": 309, "Female": 342},
        "framing_group": {"BASIC": 326, "RISK": 325},
    }
    for column, counts in expected_counts.items():
        assert respondent_meta.groupby(column).size().to_dict() == counts

    grid = pd.read_parquet(PATH_GRID)
    grid["case_id"] = grid.case_id.astype(str)
    scenarios = ["BASE"] + [s for values in ATTRIBUTE_SCENARIOS.values() for s in values if s != "BASE"]
    scenarios = list(dict.fromkeys(scenarios))
    grid = grid.loc[grid.scenario_id.isin(scenarios)].copy()
    assert grid.loc[grid.scenario_id.eq("BASE")].case_id.nunique() == 3468

    scores = pd.read_parquet(
        PATH_LLM,
        columns=["case_id", "scenario_id", "model", "margin", "free_argmax"],
    )
    scores["case_id"] = scores.case_id.astype(str)
    scores = scores.loc[
        scores.model.isin(QWEN_KEYS)
        & scores.scenario_id.isin(scenarios)
    ].copy()
    scores["system"] = scores.model.map(QWEN_KEYS)
    scores["p"] = expit(scores.margin.to_numpy(float))
    assert scores.p.between(0, 1).all()
    assert len(scores) == len(QWEN_ORDER) * len(scenarios) * 3468
    assert scores.groupby(["system", "scenario_id"]).size().eq(3468).all()

    response = _build_response_curves(scores, grid, test)
    elasticity = _build_elasticity_respondents(scores, grid, test)
    vot = _build_vot_respondents(scores, grid, test)
    elasticity_case, derivative_case = _build_case_regression_frames(scores, grid, test)
    human_heterogeneity = _build_human_heterogeneous_reference(grid)
    prediction = _build_prediction_table(scores, test)

    response.to_csv(OUT / "Table_S01_probability_response_curves.csv", index=False)
    elasticity.to_parquet(OUT / "Table_S02_direct_fare_elasticity_respondent.parquet", index=False)
    vot.to_parquet(OUT / "Table_S03_direct_vot_respondent.parquet", index=False)
    prediction.to_csv(OUT / "Table_01_prediction_performance.csv", index=False)

    return {
        "test": test,
        "grid": grid,
        "scores": scores,
        "respondent_meta": respondent_meta,
        "response": response,
        "elasticity": elasticity,
        "vot": vot,
        "elasticity_case": elasticity_case,
        "derivative_case": derivative_case,
        "human_hetero_elasticity": human_heterogeneity["elasticity_respondent"],
        "human_hetero_vot": human_heterogeneity["vot_respondent"],
        "human_hetero_elasticity_case": human_heterogeneity["elasticity_case"],
        "human_hetero_derivative_case": human_heterogeneity["derivative_case"],
        "human_hetero_summary": human_heterogeneity["summary"],
        "prediction": prediction,
        "case_order": case_order,
    }


def _build_prediction_table(scores: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(FINAL / "01_EXP1" / "table1_predictive_performance.csv")
    labels = test.set_index("case_id").y.astype(int)
    for system in QWEN_ORDER:
        frame = (
            scores.loc[scores.system.eq(system) & scores.scenario_id.eq("BASE")]
            .set_index("case_id").loc[labels.index]
        )
        probability = frame.p.clip(EPS, 1 - EPS).to_numpy(float)
        hard = np.where(
            frame.margin.to_numpy(float) > 0, 1,
            np.where(frame.margin.to_numpy(float) < 0, 0,
                     frame.free_argmax.astype(str).str.upper().eq("AV").astype(int).to_numpy())
        )
        mask = table.Model.eq(system)
        table.loc[mask, "Accuracy"] = accuracy_score(labels, hard)
        table.loc[mask, "AV-F1"] = f1_score(labels, hard, pos_label=1)
        table.loc[mask, "Macro-F1"] = f1_score(labels, hard, average="macro")
        table.loc[mask, "Log Loss"] = log_loss(labels, probability, labels=[0, 1])
    table["order"] = table.Model.map({model: i for i, model in enumerate(MODEL_ORDER)})
    table = table.sort_values("order").drop(columns="order").reset_index(drop=True)
    return table


def _build_response_curves(scores: pd.DataFrame, grid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    old_parts = [
        pd.read_csv(FINAL / "02_EXP2" / "tableS_fare_response_curve.csv"),
        pd.read_csv(FINAL / "02_EXP2" / "tableS_ivt_response_curve.csv"),
        pd.read_csv(FINAL / "02_EXP2" / "tableS_wait_common_support_response.csv"),
    ]
    old = pd.concat(old_parts, ignore_index=True)
    old = old.loc[~old.system.isin(QWEN_ORDER)].copy()
    base = scores.loc[scores.scenario_id.eq("BASE"), ["case_id", "system", "p"]].rename(
        columns={"p": "p_base"}
    )
    test_meta = test[["case_id", "respondent_id"]]
    rows: list[dict[str, object]] = []
    wait_grid = grid.loc[grid.scenario_id.isin(ATTRIBUTE_SCENARIOS["Wait"])].copy()
    wait_common = (
        wait_grid.groupby("case_id").physically_valid.all()
    )
    wait_common_ids = set(wait_common.loc[wait_common].index.astype(str))
    assert len(wait_common_ids) == 1631
    for attribute, scenarios in ATTRIBUTE_SCENARIOS.items():
        physical_column = PHYSICAL_COLUMN[attribute]
        for scenario in scenarios:
            scenario_scores = scores.loc[
                scores.scenario_id.eq(scenario), ["case_id", "system", "p"]
            ].copy()
            scenario_scores = scenario_scores.merge(base, on=["case_id", "system"], validate="one_to_one")
            g = grid.loc[grid.scenario_id.eq(scenario), [
                "case_id", "edit_amount", "within_support", "physically_valid", physical_column,
            ]].rename(columns={physical_column: "physical_value"})
            scenario_scores = (
                scenario_scores.merge(test_meta, on="case_id", validate="many_to_one")
                .merge(g, on="case_id", validate="many_to_one")
            )
            if attribute == "Wait":
                scenario_scores = scenario_scores.loc[scenario_scores.case_id.isin(wait_common_ids)].copy()
            else:
                scenario_scores = scenario_scores.loc[scenario_scores.physically_valid].copy()
            scenario_scores["delta_p"] = scenario_scores.p - scenario_scores.p_base
            respondent = (
                scenario_scores.groupby(["system", "respondent_id"], as_index=False)
                .agg(
                    mean_p=("p", "mean"), mean_p_base=("p_base", "mean"),
                    mean_delta=("delta_p", "mean"), mean_edit=("edit_amount", "mean"),
                    mean_physical=("physical_value", "mean"),
                    support=("within_support", "mean"),
                )
            )
            rung = 0 if scenario == "BASE" else int(scenario.split("_")[1].replace("M", "-").replace("P", ""))
            for system, part in respondent.groupby("system", observed=True):
                rows.append({
                    "system": system, "family": "LLM", "attribute": attribute,
                    "scenario_id": scenario, "rung": rung,
                    "n_respondents": int(part.respondent_id.nunique()),
                    "mean_p": float(part.mean_p.mean()),
                    "mean_p_base_matched": float(part.mean_p_base.mean()),
                    "mean_delta_p": float(part.mean_delta.mean()),
                    "mean_edit_amount": float(part.mean_edit.mean()),
                    "mean_physical_value": float(part.mean_physical.mean()),
                    "mean_within_support_rate": float(part.support.mean()),
                })
    qwen = pd.DataFrame(rows)
    combined = pd.concat([old, qwen], ignore_index=True)
    combined["model_order"] = combined.system.map(
        {"Human reference": -1, **{model: i for i, model in enumerate(MODEL_ORDER)}}
    )
    return combined.sort_values(["attribute", "model_order", "rung"]).drop(columns="model_order")


def _build_elasticity_respondents(scores: pd.DataFrame, grid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_parquet(FINAL / "02_EXP2" / "tableS_overall_fare_elasticity_respondent.parquet")
    existing = existing.loc[~existing.system.isin(QWEN_ORDER)].copy()
    pivot = scores.loc[scores.scenario_id.isin(["BASE", "F_P1"])].pivot(
        index=["system", "case_id"], columns="scenario_id", values="p"
    ).reset_index()
    base_grid = grid.loc[grid.scenario_id.eq("BASE"), ["case_id", "fare_cf"]].rename(columns={"fare_cf": "fare_base"})
    p1_grid = grid.loc[grid.scenario_id.eq("F_P1"), ["case_id", "fare_cf", "physically_valid"]].rename(
        columns={"fare_cf": "fare_p1"}
    )
    pivot = (
        pivot.merge(test[["case_id", "respondent_id"]], on="case_id", validate="many_to_one")
        .merge(base_grid, on="case_id", validate="many_to_one")
        .merge(p1_grid, on="case_id", validate="many_to_one")
    )
    pivot = pivot.loc[pivot.physically_valid].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pivot["elasticity"] = ((pivot.F_P1 - pivot.BASE) / ((pivot.F_P1 + pivot.BASE) / 2)) / (
            (pivot.fare_p1 - pivot.fare_base) / ((pivot.fare_p1 + pivot.fare_base) / 2)
        )
    pivot["delta_p"] = pivot.F_P1 - pivot.BASE
    qwen = (
        pivot.groupby(["system", "respondent_id"], as_index=False)
        .agg(elasticity=("elasticity", "mean"), delta_p=("delta_p", "mean"),
             n_distance_tasks=("case_id", "nunique"))
    )
    qwen["family"] = "LLM"
    qwen["n_seed_values"] = 1
    qwen = qwen[["system", "family", "respondent_id", "elasticity", "delta_p",
                 "n_distance_tasks", "n_seed_values"]]
    out = pd.concat([existing, qwen], ignore_index=True)
    assert out.groupby("system").respondent_id.nunique().eq(651).all()
    return out


def _build_vot_respondents(scores: pd.DataFrame, grid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_parquet(FINAL / "03_EXP3" / "direct_cf_overall_respondent_vot.parquet")
    existing = existing.loc[~existing.model.isin(QWEN_ORDER)].copy()
    ids = ["F_M1", "F_P1", "R_M1", "R_P1"]
    pivot = scores.loc[scores.scenario_id.isin(ids)].pivot(
        index=["system", "case_id"], columns="scenario_id", values="p"
    ).reset_index()
    grid_wide = grid.loc[grid.scenario_id.isin(ids)].pivot(
        index="case_id", columns="scenario_id", values=["fare_cf", "ride_cf"]
    )
    fare_den = grid_wide["fare_cf"]["F_P1"] - grid_wide["fare_cf"]["F_M1"]
    time_den = grid_wide["ride_cf"]["R_P1"] - grid_wide["ride_cf"]["R_M1"]
    pivot = pivot.merge(test[["case_id", "respondent_id"]], on="case_id", validate="many_to_one")
    pivot["raw_DF"] = (pivot.F_P1 - pivot.F_M1) / pivot.case_id.map(fare_den)
    pivot["raw_DT"] = (pivot.R_P1 - pivot.R_M1) / pivot.case_id.map(time_den)
    respondent = (
        pivot.groupby(["system", "respondent_id"], as_index=False)
        .agg(raw_DF=("raw_DF", "mean"), raw_DT=("raw_DT", "mean"), n_tasks=("case_id", "nunique"))
    )
    respondent["raw_finite"] = np.isfinite(respondent.raw_DF) & np.isfinite(respondent.raw_DT)
    respondent["raw_near_zero_fare"] = respondent.raw_finite & respondent.raw_DF.abs().lt(RAW_FARE_THRESHOLD)
    respondent["raw_computable"] = respondent.raw_finite & ~respondent.raw_near_zero_fare
    respondent["raw_fare_negative"] = respondent.raw_computable & respondent.raw_DF.lt(0)
    respondent["raw_ivt_negative"] = respondent.raw_computable & respondent.raw_DT.lt(0)
    respondent["raw_joint_conventional"] = (
        respondent.raw_computable & respondent.raw_DF.lt(0) & respondent.raw_DT.lt(0)
    )
    respondent["raw_vot"] = np.where(
        respondent.raw_computable, 60 * respondent.raw_DT / respondent.raw_DF, np.nan
    )
    respondent["raw_negative_vot"] = respondent.raw_computable & respondent.raw_vot.lt(0)
    respondent["raw_positive_vot"] = respondent.raw_computable & respondent.raw_vot.gt(0)
    respondent["model"] = respondent.pop("system")
    respondent["family"] = "LLM"
    for column in [
        "logit_DF", "logit_DT", "logit_finite", "logit_near_zero_fare",
        "logit_computable", "logit_fare_negative", "logit_ivt_negative",
        "logit_joint_conventional", "logit_vot", "logit_negative_vot",
        "logit_positive_vot",
    ]:
        respondent[column] = np.nan
    respondent = respondent[existing.columns]
    out = pd.concat([existing, respondent], ignore_index=True)
    assert out.groupby("model").respondent_id.nunique().eq(651).all()
    return out


def _build_case_regression_frames(
    scores: pd.DataFrame, grid: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_elasticity = pd.read_parquet(
        FINAL / "02_EXP2" / "tableS_distance_respondent_response.parquet"
    )
    existing_elasticity = existing_elasticity.loc[
        existing_elasticity.attribute.eq("Fare")
        & existing_elasticity.support_rule.eq("all_physically_valid")
        & ~existing_elasticity.system.isin(QWEN_ORDER)
    ].copy()
    existing_elasticity["case_id"] = (
        existing_elasticity.respondent_id.astype(str)
        + "_" + existing_elasticity.distance_band.astype(str)
    )
    existing_elasticity = existing_elasticity[[
        "system", "family", "case_id", "respondent_id", "distance_band", "elasticity"
    ]]

    probability = scores.loc[scores.scenario_id.isin(["BASE", "F_P1"])].pivot(
        index=["system", "case_id"], columns="scenario_id", values="p"
    ).reset_index()
    base_grid = grid.loc[grid.scenario_id.eq("BASE"), ["case_id", "fare_cf"]].rename(
        columns={"fare_cf": "fare_base"}
    )
    p1_grid = grid.loc[grid.scenario_id.eq("F_P1"), ["case_id", "fare_cf", "physically_valid"]].rename(
        columns={"fare_cf": "fare_p1"}
    )
    qwen_elasticity = (
        probability.merge(test[["case_id", "respondent_id"]], on="case_id", validate="many_to_one")
        .merge(base_grid, on="case_id", validate="many_to_one")
        .merge(p1_grid, on="case_id", validate="many_to_one")
    )
    qwen_elasticity = qwen_elasticity.loc[qwen_elasticity.physically_valid].copy()
    qwen_elasticity["distance_band"] = qwen_elasticity.case_id.str.rsplit("_", n=1).str[-1]
    qwen_elasticity["elasticity"] = (
        ((qwen_elasticity.F_P1 - qwen_elasticity.BASE) / ((qwen_elasticity.F_P1 + qwen_elasticity.BASE) / 2))
        / ((qwen_elasticity.fare_p1 - qwen_elasticity.fare_base) / ((qwen_elasticity.fare_p1 + qwen_elasticity.fare_base) / 2))
    )
    qwen_elasticity["family"] = "LLM"
    qwen_elasticity = qwen_elasticity[[
        "system", "family", "case_id", "respondent_id", "distance_band", "elasticity"
    ]]
    elasticity_case = pd.concat([existing_elasticity, qwen_elasticity], ignore_index=True)
    assert elasticity_case.groupby("system").size().eq(3468).all()

    existing_derivative = pd.read_parquet(FINAL / "03_EXP3" / "direct_cf_case_derivatives.parquet")
    existing_derivative = existing_derivative.loc[~existing_derivative.model.isin(QWEN_ORDER)].copy()
    central = ["F_M1", "F_P1", "R_M1", "R_P1"]
    qwen_pivot = scores.loc[scores.scenario_id.isin(central)].pivot(
        index=["system", "case_id"], columns="scenario_id", values="p"
    ).reset_index()
    grid_wide = grid.loc[grid.scenario_id.isin(central)].pivot(
        index="case_id", columns="scenario_id", values=["fare_cf", "ride_cf"]
    )
    fare_den = grid_wide["fare_cf"]["F_P1"] - grid_wide["fare_cf"]["F_M1"]
    time_den = grid_wide["ride_cf"]["R_P1"] - grid_wide["ride_cf"]["R_M1"]
    qwen_derivative = qwen_pivot.merge(
        test[["case_id", "respondent_id"]], on="case_id", validate="many_to_one"
    )
    qwen_derivative["distance_band"] = qwen_derivative.case_id.str.rsplit("_", n=1).str[-1]
    qwen_derivative["raw_DF"] = (
        (qwen_derivative.F_P1 - qwen_derivative.F_M1) / qwen_derivative.case_id.map(fare_den)
    )
    qwen_derivative["raw_DT"] = (
        (qwen_derivative.R_P1 - qwen_derivative.R_M1) / qwen_derivative.case_id.map(time_den)
    )
    qwen_derivative["model"] = qwen_derivative.pop("system")
    qwen_derivative["family"] = "LLM"
    qwen_derivative["logit_DF"] = np.nan
    qwen_derivative["logit_DT"] = np.nan
    qwen_derivative = qwen_derivative[existing_derivative.columns]
    derivative_case = pd.concat([existing_derivative, qwen_derivative], ignore_index=True)
    assert derivative_case.groupby("model").size().eq(3468).all()
    return elasticity_case, derivative_case


def _level_text(value) -> str:
    if pd.isna(value):
        return "__missing__"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _prepare_human_cases(cases: pd.DataFrame, specification: dict[str, object]) -> pd.DataFrame:
    cases = cases.copy().reset_index(drop=True)
    cases["fare_AV"] = cases.av_cost_1000won.astype(float)
    cases["ivt_AV"] = cases.av_travel_time_min.astype(float)
    cases["wait_AV"] = cases.av_wait_time_min.astype(float)
    for target, part in [("fare_KEEP", "fare"), ("ivt_KEEP", "ivt"), ("wait_KEEP", "wait")]:
        cases[target] = np.nan
        for mode, mapping in specification["keep_los_construction"].items():
            mask = cases.current_mode_2026.eq(mode)
            cases.loc[mask, target] = cases.loc[mask, mapping[part]].to_numpy(float)
        cases[target + "_neg"] = -cases[target]
    return cases


def _build_human_base_design(cases: pd.DataFrame, specification: dict[str, object]) -> pd.DataFrame:
    columns = list(specification["design_columns"])
    design = pd.DataFrame(index=cases.index)
    design["const"] = 1.0
    for column in specification["numeric_terms"]:
        design[column] = pd.to_numeric(cases[column], errors="raise").astype(float)
    for term in columns:
        if "[" in term:
            variable, level = term[:-1].split("[", 1)
            design[term] = cases[variable].map(_level_text).eq(level).astype(float)
    return design[columns].astype(float)


def _load_human_partition(
    profile_path: Path, specification: dict[str, object], panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    profiles = pd.read_parquet(profile_path)
    order = profiles.case_id.astype(str).tolist()
    cases = _prepare_human_cases(panel.set_index("case_id").loc[order].reset_index(), specification)
    design = _build_human_base_design(cases, specification)
    y = profiles.set_index("case_id").loc[order, "y"].astype(int).to_numpy()
    return cases, design, y


def _add_joint_group_interactions(cases: pd.DataFrame, base_design: pd.DataFrame) -> pd.DataFrame:
    design = base_design.reset_index(drop=True).copy()
    indicators = {
        "age50": cases.age.ge(50).astype(float).to_numpy(),
        "female": cases.gender_label.eq("Female").astype(float).to_numpy(),
        "risk": cases.BR_label.eq("RISK").astype(float).to_numpy(),
    }
    extra = []
    for name, values in indicators.items():
        main = f"{name}_main"
        fare = f"fare_AV_x_{name}"
        ivt = f"ivt_AV_x_{name}"
        design[main] = values
        design[fare] = design.fare_AV.to_numpy(float) * values
        design[ivt] = design.ivt_AV.to_numpy(float) * values
        extra.extend([main, fare, ivt])
    output = design[list(base_design.columns) + extra].astype(float)
    assert np.linalg.matrix_rank(output.to_numpy()) == output.shape[1]
    return output


def _build_human_heterogeneous_reference(grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    specification = json.loads(PATH_HUMAN_SPEC.read_text(encoding="utf-8"))[
        "behavioral_reference_specification"
    ]
    panel = pd.read_parquet(PATH_PANEL)
    panel["case_id"] = panel.respondent_id.astype(str) + "_" + panel.distance_band.astype(str)
    train_cases, train_base, train_y = _load_human_partition(PATH_TRAIN, specification, panel)
    test_cases, test_base, _ = _load_human_partition(PATH_TEST, specification, panel)
    assert train_cases.respondent_id.nunique() == 1518
    assert test_cases.respondent_id.nunique() == 651
    train_design = _add_joint_group_interactions(train_cases, train_base)
    test_design = _add_joint_group_interactions(test_cases, test_base)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fit = sm.Logit(train_y, train_design).fit(
            disp=0, maxiter=600, cov_type="cluster",
            cov_kwds={"groups": train_cases.respondent_id.to_numpy()},
        )
    assert bool(fit.mle_retvals.get("converged", False))

    base_lp = test_design.to_numpy(float) @ fit.params.to_numpy(float)
    indicator = pd.DataFrame({
        "case_id": test_cases.case_id.astype(str),
        "respondent_id": test_cases.respondent_id.to_numpy(),
        "distance_band": test_cases.distance_band.astype(str).to_numpy(),
        "age_group": np.where(test_cases.age.ge(50), "Age 50+", "Age <50"),
        "sex_group": test_cases.gender_label.astype(str).to_numpy(),
        "framing_group": test_cases.BR_label.astype(str).to_numpy(),
        "age50": test_cases.age.ge(50).astype(float).to_numpy(),
        "female": test_cases.gender_label.eq("Female").astype(float).to_numpy(),
        "risk": test_cases.BR_label.eq("RISK").astype(float).to_numpy(),
        "base_lp": base_lp,
    })
    indicator["beta_fare"] = float(fit.params["fare_AV"])
    indicator["beta_ivt"] = float(fit.params["ivt_AV"])
    for name in ["age50", "female", "risk"]:
        indicator["beta_fare"] += indicator[name] * float(fit.params[f"fare_AV_x_{name}"])
        indicator["beta_ivt"] += indicator[name] * float(fit.params[f"ivt_AV_x_{name}"])

    scenario_ids = ["BASE", "F_M1", "F_P1", "R_M1", "R_P1"]
    blocks = {
        scenario: grid.loc[grid.scenario_id.eq(scenario)].set_index("case_id").loc[indicator.case_id]
        for scenario in scenario_ids
    }
    probability = indicator[["case_id", "respondent_id", "distance_band", "age_group", "sex_group", "framing_group"]].copy()
    probability["BASE"] = expit(indicator.base_lp.to_numpy(float))
    for scenario in ["F_M1", "F_P1"]:
        edit = blocks[scenario].fare_cf.to_numpy(float) - blocks["BASE"].fare_cf.to_numpy(float)
        probability[scenario] = expit(indicator.base_lp + indicator.beta_fare * edit)
    for scenario in ["R_M1", "R_P1"]:
        edit = blocks[scenario].ride_cf.to_numpy(float) - blocks["BASE"].ride_cf.to_numpy(float)
        probability[scenario] = expit(indicator.base_lp + indicator.beta_ivt * edit)

    fare_base = blocks["BASE"].fare_cf.to_numpy(float)
    fare_p1 = blocks["F_P1"].fare_cf.to_numpy(float)
    probability["elasticity"] = (
        ((probability.F_P1 - probability.BASE) / ((probability.F_P1 + probability.BASE) / 2))
        / ((fare_p1 - fare_base) / ((fare_p1 + fare_base) / 2))
    )
    fare_den = blocks["F_P1"].fare_cf.to_numpy(float) - blocks["F_M1"].fare_cf.to_numpy(float)
    time_den = blocks["R_P1"].ride_cf.to_numpy(float) - blocks["R_M1"].ride_cf.to_numpy(float)
    probability["raw_DF"] = (probability.F_P1 - probability.F_M1) / fare_den
    probability["raw_DT"] = (probability.R_P1 - probability.R_M1) / time_den

    elasticity_case = probability[[
        "case_id", "respondent_id", "distance_band", "elasticity"
    ]].copy()
    elasticity_case.insert(0, "family", "Human")
    elasticity_case.insert(0, "system", "Human reference")
    elasticity_respondent = (
        probability.groupby("respondent_id", as_index=False)
        .agg(elasticity=("elasticity", "mean"), n_distance_tasks=("case_id", "nunique"))
    )
    elasticity_respondent.insert(0, "family", "Human")
    elasticity_respondent.insert(0, "system", "Human reference")

    derivative_case = probability[[
        "case_id", "raw_DF", "raw_DT", "respondent_id", "distance_band"
    ]].copy()
    derivative_case.insert(0, "family", "Human")
    derivative_case.insert(0, "model", "Human same-CF reference")
    derivative_case["logit_DF"] = np.nan
    derivative_case["logit_DT"] = np.nan
    derivative_case = derivative_case[[
        "model", "family", "case_id", "raw_DF", "raw_DT", "logit_DF", "logit_DT",
        "respondent_id", "distance_band",
    ]]
    vot_respondent = (
        probability.groupby("respondent_id", as_index=False)
        .agg(raw_DF=("raw_DF", "mean"), raw_DT=("raw_DT", "mean"), n_tasks=("case_id", "nunique"))
    )
    vot_respondent["raw_finite"] = np.isfinite(vot_respondent.raw_DF) & np.isfinite(vot_respondent.raw_DT)
    vot_respondent["raw_near_zero_fare"] = vot_respondent.raw_finite & vot_respondent.raw_DF.abs().lt(RAW_FARE_THRESHOLD)
    vot_respondent["raw_computable"] = vot_respondent.raw_finite & ~vot_respondent.raw_near_zero_fare
    vot_respondent["raw_fare_negative"] = vot_respondent.raw_computable & vot_respondent.raw_DF.lt(0)
    vot_respondent["raw_ivt_negative"] = vot_respondent.raw_computable & vot_respondent.raw_DT.lt(0)
    vot_respondent["raw_joint_conventional"] = (
        vot_respondent.raw_computable & vot_respondent.raw_DF.lt(0) & vot_respondent.raw_DT.lt(0)
    )
    vot_respondent["raw_vot"] = np.where(
        vot_respondent.raw_computable, 60 * vot_respondent.raw_DT / vot_respondent.raw_DF, np.nan
    )
    vot_respondent["raw_negative_vot"] = vot_respondent.raw_computable & vot_respondent.raw_vot.lt(0)
    vot_respondent["raw_positive_vot"] = vot_respondent.raw_computable & vot_respondent.raw_vot.gt(0)
    vot_respondent.insert(0, "family", "Human")
    vot_respondent.insert(0, "model", "Human same-CF reference")

    coefficients = pd.DataFrame({
        "term": fit.params.index,
        "coefficient": fit.params.to_numpy(float),
        "cluster_se": fit.bse.to_numpy(float),
        "p_value": fit.pvalues.to_numpy(float),
    })
    coefficients.to_csv(OUT / "Table_S05_human_joint_heterogeneity_mnl_coefficients.csv", index=False)

    interaction_rows = []
    parameter_names = list(fit.params.index)
    for name, dimension in [("age50", "age_group"), ("female", "sex_group"), ("risk", "framing_group")]:
        fare_term = f"fare_AV_x_{name}"
        ivt_term = f"ivt_AV_x_{name}"
        restriction = np.zeros((2, len(parameter_names)))
        restriction[0, parameter_names.index(fare_term)] = 1.0
        restriction[1, parameter_names.index(ivt_term)] = 1.0
        joint = fit.wald_test(restriction, scalar=True)
        interaction_rows.append({
            "Dimension": dimension,
            "Fare interaction coefficient": float(fit.params[fare_term]),
            "Fare interaction p-value": float(fit.pvalues[fare_term]),
            "IVT interaction coefficient": float(fit.params[ivt_term]),
            "IVT interaction p-value": float(fit.pvalues[ivt_term]),
            "Joint Fare+IVT chi-square": float(joint.statistic),
            "Joint Fare+IVT df": 2,
            "Joint Fare+IVT p-value": float(joint.pvalue),
        })
    interaction_tests = pd.DataFrame(interaction_rows)
    interaction_tests.to_csv(OUT / "Table_S07_human_joint_heterogeneity_tests.csv", index=False)

    summary_rows = []
    merged_elasticity = elasticity_respondent.merge(
        indicator[["respondent_id", "age_group", "sex_group", "framing_group"]].drop_duplicates("respondent_id"),
        on="respondent_id", validate="one_to_one",
    )
    merged_vot = vot_respondent.merge(
        indicator[["respondent_id", "age_group", "sex_group", "framing_group"]].drop_duplicates("respondent_id"),
        on="respondent_id", validate="one_to_one",
    )
    for dimension, spec in GROUP_SPECS.items():
        for group in spec["order"]:
            e = merged_elasticity.loc[merged_elasticity[dimension].eq(group), "elasticity"]
            v_all = merged_vot.loc[merged_vot[dimension].eq(group)]
            v_valid = v_all.loc[v_all.raw_computable]
            summary_rows.append({
                "Dimension": dimension, "Group": group,
                "N respondents": int(e.size),
                "Mean Direct Fare elasticity": float(e.mean()),
                "Median Direct Fare elasticity": float(e.median()),
                "Population Direct VOT-eq.": float(60 * v_all.raw_DT.mean() / v_all.raw_DF.mean()),
                "Median Direct VOT-eq.": float(v_valid.raw_vot.median()),
                "Joint conventional-sign (%)": 100 * float(v_valid.raw_joint_conventional.mean()),
            })
    summary = pd.DataFrame(summary_rows).merge(
        interaction_tests[["Dimension", "Fare interaction p-value", "IVT interaction p-value", "Joint Fare+IVT p-value"]],
        on="Dimension", how="left", validate="many_to_one",
    )
    summary.to_csv(OUT / "Table_S06_human_joint_heterogeneity_direct_summary.csv", index=False)
    return {
        "elasticity_case": elasticity_case,
        "elasticity_respondent": elasticity_respondent,
        "derivative_case": derivative_case,
        "vot_respondent": vot_respondent,
        "summary": summary,
    }


def prediction_figure(table: pd.DataFrame) -> tuple[Path, Path]:
    metrics = [("AV-F1", True), ("Accuracy", True), ("Macro-F1", True), ("Log Loss", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 6.7), sharey=True)
    y = np.arange(len(MODEL_ORDER))
    for ax, (metric, higher) in zip(axes, metrics):
        values = table.set_index("Model").loc[MODEL_ORDER, metric].to_numpy(float)
        colors = [FAMILY_COLOR[_model_family(model)] for model in MODEL_ORDER]
        ax.barh(y, values, color=colors, alpha=0.88, height=0.62)
        ax.set_title(f"{metric} ({'higher' if higher else 'lower'} is better)", fontsize=10.5)
        ax.set_xlabel(metric)
        ax.axvline(0, color="#555555", linewidth=0.7)
        for yi, value in enumerate(values):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=7.6)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, MODEL_ORDER)
    axes[0].invert_yaxis()
    fig.legend(
        handles=[Patch(facecolor=color, label=family) for family, color in FAMILY_COLOR.items()],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle("Figure 1. Held-out observed-choice prediction", fontsize=14.5, y=0.995)
    fig.text(
        0.5, 0.018,
        "Qwen Log Loss uses pair-normalized raw AV-versus-existing-mode probability; hard-choice metrics use the registered decision rule.",
        ha="center", fontsize=8.2, color="#444444",
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.87, bottom=0.13, wspace=0.25)
    return _save_figure(fig, "Figure_01_prediction")


def response_figure(response: pd.DataFrame) -> tuple[Path, Path]:
    systems = ["Human reference"] + QWEN_ORDER
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.3))
    for col, attribute in enumerate(["Fare", "IVT", "Wait"]):
        data = response.loc[response.attribute.eq(attribute) & response.system.isin(systems)].copy()
        for system in systems:
            part = data.loc[data.system.eq(system)].sort_values("rung")
            style = dict(color="black", marker="*", linestyle="-", linewidth=2.3, markersize=8)
            if system != "Human reference":
                style = {**QWEN_STYLE[system], "linewidth": 1.8, "markersize": 5.5}
            axes[0, col].plot(part.mean_physical_value, part.mean_p, label=system, **style)
            axes[1, col].plot(part.mean_physical_value, part.mean_delta_p, label=system, **style)
        base = data.loc[data.rung.eq(0), "mean_physical_value"].mean()
        for row in range(2):
            axes[row, col].axvline(base, color="#777777", linestyle=":", linewidth=1.0)
            axes[row, col].grid(color="#DADADA", linewidth=0.65)
        axes[0, col].set_title(["A. Fare", "B. IVT", "C. Wait/access"][col], fontsize=11.5)
        axes[1, col].axhline(0, color="#555555", linewidth=0.8)
        axes[1, col].set_xlabel(X_LABEL[attribute])
        axes[0, col].set_ylabel("Mean P(AV)" if col == 0 else "")
        axes[1, col].set_ylabel("BASE-centered Delta P(AV)" if col == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Figure 2. Direct probability responses to Fare, IVT, and Wait/access changes", fontsize=14.5, y=0.995)
    fig.text(
        0.5, 0.018,
        "Same respondents and trips are retained; only the focal AV attribute changes. Wait/access is exploratory because the Human coefficient is unresolved.",
        ha="center", fontsize=8.2, color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.10, hspace=0.24, wspace=0.22)
    return _save_figure(fig, "Figure_02_probability_response_fare_ivt_wait")


def direct_fare_table(elasticity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    human_mean = float(elasticity.loc[elasticity.system.eq("Human reference"), "elasticity"].mean())
    order = ["Human reference"] + MODEL_ORDER
    for model in order:
        values = elasticity.loc[elasticity.system.eq(model), "elasticity"].dropna()
        rows.append({
            "Model": model,
            "N": int(values.size),
            "Negative response (%)": 100 * float(values.lt(0).mean()),
            "Mean elasticity": float(values.mean()),
            "Median elasticity": float(values.median()),
            "Q1": float(values.quantile(0.25)),
            "Q3": float(values.quantile(0.75)),
            "Absolute gap from Human mean": 0.0 if model == "Human reference" else abs(float(values.mean()) - human_mean),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "Table_02_direct_fare_response.csv", index=False)
    return frame


def subgroup_elasticity_figure(
    elasticity: pd.DataFrame,
    human_elasticity: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
    figure_number: int,
) -> tuple[Path, Path]:
    spec = GROUP_SPECS[group_col]
    frame = elasticity.loc[~elasticity.system.eq("Human reference")].merge(
        meta[["respondent_id", group_col]], on="respondent_id", validate="many_to_one"
    )
    human_frame = human_elasticity.merge(
        meta[["respondent_id", group_col]], on="respondent_id", validate="many_to_one"
    )
    finite = frame.loc[np.isfinite(frame.elasticity)].copy()
    summary = _percentile_summary(finite, "system", group_col, "elasticity")
    summary.to_csv(OUT / f"Table_S_Figure_{figure_number:02d}_{group_col}_elasticity.csv", index=False)
    low, high = float(summary.p05.min()), float(summary.p95.max())
    pad = 0.06 * max(high - low, 1.0)
    low, high = max(-12.0, low - pad), min(4.0, high + pad)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.0), sharex=True, sharey=True)
    for panel_index, (ax, group_value) in enumerate(zip(axes, spec["order"])):
        human = human_frame.loc[human_frame[group_col].eq(group_value), "elasticity"].dropna()
        hq1, hmed, hq3 = human.quantile([0.25, 0.5, 0.75])
        ax.axvspan(hq1, hq3, color="#777777", alpha=0.10, zorder=0)
        ax.axvline(hmed, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(0, color="#555555", linewidth=0.9)
        for y, model in enumerate(MODEL_ORDER):
            values = finite.loc[finite.system.eq(model) & finite[group_col].eq(group_value), "elasticity"].to_numpy(float)
            _style_boxplot(ax, values, y, COLORS[model])
            ax.scatter(np.nanmedian(values), y, s=30, color=COLORS[model], edgecolor="white", linewidth=0.55, zorder=4)
        ax.set_title(
            f"{chr(65 + panel_index)}. {group_value} (n={human.size})\nHuman Direct median = {hmed:.2f}",
            fontsize=11.0, fontweight="semibold", pad=10,
        )
        ax.set_xlim(low, high)
        ax.set_xlabel("Direct AV Fare elasticity")
        ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Patch(facecolor="#7C9DB5", edgecolor="#4D7897", alpha=0.25, label="Candidate IQR (whiskers p05-p95)"),
            Line2D([], [], marker="o", color="none", markerfacecolor="#4D7897", markeredgecolor="white", label="Candidate median"),
            Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Human subgroup median"),
            Patch(facecolor="#777777", edgecolor="none", alpha=0.10, label="Human subgroup IQR"),
        ],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(f"Figure {figure_number}. Direct Fare elasticity by {spec['title'].lower()}", fontsize=14.5, y=0.995)
    fig.text(0.5, 0.025, spec["note"], ha="center", fontsize=8.3, color="#444444")
    fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.12, wspace=0.08)
    return _save_figure(fig, f"Figure_{figure_number:02d}_fare_elasticity_{group_col}")


def _vot_summary(vot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["Human same-CF reference"] + MODEL_ORDER
    for model in order:
        all_rows = vot.loc[vot.model.eq(model)]
        valid = all_rows.loc[all_rows.raw_computable & np.isfinite(all_rows.raw_vot)]
        finite = all_rows.loc[np.isfinite(all_rows.raw_DF) & np.isfinite(all_rows.raw_DT)]
        rows.append({
            "Model": model,
            "N respondents": int(all_rows.respondent_id.nunique()),
            "Computable (%)": 100 * float(all_rows.raw_computable.mean()),
            "Fare-negative (%)": 100 * float(valid.raw_DF.lt(0).mean()),
            "IVT-negative (%)": 100 * float(valid.raw_DT.lt(0).mean()),
            "Joint conventional-sign (%)": 100 * float((valid.raw_DF.lt(0) & valid.raw_DT.lt(0)).mean()),
            "Population Direct VOT-eq.": float(60 * finite.raw_DT.mean() / finite.raw_DF.mean()),
            "Median": float(valid.raw_vot.median()),
            "Q1": float(valid.raw_vot.quantile(0.25)),
            "Q3": float(valid.raw_vot.quantile(0.75)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "Table_03_direct_vot_validity.csv", index=False)
    return frame


def overall_vot_figure(vot: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, Path]:
    order = ["Human same-CF reference"] + MODEL_ORDER
    color = {"Human same-CF reference": "#7A3E1D", **COLORS}
    valid = vot.loc[vot.raw_computable & np.isfinite(vot.raw_vot)].copy()
    q02, q98 = valid.raw_vot.quantile([0.02, 0.98])
    low, high = max(-80.0, float(q02)), min(100.0, float(q98))
    low, high = min(low, -20.0), max(high, 30.0)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
    for y, model in enumerate(order):
        values = valid.loc[valid.model.eq(model), "raw_vot"].to_numpy(float)
        _style_boxplot(axes[0], values, y, color[model])
        row = summary.loc[summary.Model.eq(model)].iloc[0]
        axes[0].scatter(row["Median"], y, s=25, color=color[model], edgecolor="white", linewidth=0.5, zorder=4)
        axes[0].scatter(row["Population Direct VOT-eq."], y, s=45, marker="D", color=color[model], edgecolor="white", linewidth=0.5, zorder=5)
    axes[0].axvline(0, color="#555555", linewidth=0.9)
    axes[0].set_xlim(low, high)
    axes[0].set_yticks(range(len(order)), ["Human reference"] + MODEL_ORDER)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Signed Direct VOT-equivalent (kKRW/hour)")
    axes[0].set_title("A. Respondent distribution and population ratio")
    axes[0].grid(axis="x", color="#D7D7D7", linewidth=0.7)
    axes[0].grid(axis="y", visible=False)

    rate_columns = ["Computable (%)", "Joint conventional-sign (%)"]
    markers = ["o", "s"]
    rate_colors = ["#2F5D7E", "#B8861B"]
    for xcol, marker, c in zip(rate_columns, markers, rate_colors):
        axes[1].scatter(summary[xcol], np.arange(len(order)), marker=marker, s=35, color=c, label=xcol)
    axes[1].axvline(100, color="#888888", linewidth=0.8)
    axes[1].set_xlim(0, 102)
    axes[1].set_yticks(range(len(order)), [""] * len(order))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Respondents (%)")
    axes[1].set_title("B. Computability and component-sign validity")
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].grid(axis="x", color="#D7D7D7", linewidth=0.7)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Figure 6. Overall Direct VOT-equivalent and sign validity", fontsize=14.5, y=0.995)
    fig.text(
        0.5, 0.025,
        "A positive ratio is interpreted economically only when both Fare and IVT derivatives have conventional negative signs.",
        ha="center", fontsize=8.3, color="#444444",
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.90, bottom=0.12, wspace=0.18)
    return _save_figure(fig, "Figure_06_overall_direct_vot")


def subgroup_vot_figure(
    vot: pd.DataFrame,
    human_vot: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
    figure_number: int,
) -> tuple[Path, Path]:
    spec = GROUP_SPECS[group_col]
    frame = vot.loc[~vot.model.eq("Human same-CF reference")].merge(
        meta[["respondent_id", group_col]], on="respondent_id", validate="many_to_one"
    )
    human_frame = human_vot.merge(
        meta[["respondent_id", group_col]], on="respondent_id", validate="many_to_one"
    )
    valid = frame.loc[frame.raw_computable & np.isfinite(frame.raw_vot)].copy()
    finite_slopes = frame.loc[np.isfinite(frame.raw_DF) & np.isfinite(frame.raw_DT)].copy()
    q02, q98 = valid.loc[valid.model.isin(MODEL_ORDER), "raw_vot"].quantile([0.02, 0.98])
    low, high = max(-80.0, float(q02)), min(100.0, float(q98))
    low, high = min(low, -20.0), max(high, 30.0)
    ratio = (
        finite_slopes.groupby(["model", group_col], observed=True)[["raw_DF", "raw_DT"]]
        .mean().reset_index()
    )
    ratio["population_ratio_vot"] = 60 * ratio.raw_DT / ratio.raw_DF
    descriptive = _percentile_summary(valid, "model", group_col, "raw_vot").merge(
        ratio[["model", group_col, "population_ratio_vot"]], on=["model", group_col], validate="one_to_one"
    )
    descriptive.to_csv(OUT / f"Table_S_Figure_{figure_number:02d}_{group_col}_vot.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.0), sharex=True, sharey=True)
    for panel_index, (ax, group_value) in enumerate(zip(axes, spec["order"])):
        human_all = human_frame.loc[
            human_frame[group_col].eq(group_value)
            & np.isfinite(human_frame.raw_DF)
            & np.isfinite(human_frame.raw_DT)
        ].copy()
        human = human_all.loc[
            human_all.raw_computable & np.isfinite(human_all.raw_vot), "raw_vot"
        ]
        human_population_ratio = float(60 * human_all.raw_DT.mean() / human_all.raw_DF.mean())
        hq1, hmed, hq3 = human.quantile([0.25, 0.5, 0.75])
        ax.axvspan(hq1, hq3, color="#777777", alpha=0.10, zorder=0)
        ax.axvline(human_population_ratio, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(0, color="#555555", linewidth=0.9)
        panel_ratio = ratio.loc[ratio[group_col].eq(group_value)].set_index("model")
        for y, model in enumerate(MODEL_ORDER):
            values = valid.loc[valid.model.eq(model) & valid[group_col].eq(group_value), "raw_vot"].to_numpy(float)
            _style_boxplot(ax, values, y, COLORS[model])
            ax.scatter(np.nanmedian(values), y, s=28, color=COLORS[model], edgecolor="white", linewidth=0.55, zorder=4)
            if model in panel_ratio.index:
                ax.scatter(panel_ratio.loc[model, "population_ratio_vot"], y, s=44, marker="D", color=COLORS[model], edgecolor="white", linewidth=0.55, zorder=5)
        ax.set_title(
            f"{chr(65 + panel_index)}. {group_value} (n={human_all.respondent_id.nunique()})\n"
            f"Human Direct population ratio = {human_population_ratio:.2f}",
            fontsize=11.0, fontweight="semibold", pad=10,
        )
        ax.set_xlim(low, high)
        ax.set_xlabel("Signed Direct VOT-equivalent (kKRW/hour)")
        ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Patch(facecolor="#7C9DB5", edgecolor="#4D7897", alpha=0.25, label="Candidate IQR (whiskers p05-p95)"),
            Line2D([], [], marker="o", color="none", markerfacecolor="#4D7897", markeredgecolor="white", label="Candidate median"),
            Line2D([], [], marker="D", color="none", markerfacecolor="#4D7897", markeredgecolor="white", label="Candidate population ratio"),
            Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Human subgroup population ratio"),
        ],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(f"Figure {figure_number}. Direct VOT-equivalent by {spec['title'].lower()}", fontsize=14.5, y=0.995)
    fig.text(
        0.5, 0.025, spec["note"] + " Negative ratios remain sign-invalid diagnostics.",
        ha="center", fontsize=8.2, color="#444444",
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.12, wspace=0.08)
    return _save_figure(fig, f"Figure_{figure_number:02d}_direct_vot_{group_col}")


def direct_response_regression_table(
    elasticity_case: pd.DataFrame,
    derivative_case: pd.DataFrame,
    human_elasticity_case: pd.DataFrame,
    human_derivative_case: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg_models = ["Human reference"] + QWEN_ORDER
    elasticity_part = elasticity_case.loc[elasticity_case.system.isin(QWEN_ORDER), [
        "system", "case_id", "respondent_id", "distance_band", "elasticity"
    ]].rename(columns={"system": "Model"})
    elasticity_part = pd.concat([
        human_elasticity_case[["system", "case_id", "respondent_id", "distance_band", "elasticity"]]
        .rename(columns={"system": "Model"}),
        elasticity_part,
    ], ignore_index=True)
    derivative_part = derivative_case.loc[
        derivative_case.model.isin(QWEN_ORDER),
        ["model", "case_id", "respondent_id", "distance_band", "raw_DF", "raw_DT"],
    ].rename(columns={"model": "Model"})
    human_derivative = human_derivative_case[[
        "model", "case_id", "respondent_id", "distance_band", "raw_DF", "raw_DT"
    ]].rename(columns={"model": "Model"})
    human_derivative["Model"] = "Human reference"
    derivative_part = pd.concat([human_derivative, derivative_part], ignore_index=True)
    frame = elasticity_part.merge(
        derivative_part,
        on=["Model", "case_id", "respondent_id", "distance_band"],
        validate="one_to_one",
    )
    frame = frame.merge(meta, on="respondent_id", validate="many_to_one")
    frame["Age 50+"] = frame.age_group.eq("Age 50+").astype(float)
    frame["Female"] = frame.sex_group.eq("Female").astype(float)
    frame["RISK"] = frame.framing_group.eq("RISK").astype(float)
    distance_dummies = pd.get_dummies(frame.distance_band, prefix="Distance", drop_first=True, dtype=float)
    frame = pd.concat([frame.reset_index(drop=True), distance_dummies.reset_index(drop=True)], axis=1)
    focal_predictors = ["Age 50+", "Female", "RISK"]
    controls = sorted(distance_dummies.columns.tolist())
    predictors = focal_predictors + controls
    outcomes = [("Direct Fare elasticity", "elasticity"), ("Fare derivative D_F", "raw_DF"), ("IVT derivative D_T", "raw_DT")]
    rows = []
    for model in reg_models:
        model_data = frame.loc[frame.Model.eq(model)].copy()
        x = sm.add_constant(model_data[predictors], has_constant="add")
        for outcome_label, outcome in outcomes:
            fit = sm.OLS(model_data[outcome].astype(float), x).fit(
                cov_type="cluster", cov_kwds={"groups": model_data.respondent_id.to_numpy()}
            )
            for predictor in focal_predictors:
                rows.append({
                    "Outcome": outcome_label, "Predictor": predictor, "Model": model,
                    "Coefficient": float(fit.params[predictor]),
                    "Robust SE": float(fit.bse[predictor]),
                    "p-value": float(fit.pvalues[predictor]),
                    "N cases": int(fit.nobs),
                    "N respondents": int(model_data.respondent_id.nunique()),
                    "R-squared": float(fit.rsquared),
                })
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "Table_04_direct_response_regression_long.csv", index=False)
    display_rows = []
    for outcome_label, _ in outcomes:
        for predictor in focal_predictors:
            row = {"Outcome": outcome_label, "Predictor": predictor}
            for model in reg_models:
                value = long.loc[
                    long.Outcome.eq(outcome_label) & long.Predictor.eq(predictor) & long.Model.eq(model)
                ].iloc[0]
                row[model] = f"{value['Coefficient']:.4f} ({value['Robust SE']:.4f})"
            display_rows.append(row)
    wide = pd.DataFrame(display_rows)
    wide.to_csv(OUT / "Table_04_direct_response_regression.csv", index=False)
    return long, wide


def choice_implied_cell_vot_feasibility_audit(
    scores: pd.DataFrame,
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attempt the Silicon two-stage first step using existing generated choices.

    A setting cell is defined by observed income, gender, age group, education,
    vehicle ownership, trip purpose, survey framing, and distance-specific
    existing mode. Only generated hard choices from BASE and the six Fare and
    six IVT counterfactuals are used; probabilities are not fitted.
    """
    scenario_ids = [
        "BASE", "F_M3", "F_M2", "F_M1", "F_P1", "F_P2", "F_P3",
        "R_M3", "R_M2", "R_M1", "R_P1", "R_P2", "R_P3",
    ]
    generated = scores.loc[scores.scenario_id.isin(scenario_ids), [
        "case_id", "scenario_id", "system", "margin", "free_argmax"
    ]].copy()
    generated["choice"] = np.where(
        generated.margin.gt(0), 1,
        np.where(
            generated.margin.lt(0), 0,
            generated.free_argmax.astype(str).str.upper().eq("AV").astype(int),
        ),
    )
    scenario = grid.loc[grid.scenario_id.isin(scenario_ids), [
        "case_id", "scenario_id", "fare_cf", "ride_cf", "physically_valid"
    ]].copy()
    generated = generated.merge(
        scenario, on=["case_id", "scenario_id"], how="left", validate="many_to_one"
    )
    generated = generated.loc[generated.physically_valid].copy()

    panel = pd.read_parquet(PATH_PANEL)
    panel["case_id"] = panel.respondent_id.astype(str) + "_" + panel.distance_band.astype(str)
    test_cases = set(pd.read_parquet(PATH_TEST, columns=["case_id"]).case_id.astype(str))
    panel = panel.loc[panel.case_id.isin(test_cases)].copy()
    panel["income"] = pd.to_numeric(panel.D5, errors="coerce")
    panel = panel.loc[panel.income.isin([1, 2, 3, 4])].copy()
    panel["gender"] = panel.gender_label.astype(str)
    panel["age_group"] = np.where(panel.age.ge(50), "Age 50+", "Age <50")
    panel["education"] = np.where(
        pd.to_numeric(panel.D2, errors="coerce").eq(1),
        "High school or lower", "College or higher",
    )
    panel["trip_purpose"] = panel.QQ1.astype(str)
    panel["vehicle_ownership"] = panel.D6.astype(str)
    panel["framing"] = panel.BR_label.astype(str)
    panel["existing_mode"] = panel.current_mode_2026.astype(str)
    cell_columns = [
        "income", "gender", "age_group", "education", "vehicle_ownership",
        "trip_purpose", "framing", "existing_mode",
    ]
    metadata = panel[["case_id", "respondent_id"] + cell_columns].drop_duplicates("case_id")
    cell_support = (
        metadata.groupby(cell_columns, dropna=False)
        .agg(n_base_cases=("case_id", "nunique"), n_respondents=("respondent_id", "nunique"))
        .reset_index()
    )
    eligible = cell_support.loc[
        cell_support.n_respondents.ge(3) & cell_support.n_base_cases.ge(6)
    ].copy()
    assert len(eligible) > 0
    generated = (
        generated.merge(metadata, on="case_id", how="inner", validate="many_to_one")
        .merge(eligible, on=cell_columns, how="inner", validate="many_to_one")
    )

    fit_rows = []
    for keys, data in generated.groupby(["system"] + cell_columns, observed=True, sort=False):
        model, *cell_values = keys
        row = {
            "Model": model,
            **dict(zip(cell_columns, cell_values)),
            "N choice rows": int(len(data)),
            "N base cases": int(data.n_base_cases.iloc[0]),
            "N respondents": int(data.n_respondents.iloc[0]),
            "AV choice rate": float(data.choice.mean()),
        }
        if data.choice.nunique() < 2:
            row.update({
                "fit_status": "CONSTANT_CHOICE", "converged": False,
                "beta_fare": np.nan, "beta_ivt": np.nan,
                "VOT_choice": np.nan, "vot_status": "NE",
            })
            fit_rows.append(row)
            continue
        design = sm.add_constant(data[["fare_cf", "ride_cf"]].astype(float), has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.Logit(data.choice.astype(int), design).fit(disp=0, maxiter=300)
            converged = bool(fit.mle_retvals.get("converged", False))
            beta_fare = float(fit.params.fare_cf)
            beta_ivt = float(fit.params.ride_cf)
            vot = 60 * beta_ivt / beta_fare if beta_fare != 0 else np.nan
            if not converged or not np.isfinite(beta_fare) or not np.isfinite(beta_ivt):
                fit_status, vot_status = "FIT_FAILED", "NE"
            elif beta_fare < 0 and beta_ivt < 0 and np.isfinite(vot) and vot > 0:
                fit_status, vot_status = "ESTIMATED", "VALID"
            else:
                fit_status, vot_status = "ESTIMATED", "SIGN_INVALID"
            row.update({
                "fit_status": fit_status, "converged": converged,
                "beta_fare": beta_fare, "beta_ivt": beta_ivt,
                "se_fare": float(fit.bse.fare_cf), "se_ivt": float(fit.bse.ride_cf),
                "p_fare": float(fit.pvalues.fare_cf), "p_ivt": float(fit.pvalues.ride_cf),
                "VOT_choice": float(vot), "vot_status": vot_status,
            })
        except Exception as error:
            row.update({
                "fit_status": "FIT_FAILED", "converged": False,
                "beta_fare": np.nan, "beta_ivt": np.nan,
                "VOT_choice": np.nan, "vot_status": "NE",
                "failure_reason": type(error).__name__,
            })
        fit_rows.append(row)

    detail = pd.DataFrame(fit_rows)
    detail.to_csv(OUT / "Table_S09_choice_implied_cell_vot_fit_audit.csv", index=False)
    summary_rows = []
    for model in QWEN_ORDER:
        model_rows = detail.loc[detail.Model.eq(model)]
        valid = model_rows.vot_status.eq("VALID")
        valid_rows = model_rows.loc[valid]
        level_counts = {column: int(valid_rows[column].nunique()) for column in cell_columns}
        coverage = "/".join(str(level_counts[column]) for column in cell_columns)
        estimable = int(valid.sum()) >= 15 and all(count >= 2 for count in level_counts.values())
        summary_rows.append({
            "Model": model,
            "Eligible factor cells": int(len(model_rows)),
            "Constant-choice cells": int(model_rows.fit_status.eq("CONSTANT_CHOICE").sum()),
            "Fit-failed cells": int(model_rows.fit_status.eq("FIT_FAILED").sum()),
            "Sign-invalid VOT cells": int(model_rows.vot_status.eq("SIGN_INVALID").sum()),
            "Valid VOT cells": int(valid.sum()),
            "Valid-cell rate (%)": 100 * float(valid.mean()),
            "Valid factor-level coverage (I/G/A/E/V/P/R/M)": coverage,
            "Second-stage regression": "ESTIMABLE" if estimable else "NOT ESTIMABLE",
        })
    summary = pd.DataFrame(summary_rows)
    valid_detail = detail.loc[detail.vot_status.eq("VALID")].copy()
    summary.to_csv(OUT / "Table_05A_choice_implied_vot_feasibility.csv", index=False)
    valid_detail.to_csv(OUT / "Table_05B_valid_choice_implied_vot_cells.csv", index=False)
    return summary, valid_detail


def expanded_generated_choice_logit_table(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a common expanded Logit to observed Human and Qwen BASE hard choices."""
    specification = json.loads(PATH_HUMAN_SPEC.read_text(encoding="utf-8"))[
        "behavioral_reference_specification"
    ]
    panel = pd.read_parquet(PATH_PANEL)
    panel["case_id"] = panel.respondent_id.astype(str) + "_" + panel.distance_band.astype(str)
    train_cases, _, train_y = _load_human_partition(PATH_TRAIN, specification, panel)
    test_cases, _, _ = _load_human_partition(PATH_TEST, specification, panel)

    def build_design(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        design = pd.DataFrame(index=cases.index)
        design["const"] = 1.0
        for column in [
            "fare_AV", "ivt_AV", "wait_AV",
            "fare_KEEP_neg", "ivt_KEEP_neg", "wait_KEEP_neg",
        ]:
            design[column] = cases[column].astype(float)
        design["Male"] = cases.gender_label.eq("Male").astype(float)
        design["Age_50plus"] = cases.age.ge(50).astype(float)
        design["RISK"] = cases.BR_label.eq("RISK").astype(float)

        categorical = {
            "Income": ("D5", ["1", "2", "3", "4", "5"], "income"),
            "Education": ("D2", ["1", "2", "3"], "education"),
            "Vehicle ownership": ("D6", ["1", "2", "3"], "vehicles"),
            "Trip purpose": ("QQ1", ["1", "2", "3"], "purpose"),
            "Existing mode": ("current_mode_2026", ["PT", "Car", "PM", "Walk"], "mode"),
            "Distance controls": ("distance_band", ["D1", "D2", "D3", "D4", "D5", "D6"], "distance"),
        }
        blocks: dict[str, list[str]] = {
            "AV LOS": ["fare_AV", "ivt_AV", "wait_AV"],
            "Existing-mode LOS controls": ["fare_KEEP_neg", "ivt_KEEP_neg", "wait_KEEP_neg"],
            "Demographics": ["Male", "Age_50plus", "RISK"],
        }
        for block, (source, categories, prefix) in categorical.items():
            values = pd.Categorical(cases[source].map(_level_text), categories=categories)
            dummies = pd.get_dummies(values, prefix=prefix, drop_first=True, dtype=float)
            dummies.index = cases.index
            design = pd.concat([design, dummies], axis=1)
            blocks[block] = dummies.columns.tolist()
        design = design.astype(float)
        assert np.linalg.matrix_rank(design.to_numpy()) == design.shape[1]
        return design, blocks

    train_design, blocks = build_design(train_cases)
    test_design, test_blocks = build_design(test_cases)
    assert list(train_design.columns) == list(test_design.columns)
    assert blocks == test_blocks

    base_scores = scores.loc[scores.scenario_id.eq("BASE")].copy()
    test_order = test_cases.case_id.astype(str).tolist()
    outcomes: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "H-MNL": (np.asarray(train_y, dtype=int), train_cases.respondent_id.to_numpy()),
    }
    for model in QWEN_ORDER:
        frame = base_scores.loc[base_scores.system.eq(model)].set_index("case_id").loc[test_order]
        choice = np.where(
            frame.margin.to_numpy(float) > 0, 1,
            np.where(
                frame.margin.to_numpy(float) < 0, 0,
                frame.free_argmax.astype(str).str.upper().eq("AV").astype(int).to_numpy(),
            ),
        )
        outcomes[model] = (choice.astype(int), test_cases.respondent_id.to_numpy())

    fits = {}
    long_rows = []
    joint_rows = []
    for model, (choice, respondents) in outcomes.items():
        design = train_design if model == "H-MNL" else test_design
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(choice, design).fit(
                disp=0, maxiter=800, cov_type="cluster",
                cov_kwds={"groups": respondents},
            )
        assert bool(fit.mle_retvals.get("converged", False)), model
        fits[model] = fit
        for term in design.columns:
            long_rows.append({
                "Model": model, "Term": term,
                "Coefficient": float(fit.params[term]),
                "Cluster SE": float(fit.bse[term]),
                "p-value": float(fit.pvalues[term]),
                "N cases": int(fit.nobs),
                "N respondents": int(pd.Series(respondents).nunique()),
                "McFadden R-squared": float(fit.prsquared),
                "Generated AV share": float(np.mean(choice)),
            })
        parameter_names = list(design.columns)
        for block, terms in blocks.items():
            restriction = np.zeros((len(terms), len(parameter_names)))
            for index, term in enumerate(terms):
                restriction[index, parameter_names.index(term)] = 1.0
            test = fit.wald_test(restriction, scalar=True)
            joint_rows.append({
                "Model": model, "Block": block, "df": len(terms),
                "Chi-square": float(test.statistic),
                "Joint p-value": float(test.pvalue),
            })

    long = pd.DataFrame(long_rows)
    joint = pd.DataFrame(joint_rows)
    long.to_csv(OUT / "Table_05_generated_choice_logit_coefficients_long.csv", index=False)
    joint.to_csv(OUT / "Table_S12_generated_choice_logit_joint_tests.csv", index=False)

    display_terms = [
        ("LOS", "AV Fare", "fare_AV"),
        ("LOS", "AV IVT", "ivt_AV"),
        ("LOS", "AV Wait/access", "wait_AV"),
        ("Demographic", "Male (ref. Female)", "Male"),
        ("Demographic", "Age 50+ (ref. <50)", "Age_50plus"),
        ("Income", "3--<5m KRW (ref. <3m)", "income_2"),
        ("Income", "5--<7m KRW", "income_3"),
        ("Income", ">=7m KRW", "income_4"),
        ("Income", "Income nonresponse", "income_5"),
        ("Education", "College (ref. high school or lower)", "education_2"),
        ("Education", "Graduate school or higher", "education_3"),
        ("Vehicle ownership", "One vehicle (ref. zero)", "vehicles_2"),
        ("Vehicle ownership", "Two or more vehicles", "vehicles_3"),
        ("Trip purpose", "School commute (ref. work commute)", "purpose_2"),
        ("Trip purpose", "Other purpose", "purpose_3"),
        ("Existing mode", "Car (ref. PT)", "mode_Car"),
        ("Existing mode", "PM", "mode_PM"),
        ("Existing mode", "Walk", "mode_Walk"),
        ("Survey framing", "RISK (ref. BASIC)", "RISK"),
    ]

    def stars(p_value: float) -> str:
        if p_value < 0.01:
            return "***"
        if p_value < 0.05:
            return "**"
        if p_value < 0.10:
            return "*"
        return ""

    model_columns = ["H-MNL"] + QWEN_ORDER
    display_rows = []
    for block, label, term in display_terms:
        row = {"Block": block, "Variable": label}
        for model in model_columns:
            value = long.loc[long.Model.eq(model) & long.Term.eq(term)].iloc[0]
            row[model] = (
                f"{value['Coefficient']:.3f} ({value['Cluster SE']:.3f})"
                f"{stars(float(value['p-value']))}"
            )
        display_rows.append(row)
    for block in ["Income", "Education", "Vehicle ownership", "Trip purpose", "Existing mode"]:
        row = {"Block": block, "Variable": f"{block} joint p-value"}
        for model in model_columns:
            value = joint.loc[joint.Model.eq(model) & joint.Block.eq(block)].iloc[0]
            row[model] = f"p={value['Joint p-value']:.3f}"
        display_rows.append(row)
    for statistic, source in [
        ("N cases", "N cases"),
        ("N respondents", "N respondents"),
        ("McFadden R-squared", "McFadden R-squared"),
        ("Generated AV share", "Generated AV share"),
    ]:
        row = {"Block": "Fit", "Variable": statistic}
        for model in model_columns:
            value = long.loc[long.Model.eq(model)].iloc[0][source]
            row[model] = (
                f"{int(value)}" if statistic.startswith("N ") else f"{float(value):.3f}"
            )
        display_rows.append(row)
    display = pd.DataFrame(display_rows)
    display.to_csv(OUT / "Table_05_generated_choice_logit_coefficients.csv", index=False)
    return long, joint, display


def _direct_vot_ratio_sensitivity_regression(
    derivative_case: pd.DataFrame,
    human_derivative_case: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-Silicon Direct-ratio sensitivity retained outside the main notebook.

    This is a sensitivity analysis. It is deliberately separate from the main
    component regressions because the valid-case filter conditions on both
    derivatives and because case-level VOT ratios can be heavy-tailed.
    """
    models = ["H-MNL"] + QWEN_ORDER
    qwen = derivative_case.loc[
        derivative_case.model.isin(QWEN_ORDER),
        ["model", "case_id", "respondent_id", "distance_band", "raw_DF", "raw_DT"],
    ].copy()
    human = human_derivative_case[[
        "case_id", "respondent_id", "distance_band", "raw_DF", "raw_DT"
    ]].copy()
    human.insert(0, "model", "H-MNL")
    frame = pd.concat([human, qwen], ignore_index=True)

    panel = pd.read_parquet(PATH_PANEL)
    panel["case_id"] = panel.respondent_id.astype(str) + "_" + panel.distance_band.astype(str)
    metadata = panel[[
        "case_id", "age", "gender_label", "BR_label", "D5", "SQ5", "QQ1",
        "current_mode_2026", "distance_band",
    ]].drop_duplicates("case_id")
    frame = frame.drop(columns="distance_band").merge(
        metadata, on="case_id", how="left", validate="many_to_one"
    )
    assert frame[["age", "gender_label", "BR_label", "D5", "SQ5", "QQ1"]].notna().all().all()
    income_midpoint = {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0}
    frame["income_million_krw"] = pd.to_numeric(frame.D5, errors="coerce").map(income_midpoint)
    frame["vot_case"] = 60 * frame.raw_DT / frame.raw_DF
    frame["valid_vot_case"] = (
        np.isfinite(frame.raw_DF)
        & np.isfinite(frame.raw_DT)
        & frame.raw_DF.abs().ge(RAW_FARE_THRESHOLD)
        & frame.raw_DF.lt(0)
        & frame.raw_DT.lt(0)
        & frame.vot_case.gt(0)
        & frame.income_million_krw.notna()
    )
    frame = frame.loc[frame.valid_vot_case].copy()
    frame["log_vot"] = np.log(frame.vot_case)
    frame["log_income"] = np.log(frame.income_million_krw)
    frame["age_per_10y"] = frame.age.astype(float) / 10.0
    frame["Female"] = frame.gender_label.eq("Female").astype(float)
    frame["RISK"] = frame.BR_label.eq("RISK").astype(float)

    categorical_specs = {
        "Education": ("SQ5", ["1", "2", "3"]),
        "Trip purpose": ("QQ1", ["1", "2", "3"]),
        "Distance context": ("distance_band", ["D1", "D2", "D3", "D4", "D5", "D6"]),
        "Existing mode": ("current_mode_2026", ["PT", "Car", "PM", "Walk"]),
    }
    rows = []
    block_rows = []
    for model in models:
        data = frame.loc[frame.model.eq(model)].copy()
        design = pd.DataFrame({
            "const": 1.0,
            "log_income": data.log_income.astype(float),
            "Female": data.Female.astype(float),
            "age_per_10y": data.age_per_10y.astype(float),
            "RISK": data.RISK.astype(float),
        }, index=data.index)
        blocks: dict[str, list[str]] = {}
        for label, (column, categories) in categorical_specs.items():
            categorical = pd.Categorical(data[column].astype(str), categories=categories)
            dummies = pd.get_dummies(categorical, prefix=column, drop_first=True, dtype=float)
            dummies.index = data.index
            design = pd.concat([design, dummies], axis=1)
            blocks[label] = dummies.columns.tolist()
        design = design.astype(float)
        assert np.linalg.matrix_rank(design.to_numpy()) == design.shape[1]
        fit = sm.OLS(data.log_vot.astype(float), design).fit(
            cov_type="cluster", cov_kwds={"groups": data.respondent_id.to_numpy()}
        )
        for term in design.columns:
            rows.append({
                "Model": model,
                "Term": term,
                "Coefficient": float(fit.params[term]),
                "Cluster SE": float(fit.bse[term]),
                "p-value": float(fit.pvalues[term]),
                "N cases": int(fit.nobs),
                "N respondents": int(data.respondent_id.nunique()),
                "R-squared": float(fit.rsquared),
            })
        names = list(design.columns)
        for block, terms in blocks.items():
            restriction = np.zeros((len(terms), len(names)))
            for index, term in enumerate(terms):
                restriction[index, names.index(term)] = 1.0
            test = fit.wald_test(restriction, scalar=True)
            block_rows.append({
                "Model": model,
                "Block": block,
                "df": len(terms),
                "Chi-square": float(test.statistic),
                "Joint p-value": float(test.pvalue),
                "N cases": int(fit.nobs),
                "N respondents": int(data.respondent_id.nunique()),
                "R-squared": float(fit.rsquared),
            })

    long = pd.DataFrame(rows)
    blocks = pd.DataFrame(block_rows)
    long.to_csv(OUT / "Table_S10_direct_vot_ratio_regression_long.csv", index=False)
    blocks.to_csv(OUT / "Table_S11_direct_vot_ratio_joint_tests.csv", index=False)

    display_rows = []
    continuous_rows = [
        ("ln(Household income)", "log_income"),
        ("Female", "Female"),
        ("Age per 10 years", "age_per_10y"),
        ("RISK framing", "RISK"),
    ]
    for label, term in continuous_rows:
        row = {"Variable": label}
        for model in models:
            value = long.loc[long.Model.eq(model) & long.Term.eq(term)].iloc[0]
            row[model] = f"{value['Coefficient']:.3f} ({value['Cluster SE']:.3f})"
        display_rows.append(row)
    for block in categorical_specs:
        row = {"Variable": f"{block} (joint p)"}
        for model in models:
            value = blocks.loc[blocks.Model.eq(model) & blocks.Block.eq(block)].iloc[0]
            row[model] = f"p={value['Joint p-value']:.3f}"
        display_rows.append(row)
    for statistic, column in [
        ("N cases", "N cases"), ("N respondents", "N respondents"), ("R-squared", "R-squared")
    ]:
        row = {"Variable": statistic}
        for model in models:
            value = long.loc[long.Model.eq(model)].iloc[0][column]
            row[model] = f"{int(value)}" if statistic.startswith("N ") else f"{value:.3f}"
        display_rows.append(row)
    wide = pd.DataFrame(display_rows)
    wide.to_csv(OUT / "Table_S10_direct_vot_ratio_regression.csv", index=False)
    return long, wide


def gc_mnl_table() -> pd.DataFrame:
    diagnostics = pd.read_csv(FINAL / "tables" / "Table_S_EXP4_class_balanced_raw_coefficient_diagnostics.csv")
    diagnostics = diagnostics.loc[
        diagnostics.projection.eq("BASE_MNL") & diagnostics.Model.isin(QWEN_ORDER)
    ].copy()
    elasticity = pd.read_csv(FINAL / "tables" / "Table_09B_EXP4_class_balanced_raw_elasticity.csv").set_index("Model")
    vot = pd.read_csv(FINAL / "tables" / "Table_09_EXP4_class_balanced_raw_vot.csv").set_index("Model")
    human_coef = pd.read_csv(FINAL / "03_EXP3" / "tableS_human_mnl_coefficients.csv")
    human_coef = human_coef.loc[human_coef["sample"].eq("development/train")].set_index("term")
    human_ref = json.loads((FINAL / "04_EXP4" / "CF_MNL_ROBUSTNESS" / "human_references.json").read_text())
    rows = [{
        "Model": "H-MNL",
        "Fare coefficient": float(human_coef.loc["fare_AV", "coefficient"]),
        "Fare cluster SE": float(human_coef.loc["fare_AV", "cluster_se"]),
        "Fare p-value": float(human_coef.loc["fare_AV", "p_value"]),
        "IVT coefficient": float(human_coef.loc["ivt_AV", "coefficient"]),
        "IVT cluster SE": float(human_coef.loc["ivt_AV", "cluster_se"]),
        "IVT p-value": float(human_coef.loc["ivt_AV", "p_value"]),
        "GC-MNL Fare elasticity": float(human_ref["human_structural_elasticity_mean"]),
        "GC-MNL VOT": float(human_ref["human_structural_vot"]),
        "Fieller status": str(human_ref["human_vot_status"]),
    }]
    for item in diagnostics.itertuples(index=False):
        fieller_zero = bool(vot.loc[item.Model, "BASE set includes zero"] == "yes")
        status = "WEAKLY_IDENTIFIED" if fieller_zero else "RESOLVED"
        if item.beta_fare_av >= 0 or item.beta_ivt_av >= 0:
            status = "SIGN_INVALID"
        rows.append({
            "Model": item.Model,
            "Fare coefficient": item.beta_fare_av,
            "Fare cluster SE": item.se_fare_av,
            "Fare p-value": item.p_fare_av,
            "IVT coefficient": item.beta_ivt_av,
            "IVT cluster SE": item.se_ivt_av,
            "IVT p-value": item.p_ivt_av,
            "GC-MNL Fare elasticity": float(elasticity.loc[item.Model, "BASE-MNL elasticity"]),
            "GC-MNL VOT": float(vot.loc[item.Model, "BASE-MNL VOT"]),
            "Fieller status": status,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "Table_06_gc_mnl_behavioral_measures.csv", index=False)
    return frame


def correspondence_table(
    fare: pd.DataFrame, vot_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gc_elasticity = pd.read_csv(FINAL / "tables" / "Table_09B_EXP4_class_balanced_raw_elasticity.csv").set_index("Model")
    gc_vot = pd.read_csv(FINAL / "tables" / "Table_09_EXP4_class_balanced_raw_vot.csv").set_index("Model")
    fare_index = fare.set_index("Model")
    vot_index = vot_summary.set_index("Model")
    rows = []
    robustness = []
    for model in MODEL_ORDER:
        direct_e = float(fare_index.loc[model, "Mean elasticity"])
        gc_e = float(gc_elasticity.loc[model, "BASE-MNL elasticity"])
        direct_v = float(vot_index.loc[model, "Population Direct VOT-eq."])
        gc_v = float(gc_vot.loc[model, "BASE-MNL VOT"])
        status = "WEAKLY_IDENTIFIED" if gc_vot.loc[model, "BASE set includes zero"] == "yes" else "RESOLVED"
        rows.append({
            "Model": model,
            "GC-MNL Fare elasticity": gc_e,
            "Direct Fare elasticity": direct_e,
            "Absolute elasticity gap": abs(gc_e - direct_e),
            "GC-MNL VOT": gc_v,
            "Direct VOT-eq.": direct_v,
            "Absolute VOT gap": abs(gc_v - direct_v),
            "GC-MNL VOT status": status,
        })
        cf_e = float(gc_elasticity.loc[model, "CF-MNL elasticity"])
        cf_v = float(gc_vot.loc[model, "CF-MNL VOT"])
        robustness.append({
            "Model": model, "CF-MNL Fare elasticity": cf_e,
            "Absolute CF-Direct elasticity gap": abs(cf_e - direct_e),
            "CF-MNL VOT": cf_v, "Absolute CF-Direct VOT gap": abs(cf_v - direct_v),
        })
    main = pd.DataFrame(rows)
    robustness = pd.DataFrame(robustness)
    main.to_csv(OUT / "Table_07_gc_mnl_direct_correspondence.csv", index=False)
    robustness.to_csv(OUT / "Table_S04_cf_mnl_robustness.csv", index=False)
    return main, robustness


def correspondence_elasticity_figure(
    correspondence: pd.DataFrame,
    cf_robustness: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> tuple[Path, Path]:
    human = json.loads((FINAL / "04_EXP4" / "CF_MNL_ROBUSTNESS" / "human_references.json").read_text())
    direct_color = "#2F78A0"
    cf = cf_robustness.set_index("Model")
    comparison = correspondence.set_index("Model")
    human_values = elasticity.loc[elasticity.system.eq("Human reference"), "elasticity"].dropna()
    human_q1, human_q3 = human_values.quantile([0.25, 0.75])
    all_values = elasticity.loc[elasticity.system.isin(MODEL_ORDER), "elasticity"].dropna()
    plot_values = list(all_values.quantile([0.01, 0.99]))
    plot_values += correspondence[["GC-MNL Fare elasticity", "Direct Fare elasticity"]].to_numpy(float).ravel().tolist()
    plot_values += cf_robustness["CF-MNL Fare elasticity"].to_numpy(float).tolist()
    plot_values += [human["human_structural_elasticity_mean"], human["human_direct_elasticity_mean"]]
    low, high = min(plot_values), max(plot_values)
    pad = 0.08 * max(high - low, 1.0)
    panel_specs = [
        ("A. Conventional prediction systems", MODEL_ORDER[:6]),
        ("B. Qwen systems", QWEN_ORDER),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.0), sharex=True)
    for ax, (panel_title, models) in zip(axes, panel_specs):
        for yi, model in enumerate(models):
            values = elasticity.loc[elasticity.system.eq(model), "elasticity"].dropna().to_numpy(float)
            _style_boxplot(ax, values, yi, direct_color)
            row = comparison.loc[model]
            ax.scatter(row["Direct Fare elasticity"], yi, marker="D", s=54,
                       color=direct_color, edgecolor="white", linewidth=0.6, zorder=5)
            ax.scatter(row["GC-MNL Fare elasticity"], yi, marker="s", s=58,
                       facecolor="white", edgecolor="#333333", linewidth=1.3, zorder=6)
            ax.scatter(cf.loc[model, "CF-MNL Fare elasticity"], yi, marker="o", s=52,
                       facecolor="#E6A04B", edgecolor="#8C4E16", linewidth=1.0, zorder=6)
        ax.axvspan(human_q1, human_q3, color="#777777", alpha=0.08, zorder=0)
        ax.axvline(human["human_structural_elasticity_mean"], color="#9A5D35", linestyle="--", linewidth=1.35)
        ax.axvline(human["human_direct_elasticity_mean"], color="#1D8588", linestyle=":", linewidth=1.55)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        ax.set_xlim(low - pad, high + pad)
        ax.set_xlabel("AV Fare elasticity")
        ax.set_title(panel_title, fontsize=12.0, fontweight="semibold", pad=10)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 10. Direct, GC-MNL, and CF-MNL Fare elasticity", fontsize=14.2, y=0.995)
    fig.legend(
        handles=[
            Patch(facecolor=direct_color, edgecolor=direct_color, alpha=0.23,
                  label="Direct respondent distribution (IQR; whiskers p05-p95)"),
            Line2D([], [], marker="D", color="none", markerfacecolor=direct_color,
                   markeredgecolor="white", label="Direct population mean"),
            Line2D([], [], marker="s", color="none", markerfacecolor="white",
                   markeredgecolor="#333333", label="GC-MNL"),
            Line2D([], [], marker="o", color="none", markerfacecolor="#E6A04B",
                   markeredgecolor="#8C4E16", label="CF-MNL"),
            Line2D([], [], color="#9A5D35", linestyle="--", label="H-MNL GC coordinate"),
            Line2D([], [], color="#1D8588", linestyle=":", label="Human Direct coordinate"),
        ],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94),
    )
    fig.text(0.5, 0.025,
             "Direct boxes summarize respondent heterogeneity, not confidence intervals; GC-MNL uses original-condition generated choices and CF-MNL adds LOS-counterfactual generated choices.",
             ha="center", fontsize=8.1, color="#444444")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.12, wspace=0.30)
    return _save_figure(fig, "Figure_10_gc_mnl_direct_elasticity")


def correspondence_vot_figure(
    correspondence: pd.DataFrame,
    cf_robustness: pd.DataFrame,
    vot: pd.DataFrame,
) -> tuple[Path, Path]:
    human = json.loads((FINAL / "04_EXP4" / "CF_MNL_ROBUSTNESS" / "human_references.json").read_text())
    direct_color = "#2F78A0"
    cf = cf_robustness.set_index("Model")
    comparison = correspondence.set_index("Model")
    valid_all = vot.loc[vot.model.isin(MODEL_ORDER) & vot.raw_computable & np.isfinite(vot.raw_vot)]
    plot_values = list(valid_all.raw_vot.quantile([0.01, 0.99]))
    plot_values += correspondence[["GC-MNL VOT", "Direct VOT-eq."]].to_numpy(float).ravel().tolist()
    plot_values += cf_robustness["CF-MNL VOT"].to_numpy(float).tolist()
    plot_values += [human["human_fieller_low"], human["human_fieller_high"], human["human_structural_vot"]]
    # Keep the policy-relevant points and IQRs readable. Extreme respondent
    # whiskers remain in the source table and may extend beyond the display.
    low, high = -50.0, 80.0
    pad = 0.0
    panel_specs = [
        ("A. Conventional prediction systems", MODEL_ORDER[:6]),
        ("B. Qwen systems", QWEN_ORDER),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.0), sharex=True)
    for ax, (panel_title, models) in zip(axes, panel_specs):
        ax.axvspan(human["human_fieller_low"], human["human_fieller_high"],
                   color="#9A5D35", alpha=0.08, zorder=0)
        ax.axvline(human["human_structural_vot"], color="#9A5D35", linestyle="--", linewidth=1.4)
        ax.axvline(0, color="#555555", linewidth=0.8)
        for yi, model in enumerate(models):
            values = vot.loc[
                vot.model.eq(model) & vot.raw_computable & np.isfinite(vot.raw_vot), "raw_vot"
            ].to_numpy(float)
            _style_boxplot(ax, values, yi, direct_color)
            row = comparison.loc[model]
            ax.scatter(row["Direct VOT-eq."], yi, marker="D", s=54,
                       color=direct_color, edgecolor="white", linewidth=0.6, zorder=5)
            ax.scatter(row["GC-MNL VOT"], yi, marker="s", s=58,
                       facecolor="white", edgecolor="#333333", linewidth=1.3, zorder=6)
            ax.scatter(cf.loc[model, "CF-MNL VOT"], yi, marker="o", s=52,
                       facecolor="#E6A04B", edgecolor="#8C4E16", linewidth=1.0, zorder=6)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        ax.set_xlim(low - pad, high + pad)
        ax.set_xlabel("VOT / VOT-equivalent (kKRW/hour)")
        ax.set_title(panel_title, fontsize=12.0, fontweight="semibold", pad=10)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.7)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 11. Direct, GC-MNL, and CF-MNL VOT comparison", fontsize=14.2, y=0.995)
    fig.legend(
        handles=[
            Patch(facecolor=direct_color, edgecolor=direct_color, alpha=0.23,
                  label="Direct respondent distribution (IQR; whiskers p05-p95)"),
            Line2D([], [], marker="D", color="none", markerfacecolor=direct_color,
                   markeredgecolor="white", label="Direct population ratio"),
            Line2D([], [], marker="s", color="none", markerfacecolor="white",
                   markeredgecolor="#333333", label="GC-MNL VOT"),
            Line2D([], [], marker="o", color="none", markerfacecolor="#E6A04B",
                   markeredgecolor="#8C4E16", label="CF-MNL VOT"),
            Line2D([], [], color="#9A5D35", linestyle="--", label="H-MNL VOT reference"),
            Patch(facecolor="#9A5D35", alpha=0.08, label="Human Fieller set"),
        ],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94),
    )
    fig.text(0.5, 0.025,
             "Direct boxes show respondent distributions; extreme p05-p95 whiskers may extend beyond the displayed range. Diamonds are ratios of population-mean slopes; Fieller shading is uncertainty context.",
             ha="center", fontsize=8.1, color="#444444")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.78, bottom=0.12, wspace=0.30)
    return _save_figure(fig, "Figure_11_gc_mnl_direct_vot")


def build_manifest() -> pd.DataFrame:
    files = sorted(
        [path for path in OUT.iterdir() if path.is_file() and path.name not in {"manifest.json"}],
        key=lambda path: path.name.lower(),
    )
    rows = []
    for path in files:
        rows.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = pd.DataFrame(rows)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return manifest


def build_contact_sheet() -> Path:
    figure_paths = sorted(OUT.glob("Figure_*.png"))
    assert len(figure_paths) == 11
    cell_width, cell_height = 1100, 720
    columns = 2
    rows = math.ceil(len(figure_paths) / columns)
    canvas = PILImage.new("RGB", (columns * cell_width, rows * cell_height), "#E9E9E9")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, path in enumerate(figure_paths):
        row, column = divmod(index, columns)
        left, top = column * cell_width, row * cell_height
        image = PILImage.open(path).convert("RGB")
        fitted = ImageOps.contain(image, (cell_width - 40, cell_height - 70))
        x = left + (cell_width - fitted.width) // 2
        y = top + 42 + (cell_height - 70 - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.text((left + 16, top + 14), path.stem, fill="#202020", font=font)
    output = OUT / "FIGURES_CONTACT_SHEET.png"
    canvas.save(output, dpi=(150, 150))
    return output


def validate_outputs(state: dict[str, object]) -> pd.DataFrame:
    checks = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    prediction = state["prediction"]
    response = state["response"]
    elasticity = state["elasticity"]
    vot = state["vot"]
    check("prediction rows", len(prediction) == 10, f"rows={len(prediction)}")
    check("response attributes", set(response.attribute) == {"Fare", "IVT", "Wait"}, str(sorted(response.attribute.unique())))
    check("response systems", response.system.nunique() == 11, f"systems={response.system.nunique()}")
    check("elasticity respondent coverage", elasticity.groupby("system").respondent_id.nunique().eq(651).all(), "651 per system")
    check("VOT respondent coverage", vot.groupby("model").respondent_id.nunique().eq(651).all(), "651 per model")
    check("Qwen probability definition", np.allclose(state["scores"].p, expit(state["scores"].margin), atol=1e-14), "p=expit(margin)")
    human_group = state["human_hetero_summary"]
    check("Human heterogeneous reference rows", len(human_group) == 6, f"rows={len(human_group)}")
    age_vot = human_group.loc[human_group.Dimension.eq("age_group")].set_index("Group")["Population Direct VOT-eq."]
    check(
        "Human age VOT reference is interaction-specific",
        abs(float(age_vot.loc["Age <50"] - age_vot.loc["Age 50+"])) > 1.0,
        f"Age<50={age_vot.loc['Age <50']:.3f}; Age50+={age_vot.loc['Age 50+']:.3f}",
    )
    expected_png = 11
    expected_pdf = 11
    check("figure PNG count", len(list(OUT.glob("Figure_*.png"))) == expected_png, f"count={len(list(OUT.glob('Figure_*.png')))}")
    check("figure PDF count", len(list(OUT.glob("Figure_*.pdf"))) == expected_pdf, f"count={len(list(OUT.glob('Figure_*.pdf')))}")
    vot_audit_path = OUT / "Table_05A_choice_implied_vot_feasibility.csv"
    vot_audit = pd.read_csv(vot_audit_path) if vot_audit_path.exists() else pd.DataFrame()
    check(
        "Supplementary choice-VOT feasibility table",
        len(vot_audit) == 4
        and vot_audit["Eligible factor cells"].nunique() == 1
        and int(vot_audit["Eligible factor cells"].iloc[0]) > 0,
        f"rows={len(vot_audit)}; valid_total={int(vot_audit['Valid VOT cells'].sum()) if len(vot_audit) else 0}",
    )
    expanded_logit_path = OUT / "Table_05_generated_choice_logit_coefficients.csv"
    expanded_logit = pd.read_csv(expanded_logit_path) if expanded_logit_path.exists() else pd.DataFrame()
    check(
        "Expanded generated-choice Logit coefficient table",
        len(expanded_logit) == 28
        and {"H-MNL", *QWEN_ORDER}.issubset(expanded_logit.columns),
        f"rows={len(expanded_logit)}",
    )
    correspondence_path = OUT / "Table_07_gc_mnl_direct_correspondence.csv"
    correspondence = pd.read_csv(correspondence_path) if correspondence_path.exists() else pd.DataFrame()
    check(
        "Cross-system correspondence rows",
        len(correspondence) == 10,
        f"rows={len(correspondence)}",
    )
    report = pd.DataFrame(checks)
    report.to_csv(OUT / "VALIDATION_CHECKS.csv", index=False)
    assert report.passed.all(), report.loc[~report.passed]
    return report
