# -*- coding: utf-8 -*-
"""Exp 1-3 분석 코어.

predictions/ 아래의 모든 시스템 예측을 같은 형식으로 읽어 지표를 만든다.
tabular 은 predictions/tabular/*.parquet, LLM 은 predictions/llm_scores.parquet 이며
둘 다 (case_id, scenario_id, world, info, model, seed, p) 스키마로 정규화한다.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from scipy.stats import spearmanr, wasserstein_distance

import metrics as M
from common import CONFIG, DATA, ROOT, load_config

PRED = ROOT / "predictions"
TRUTH, SCEN, MODEL_IN = DATA / "truth", DATA / "scenarios", DATA / "model_inputs"
def _grid_base():
    """case 별 기저 속성값(원 단위). 탄력성 분모와 사다리 스텝 조회에 쓴다."""
    g = pd.read_parquet(SCEN / "test_scenario_grid.parquet")
    cols = [c for c in ("distance_band", "fare_base", "ride_base", "wait_base", "rel_base")
            if c in g.columns]
    return g.drop_duplicates("case_id").set_index("case_id")[cols]


_FINE = {}


def _analytic_slopes(world, attr):
    """p10b 가 h->0 에서 낸 사례별 참 기울기. 없으면 None.

    격자의 h 는 설계범위의 1/3 이라 작지 않다. 곡률이 있는 세계(NL, RC)에서는 중앙차분이
    미분값을 최대 23% 과소평가한다(results/tables/TA1_step_size.md). 모형 대 참값 비교는
    양쪽에 같은 연산자를 쓰므로 영향이 없지만, 참 기울기의 '진짜 크기'를 함께 보여줘야
    독자가 격자 때문에 생긴 간격을 판단할 수 있다. 참값은 어디서나 공짜로 계산되므로
    이 열은 추가 비용이 없다.
    """
    if not _FINE:
        f = TRUTH / "analytic_slopes.parquet"
        if not f.exists():
            _FINE["__missing__"] = True
            return None
        d = pd.read_parquet(f)
        for (w, a), g in d.groupby(["world", "attribute"]):
            _FINE[(w, a)] = g.set_index("case_id")["slope_fine"]
    return _FINE.get((world, attr))


def _common_cohort(grid, index, scenarios):
    """모든 정책 시나리오에서 물리적으로 가능한 사례만 남기는 마스크 (명세 6절 C_common).

    시나리오마다 유효 사례가 다르면 점유율 변화를 나란히 놓을 수 없다. 예컨대 S3(차내시간
    -15분)은 짧은 통행에서 시간이 음수가 되어 2,354건만 남는데, S1(요금 -30%)은 3,468건이
    전부 살아 있다. 각자의 모집단에서 재면 ΔMS 차이가 정책 때문인지 사람이 달라서인지
    구분되지 않는다.
    """
    t = grid.set_index(["case_id", "scenario_id"]).analysis_tier
    idx = list(index)
    ok = np.ones(len(idx), bool)
    for s in scenarios:
        try:
            ok &= (t.loc[list(zip(idx, [s] * len(idx)))].to_numpy() != "INVALID")
        except KeyError:
            pass
    return ok


def _legs_valid(grid, index, prefix):
    """중앙차분의 두 다리(P1, M1)가 **둘 다 물리적으로 가능한** 사례 마스크.

    **왜 필요한가.** 기울기는 x+h 와 x-h 두 지점의 확률 차이다. 한쪽이 존재할 수 없는
    조건이면 그 차이는 기울기가 아니다. 신뢰도가 대표적이다 - 설계범위가 99.99~100 으로
    폭이 0.01 뿐이라, 기준값이 100 인 사례는 x+h 가 100.001 이 되어 확률로 존재할 수 없다
    (전체의 11%). 예전에는 tier 를 전혀 거르지 않아 자율주행 고유 지표인 신뢰도 WTP 가
    그런 다리로 계산됐다.
    """
    t = grid.set_index(["case_id", "scenario_id"]).analysis_tier
    idx = list(index)
    ok = np.ones(len(idx), bool)
    for s in ("%s_P1" % prefix, "%s_M1" % prefix):
        pairs = list(zip(idx, [s] * len(idx)))
        try:
            ok &= (t.loc[pairs].to_numpy() != "INVALID")
        except KeyError:
            pass                      # 그 사다리가 격자에 없다 - 거를 것도 없다
    return ok


def _ladders():
    """거리대별 스텝. 유한차분 분모라 반드시 격자와 같은 값을 써야 한다."""
    import yaml
    with open(CONFIG / "trc_band_ladders.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


LADDER = {"fare": "F", "ride": "R", "wait": "W", "rel": "A"}
# rel = AV 신뢰도(정상 완료 가능성 %). 거리대와 무관하게 스텝이 하나다.
KEY = ["world", "info", "model", "seed"]


def ladder_cols(prefix, include_base=True):
    c = ["%s_M3" % prefix, "%s_M2" % prefix, "%s_M1" % prefix,
         "%s_P1" % prefix, "%s_P2" % prefix, "%s_P3" % prefix]
    return (c[:3] + ["BASE"] + c[3:]) if include_base else c


def load_predictions() -> pd.DataFrame:
    frames = []
    d = PRED / "tabular"
    if d.exists():
        for f in sorted(d.glob("*.parquet")):
            frames.append(pd.read_parquet(f))
    f = PRED / "llm_scores.parquet"
    if f.exists():
        t = pd.read_parquet(f)
        # p18 은 이미 model = "{family}_{state}" 를 쓴다. model_state 를 model 로
        # rename 하면 같은 이름의 열이 둘 생겨 groupby 가 깨지고, 게다가 family 정보가
        # 사라져 qwen 과 llama 를 구분할 수 없게 된다. 있는 model 열을 그대로 쓴다.
        t = t.drop(columns=[c for c in ("model_state", "model_family") if c in t.columns])
        t = t.rename(columns={"information_condition": "info",
                              "calibrated_probability": "p"})
        # **choice = 모델이 실제로 고른 답** (margin>0, 후보 둘 중 점수가 높은 쪽).
        # 정확도·F1 같은 분류 지표는 이걸로 매긴다. 보정확률을 0.5 에서 자르면 Platt 이
        # 작동점을 옮겨 버려서, 같은 예측인데 F1 이 0.4954(원선택) -> 0.4154(보정 0.5)
        # 로 떨어졌다 (HUMAN qwen_mid_sft, 2026-08-15 실측). 보정은 확률 지표
        # (log loss, 확률회복, 반응곡선) 전용이다. tabular 는 이 열이 없고, 그쪽의
        # 원선택은 자기 확률의 argmax = p>=0.5 그대로라 예전과 같다.
        # **동점은 free_argmax 로 깬다** (2026-08-17, D248). margin 은 fp16·4bit 에서
        # 정확히 0 이 되는 일이 드물지 않다 - SFT BASE 10,404행 중 287행(2.76%). 그
        # 287행 전부에서 모델이 실제로 생성한 글자는 AV 인데(p18b 로 확인), margin>0 은
        # 거짓이 되어 Keep 으로 기록됐다. 즉 "모델의 원선택" 이라는 D239 의 원칙과
        # 어긋난다. free_argmax 는 어휘 전체 1등이고, p18b 생성물과 80/80 일치했다.
        # 옮겨지는 값: HUMAN SFT AV-F1 0.4954 -> 0.4868 (낮아진다 - 유리한 쪽이 아니다).
        tie = t["margin"].abs() < 1e-9
        t["choice"] = np.where(tie, (t["free_argmax"] == "AV"), t["margin"] > 0).astype(float)
        frames.append(t[["case_id", "scenario_id", "world", "info", "model", "seed",
                         "p", "choice"]])
    if not frames:
        raise FileNotFoundError("predictions/ 에 예측이 없다")
    out = pd.concat(frames, ignore_index=True)
    dup = [c for c in out.columns if list(out.columns).count(c) > 1]
    if dup:
        raise SystemExit("예측 프레임에 중복 열 %s" % sorted(set(dup)))
    _check_scenario_coverage(out)
    _check_information_condition(out)
    return out


def _check_information_condition(pred):
    """`pro` 예측이 하나라도 섞이면 중단한다.

    2026-08-07 부터 정보조건은 retro 하나다. 그런데 예측은 (world, info, model, seed) 마다
    파일 하나이고 옛 파일이 남아 있으면 조용히 평균에 들어간다. 27조건 예측이 섞여
    "어떤 모형은 신뢰도 사다리가 있고 어떤 모형은 없는" 상태로 평균이 나갔던 일이
    이미 한 번 있었다. 같은 실패를 정보조건에서도 막는다.
    """
    import os
    want = set(load_config()["information_conditions"])
    bad = sorted(set(pred["info"].unique()) - want)
    if not bad:
        return
    n = pred.groupby(["world", "info", "model", "seed"]).size()
    off = n.reset_index()
    off = off[off["info"].isin(bad)]
    msg = ("정보조건 %s 예측이 %d벌 섞여 있다 (허용: %s).\n"
           % (bad, len(off), sorted(want))
           + off[["world", "info", "model", "seed"]].to_string(index=False)
           + "\n\n2026-08-07 부터 retro 하나만 쓴다. 옛 파일은"
             "\n  artifacts/archive/20260807/predictions/  로 옮겨야 한다."
             "\n  (검사만 건너뛰려면 TRC_ALLOW_STALE_PREDICTIONS=1)")
    if os.environ.get("TRC_ALLOW_STALE_PREDICTIONS") == "1":
        print("  [경고] " + msg.replace("\n", "\n  "))
        return
    raise SystemExit(msg)


def _check_scenario_coverage(pred):
    """모든 예측이 같은 시나리오 집합을 담고 있는지 확인한다.

    **왜 필요한가.** predictions/tabular/ 는 (world, info, model, seed) 마다 파일 하나이고
    다시 돌리면 덮어쓴다. 격자를 27조건에서 33조건으로 늘린 뒤 재학습이 중간에 죽으면
    새 파일과 옛 파일이 한 디렉터리에 공존하는데, 예전에는 이걸 그대로 concat 해서
    "어떤 모형은 신뢰도 사다리가 있고 어떤 모형은 없는" 상태로 평균을 냈다. 표에는
    아무 경고 없이 숫자가 찍히고, 모형 간 비교가 조용히 무의미해진다. 실제로
    2026-08-06 재학습 도중에 90벌 중 33벌이 옛 27조건이었다.

    강제로 넘기려면 TRC_ALLOW_STALE_PREDICTIONS=1 (스모크 테스트용).
    """
    import os
    want = int(load_config()["inference_grid"]["conditions_per_case"])
    n = pred.groupby(KEY).scenario_id.nunique()
    bad = n[n != want]
    if not len(bad):
        return
    msg = ("예측 %d벌(전체 %d벌 중)의 시나리오 수가 기대치 %d개와 다르다.\n"
           % (len(bad), len(n), want)
           + bad.rename("scenario_count").reset_index().to_string(index=False)
           + "\n\n격자를 바꾼 뒤 재학습이 끝나지 않았거나 중간에 죽은 상태다."
             "\n  python pipeline/p13_tabular_full.py --force   로 다시 채운 뒤 분석하라."
             "\n  (검사만 건너뛰려면 TRC_ALLOW_STALE_PREDICTIONS=1)")
    if os.environ.get("TRC_ALLOW_STALE_PREDICTIONS") == "1":
        print("  [경고] " + msg.replace("\n", "\n  "))
        return
    raise SystemExit(msg)


def load_truth():
    t = pd.read_parquet(TRUTH / "counterfactual_truth.parquet")
    g = pd.read_parquet(SCEN / "test_scenario_grid.parquet")[
        ["case_id", "scenario_id", "distance_band", "analysis_tier"]]
    return t, g


def _wide(df, value):
    return df.pivot_table(index="case_id", columns="scenario_id", values=value,
                          aggfunc="first")


def exp1_table(preds, truth) -> pd.DataFrame:
    """baseline 조건에서의 선택예측·확률회복."""
    cfg = load_config()
    rows = []
    for keys, g in preds[preds.scenario_id == "BASE"].groupby(KEY, sort=False):
        world, info, model, seed = keys
        mi = pd.read_parquet(MODEL_IN / ("%s_%s_test.parquet" % (world, info)))
        mi = mi.set_index("case_id").loc[g.case_id]
        if world == "HUMAN":
            pt = None
        else:
            t = truth[(truth.world == world) & (truth.scenario_id == "BASE")]
            pt = t.set_index("case_id").loc[g.case_id, "p_true_marg"].to_numpy()
        ch = (g.choice.to_numpy() if "choice" in g.columns and g.choice.notna().all()
              else None)
        m = M.exp1(mi.y.to_numpy(), g.p.to_numpy(), pt, mi.respondent_id.to_numpy(),
                   choice=ch)
        m.update(dict(zip(KEY, keys)))
        rows.append(m)
    return pd.DataFrame(rows)


def exp2_table(preds, truth, grid, tier=None) -> pd.DataFrame:
    cfg = load_config()
    thr = cfg["thresholds"]
    tiers = grid.set_index(["case_id", "scenario_id"]).analysis_tier
    rows = []
    for keys, g in preds.groupby(KEY, sort=False):
        world, info, model, seed = keys
        if world == "HUMAN":
            continue
        pm = _wide(g, "p")
        t = truth[truth.world == world]
        pt = _wide(t, "p_true_marg").loc[pm.index]
        for attr, pre in LADDER.items():
            cols = [c for c in ladder_cols(pre) if c in pm.columns and c in pt.columns]
            edits = [c for c in cols if c != "BASE"]
            if not edits or "BASE" not in cols:
                # 이 사다리가 통째로 없다. 정상 상태에서는 load_predictions 의
                # 검사에 먼저 걸리므로 여기 오면 격자·참값 쪽이 어긋난 것이다.
                continue
            # **tier=None 이 '전부' 를 뜻하면 안 된다.** 예전에는 전부 통과시켜서 대표
            # 숫자에 물리적으로 불가능한 조건(대기시간 음수 3,674행, 신뢰도 100% 초과
            # 3,102행)이 섞였다. 명세 5절은 invalid 를 '생성하지 않음' 으로 규정하므로
            # 기본값은 INVALID 제외이고, 그 사실이 보이도록 라벨을 VALID 로 쓴다.
            tmat = np.column_stack([
                tiers.loc[list(zip(pm.index, [c] * len(pm)))].to_numpy() for c in edits])
            keep = (tmat != "INVALID") if tier is None else (tmat == tier)
            dm = (pm[edits].to_numpy() - pm[["BASE"]].to_numpy())[keep]
            dt = (pt[edits].to_numpy() - pt[["BASE"]].to_numpy())[keep]
            if not len(dt):
                continue
            r = M.exp2(dm, dt, float(thr["response_direction"]))
            # 곡선 지표는 사다리 7점이 한 줄로 이어져야 뜻이 있다. 일부 단계만 빼면
            # 곡선 모양이 깨지므로, **모든 단계가 유효한 사례**에서만 낸다.
            curve_ok = (tmat != "INVALID").all(axis=1)
            r["n_curve_cases"] = int(curve_ok.sum())
            if curve_ok.any():
                cm, ct = pm[cols].to_numpy()[curve_ok], pt[cols].to_numpy()[curve_ok]
                r["monotonicity_violation_rate"] = M.monotonicity_violation_rate(
                    cm, float(thr["monotonicity"]))
                r["ladder_curve_error"] = M.ladder_curve_error(cm, ct)
            else:
                r["monotonicity_violation_rate"] = np.nan
                r["ladder_curve_error"] = np.nan
            r.update(dict(zip(KEY, keys)))
            r["attribute"], r["tier"] = attr, tier or "VALID"
            rows.append(r)
    return pd.DataFrame(rows)


def slope_table(preds, truth, grid) -> pd.DataFrame:
    """한계효과(절대)와 탄력성(상대). 참값이 있으면 참값 쪽도 같이 낸다.

    **왜 따로 두나.** exp2_table 은 참값 대비 비율(진폭비)만 낸다. 그러면 (1) 절대 크기를
    원고에 못 쓰고 (2) 참값이 없는 인간자료에서 아무것도 못 낸다. 여기서는 사다리 +-1h
    두 점으로 중앙차분을 만들어 기울기 자체를 보고한다.
    """
    cfg = load_config(); thr = cfg["thresholds"]
    lad = _ladders()
    rows = []
    # **격자 원본을 직접 읽는다.** load_truth() 가 넘겨주는 grid 는 네 열(case_id,
    # scenario_id, distance_band, analysis_tier)뿐이라 fare_base 등 기저값이 없다.
    gbase = _grid_base()
    for keys, g in preds.groupby(KEY, sort=False):
        world, info, model, seed = keys
        pm = _wide(g, "p")
        pt = None
        if world != "HUMAN":
            t = truth[truth.world == world]
            w = _wide(t, "p_true_marg")
            pt = w.loc[w.index.intersection(pm.index)]
        bands = gbase.distance_band.reindex(pm.index).to_numpy()
        for attr, pre in LADDER.items():
            c_p, c_m = "%s_P1" % pre, "%s_M1" % pre
            if c_p not in pm.columns or c_m not in pm.columns:
                continue
            h = np.array([float(lad["steps"][attr][b]) for b in bands])
            d = M.local_slopes(pm[c_p].to_numpy(), pm[c_m].to_numpy(), h)
            # 존재할 수 없는 지점으로 만든 기울기는 기울기가 아니다. 걸러낸 뒤 평균한다.
            legs = _legs_valid(grid, pm.index, pre)
            d = np.where(legs, d, np.nan)
            col = {"fare": "fare_base", "ride": "ride_base",
                   "wait": "wait_base", "rel": "rel_base"}[attr]
            xb = gbase[col].reindex(pm.index).to_numpy(float)
            pb = pm["BASE"].to_numpy()
            # **신뢰도에는 탄력성을 내지 않는다.** 탄력성은 "x 를 1% 올리면" 인데 신뢰도는
            # 99.99~100 사이라 1% 를 올리면 100.99 가 되어 확률로서 존재할 수 없는 값이다.
            # x/P 가 526 쯤이라 숫자는 나오지만 해석할 수 없는 값이다. 신뢰도는 한계효과와
            # WTP 로만 본다.
            e = M.elasticity(d, xb, pb) if attr != "rel" else np.full(len(d), np.nan)
            r = {"attribute": attr, "n": int(len(d)),
                 "both_legs_valid_rate": float(legs.mean()),
                 "step_h_median": float(np.median(h)),
                 "marginal_effect": float(np.nanmean(d)),
                 "marginal_effect_median": float(np.nanmedian(d)),
                 "elasticity": float(np.nanmedian(e)),
                 "x_base_median": float(np.nanmedian(xb))}
            if pt is not None and c_p in pt.columns:
                idx = pm.index.intersection(pt.index)
                ht = np.array([float(lad["steps"][attr][b])
                               for b in gbase.distance_band.reindex(idx)])
                dt = M.local_slopes(pt.loc[idx, c_p].to_numpy(),
                                    pt.loc[idx, c_m].to_numpy(), ht)
                et = M.elasticity(dt, gbase[col].reindex(idx).to_numpy(float),
                                  pt.loc[idx, "BASE"].to_numpy())
                r["marginal_effect_true"] = float(np.nanmean(dt))
                r["elasticity_true"] = float(np.nanmedian(et))
                fs = _analytic_slopes(world, attr)
                if fs is not None:
                    r["marginal_effect_true_fine"] = float(np.nanmean(
                        fs.reindex(idx).to_numpy()))
            r.update(dict(zip(KEY, keys)))
            rows.append(r)
    return pd.DataFrame(rows)


def wtp_table(preds, truth, grid) -> pd.DataFrame:
    """WTP 세 가지 - 차내시간(VOT), 대기시간, AV 신뢰도.

    전부 같은 형태다: 분자 속성의 기울기를 요금 기울기로 나눈다. 요금이 분모이므로
    **요금 기울기가 음수이고 충분히 크지 않으면 계산하지 않는다** - 0 근처에서 발산한다.
    유효 비율(estimable_rate)을 함께 보고하는 이유가 그것이고, 나무 모형은 계단식 출력이라
    이 비율이 낮게 나온다. 그 자체가 결과다.
    """
    cfg = load_config(); thr = cfg["thresholds"]
    kap = float(thr["vot_denominator"])
    lad = _ladders()
    gbase = _grid_base()
    SPEC = [("vot_ride", "ride", 60.0, "kKRW/hour"),
            ("vot_wait", "wait", 60.0, "kKRW/hour"),
            ("wtp_reliability", "rel", 0.001, "kKRW per 0.001%p")]
    rows = []
    for keys, g in preds.groupby(KEY, sort=False):
        world, info, model, seed = keys
        pm = _wide(g, "p")
        bands = gbase.distance_band.reindex(pm.index).to_numpy()
        hf = np.array([float(lad["steps"]["fare"][b]) for b in bands])
        if "F_P1" not in pm.columns:
            continue
        dF = M.local_slopes(pm["F_P1"].to_numpy(), pm["F_M1"].to_numpy(), hf)
        pt = None
        if world != "HUMAN":
            t = truth[truth.world == world]
            w = _wide(t, "p_true_marg")
            pt = w.loc[w.index.intersection(pm.index)]
        for name, attr, scale, unit in SPEC:
            pre = LADDER[attr]
            if "%s_P1" % pre not in pm.columns:
                continue
            h = np.array([float(lad["steps"][attr][b]) for b in bands])
            dN = M.local_slopes(pm["%s_P1" % pre].to_numpy(),
                                pm["%s_M1" % pre].to_numpy(), h)
            # 분자·분모 **양쪽 사다리 모두** 두 다리가 유효해야 WTP 가 성립한다.
            legs = _legs_valid(grid, pm.index, pre) & _legs_valid(grid, pm.index, "F")
            dN = np.where(legs, dN, np.nan)
            v, ok = M.wtp_from_slopes(dN, dF, kap, scale)
            ok &= legs
            r = {"quantity": name, "attribute": attr, "unit": unit,
                 "both_legs_valid_rate": float(legs.mean()),
                 "estimable_rate": float(ok.mean()),
                 "median_model": float(np.nanmedian(v)) if ok.any() else np.nan,
                 "wrong_sign_rate": float(np.mean(~ok))}
            if pt is not None and "F_P1" in pt.columns:
                idx = pm.index.intersection(pt.index)
                bb = gbase.distance_band.reindex(idx)
                hft = np.array([float(lad["steps"]["fare"][b]) for b in bb])
                ht = np.array([float(lad["steps"][attr][b]) for b in bb])
                dFt = M.local_slopes(pt.loc[idx, "F_P1"].to_numpy(),
                                     pt.loc[idx, "F_M1"].to_numpy(), hft)
                dNt = M.local_slopes(pt.loc[idx, "%s_P1" % pre].to_numpy(),
                                     pt.loc[idx, "%s_M1" % pre].to_numpy(), ht)
                vt, okt = M.wtp_from_slopes(dNt, dFt, kap, scale)
                r["median_true"] = float(np.nanmedian(vt)) if okt.any() else np.nan
                # 참값 기울기를 함께 남긴다. **분자의 참 효과가 0 이면 WTP 자체가 정의되지
                # 않는다.** 실측: HOM 의 대기시간 계수는 부호제약 경계에 붙은 -1e-06 이라
                # 참 한계효과가 -0.000000 이다. 그 세계의 WTP_wait 은 "모형이 틀렸다" 가
                # 아니라 "잴 것이 없다" 로 읽어야 한다.
                r["true_slope_numerator"] = float(np.nanmean(dNt))
                r["true_slope_fare"] = float(np.nanmean(dFt))
                both = ok & okt
                r["mae"] = (float(np.nanmean(np.abs(v[both] - vt[both])))
                            if both.any() else np.nan)
                # h->0 에서의 참 기울기로도 같은 WTP 를 낸다. 모형과의 비교는 위쪽(같은 h)
                # 이 맞고, 이 열은 "격자 h 때문에 참값 자체가 얼마나 눌렸나" 를 보여준다.
                fN, fF = _analytic_slopes(world, attr), _analytic_slopes(world, "fare")
                if fN is not None and fF is not None:
                    dNf, dFf = fN.reindex(idx).to_numpy(), fF.reindex(idx).to_numpy()
                    vf, okf = M.wtp_from_slopes(dNf, dFf, kap, scale)
                    r["true_slope_numerator_fine"] = float(np.nanmean(dNf))
                    r["median_true_fine"] = (float(np.nanmedian(vf))
                                             if okf.any() else np.nan)
            r.update(dict(zip(KEY, keys)))
            rows.append(r)
    return pd.DataFrame(rows)


def exp3_table(preds, truth, grid) -> pd.DataFrame:
    cfg = load_config()
    thr = cfg["thresholds"]
    with open(CONFIG / "trc_band_ladders.yaml", encoding="utf-8") as f:
        lad = yaml.safe_load(f)
    band = grid.drop_duplicates("case_id").set_index("case_id").distance_band
    rows = []
    for keys, g in preds.groupby(KEY, sort=False):
        world, info, model, seed = keys
        if world == "HUMAN":
            continue
        pm = _wide(g, "p")
        t = truth[truth.world == world]
        pt = _wide(t, "p_true_marg").loc[pm.index]
        mi = pd.read_parquet(MODEL_IN / ("%s_%s_test.parquet" % (world, info)))
        rid = mi.set_index("case_id").loc[pm.index, "respondent_id"].to_numpy()

        # **공통 유효 코호트에서만 집계한다** (명세 6절 C_common = ∩(r=0..8) E(S_r)).
        # 시나리오마다 물리적으로 가능한 사례가 다르다 - 실측으로 S1/S2/S5/S7 은 3,468건인데
        # S3 은 2,354건이다. 각자의 모집단에서 점유율을 재면 시나리오 간 ΔMS 비교가 정책
        # 차이와 모집단 차이를 섞는다. 게다가 예전에는 무효 조건까지 평균에 들어가서
        # S3 점유율의 32% 가 존재할 수 없는 조건에서 나왔다.
        scen = [c for c in pm.columns if re.fullmatch(r"S\d+", c)]
        cohort = _common_cohort(grid, pm.index, scen + ["BASE"])
        keep = cohort.to_numpy() if hasattr(cohort, "to_numpy") else cohort
        pm_c, pt_c, rid_c = pm[keep], pt[keep], rid[keep]
        bm = M.scenario_share(pm_c["BASE"].to_numpy(), rid_c)
        bt = M.scenario_share(pt_c["BASE"].to_numpy(), rid_c)
        for s in scen:
            msm = M.scenario_share(pm_c[s].to_numpy(), rid_c)
            mst = M.scenario_share(pt_c[s].to_numpy(), rid_c)
            # JSD 를 소프트·하드 둘 다 낸다. **둘이 갈리면 신호다** - 소프트가 맞는데
            # 하드가 틀리면 확률은 맞고 0.5 로 자르는 지점이 틀렸다는 뜻이고, 그건
            # 보정으로 고쳐지는 문제다. L1 차이만 보면 이 구분이 안 된다.
            hm = float(np.mean(pm_c[s].to_numpy() >= 0.5))
            ht = float(np.mean(pt_c[s].to_numpy() >= 0.5))
            rows.append({**dict(zip(KEY, keys)), "quantity": "share", "scenario": s,
                         "n_common_cohort": int(keep.sum()), "n_all_cases": int(len(pm)),
                         "ms_model": msm, "ms_true": mst,
                         "delta_model": msm - bm, "delta_true": mst - bt,
                         "error": M.share_change_error(msm, mst, bm, bt),
                         "jsd_soft": M.share_jsd(msm, mst),
                         "jsd_hard": M.share_jsd(hm, ht),
                         "hard_share_model": hm, "hard_share_true": ht})
        h = {a: np.array([lad["steps"][a][band[c]] for c in pm.index])
             for a in ("fare", "ride")}
        dF = M.local_slopes(pm["F_P1"].to_numpy(), pm["F_M1"].to_numpy(), h["fare"])
        dR = M.local_slopes(pm["R_P1"].to_numpy(), pm["R_M1"].to_numpy(), h["ride"])
        # 두 사다리 모두 다리가 유효한 사례에서만 VOT 를 낸다 (WTP 와 같은 이유).
        vlegs = _legs_valid(grid, pm.index, "F") & _legs_valid(grid, pm.index, "R")
        dF, dR = np.where(vlegs, dF, np.nan), np.where(vlegs, dR, np.nan)
        vm, okm = M.vot_from_slopes(dF, dR, float(thr["vot_denominator"]))
        okm &= vlegs
        tF = M.local_slopes(pt["F_P1"].to_numpy(), pt["F_M1"].to_numpy(), h["fare"])
        tR = M.local_slopes(pt["R_P1"].to_numpy(), pt["R_M1"].to_numpy(), h["ride"])
        vt, okt = M.vot_from_slopes(tF, tR, float(thr["vot_denominator"]))
        both = okm & okt
        rows.append({**dict(zip(KEY, keys)), "quantity": "vot", "scenario": "-",
                     "vot_estimable_rate": float(okm.mean()),
                     "vot_median_model": float(np.nanmedian(vm)),
                     "vot_median_true": float(np.nanmedian(vt)),
                     "vot_mae": float(np.nanmean(np.abs(vm[both] - vt[both])))
                     if both.any() else np.nan,
                     "wrong_sign_rate": float(np.mean(~okm))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- 이질성 (Exp 3 확장)
#
# 계획서 Phase 21 의 미구현 항목이었다.
#   LC / NL : 계층별 반응 대비, 대비 MAE, 압축비
#   RC      : 개인 VOT 분포의 평균·중앙값·분산비·Wasserstein, 보조로 개인 순위 상관
#
# NL 은 계층 구조를 LC 에서 물려받는다 (configs: dgp.nl.inherit_class_from = LC).
# RC 의 비교 대상은 유한차분으로 만든 참 VOT 가 아니라 **구조적 개인 VOT**(vot_true) 다.
# 주변확률의 유한차분은 취향분포를 적분한 뒤의 population 기울기라 개인값이 아니기 때문이다.
# 후보모형은 관측변수만 보므로 개인 VOT 를 원리적으로 다 복원할 수 없다 — 이 격차 자체가
# Proposition 2 가 말하는 비식별이고, 그래서 여기 수치는 "얼마나 못 하는가" 의 측정이다.

DGP_DIR = ROOT / "artifacts" / "dgp"


def _resp_map(world, info):
    mi = pd.read_parquet(MODEL_IN / ("%s_%s_test.parquet" % (world, info)))
    return mi.set_index("case_id").respondent_id


def _respondent_response(pw, cols_edit, rid):
    """응답자별 평균 반응 ΔP (respondent-first)."""
    d = (pw[cols_edit].to_numpy() - pw[["BASE"]].to_numpy()).mean(axis=1)
    return pd.Series(d, index=pw.index).groupby(rid.loc[pw.index]).mean()


def heterogeneity_table(preds, truth) -> pd.DataFrame:
    cfg = load_config()
    thr = cfg["thresholds"]
    lc = pd.read_parquet(DGP_DIR / "LC_latent_states.parquet").set_index("respondent_id")
    rc = pd.read_parquet(DGP_DIR / "RC_latent_states.parquet").set_index("respondent_id")
    with open(CONFIG / "trc_band_ladders.yaml", encoding="utf-8") as f:
        lad = yaml.safe_load(f)
    rows = []
    for keys, g in preds.groupby(KEY, sort=False):
        world, info, model, seed = keys
        if world not in ("LC", "NL", "RC"):
            continue
        pm = _wide(g, "p")
        pt = _wide(truth[truth.world == world], "p_true_marg").loc[pm.index]
        rid = _resp_map(world, info)

        if world in ("LC", "NL"):
            cls = lc.class_true.reindex(rid.loc[pm.index].to_numpy())
            for attr, pre in LADDER.items():
                edits = [c for c in ladder_cols(pre, False) if c in pm.columns]
                if not edits:
                    continue
                rm = _respondent_response(pm, edits, rid)
                rt = _respondent_response(pt, edits, rid)
                k = lc.class_true.reindex(rm.index)
                gm = rm.groupby(k).mean().sort_index()
                gt = rt.groupby(k).mean().sort_index()
                if len(gm) < 2:
                    continue
                cm, ct = M.class_contrast(gm.to_numpy()), M.class_contrast(gt.to_numpy())
                rows.append({**dict(zip(KEY, keys)), "quantity": "class_response",
                             "attribute": attr, "n_classes": int(len(gm)),
                             "contrast_mae": float(np.mean(
                                 [abs(cm[p] - ct[p]) for p in cm])),
                             "compression_ratio": M.heterogeneity_compression(
                                 gm.to_numpy(), gt.to_numpy()),
                             "class_means_model": ";".join("%.4f" % v for v in gm),
                             "class_means_true": ";".join("%.4f" % v for v in gt),
                             "contrast_model": ";".join("%s=%.4f" % (p, cm[p]) for p in cm),
                             "contrast_true": ";".join("%s=%.4f" % (p, ct[p]) for p in ct)})
        else:  # RC — 개인 VOT 분포
            band = pd.read_parquet(SCEN / "test_scenario_grid.parquet")                      .drop_duplicates("case_id").set_index("case_id").distance_band
            h = {a: np.array([lad["steps"][a][band[c]] for c in pm.index])
                 for a in ("fare", "ride")}
            dF = M.local_slopes(pm["F_P1"].to_numpy(), pm["F_M1"].to_numpy(), h["fare"])
            dR = M.local_slopes(pm["R_P1"].to_numpy(), pm["R_M1"].to_numpy(), h["ride"])
            vm, ok = M.vot_from_slopes(dF, dR, float(thr["vot_denominator"]))
            s = pd.Series(np.where(ok, vm, np.nan), index=pm.index).groupby(
                rid.loc[pm.index]).mean()
            vt = rc.vot_true.reindex(s.index)
            m = s.notna() & vt.notna()
            if m.sum() < 10:
                continue
            a, b = s[m].to_numpy(), vt[m].to_numpy()
            q = pd.qcut(b, 3, labels=["low", "mid", "high"], duplicates="drop")
            sub = {str(lv): float(a[q == lv].mean()) for lv in q.categories}
            subt = {str(lv): float(b[q == lv].mean()) for lv in q.categories}
            rows.append({**dict(zip(KEY, keys)), "quantity": "rc_vot_distribution",
                         "attribute": "vot", "n_respondents": int(m.sum()),
                         "estimable_share": float(m.mean()),
                         "mean_model": float(a.mean()), "mean_true": float(b.mean()),
                         "median_model": float(np.median(a)),
                         "median_true": float(np.median(b)),
                         "variance_ratio": float(a.var(ddof=0) / b.var(ddof=0))
                         if b.var(ddof=0) > 0 else np.nan,
                         "wasserstein": float(wasserstein_distance(a, b)),
                         "rank_spearman": float(spearmanr(a, b).statistic),
                         "subgroup_mean_model": ";".join("%s=%.2f" % (k, v)
                                                         for k, v in sub.items()),
                         "subgroup_mean_true": ";".join("%s=%.2f" % (k, v)
                                                        for k, v in subt.items())})
    return pd.DataFrame(rows)
