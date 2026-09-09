# -*- coding: utf-8 -*-
"""CV5 전용 시나리오 ground truth - 전체 11,584 사례 격자 위에서 계산한다.

**기존 산출물과 섞지 않는다.** p10 이 만드는 data/truth/*.parquet 은 손대지 않고,
여기서는 data/truth/cv5_*.parquet 을 따로 쓴다. 주 실험(고정 분할)은 옛 진실값을,
CV5 실험만 이 진실값을 읽는다.

p10 과 다른 점은 **격자와 파티션 두 곳뿐**이다:
    p10   test_scenario_grid.parquet / build_design("test")
    여기  cv5_scenario_grid.parquet  / build_design(None)
합성세계 DGP·효용식·임계값 처리는 한 줄도 바꾸지 않는다.
"""


from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

import worlds as W
from common import CONFIG, DATA, REPORTS, ROOT, ensure_dirs, load_config, write_json
from design import build_design
from p08_baseline_truth import context_for, load_world

TRUTH = DATA / "truth"
DGP_DIR = ROOT / "artifacts" / "dgp"


def run() -> dict:
    print("\n=== Phase 10. 시나리오 ground truth ===")
    cfg = load_config()
    with open(CONFIG / "trc_band_ladders.yaml", encoding="utf-8") as f:
        lad = yaml.safe_load(f)
    ensure_dirs(TRUTH, REPORTS)

    grid = pd.read_parquet(DATA / "scenarios" / "cv5_scenario_grid.parquet")
    d = build_design(None)
    si = d.service_idx
    states = {"LC": pd.read_parquet(DGP_DIR / "LC_latent_states.parquet"),
              "RC": pd.read_parquet(DGP_DIR / "RC_latent_states.parquet")}
    U = pd.read_parquet(DGP_DIR / "common_choice_thresholds.parquet").set_index("case_id")
    u_case = U.loc[d.case_id, "u_choice"].to_numpy()

    pos = {c: i for i, c in enumerate(d.case_id)}
    scen_ids = list(dict.fromkeys(grid.scenario_id))
    print("  %d 조건 × %d cases, 세계 %s" % (len(scen_ids), d.n, cfg["synthetic_worlds"]))

    out = []
    thr = cfg["thresholds"]
    summary = {}

    for world in cfg["synthetic_worlds"]:
        wj = load_world(world)
        par = wj["params"]
        ctx = context_for(world, d, states, cfg)
        base_cond, base_marg = W.world_probs(world, d.X, si, par, ctx)

        for sid in scen_ids:
            g = grid[grid.scenario_id == sid]
            idx = np.array([pos[c] for c in g.case_id])
            X = d.X[idx].copy()
            # 격자는 원 단위, 설계행렬은 거리대 내 중심화 척도다. 중심값을 빼서 맞춘다.
            bands = g.distance_band.to_numpy()
            for key, col in (("fare", "fare_cf"), ("ride", "ride_cf"), ("wait", "wait_cf")):
                v = g[col].to_numpy(float)
                if d.band_center:
                    cen = d.band_center[cfg["dgp_specification"]["service_attributes"][key]]
                    v = v - np.array([cen[b] for b in bands])
                X[:, si[key]] = v

            # 신뢰도는 **표준화된 통제변수**다. 격자는 원 단위(%)이므로 설계 척도로 옮긴다.
            # 이게 빠지면 편집이 무시되고 A_* 조건의 확률이 BASE 와 같아진다.
            if "rel_cf" in g.columns and "av_service_reliability_pct" in d.names:
                j = d.names.index("av_service_reliability_pct")
                sc = d.scaler.get("av_service_reliability_pct")
                v = g["rel_cf"].to_numpy(float)
                if sc:
                    v = (v - sc["mean"]) / sc["sd"]
                X[:, j] = v

            c2 = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (d.n,) else v)
                  for k, v in ctx.items()}
            cond, marg = W.world_probs(world, X, si, par, c2)
            out.append(pd.DataFrame({
                "case_id": g.case_id.to_numpy(), "scenario_id": sid, "world": world,
                "distance_band": g.distance_band.to_numpy(),
                "analysis_tier": g.analysis_tier.to_numpy(),
                "p_true_cond": cond, "p_true_marg": marg,
                "delta_cond": cond - base_cond[idx],
                "delta_marg": marg - base_marg[idx],
                "y_cf": (u_case[idx] < cond).astype(int),
            }))
        print("  [ok] %s 계산 완료" % world)

    truth = pd.concat(out, ignore_index=True)
    truth.to_parquet(TRUTH / "cv5_counterfactual_truth.parquet", index=False)

    # ---- 국소 기울기와 VOT (사다리 첫 단계 h) ----------------------------
    band = dict(zip(d.case_id, d.band))
    beh = []
    for world in cfg["synthetic_worlds"]:
        t = truth[truth.world == world]
        piv = t.pivot_table(index="case_id", columns="scenario_id",
                            values="p_true_marg", aggfunc="first")
        h = {a: np.array([lad["steps"][a][band[c]] for c in piv.index]) for a in
             ("fare", "ride", "wait")}
        D = {}
        for a, (pcol, mcol) in (("fare", ("F_P1", "F_M1")), ("ride", ("R_P1", "R_M1")),
                                ("wait", ("W_P1", "W_M1"))):
            D[a] = (piv[pcol].to_numpy() - piv[mcol].to_numpy()) / (2 * h[a])
        ok = (D["fare"] < 0) & (D["ride"] < 0) & (np.abs(D["fare"]) >= float(thr["vot_denominator"]))
        vot = np.where(ok, 60 * D["ride"] / np.where(D["fare"] == 0, np.nan, D["fare"]), np.nan)
        beh.append(pd.DataFrame({"case_id": piv.index, "world": world,
                                 "D_fare": D["fare"], "D_ride": D["ride"], "D_wait": D["wait"],
                                 "vot_true": vot, "vot_estimable": ok}))
    behav = pd.concat(beh, ignore_index=True)
    behav.to_parquet(TRUTH / "cv5_behavioral_truth.parquet", index=False)

    # ---- 시나리오 share (respondent-first) -------------------------------
    r_of = dict(zip(d.case_id, d.respondent_ids[d.respondent]))
    truth["respondent_id"] = truth.case_id.map(r_of)
    sh = (truth.groupby(["world", "scenario_id", "respondent_id"])["p_true_marg"].mean()
               .groupby(["world", "scenario_id"]).mean().rename("ms_av").reset_index())
    base_ms = sh[sh.scenario_id == "BASE"].set_index("world")["ms_av"]
    sh["delta_ms"] = sh.ms_av - sh.world.map(base_ms)
    sh.to_parquet(TRUTH / "cv5_scenario_share_truth.parquet", index=False)

    for world in cfg["synthetic_worlds"]:
        b = behav[behav.world == world]
        s = sh[(sh.world == world) & (sh.scenario_id.str.startswith("S"))]
        summary[world] = {
            "baseline_share": float(base_ms[world]),
            "vot_estimable_share": float(b.vot_estimable.mean()),
            "vot_median": float(np.nanmedian(b.vot_true)),
            "delta_ms_range": [float(s.delta_ms.min()), float(s.delta_ms.max())],
            "mean_abs_delta_marg": float(truth[(truth.world == world) &
                                               (truth.scenario_id != "BASE")]
                                         .delta_marg.abs().mean()),
        }
        print("  %-4s baseline share %.4f | VOT 추정가능 %.1f%% 중앙값 %.2f | "
              "ΔMS [%.4f, %.4f]"
              % (world, summary[world]["baseline_share"],
                 100 * summary[world]["vot_estimable_share"], summary[world]["vot_median"],
                 *summary[world]["delta_ms_range"]))

    write_json(REPORTS / "scenario_truth.json", summary)
    print("  [ok] truth 3종 저장 (%d 행)" % len(truth))
    return summary


if __name__ == "__main__":
    run()
