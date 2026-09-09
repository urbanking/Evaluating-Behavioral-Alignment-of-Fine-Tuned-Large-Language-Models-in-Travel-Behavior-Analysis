# -*- coding: utf-8 -*-
"""CV5 33조건 채점 결과 -> 실험 2 원값 표 (RAW-FIRST).

CLAUDE.md 의 RAW-METRIC FIRST 규칙을 따른다: 파생 요약(방향 일치 여부) 이전에
사다리 칸마다의 **실제 mean Delta P(AV)** 를 먼저 낸다.

확률. D321 이 등록한 Qwen 확률 sigmoid(q_AV - q_Keep) 을 쓴다. 사후 보정 없음.
Delta P(AV) = P(edited) - P(BASE), 사례별로 짝지어 계산한 뒤 평균낸다.

측정 바닥. margin 이 fp16 log-softmax 때문에 0.125 단위로만 움직인다(D324).
사례 하나의 변화는 못 읽는다. 그래서 fold 당 약 2,300 사례의 평균으로만 읽는다.
표에 그 사실을 같이 적는다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions" / "cv5_llm"
OUT = ROOT / "results"
FOLDS = (1, 2, 3, 4, 5)
LADDER = {"fare": ["F_M3", "F_M2", "F_M1", "BASE", "F_P1", "F_P2", "F_P3"],
          "ride": ["R_M3", "R_M2", "R_M1", "BASE", "R_P1", "R_P2", "R_P3"],
          "wait": ["W_M3", "W_M2", "W_M1", "BASE", "W_P1", "W_P2", "W_P3"],
          "rel":  ["A_M3", "A_M2", "A_M1", "BASE", "A_P1", "A_P2", "A_P3"]}
BUNDLES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def main(family="qwen_mid"):
    files = {f: PRED / ("%s_f%d_scenarios.parquet" % (family, f)) for f in FOLDS}
    have = {f: p for f, p in files.items() if p.exists()}
    print("%s: %d/5 폴드 존재" % (family, len(have)))
    if len(have) < 5:
        print("  없는 폴드: %s -> 아래 값은 %d 폴드만의 것이다."
              % ([f for f in FOLDS if f not in have], len(have)))
    if not have:
        return 1

    frames = []
    for f, p in sorted(have.items()):
        d = pd.read_parquet(p)
        d["p"] = 1.0 / (1.0 + np.exp(-d["margin"].to_numpy(float)))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print("  총 %d행 | 사례 %d | 조건 %d | 후보질량 평균 %.4f"
          % (len(df), df.case_id.nunique(), df.scenario_id.nunique(), df.candidate_mass.mean()))

    base = df[df.scenario_id == "BASE"].set_index(["fold", "case_id"])["p"]
    df = df.set_index(["fold", "case_id"])
    df["p_base"] = base
    df["dp"] = df["p"] - df["p_base"]
    df = df.reset_index()
    print("  BASE 평균 P(AV) = %.4f  (실제 AV율 %.4f)"
          % (base.mean(), df[df.scenario_id == "BASE"].y.mean()))

    OUT.mkdir(exist_ok=True)
    # ---- 표 1: 사다리 칸별 원값 (fold 평균 +/- fold 표준편차) ----------------
    rows = []
    for attr, sids in LADDER.items():
        for sid in sids:
            g = df[df.scenario_id == sid]
            if not len(g):
                continue
            per_fold = g.groupby("fold")["dp"].mean()
            rows.append({"attribute": attr, "scenario_id": sid,
                         "rung": sids.index(sid) - 3,
                         "n_cases": int(g.case_id.nunique()),
                         "mean_dP": float(per_fold.mean()),
                         "sd_across_folds": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else np.nan,
                         "mean_P": float(g["p"].mean()),
                         **{("f%d" % f): float(per_fold.get(f, np.nan)) for f in FOLDS}})
    lad = pd.DataFrame(rows)
    lad.to_csv(OUT / ("cv5_llm_%s_exp2_ladder_raw.csv" % family), index=False)

    for attr in LADDER:
        sub = lad[lad.attribute == attr]
        if not len(sub):
            continue
        print("\n--- %s 사다리: mean Delta P(AV), fold 평균 +/- fold SD ---" % attr.upper())
        print("  rung  scenario   mean dP      fold SD    mean P(AV)")
        for _, r in sub.iterrows():
            print("  %+4d  %-9s %+.4f     %s     %.4f"
                  % (r["rung"], r["scenario_id"], r["mean_dP"],
                     ("  --  " if np.isnan(r["sd_across_folds"]) else "%.4f" % r["sd_across_folds"]),
                     r["mean_P"]))

    # ---- 표 2: 정책 번들 ----------------------------------------------------
    brows = []
    for sid in BUNDLES:
        g = df[df.scenario_id == sid]
        if not len(g):
            continue
        pf = g.groupby("fold")["dp"].mean()
        brows.append({"scenario_id": sid, "n_cases": int(g.case_id.nunique()),
                      "mean_dP": float(pf.mean()),
                      "sd_across_folds": float(pf.std(ddof=1)) if len(pf) > 1 else np.nan,
                      "mean_P": float(g["p"].mean())})
    bun = pd.DataFrame(brows)
    bun.to_csv(OUT / ("cv5_llm_%s_exp2_bundles_raw.csv" % family), index=False)
    print("\n--- 정책 번들: mean Delta P(AV) ---")
    for _, r in bun.iterrows():
        print("  %-3s  %+.4f  (fold SD %.4f)  mean P=%.4f"
              % (r["scenario_id"], r["mean_dP"], r["sd_across_folds"], r["mean_P"]))

    # ---- 파생 요약(2차): 단조성과 방향 -------------------------------------
    print("\n--- 파생 요약 (2차 지표) ---")
    for attr in ("fare", "ride", "wait"):
        sub = lad[lad.attribute == attr].sort_values("rung")
        v = sub.mean_dP.to_numpy()
        mono = bool(np.all(np.diff(v) <= 1e-9))
        print("  %-5s: -3칸 %+.4f -> +3칸 %+.4f | 단조감소 %s"
              % (attr, v[0], v[-1], "예" if mono else "아니오"))
    print("\n  주의(D324): margin 이 0.125 단위로만 움직인다. 위 값은 fold 당 약 2,300")
    print("  사례를 평균낸 것이며, 사례 하나 수준의 변화는 이 방식으로 읽을 수 없다.")
    print("\n저장: %s" % (OUT / ("cv5_llm_%s_exp2_ladder_raw.csv" % family)))
    print("저장: %s" % (OUT / ("cv5_llm_%s_exp2_bundles_raw.csv" % family)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "qwen_mid"))
