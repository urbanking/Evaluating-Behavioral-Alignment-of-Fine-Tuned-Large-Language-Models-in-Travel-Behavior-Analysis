# -*- coding: utf-8 -*-
"""Phase 8. baseline 확률과 합성 라벨 생성.

라벨은 conditional 진실에서 만든다. marginal 로 만들면 뽑아 둔 잠재상태가 선택 생성에
아무 역할을 하지 않아 이질성 세계라는 설계 취지가 무너진다.
확률 회복 평가 target 은 marginal 이다 (관측정보만 보는 예측기의 Bayes 최적).

출력
  data/synthetic/baseline_truth_{HOM,LC,NL,RC}.parquet
  data/model_inputs/{world}_{info}_{partition}.parquet
  reports/data_audit/baseline_truth.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import worlds as W
from common import DATA, REPORTS, ROOT, VIEWS, ensure_dirs, load_config, write_json
from design import build_design, respondent_level

DGP_DIR = ROOT / "artifacts" / "dgp"
SYN = DATA / "synthetic"
MODEL_IN = DATA / "model_inputs"

# 거주지(SQ1_1/SQ1_2)는 설문 원본대로 한글이고, 통행 O/D(trip_*_sido/sigungu)는 영문이다.
# 그대로 두면 표 모형이 '경기도' 와 'Gyeonggi-do' 를 무관한 두 범주로 본다 (교집합 0 확인).
# "경기도에 살면서 경기도로 출근한다" 는 정보가 통째로 사라지는데, LLM 프롬프트는
# 양쪽을 모두 영문으로 바꿔 넣으므로 이 정보를 본다 - 두 모형이 서로 다른 것을 보게 된다.
# 원본 view 는 한글 그대로 두고(설문에 충실), 두 모형이 공유하는 model_inputs 에서만 통일한다.
REGION_COLS = ("SQ1_1", "SQ1_2")


def unify_regions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """거주지 한글 지명을 통행 O/D 와 같은 로마자 표기로 바꾼다. (바뀐 셀 수를 함께 반환)"""
    import p14_llm_prompts as P14
    reg = P14.labels().get("regions", {})
    n = 0
    for c in REGION_COLS:
        if c not in df.columns:
            continue
        s = df[c].astype("object")
        new = s.map(lambda v: reg.get(str(v), v) if pd.notna(v) else v)
        n += int((new.astype(str) != s.astype(str)).sum())
        df[c] = new
    return df, n


def load_world(name):
    return json.loads((DGP_DIR / ("%s_frozen.json" % name)).read_text(encoding="utf-8"))


def cd_context(d, cfg):
    """CD 가 로그를 취하려면 원 단위 수준이 필요하다. 중심화를 되돌릴 오프셋과 거리대 코드.

    **모양을 (n, p) 로 맞추는 것이 중요하다.** Phase 10 은 시나리오마다 케이스를 부분집합
    하면서 ctx 의 값 중 첫 축이 n 인 ndarray 만 같이 자른다. 속성별 dict 로 넘기면 잘리지
    않아 행이 어긋난다.
    """
    attr = cfg["dgp_specification"]["service_attributes"]
    off = np.zeros((d.n, d.X.shape[1]), float)
    for k, col in attr.items():
        if col in d.band_center:
            cen = d.band_center[col]
            off[:, d.service_idx[k]] = np.array([cen[str(b)] for b in d.band])
    # 거리대는 **라벨로** 넘긴다. 정수 코드로 넘기면 development 와 test 에서 등장하는
    # 거리대 집합이 다를 때 코드가 밀려 조용히 다른 중심값을 빼게 된다.
    return {"offset": off, "band_code": np.array([str(b) for b in d.band], dtype=object)}


def context_for(world, d, states, cfg):
    """세계별 잠재상태를 케이스 단위로 정렬한다."""
    # **모르는 세계는 조용히 통과시키지 않는다.** 마지막 분기가 RC 용 fall-through 라서,
    # 세계를 새로 추가하고 여기 등록하지 않으면 그 세계가 RC 의 잠재상태(개인별 취향 draw)를
    # 받아 버린다. 예외가 나면 다행이고, 안 나면 조용히 틀린 참값이 만들어진다.
    # 실제로 2026-08-07 PBCD 추가 때 이 경로로 떨어졌다.
    known = {"HOM", "PB", "CD", "PBCD", "LC", "NL", "RC"}
    if world not in known:
        raise SystemExit("context_for: 모르는 세계 %r — p08 의 known 집합에 등록하라" % world)
    if world in ("HOM", "PB"):
        return {}
    if world in ("CD", "PBCD"):
        return cd_context(d, cfg)
    if world in ("LC", "NL"):
        lc = states["LC"].set_index("respondent_id")
        cls = lc.loc[d.respondent_ids, "class_true"].to_numpy(int)[d.respondent]
        K = sum(1 for c in lc.columns if c.startswith("pi_C"))
        pi = lc.loc[d.respondent_ids, ["pi_C%d" % k for k in range(K)]].to_numpy()[d.respondent]
        return {"cls": cls, "pi": pi}
    rc = states["RC"].set_index("respondent_id")
    Zr = respondent_level(d.Z, d.respondent, d.n_resp)
    nodes = W.quadrature_nodes(int(cfg["dgp"]["rc"].get("n_quadrature_nodes", 400)),
                               int(cfg["seeds"]["random_coefficient_draw"]))
    return {"lam": rc.loc[d.respondent_ids, "lambda_true"].to_numpy()[d.respondent],
            "vot": rc.loc[d.respondent_ids, "vot_true"].to_numpy()[d.respondent],
            "wtw": rc.loc[d.respondent_ids, "wtpw_true"].to_numpy()[d.respondent],
            "Q": d.Z, "nodes": nodes}


def run() -> dict:
    print("\n=== Phase 8. baseline 확률과 라벨 ===")
    cfg = load_config()
    ensure_dirs(SYN, MODEL_IN, REPORTS)

    U = pd.read_parquet(DGP_DIR / "common_choice_thresholds.parquet").set_index("case_id")
    states = {"LC": pd.read_parquet(DGP_DIR / "LC_latent_states.parquet"),
              "RC": pd.read_parquet(DGP_DIR / "RC_latent_states.parquet")}
    designs = {p: build_design(p) for p in ("development", "test")}

    summary = {}
    for world in cfg["synthetic_worlds"]:
        wj = load_world(world)
        par, si = wj["params"], wj["service_idx"]
        frames = []
        for part, d in designs.items():
            ctx = context_for(world, d, states, cfg)
            cond, marg = W.world_probs(world, d.X, si, par, ctx)
            u = U.loc[d.case_id, "u_choice"].to_numpy()
            y = (u < cond).astype(int)
            frames.append(pd.DataFrame({
                "case_id": d.case_id, "respondent_id": d.respondent_ids[d.respondent],
                "partition": part, "world": world,
                "p_true_cond": cond, "p_true_marg": marg,
                "y_synthetic": y, "u_choice": u}))
        t = pd.concat(frames, ignore_index=True)
        t.to_parquet(SYN / ("baseline_truth_%s.parquet" % world), index=False)

        dev = t[t.partition == "development"]
        rf = dev.groupby("respondent_id")[["p_true_marg", "y_synthetic"]].mean().mean()
        summary[world] = {
            "n": len(t),
            "expected_share_dev_respondent_first": float(rf["p_true_marg"]),
            "realized_label_share_dev_respondent_first": float(rf["y_synthetic"]),
            "target": wj["target"],
            "deviation": float(rf["p_true_marg"] - wj["target"]),
            "cond_marg_corr": float(np.corrcoef(t.p_true_cond, t.p_true_marg)[0, 1]),
            "av_counts_dev": int(dev.y_synthetic.sum()),
            "av_counts_test": int(t[t.partition == "test"].y_synthetic.sum()),
        }
        print("  %-4s expected %.4f / realized %.4f (target %.4f, 편차 %+.5f)  "
              "cond~marg r=%.3f  AV dev %d / test %d"
              % (world, summary[world]["expected_share_dev_respondent_first"],
                 summary[world]["realized_label_share_dev_respondent_first"],
                 wj["target"], summary[world]["deviation"],
                 summary[world]["cond_marg_corr"],
                 summary[world]["av_counts_dev"], summary[world]["av_counts_test"]))

    # ---- pro / retro 학습 view 에 라벨 join -----------------------------
    views = {i: pd.read_parquet(VIEWS / i / "baseline_cases.parquet")
             for i in cfg["information_conditions"]}
    n_reg = 0
    for i in views:
        views[i], k = unify_regions(views[i])
        n_reg += k
    made = []
    for world in cfg["synthetic_worlds"] + ["HUMAN"]:
        if world == "HUMAN":
            lab = None
        else:
            lab = (pd.read_parquet(SYN / ("baseline_truth_%s.parquet" % world))
                   [["case_id", "y_synthetic"]].set_index("case_id"))
        for info, v in views.items():
            df = v.copy()
            if lab is not None:
                df["y"] = lab.loc[df.case_id, "y_synthetic"].to_numpy()
            else:
                df["y"] = df["human_label"].to_numpy()
            for part in ("development", "test"):
                sub = df[df.partition == part]
                f = MODEL_IN / ("%s_%s_%s.parquet" % (world, info, part))
                sub.to_parquet(f, index=False)
                made.append(f.name)

    # 라벨 무결성.
    #
    # 2026-08-07 이전에는 pro 와 retro 두 view 를 만들어 `y_pro == y_retro` 를 확인했다.
    # 정보조건이 retro 하나가 되면서 그 대조가 불가능하다. 대신 **모형입력의 라벨이
    # baseline_truth 와 정확히 같은지**를 본다. 원래 확인하려던 것("라벨은 정보조건이
    # 아니라 세계가 정한다")을 더 직접적으로 검사한다.
    infos = list(cfg["information_conditions"])
    for world in cfg["synthetic_worlds"]:
        bt = (pd.read_parquet(SYN / ("baseline_truth_%s.parquet" % world))
              [["case_id", "y_synthetic"]].set_index("case_id"))
        for info in infos:
            for part in ("development", "test"):
                mi_ = pd.read_parquet(MODEL_IN / ("%s_%s_%s.parquet" % (world, info, part)))
                exp = bt.loc[mi_.case_id, "y_synthetic"].to_numpy()
                assert (mi_.y.to_numpy() == exp).all(), "%s %s %s" % (world, info, part)
    if len(infos) > 1:                       # 정보조건이 둘 이상이면 서로 같은지도 본다
        for world in cfg["synthetic_worlds"] + ["HUMAN"]:
            ref = pd.read_parquet(MODEL_IN / ("%s_%s_development.parquet" % (world, infos[0])))
            for info in infos[1:]:
                b = pd.read_parquet(MODEL_IN / ("%s_%s_development.parquet" % (world, info)))
                assert (ref.case_id.values == b.case_id.values).all() \
                    and (ref.y.values == b.y.values).all(), world
    print("  [PASS] 모형입력 라벨 == baseline_truth (정보조건 %s)" % infos)

    # 지역 어휘가 실제로 합쳐졌는지 확인한다. 통일 전에는 교집합이 0 이었다.
    chk = pd.read_parquet(MODEL_IN / ("HUMAN_%s_development.parquet" % infos[0]))
    ov = len(set(chk.SQ1_2.dropna()) & set(chk.trip_origin_sigungu.dropna()))
    left = [v for v in chk.SQ1_2.dropna().unique() if any("가" <= ch <= "힣" for ch in str(v))]
    if ov == 0 or left:
        raise SystemExit("지역 표기 통일 실패 - 교집합 %d, 한글 잔존 %s" % (ov, left[:5]))
    print("  [PASS] 지역 표기 통일  셀 %s개 변환 / 거주·통행 시군구 어휘 교집합 %d개"
          % (f"{n_reg:,}", ov))
    print("  [ok] model_inputs %d 개 생성" % len(made))

    write_json(REPORTS / "baseline_truth.json", summary)
    return summary


if __name__ == "__main__":
    run()
