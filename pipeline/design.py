# -*- coding: utf-8 -*-
"""DGP anchor 추정용 설계행렬 빌더.

명세는 configs/trc_experiment.yaml 의 dgp_specification 에서만 읽는다.
코드에 변수명이나 상수를 하드코딩하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import load_config, load_panel, restricted_mask


@dataclass
class Design:
    """설계행렬과 그 메타데이터."""
    X: np.ndarray                    # (n, p) 절편 제외
    names: list[str]
    service_idx: dict[str, int]      # fare/ride/wait -> X 열 인덱스
    y: np.ndarray
    partition: np.ndarray            # 'development' / 'test'
    respondent: np.ndarray           # 응답자 코드 (0..n_resp-1)
    respondent_ids: np.ndarray
    case_id: np.ndarray
    band: np.ndarray
    Z: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # membership
    z_names: list[str] = field(default_factory=list)
    scaler: dict = field(default_factory=dict)
    dropped: list[dict] = field(default_factory=list)
    band_center: dict = field(default_factory=dict)   # 거리대 내 중심화에 쓴 중심값

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def n_resp(self) -> int:
        return int(self.respondent.max()) + 1


def _numeric_block(df, cols, spec, scaler, prefix, dropped, fit_mask=None):
    mats, names = [], []
    for c in cols:
        if c not in df.columns:
            dropped.append({"column": c, "reason": "absent_from_panel", "block": prefix})
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() == 0 or v.nunique(dropna=True) < 2:
            dropped.append({"column": c, "reason": "degenerate", "block": prefix})
            continue
        fm = np.ones(len(v), bool) if fit_mask is None else fit_mask
        if spec["missing"]["numeric"] == "median":
            v = v.fillna(float(v[fm].median()))
        if spec.get("standardize_numeric", True):
            mu, sd = float(v[fm].mean()), float(v[fm].std(ddof=0))
            sd = sd if sd > 0 else 1.0
            scaler[c] = {"mean": mu, "sd": sd}
            v = (v - mu) / sd
        mats.append(v.to_numpy(float))
        names.append(c)
    return mats, names


def _categorical_block(df, cols, spec, prefix, dropped):
    mats, names = [], []
    cap = int(spec["max_categorical_levels"])
    for c in cols:
        if c not in df.columns:
            dropped.append({"column": c, "reason": "absent_from_panel", "block": prefix})
            continue
        s = df[c].astype("object")
        if spec["missing"]["categorical"] == "explicit_level":
            s = s.where(s.notna(), "__missing__")
        s = s.astype(str)
        levels = sorted(s.unique())
        if len(levels) < 2:
            dropped.append({"column": c, "reason": "single_level", "block": prefix})
            continue
        if len(levels) > cap:
            dropped.append({"column": c, "reason": "too_many_levels(%d>%d)" % (len(levels), cap),
                            "block": prefix})
            continue
        for lv in levels[1:]:                      # 첫 수준을 기준으로 뺀다
            mats.append((s == lv).to_numpy(float))
            names.append("%s=%s" % (c, lv))
    return mats, names


def build_design(partition: str | None = "development") -> Design:
    """제한 도메인 전체로 설계행렬을 만들고 partition 으로 잘라 준다.

    표준화 스케일러는 **development 행에서만** 적합한다. 파티션마다 따로 적합하면
    test 확률이 anchor 추정 때와 다른 척도로 계산되어 조용히 어긋난다.
    """
    cfg = load_config()
    spec = cfg["dgp_specification"]

    df_all = load_panel()
    df_all = df_all[restricted_mask(df_all)]
    df_all = df_all.sort_values(["respondent_id", "task_id"]).reset_index(drop=True)
    dev_mask = (df_all.partition == "development").to_numpy()
    df = df_all

    dropped, scaler = [], {}
    mats, names = [], []
    service_idx = {}

    # 1) 서비스 속성 — 원 단위 그대로 둔다. 표준화하면 부호제약 해석과 VOT 계산이 꼬인다.
    #    다만 거리대 내 중심화는 위치 이동이라 기울기(=계수)와 유한차분·VOT 를 바꾸지 않으면서
    #    통행 길이와의 공선성만 제거한다. 중심값은 development 에서만 계산한다.
    band_center = {}
    center = bool(spec.get("center_service_within_band", False))
    for key, col in spec["service_attributes"].items():
        v = pd.to_numeric(df[col], errors="coerce")
        v = v.fillna(float(v[dev_mask].median()))
        if center:
            m = v[dev_mask].groupby(df.loc[dev_mask, "distance_band"]).mean()
            band_center[col] = {str(k): float(x) for k, x in m.items()}
            v = v - df["distance_band"].map(m).astype(float)
        service_idx[key] = len(names)
        mats.append(v.to_numpy(float))
        names.append(col)

    # 2) 통제 + persona
    for block, node in (("control", spec["controls"]), ("persona", spec["persona"])):
        m, nm = _numeric_block(df, node.get("numeric", []), spec, scaler, block, dropped,
                               fit_mask=dev_mask)
        mats += m; names += nm
        m, nm = _categorical_block(df, node.get("categorical", []), spec, block, dropped)
        mats += m; names += nm

    X = np.column_stack(mats)

    # 3) membership 설계행렬 (응답자 수준)
    zm, znm = _numeric_block(df, spec["membership"].get("numeric", []), spec, {},
                             "membership", dropped, fit_mask=dev_mask)
    zc, zcnm = _categorical_block(df, spec["membership"].get("categorical", []), spec,
                                  "membership", dropped)
    Z = np.column_stack(zm + zc) if (zm or zc) else np.empty((len(df), 0))

    # partition 으로 자른다 (스케일러는 이미 development 기준으로 적합됨)
    keep = np.ones(len(df), bool) if not partition else (df.partition == partition).to_numpy()
    X = X[keep]
    Z = Z[keep] if Z.size else Z
    df = df[keep].reset_index(drop=True)

    codes, ids = pd.factorize(df["respondent_id"], sort=True)

    return Design(
        X=X, names=names, service_idx=service_idx, partition=df["partition"].to_numpy(),
        y=df["human_label"].to_numpy(int) if "human_label" in df
          else (df["future_mode_label"].astype(str) == "AV").to_numpy(int),
        respondent=codes.astype(int), respondent_ids=np.asarray(ids),
        case_id=df["case_id"].to_numpy(), band=df["distance_band"].to_numpy(),
        Z=Z, z_names=znm + zcnm, scaler=scaler, dropped=dropped,
        band_center=band_center,
    )


def respondent_level(Z: np.ndarray, respondent: np.ndarray, n_resp: int) -> np.ndarray:
    """task 단위 행렬을 응답자 단위로 축약한다 (응답자 내 상수 가정)."""
    out = np.zeros((n_resp, Z.shape[1]))
    first = np.zeros(n_resp, dtype=int)
    seen = np.zeros(n_resp, dtype=bool)
    for i, r in enumerate(respondent):
        if not seen[r]:
            first[r], seen[r] = i, True
    return Z[first] if Z.size else out
