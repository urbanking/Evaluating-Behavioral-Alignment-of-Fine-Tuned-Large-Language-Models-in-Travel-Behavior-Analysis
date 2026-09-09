# -*- coding: utf-8 -*-
"""동결된 네 행동세계의 확률 계산 코어.

각 세계는 임의의 설계행렬 X (baseline 이든 편집본이든) 에 대해
conditional / marginal 진실 확률을 낸다.

  P_cond : 실현된 잠재상태에 조건부. 라벨 생성에 쓴다.
  P_marg : 관측정보에 조건부(잠재상태 적분). 확률회복 평가 target 이다.

HOM 은 잠재상태가 없으므로 둘이 같다.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import norm, qmc

U_COLS = ("fare", "ride", "wait")
NL_COMPONENTS = ("fare_sq", "ride_sq", "wait_sq", "fare_x_ride", "ride_x_wait")


# --------------------------------------------------------------------- HOM

def hom_prob(X, par):
    return expit(par["alpha"] + par["shift"] + X @ np.asarray(par["beta"]))


# ---------------------------------------------------------------------- PB
#
# probit. 체계적 효용은 HOM 과 같은 선형식이고 **링크만 다르다**. 이 세계의 목적은
# 함수형태를 그대로 두고 오차분포만 바꿨을 때 후보모형의 행동 복원이 어떻게 달라지는지를
# 보는 것이다. HOM 과 짝을 이뤄 "링크" 축을 분리한다.

def pb_prob(X, par):
    return norm.cdf(par["alpha"] + par["shift"] + X @ np.asarray(par["beta"]))


# ---------------------------------------------------------------------- CD
#
# Cobb-Douglas. 링크는 로짓 그대로이고 **서비스 속성이 로그로 들어간다**. HOM 과 짝을
# 이뤄 "함수형태" 축을 분리한다.
#
# 세 가지를 문서에 남길 것.
#  (1) 설계행렬의 서비스 열은 거리대 내 중심화가 되어 있어 음수가 될 수 있다. 로그를
#      취하려면 원 단위 수준이 필요하므로 ctx 로 받은 offset 을 더해 되돌린다.
#  (2) 로그를 취한 뒤 **다시 거리대 내 중심화**한다. 중심화를 빼면 거리대 간 변동이
#      계수를 흡수해 HOM 에서와 같은 공선성 문제가 돌아온다(명세 3절).
#  (3) 사다리 끝에서 fare 가 0, wait 이 음수가 될 수 있어 로그가 정의되지 않는다.
#      격자 눈금(configs/trc_band_ladders.yaml 의 grid)을 바닥으로 둔다. 설계범위 안에서는
#      바닥이 작동하지 않고, 작동하는 칸은 이미 EXTRAPOLATION/INVALID 로 표시된 자리다.
#
# 확률 탄력성은 beta*(1-P) 가 되어 **x 에 의존하지 않는다**. 선형 세계에서는
# beta*x*(1-P) 라 사례마다 다르다. 즉 CD 세계는 탄력성 회복을 훨씬 깨끗하게 채점한다.

def cd_goods(X, si, par, offset, band_code):
    """서비스 속성을 **재화**(클수록 좋음)로 바꿔 [0,1] 로 정규화한다.

        g_k = (band_max_k(b) - x_k) / (band_max_k(b) - band_min_k(b))

    요금이 쌀수록 g 가 1 에 가깝다. 참고 논문의 x, I 는 재화이고 [0,1] 위에 있으므로
    (그래서 beta > 0), 우리 속성(요금·시간 = 비재화)을 그 형태로 옮기려면 이 변환이 필요하다.

    **반드시 거리대 안에서 정규화한다.** 전역 범위로 정규화하면 g 가 서비스 변동이 아니라
    통행 길이를 담게 된다. 설계가 통행시간을 주로 대역 사이에서 변화시켜 R2(ride~대역더미)
    = 0.977 이기 때문이다(명세 3절). 실측: 전역 정규화로 만든 곱항은 y 와 상관이 -0.090 으로
    **부호가 거꾸로** 나왔고, 우도는 그 항의 계수를 0 으로 눌렀다. 나머지 설계가 전부
    거리대 내 중심화를 쓰는 것과 같은 이유다.
    """
    G = {}
    lo, hi = par["good_min"], par["good_max"]     # {attr: {band: value}}
    bands = np.asarray([str(b) for b in band_code])
    for k, j in si.items():
        raw = np.asarray(X)[:, j] + offset[:, j]
        b_lo = np.array([float(lo[k][b]) for b in bands])
        b_hi = np.array([float(hi[k][b]) for b in bands])
        g = (b_hi - raw) / np.maximum(b_hi - b_lo, 1e-9)
        G[k] = np.clip(g, 1e-4, 1.0)          # 0 을 피한다: 로그미분이 발산
    return G


def cd_product(X, si, par, offset, band_code):
    """정규화 없는 곱  Prod_k g_k^{beta_k}  (모양만 담당)."""
    G = cd_goods(X, si, par, offset, band_code)
    b = par["beta_cd"]
    z = np.ones(len(np.asarray(X)))
    for k in si:
        z = z * (G[k] ** float(b[k]))
    return z


def cd_index(X, si, par, offset, band_code):
    """곱형태 Cobb-Douglas 효용항.  V = gamma * standardize(Prod_k g_k^{beta_k}).

    **원 단위 그대로 곱하면 추정이 안 된다 (실측).** 요금 4~28.5, 시간 4~62 스케일에서
    Prod x^beta 는 계수를 -3.8 까지 밀어 효용항이 0 으로 붕괴한다. 그래서 재화로 정규화한다.

    **재화로 바꾸는 것만으로도 부족하다 (실측).** 요금 중앙값이 6 이고 범위가 4~28.5 라
    g = (28.5-x)/24.5 가 대부분 0.9 근처에 몰린다. 거기에 작은 지수를 씌우면 곱이 거의
    상수가 되고, 우도는 다시 gamma 를 0 으로 눌러 통제변수만 남긴다.

    표준화가 그 조건수를 고친다. `gamma*(z-m)/s = (gamma/s)*z - gamma*m/s` 이므로 절편이
    두 번째 항을 흡수한다. **모형은 그대로이고 좌표만 바꾼 것이다.** beta 가 모양을,
    gamma 가 폭을 맡게 되어 둘이 서로를 죽이지 않는다.
    """
    z = cd_product(X, si, par, offset, band_code)
    m, s = float(par["z_mean"]), float(par["z_sd"])
    return float(par["gamma"]) * (z - m) / max(s, 1e-12)


def cd_utility(X, si, par, offset, band_code):
    """곱형태 효용항 + 선형 통제변수. 서비스 열은 통제 쪽에서 빼고 센다."""
    Xa = np.asarray(X, float)
    ctrl_idx = [j for j in range(Xa.shape[1]) if j not in set(si.values())]
    return (par["alpha"] + par["shift"] + cd_index(X, si, par, offset, band_code)
            + Xa[:, ctrl_idx] @ np.asarray(par["beta_ctrl"], float))


def cd_prob(X, si, par, offset, band_code):
    return expit(cd_utility(X, si, par, offset, band_code))


# ---------------------------------------------------------------------- LC

def lc_class_probs(X, par):
    """(n, K) 클래스별 확률."""
    return np.column_stack([
        expit(c["alpha"] + par["shift"] + X @ np.asarray(c["beta"]))
        for c in par["classes"]])


def lc_marg(X, par, pi):
    return np.sum(lc_class_probs(X, par) * pi, axis=1)


def lc_cond(X, par, cls):
    P = lc_class_probs(X, par)
    return P[np.arange(len(P)), cls]


# ---------------------------------------------------------------------- NL

def nl_components(X, si, u_ref):
    """[0,1] 스케일 후 중심화된 5개 비선형 성분."""
    u = {}
    for k in U_COLS:
        lo, hi = u_ref[k]["min"], u_ref[k]["max"]
        u[k] = (X[:, si[k]] - lo) / (hi - lo if hi > lo else 1.0)
    raw = {
        "fare_sq": u["fare"] ** 2,
        "ride_sq": u["ride"] ** 2,
        "wait_sq": u["wait"] ** 2,
        "fare_x_ride": u["fare"] * u["ride"],
        "ride_x_wait": u["ride"] * u["wait"],
    }
    return np.column_stack([
        (raw[q] - u_ref["components"][q]["mean"]) / u_ref["components"][q]["sd"]
        for q in NL_COMPONENTS])


def nl_class_probs(X, par, si):
    H = nl_components(X, si, par["u_ref"])
    pen = H @ (np.ones(len(NL_COMPONENTS)) * par["omega"])       # 등가중 합
    return np.column_stack([
        expit(c["alpha"] + par["shift"] + X @ np.asarray(c["beta"]) - par["c_k"][k] * pen)
        for k, c in enumerate(par["classes"])])


def nl_marg(X, par, si, pi):
    return np.sum(nl_class_probs(X, par, si) * pi, axis=1)


def nl_cond(X, par, si, cls):
    P = nl_class_probs(X, par, si)
    return P[np.arange(len(P)), cls]


# ---------------------------------------------------------------------- RC

def rc_taste(Q, par, nu):
    """nu: (m, 3) 표준정규 -> (m, 3) 취향 파라미터 (lambda, VOT, WTPW)."""
    mu = np.asarray(par["mu"]); B = np.asarray(par["B"]); L = np.asarray(par["L"])
    base = mu[None, :] + Q @ B.T
    v = base + nu @ L.T
    v = np.clip(v, -20, 20)
    return -np.exp(v[:, 0]), np.exp(v[:, 1]), np.exp(v[:, 2])


def rc_eta(X, si, ctrl_idx, par, lam, vot, wtw):
    F, R, W = X[:, si["fare"]], X[:, si["ride"]], X[:, si["wait"]]
    C = X[:, ctrl_idx]
    return (par["alpha"] + par["shift"] + C @ np.asarray(par["gamma"])
            + lam * (F + vot * R / 60.0 + wtw * W / 60.0))


def rc_cond(X, si, ctrl_idx, par, lam_i, vot_i, wtw_i):
    return expit(np.clip(rc_eta(X, si, ctrl_idx, par, lam_i, vot_i, wtw_i), -30, 30))


def quadrature_nodes(n_nodes, seed):
    """고정 노드. 재샘플링하지 않으므로 marginal 진실이 결정적이다."""
    h = qmc.Halton(d=3, scramble=True, seed=seed).random(n_nodes)
    return norm.ppf(np.clip(h, 1e-6, 1 - 1e-6))


def rc_marg(X, si, ctrl_idx, par, Q_case, nodes, chunk=64):
    """Q_case: (n, q) 케이스별 응답자 persona. nodes: (m, 3)."""
    n = len(X)
    acc = np.zeros(n)
    for s in range(0, len(nodes), chunk):
        nb = nodes[s:s + chunk]
        for j in range(len(nb)):
            lam, vot, wtw = rc_taste(Q_case, par, np.repeat(nb[j:j + 1], n, axis=0))
            acc += expit(np.clip(rc_eta(X, si, ctrl_idx, par, lam, vot, wtw), -30, 30))
    return acc / len(nodes)


# ------------------------------------------------------------- 공통 인터페이스

def world_probs(world, X, si, par, ctx):
    """(P_cond, P_marg) 반환. ctx 는 케이스별 잠재상태·persona 를 담는다."""
    if world == "HOM":
        p = hom_prob(X, par)
        return p, p
    if world == "CD":
        p = cd_prob(X, si, par, ctx["offset"], ctx["band_code"])
        return p, p
    if world == "LC":
        return lc_cond(X, par, ctx["cls"]), lc_marg(X, par, ctx["pi"])
    if world == "NL":
        return nl_cond(X, par, si, ctx["cls"]), nl_marg(X, par, si, ctx["pi"])
    if world == "RC":
        cond = rc_cond(X, si, par["control_idx"], par,
                       ctx["lam"], ctx["vot"], ctx["wtw"])
        marg = rc_marg(X, si, par["control_idx"], par, ctx["Q"], ctx["nodes"])
        return cond, marg
    raise ValueError(world)
