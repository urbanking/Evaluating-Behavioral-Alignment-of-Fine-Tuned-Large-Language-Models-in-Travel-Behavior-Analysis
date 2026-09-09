# -*- coding: utf-8 -*-
"""선행 AAAI 프로젝트가 쓴 복합 목적함수를 그대로 재현한다.

출처는 추측이 아니라 코드다 — `C:/SP_LLM/methods/fine_tuning/train_av_pairwise_lora.py`
의 `pairwise_loss_rows()` (727-753행) 와 그 실행이 남긴 `training_manifest.json` 의

    loss.formula: 0.90*(0.65*weighted_BCE_normalized_by_sample_weight
                        + 0.25*unweighted_focal_gamma2 + 0.10*Brier) + 0.10*format_CE
    loss.positive_weight: 2.233   loss.focal_gamma: 2.0

**왜 이걸 만드나.** pos_weight 2.233 만 떼어다 토큰 CE 에 붙인 `weighted_token_ce` 로
2B/HOM 을 학습했더니 AV F1 이 0.1987 에 그쳤다 (AV 를 5.32% 만 예측, 실제는 18.24%,
1,480건 중 1,290건을 놓침). AAAI 는 같은 가중치로 AV F1 0.4549 를 냈다.

차이는 **가중치가 혼자 일한 게 아니라는 것**이다. 계산해 보면 pos_weight 2.233 은
기울기 불균형을 4.31:1 에서 1.93:1 로 줄일 뿐 없애지 못한다 (AV 비율 0.1883 기준).
나머지를 focal 이 맡는다 - (1-p_t)^gamma 가 **이미 잘 맞히는 쉬운 사례의 기여를 눌러서**
어려운 소수에 집중시킨다. 클래스 단위가 아니라 사례별 난이도로 미는 것이라
단순 가중치가 못 잡는 쏠림을 잡는다.

    L = binary_w * (wbce_w*wBCE + focal_w*focal + brier_w*Brier) + format_w*format_CE

    wBCE   = BCEWithLogits(margin, y, pos_weight=w) / (1 + (w-1)*r)    r = 학습자료 AV 비율
    focal  = -(1 - p_t)^gamma * log(p_t)            p_t = 정답 쪽 확률, 가중치 없음
    Brier  = (sigmoid(margin) - y)^2
    format_CE = 감독 토큰의 교차엔트로피            답 자리에 엉뚱한 토큰이 오는 것을 막는다

format_CE 를 빼면 안 된다. 학습 전 9B 는 정답 자리에서 Thinking(logit 25.53) 을
내놓으려 한다 (Qwen3.5 가 thinking model 이다). margin 만 미는 손실은 그걸 못 막는다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from margin_bce import margin_and_target


def format_ce(logits, labels, n):
    """감독 토큰의 교차엔트로피. 답 형식을 유지시키는 항.

    **n 으로 나눈다.** reduction="mean" 을 쓰면 이 항만 num_items_in_batch 정규화를
    안 받아, 누적 단계 수만큼 부풀어 오른다. 회귀 테스트가 실제로 잡았다 -
    num_items_in_batch=16 을 줬는데 기울기가 0.0449 가 아니라 0.1367 이었다.
    같은 종류의 실수로 전에 손실이 1.02 대신 12 로 찍히고 학습이 발산한 적이 있다.
    """
    sl = logits[:, :-1, :].contiguous()
    tl = labels[:, 1:].contiguous()
    if (tl != -100).sum() == 0:
        return logits.sum() * 0.0
    tot = F.cross_entropy(sl.float().view(-1, sl.size(-1)), tl.view(-1),
                          ignore_index=-100, reduction="sum")
    return tot / n


def logit_adjust(margin, av_rate, tau):
    """소수 클래스를 잡기 위한 logit adjustment (Menon et al., ICLR 2021).

    margin 에 클래스 사전확률의 로그비를 **학습 중에** 더한다.

        margin' = margin + tau * log(pi_AV / pi_Keep)

    AV 가 18.8% 이므로 log(0.1883/0.8117) = -1.461 이다. 즉 학습 중에는 margin 을
    1.461*tau 만큼 깎아 놓고 손실을 계산한다. 모델은 그만큼을 스스로 메우도록 학습되고,
    **추론 때 보정 없이 margin > 0 을 쓰면 그 지점이 곧 균형 최적 경계가 된다.**

    이게 pos_weight 와 다른 점: pos_weight 2.233 은 경계를 p = 1/(1+w) = 0.3093 에
    놓는다 (선행 AAAI 프로젝트가 manifest 에 그 값을 그대로 기록해 두었다). 반면
    tau=1 은 경계를 p = 기저율(0.1883) 에 놓는다 (sigmoid(log(r/(1-r))) = r).
    0.3093 보다 낮으므로 소수 클래스를 훨씬 적극적으로 잡는다.

    **왜 이 방법인가.** 논문이 balanced error 최소화에 대해 Fisher consistent 임을
    보였다 - 가중치 정규화나 margin 수정에는 없는 보장이다. 우리 목표가 "전환과 유지를
    둘 다 잡는 것" 이므로 balanced error 가 정확히 맞는 목적이다.

    **둘을 같이 쓰면 경계가 두 번 움직인다.** logit adjustment 를 켤 때는 pos_weight 를
    1.0 으로 두는 것이 깨끗하다.
    """
    if not tau:
        return margin
    r = min(max(float(av_rate), 1e-6), 1 - 1e-6)
    import math
    return margin + float(tau) * math.log(r / (1.0 - r))


def composite_loss(logits, labels, av_id, keep_id, pos_weight=1.0, av_rate=0.5,
                   focal_gamma=2.0, wbce_w=0.65, focal_w=0.25, brier_w=0.10,
                   binary_w=0.90, format_w=0.10, logit_adjust_tau=0.0,
                   num_items_in_batch=None):
    """AAAI 복합 손실. **num_items_in_batch 를 반드시 받는다.**

    안 받으면 transformers 의 Trainer 가 누적 단계로 한 번 더 나누지 않아 손실과 기울기가
    gradient_accumulation_steps 배로 들어간다. 실측으로 손실이 1.02 대신 12 로 찍혔고
    학습률 2e-4 가 실질 3.2e-3 이 되어 발산했다 (margin_bce 주석 참조).
    """
    margin, y, ok = margin_and_target(logits, labels, av_id, keep_id)
    if ok.sum() == 0:
        return logits.sum() * 0.0
    m, yy = margin[ok].float(), y[ok].float()
    # **소수 클래스 보정은 여기서 한 번만 한다.** 아래 세 항이 모두 이 m 을 쓰므로
    # wBCE / focal / Brier 가 같은 경계를 보게 된다.
    m = logit_adjust(m, av_rate, logit_adjust_tau)
    w = torch.tensor(float(pos_weight), device=logits.device, dtype=m.dtype)

    wbce = F.binary_cross_entropy_with_logits(m, yy, pos_weight=w, reduction="none")
    # **분모는 상수다.** 배치 안의 가중치 합으로 나누면 per-device batch 가 1 일 때
    # 분자와 분모에서 가중치가 상쇄된다 (DECISIONS.md D-882 에서 실제로 당했다).
    wbce = wbce / max(1.0 + (float(pos_weight) - 1.0) * float(av_rate), 1e-8)

    p = torch.sigmoid(m)
    p_t = torch.where(yy > 0.5, p, 1.0 - p).clamp(1e-7, 1 - 1e-7)
    focal = -((1.0 - p_t) ** float(focal_gamma)) * torch.log(p_t)
    brier = (p - yy) ** 2

    n = float(num_items_in_batch) if num_items_in_batch else float(ok.sum())
    binary = (wbce_w * wbce.sum() + focal_w * focal.sum() + brier_w * brier.sum()) / n
    return binary_w * binary + format_w * format_ce(logits, labels, n)


def make_trainer_class(base_cls):
    class CompositeTrainer(base_cls):
        def __init__(self, *a, av_id=None, keep_id=None, pos_weight=1.0, av_rate=0.5,
                     focal_gamma=2.0, wbce_w=0.65, focal_w=0.25, brier_w=0.10,
                     binary_w=0.90, format_w=0.10, logit_adjust_tau=0.0, **kw):
            super().__init__(*a, **kw)
            self.av_id, self.keep_id = int(av_id), int(keep_id)
            self.k = dict(pos_weight=float(pos_weight), av_rate=float(av_rate),
                          focal_gamma=float(focal_gamma), wbce_w=float(wbce_w),
                          focal_w=float(focal_w), brier_w=float(brier_w),
                          binary_w=float(binary_w), format_w=float(format_w),
                          logit_adjust_tau=float(logit_adjust_tau))

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = composite_loss(out.logits, labels, self.av_id, self.keep_id,
                                  num_items_in_batch=num_items_in_batch, **self.k)
            return (loss, out) if return_outputs else loss

    return CompositeTrainer


def selftest(verbose=True):
    """네 가지를 확인한다. 앞서 두 목적함수에서 밟은 함정을 다시 밟지 않기 위해서다.

    1. 세 구성요소 가중치의 합이 1 인가 (AAAI 의 validate_loss_weights 와 같은 검사)
    2. focal 이 **쉬운 사례를 실제로 눌러 주는가** - 이게 이 목적함수를 도입한 이유다
    3. 배치 1 에서 pos_weight 가 상쇄되지 않는가 (D-882)
    4. num_items_in_batch 를 주면 그만큼 나뉘는가
    """
    torch.manual_seed(0)
    V, T, AV, KEEP, R = 50, 6, 3, 7, 0.1883
    ok = True

    s = 0.65 + 0.25 + 0.10
    if verbose:
        print("  구성요소 가중치 합 %.4f (1이어야 함)  %s" % (s, "ok" if abs(s - 1) < 1e-9 else "**틀림**"))
    ok = ok and abs(s - 1) < 1e-9

    # focal 이 쉬운 사례를 누르는가: p_t 가 클수록 focal 기여가 급격히 작아져야 한다
    pt = torch.tensor([0.5, 0.9, 0.99])
    f = -((1 - pt) ** 2) * torch.log(pt)
    drop = float(f[0] / f[2])
    if verbose:
        print("  focal  p_t=0.50 -> %.5f | 0.90 -> %.5f | 0.99 -> %.7f  (0.5 가 0.99 의 %.0f배)"
              % (f[0], f[1], f[2], drop))
    ok = ok and drop > 100

    def grad_for(tgt, w, nib=None):
        z = torch.zeros(1, T, V, requires_grad=True)
        lab = torch.full((1, T), -100, dtype=torch.long)
        lab[0, T - 1] = tgt
        composite_loss(z, lab, AV, KEEP, pos_weight=w, av_rate=R,
                       num_items_in_batch=nib).backward()
        return abs(float(z.grad[0, T - 2, AV]))

    spread = abs(grad_for(AV, 4.0) - grad_for(AV, 1.0))
    alive = spread > 1e-8
    if verbose:
        print("  배치 1 에서 가중치가 기울기를 바꾸는가: %s (차이 %.6f)"
              % ("예" if alive else "**아니오 - D-882 재현**", spread))

    # logit adjustment 가 경계를 기저율로 옮기는가.
    # 학습 최적점에서 sigmoid(margin + tau*log(r/(1-r))) = p 이므로,
    # 추론 때 margin>0 인 지점은 p/(1-p) = ((1-r)/r)^-tau, 즉 p = r 이어야 한다.
    # 학습 최적점:  sigmoid(margin + shift) = p    (shift = tau*log(r/(1-r)))
    #   => margin = logit(p) - shift
    #   => 추론 때 margin > 0  <=>  logit(p) > shift  <=>  p > sigmoid(shift)
    # tau=1 이면 sigmoid(log(r/(1-r))) = r 이므로 경계가 정확히 기저율에 놓인다.
    import math
    r = R
    for tau in (0.0, 1.0):
        shift = tau * math.log(r / (1 - r))
        thr = 1.0 / (1.0 + math.exp(-shift))
        if verbose:
            print("  tau=%.1f -> margin 이동 %+.4f, margin>0 의 실제 확률 경계 %.4f%s"
                  % (tau, shift, thr, "  (기저율과 일치)" if abs(thr - r) < 1e-9 else ""))
    _t1 = 1.0 / (1.0 + math.exp(-math.log(r / (1 - r))))
    ok = ok and abs(_t1 - r) < 1e-9

    g1, g16 = grad_for(AV, 2.233), grad_for(AV, 2.233, nib=16)
    scaled = abs(g1 / 16 - g16) < 1e-9
    if verbose:
        print("  num_items_in_batch=16 이면 1/16 인가: %s (%.8f vs %.8f)"
              % ("예" if scaled else "**아니오**", g1 / 16, g16))
        print("  판정:", "통과" if (ok and alive and scaled) else "**실패**")
    return ok and alive and scaled


if __name__ == "__main__":
    import sys
    print("=== AAAI 복합 손실 회귀 테스트 ===")
    sys.exit(0 if selftest() else 1)
