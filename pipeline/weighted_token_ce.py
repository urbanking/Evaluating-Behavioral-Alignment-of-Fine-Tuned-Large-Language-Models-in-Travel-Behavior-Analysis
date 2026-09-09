# -*- coding: utf-8 -*-
"""class 가중 토큰 교차엔트로피. 표준 SFT 와 margin_bce 의 중간.

**왜 만들었나.** 두 기존 목적함수가 각각 반대쪽 결함을 갖는다.

  token_ce    정답 토큰의 우도를 올린다. 자료가 Keep 81.2% / AV 18.8% 라 "무조건 Keep"
              이 손실을 줄이는 지름길이다. 실측으로 margin 평균이 -1.753 이었다.
  margin_bce  AV 와 Keep 두 로짓의 차이만 민다. 불균형은 제대로 다루지만 **나머지 어휘를
              전혀 건드리지 않는다.** 그래서 정답 자리에 엉뚱한 토큰이 오는 것을 못 막는다.
              실측: 학습 전 9B 는 그 자리에서 Thinking(25.53) / The(21.97) 을 내놓으려 했다
              (Qwen3.5 가 thinking model 이라 추론 블록부터 쓰려 한다). 생성
              (p18b_generate.py)을 쓸 거면 이건 실제 문제다.

이 목적함수는 **토큰 교차엔트로피를 그대로 두고 행마다 가중치만 건다.**

    L = sum_i  w_i * CE_i  /  ( N * (1 + (w-1) * r) )

    w_i = pos_weight   정답이 AV 인 행
    w_i = 1.0          정답이 Keep 인 행
    r   = 학습자료 전체의 AV 비율

**분모가 상수인 것이 핵심이다.** margin_bce 주석과 같은 이유다 - 배치 안의 가중치 합으로
나누면 per-device batch 가 1 일 때 분자와 분모에서 가중치가 상쇄되어, 가중치를 뭘 주든
손실과 기울기가 똑같아진다 (DECISIONS.md D-882 에서 실제로 겪었다).

**num_items_in_batch 를 반드시 받는다.** 안 받으면 transformers 의 Trainer 가 누적 단계로
한 번 더 나누지 않아 손실과 기울기가 gradient_accumulation_steps 배로 들어간다. 실측으로
손실이 1.02 대신 12 로 찍혔고 학습이 발산했다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from margin_bce import answer_slice


def row_weights(labels, av_id, pos_weight):
    """행마다 가중치. 정답 첫 토큰이 av_id 면 pos_weight, 아니면 1.0."""
    pos = answer_slice(labels)
    ok = pos > 0
    b = torch.arange(labels.size(0), device=labels.device)
    tgt = labels[b, torch.clamp(pos, min=0)]
    w = torch.where(tgt == av_id,
                    torch.full_like(tgt, 0, dtype=torch.float).fill_(float(pos_weight)),
                    torch.ones_like(tgt, dtype=torch.float))
    return w, ok


def weighted_token_ce_loss(logits, labels, av_id, pos_weight=1.0, av_rate=0.5,
                           num_items_in_batch=None):
    """행 가중 토큰 CE. 감독되는 모든 토큰에 그 행의 가중치를 곱한다."""
    # 로짓은 한 칸 앞에서 다음 토큰을 예측한다.
    sl = logits[:, :-1, :].contiguous()
    tl = labels[:, 1:].contiguous()
    valid = tl != -100
    if valid.sum() == 0:
        return logits.sum() * 0.0

    per_tok = F.cross_entropy(sl.float().view(-1, sl.size(-1)), tl.view(-1),
                              ignore_index=-100, reduction="none").view(tl.shape)
    w_row, _ = row_weights(labels, av_id, pos_weight)
    per = (per_tok * valid.float() * w_row.unsqueeze(1)).sum()

    denom = 1.0 + (float(pos_weight) - 1.0) * float(av_rate)
    n = float(num_items_in_batch) if num_items_in_batch else float(valid.sum())
    return per / n / denom


def make_trainer_class(base_cls):
    """HF Trainer 를 감싸 compute_loss 만 갈아끼운다."""

    class WeightedTokenCETrainer(base_cls):
        def __init__(self, *a, av_id=None, pos_weight=1.0, av_rate=0.5, **kw):
            super().__init__(*a, **kw)
            self.av_id = int(av_id)
            self.pos_weight, self.av_rate = float(pos_weight), float(av_rate)

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = weighted_token_ce_loss(out.logits, labels, self.av_id,
                                          self.pos_weight, self.av_rate,
                                          num_items_in_batch=num_items_in_batch)
            return (loss, out) if return_outputs else loss

    return WeightedTokenCETrainer


def selftest(verbose=True):
    """세 가지를 확인한다. margin_bce 와 같은 함정을 다시 밟지 않기 위해서다.

    1. 양성 대 음성 기울기 비가 정확히 pos_weight 인가
    2. per-device batch 1 에서도 가중치가 살아 있는가 (D-882 버그)
    3. num_items_in_batch 를 주면 그만큼 나뉘는가 (누적 16 에서 16배 부풀지 않는지)
    """
    torch.manual_seed(0)
    V, T, AV, KEEP = 50, 6, 3, 7
    AV_RATE = 0.188

    def grad_for(tgt, w, nib=None):
        z = torch.zeros(1, T, V, requires_grad=True)
        lab = torch.full((1, T), -100, dtype=torch.long)
        lab[0, T - 1] = tgt
        weighted_token_ce_loss(z, lab, AV, pos_weight=w, av_rate=AV_RATE,
                               num_items_in_batch=nib).backward()
        return abs(float(z.grad[0, T - 2, tgt]))

    ok = True
    if verbose:
        print("  %8s %12s %12s %9s %9s" % ("가중치", "AV행 기울기", "Keep행 기울기", "비율", "기대"))
    for w in (1.0, 2.233, 4.0):
        gp, gn = grad_for(AV, w), grad_for(KEEP, w)
        ratio = gp / gn if gn else float("nan")
        good = abs(ratio - w) < 1e-4
        ok = ok and good
        if verbose:
            print("  %8.3f %12.6f %12.6f %9.4f %9.3f  %s"
                  % (w, gp, gn, ratio, w, "ok" if good else "**불일치**"))

    spread = abs(grad_for(AV, 4.0) - grad_for(AV, 1.0))
    alive = spread > 1e-8
    if verbose:
        print("  배치 1 에서 가중치가 기울기를 바꾸는가: %s (차이 %.6f)"
              % ("예" if alive else "**아니오 - D-882 버그 재현**", spread))

    g1, g16 = grad_for(AV, 2.233), grad_for(AV, 2.233, nib=16)
    scaled = abs(g1 / 16 - g16) < 1e-9
    if verbose:
        print("  num_items_in_batch=16 이면 기울기가 1/16 인가: %s (%.8f vs %.8f)"
              % ("예" if scaled else "**아니오**", g1 / 16, g16))
        print("  판정:", "통과" if (ok and alive and scaled) else "**실패**")
    return ok and alive and scaled


if __name__ == "__main__":
    import sys
    print("=== weighted token CE 회귀 테스트 ===")
    sys.exit(0 if selftest() else 1)
