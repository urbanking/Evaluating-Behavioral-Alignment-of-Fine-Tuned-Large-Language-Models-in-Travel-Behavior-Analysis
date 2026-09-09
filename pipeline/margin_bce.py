# -*- coding: utf-8 -*-
"""margin BCE 목적함수. 표준 SFT(토큰 교차엔트로피)의 대안.

**왜 다른가.** 표준 SFT 는 "정답 글자에 높은 확률을 줘라" 를 최적화한다. 그런데 우리 자료는
82%가 Keep 이라 모델이 Keep 쪽으로 쏠린다 - 실측으로 margin 평균이 -1.753 이었다.
이 목적함수는 대신 **AV 와 Keep 의 점수 차이(margin)를 이항 분류기처럼 직접** 민다.

    margin = z_AV - z_Keep            (답변 자리의 로짓 차이)
    loss   = BCEWithLogits(margin, y, pos_weight=w) / (1 + (w-1) * av_rate)

**분모가 상수인 것이 핵심이다.** 선행 프로젝트(DECISIONS.md D-883)는 처음에 배치 안의
가중치 합으로 나눴는데, per-device batch size 가 1 이라 분자와 분모에서 가중치가 그대로
상쇄됐다 - 가중치 1.0 / 2.233 / 4.0 의 손실과 기울기가 **완전히 같았다.** 전체 학습자료의
AV 비율로 계산한 상수로 나누면 상쇄되지 않고, 양성 대 음성 기울기 비가 의도대로 유지된다.
그래서 배치 1 회귀 테스트가 필수다 (`selftest()`).

주의: 이건 **0.8B 절제실험용**이다. 전체 설계의 고정 프로토콜은 표준 SFT 이고, 규모 사다리
비교(0.8B / 2B / 9B)는 같은 목적함수로 학습한 어댑터끼리만 한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def answer_slice(labels):
    """각 행에서 **답변 첫 토큰의 위치**를 찾는다.

    labels 는 프롬프트 구간이 -100 이고 답변만 실제 토큰 id 다. 로짓은 한 칸 앞에서
    다음 토큰을 예측하므로, 답변 첫 토큰 위치가 pos 면 우리가 볼 로짓은 pos-1 이다.
    """
    valid = labels != -100
    # 각 행의 첫 유효 위치. 없는 행은 -1.
    idx = torch.where(valid.any(dim=1),
                      valid.float().argmax(dim=1),
                      torch.full((labels.size(0),), -1, device=labels.device))
    return idx.long()


def margin_and_target(logits, labels, av_id, keep_id):
    """(margin, y, 유효마스크). margin = z_AV - z_Keep, y = 1 이면 정답이 AV."""
    pos = answer_slice(labels)
    ok = (pos > 0)
    b = torch.arange(labels.size(0), device=labels.device)
    p = torch.clamp(pos - 1, min=0)
    z = logits[b, p]                                   # (B, V)
    margin = z[:, av_id] - z[:, keep_id]
    tgt = labels[b, torch.clamp(pos, min=0)]
    y = (tgt == av_id).float()
    return margin, y, ok


def margin_bce_loss(logits, labels, av_id, keep_id, pos_weight=1.0, av_rate=0.5,
                    num_items_in_batch=None):
    """정규화된 가중 BCE. 분모는 **상수**라 배치 1 에서도 가중치가 살아 있다.

    **num_items_in_batch 를 받아야 한다.** 최신 transformers 의 Trainer 는 커스텀
    compute_loss 가 이 인자를 받으면 "이미 전역 평균으로 정규화됐다" 고 보고 누적 단계로
    다시 나누지 않는다. 안 받으면 손실과 기울기가 gradient_accumulation_steps 배(우리는
    16배)로 들어간다 - 실측으로 손실이 기대 1.02 대신 12 로 찍혔고(12/16 = 0.75),
    학습률 2e-4 가 실질 3.2e-3 이 되어 발산했다. 학습률을 10분의 1 로 낮춰도 손실이
    11~12 에서 안 내려간 이유가 이것이다.
    """
    margin, y, ok = margin_and_target(logits, labels, av_id, keep_id)
    if ok.sum() == 0:
        return logits.sum() * 0.0
    w = torch.tensor(float(pos_weight), device=logits.device)
    per = F.binary_cross_entropy_with_logits(
        margin[ok].float(), y[ok].float(), pos_weight=w.float(), reduction="sum")
    denom = 1.0 + (float(pos_weight) - 1.0) * float(av_rate)
    n = float(num_items_in_batch) if num_items_in_batch else float(ok.sum())
    return per / n / denom


def make_trainer_class(base_cls):
    """HF Trainer 를 감싸 compute_loss 만 갈아끼운다."""

    class MarginBCETrainer(base_cls):
        def __init__(self, *a, av_id=None, keep_id=None, pos_weight=1.0,
                     av_rate=0.5, **kw):
            super().__init__(*a, **kw)
            self.av_id, self.keep_id = int(av_id), int(keep_id)
            self.pos_weight, self.av_rate = float(pos_weight), float(av_rate)

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = margin_bce_loss(out.logits, labels, self.av_id, self.keep_id,
                                   self.pos_weight, self.av_rate,
                                   num_items_in_batch=num_items_in_batch)
            return (loss, out) if return_outputs else loss

    return MarginBCETrainer


def selftest(verbose=True):
    """**배치 1 에서 pos_weight 가 실제로 작동하는지** 확인한다.

    선행 프로젝트가 여기서 당했다 (DECISIONS.md D-882): 배치 안의 가중치 합으로 나누는
    바람에 가중치 1.0 / 2.233 / 4.0 의 기울기가 완전히 같았다.

    **판정 기준은 "양성 대 음성 기울기 비 = w" 다.** 상수 분모는 손실 전체 크기를 조정하므로
    음성 기울기도 w 에 따라 변한다(그게 의도다 - 전역 스케일 안정화). 가중치가 하는 일은
    **비율**을 바꾸는 것이지 음성을 그대로 두는 것이 아니다.
    """
    torch.manual_seed(0)
    V, T, AV, KEEP = 50, 6, 3, 7
    AV_RATE = 0.176

    def grad_for(tgt, w):
        z = torch.zeros(1, T, V, requires_grad=True)
        lab = torch.full((1, T), -100, dtype=torch.long)
        lab[0, T - 1] = tgt
        margin_bce_loss(z, lab, AV, KEEP, pos_weight=w, av_rate=AV_RATE).backward()
        return abs(float(z.grad[0, T - 2, AV]))

    ok = True
    if verbose:
        print("  %8s %11s %11s %9s %9s" % ("가중치", "양성 기울기", "음성 기울기", "비율", "기대"))
    for w in (1.0, 2.233, 4.0):
        gp, gn = grad_for(AV, w), grad_for(KEEP, w)
        ratio = gp / gn if gn else float("nan")
        good = abs(ratio - w) < 1e-4
        ok = ok and good
        if verbose:
            print("  %8.3f %11.5f %11.5f %9.4f %9.3f  %s"
                  % (w, gp, gn, ratio, w, "ok" if good else "**불일치**"))
    # 배치 1 에서 가중치가 상쇄되지 않는지 (D-882 의 그 버그)
    spread = max(grad_for(AV, w) for w in (1.0, 4.0)) - min(grad_for(AV, w) for w in (1.0, 4.0))
    alive = spread > 1e-8
    if verbose:
        print("  배치 1 에서 가중치가 기울기를 바꾸는가: %s (차이 %.5f)"
              % ("예" if alive else "**아니오 - D-882 버그 재현**", spread))
    # num_items_in_batch 를 주면 그만큼 나뉘는가 (누적 16 에서 16배로 부풀지 않는지)
    g1 = grad_for(AV, 2.233)
    z = torch.zeros(1, T, V, requires_grad=True)
    lab = torch.full((1, T), -100, dtype=torch.long); lab[0, T - 1] = AV
    margin_bce_loss(z, lab, AV, KEEP, pos_weight=2.233, av_rate=AV_RATE,
                    num_items_in_batch=16).backward()
    g16 = abs(float(z.grad[0, T - 2, AV]))
    scaled = abs(g1 / 16 - g16) < 1e-6
    if verbose:
        print("  num_items_in_batch=16 이면 기울기가 1/16 인가: %s (%.6f vs %.6f)"
              % ("예" if scaled else "**아니오**", g1 / 16, g16))
        print("  판정:", "통과" if (ok and alive and scaled) else "**실패**")
    return ok and alive and scaled


if __name__ == "__main__":
    import sys
    print("=== margin BCE 배치 1 회귀 테스트 ===")
    sys.exit(0 if selftest() else 1)
