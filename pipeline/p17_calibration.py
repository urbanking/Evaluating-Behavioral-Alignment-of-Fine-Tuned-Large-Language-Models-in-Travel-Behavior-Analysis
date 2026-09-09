# -*- coding: utf-8 -*-
"""Phase 17. Development scoring 과 Platt calibration.

adapter 는 development 전체로 1회 학습했으므로, calibrator 만 respondent-grouped CV 로
적합한다. 원고에는 'development-only grouped calibration' 이라고 쓰고
'fully cross-fitted base-model probabilities' 라고 쓰지 않는다.

zero-shot 점수는 세계와 무관하다. 세계별 jsonl 은 프롬프트가 완전히 동일하고 라벨만
다르기 때문이다(p14 가 같은 렌더러로 만든다 — 확인함). 그래서 zero-shot 뱅크는
(model x info x partition) 하나만 만들고, 보정기만 세계별 라벨로 다시 적합한다.
SFT 는 adapter 가 세계마다 다르므로 뱅크도 세계별로 만든다.
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from common import DATA, REPORTS, ROOT, ensure_dirs, load_config, write_json
from llm_runtime import (CANDIDATES, NAMED_STYLES, candidate_scores,
                         candidate_scores_percase, load_base, margins,
                         prefix_tag, read_jsonl, render_chat, select_families)
from pathlib import Path
# **어댑터 경로는 p16 과 같은 함수로 만든다.** 여기서 "seed%d" 를 직접 조립하면
# 목적함수·프로파일 접미사가 붙은 폴더를 못 찾는다. 실제로 2026-08-08 에 2B 를
# weighted_token_ce 로 학습해 seed42_weighted_token_ce_aaai 에 저장했는데, p17 이
# seed42 를 찾아 "[skip] adapter 없음" 을 내고 **zero-shot 만 보정했다.**
from p16_sft_runs import adapter_dir

ADP = ROOT / "artifacts" / "models" / "llm"
CAL = ROOT / "artifacts" / "calibration"
BANK = ROOT / "artifacts" / "score_bank"


def jsonl_path(world, info, partition, src=None):
    """프롬프트 파일. src 를 주면 그 파일을 쓴다.

    같은 world/info 이름 아래에서 프롬프트를 다시 만들어 파일을 덮은 적이 있어서
    (2026-08-09, 2026년 거리대별 선택이유 추가), 이름만으로는 어느 판인지 알 수 없다.
    어댑터가 학습된 판과 같은 판으로 채점해야 하므로 명시적으로 고를 수 있어야 한다.
    """
    return Path(src) if src else (DATA / "llm" / ("%s_%s_%s.jsonl" % (world, info, partition)))


def world_labels(world, info, partition, src=None):
    """case_id -> label. 라벨은 세계마다 다르므로 뱅크에 굳혀두지 않고 여기서 붙인다."""
    rows = read_jsonl(jsonl_path(world, info, partition, src))
    return {r["case_id"]: int(r["label"]) for r in rows}


def score_bank(family, world, info, partition, state, adapter=None, force=False,
               src=None, answer_prefix=None, candidates=None, style="sft"):
    """점수 뱅크를 만들거나 읽는다. 라벨은 포함하지 않는다 (세계별로 갈아끼워야 하므로).

    answer_prefix / candidates 는 선행 AAAI 코드로 학습한 어댑터를 위한 것이다. 그쪽은
    'AV' 한 토큰이 아니라 {"pairwise_choice":"AV"} 를 뱉도록 배웠고, 채점도 접두사
    {"pairwise_choice":" 를 붙인 자리에서 AV 와 CURRENT 를 비교한다. 접두사를 안 붙이면
    학습된 적 없는 형식으로 묻는 셈이라 그 모델의 실력이 아니다.
    """
    ensure_dirs(BANK)
    zero_shot = adapter is None
    # **접두사를 키에 넣는다.** 접두사가 다르면 margin 이 다른 값이다. 키를 같이 쓰면
    # 접두사 없이 만든 옛 뱅크를 그대로 읽어, 접두사를 준 의미가 사라진 채 보정기가
    # 만들어진다 (p18 의 raw 폴더에서 이미 겪은 함정과 같은 것).
    from llm_runtime import STYLE_TAG, prefix_tag
    # 접두사와 **프롬프트 양식**을 키에 넣는다. 양식이 다르면 margin 이 다른 값이다.
    ptag = prefix_tag(answer_prefix) + STYLE_TAG[style]
    # zero-shot 은 세계 무관 -> 키에서 world 를 뺀다. 프롬프트는 어느 세계 파일을 읽어도 같다.
    key = ("%s_%s_%s_%s%s" % (family, state, info, partition, ptag) if zero_shot
           else "%s_%s_%s_%s_%s%s" % (family, state, world, info, partition, ptag))
    f = BANK / (key + ".parquet")
    if f.exists() and not force:
        return pd.read_parquet(f)
    cfg = load_config(); lc = cfg["llm"]
    src_path = jsonl_path("HUMAN" if zero_shot else world, info, partition, src)
    if not src_path.exists():
        raise SystemExit("프롬프트 파일이 없다: %s" % src_path)
    print("  [프롬프트] %s" % src_path.name, flush=True)
    rows = read_jsonl(src_path)
    tok, base = load_base(family, four_bit=lc["four_bit"])
    model = base
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter))
    prompts = [render_chat(tok, r["messages"][:-1]) for r in rows]
    # 이름 -> 실제 문자열. 표는 llm_runtime 에 모아 두었다 (p18 과 같은 것을 써야 한다.
    # 보정기는 그 접두사로 만든 margin 위에 적합되므로, 추론 때와 접두사가 다르면
    # 보정이 엉뚱한 척도에 걸린다).
    from llm_runtime import resolve_answer_prefix
    answer_prefix = resolve_answer_prefix(answer_prefix)
    if answer_prefix:
        prompts = [p + answer_prefix for p in prompts]
        print("  [답변접두사] %r" % answer_prefix, flush=True)
    cand = tuple(x for x in str(candidates).split(",")) if candidates else CANDIDATES
    if cand != CANDIDATES:
        print("  [후보] %s (양성=%s)" % (list(cand), cand[0]), flush=True)
    if style in NAMED_STYLES:
        # 후보가 사례마다 다른 수단 이름이다 (p18 과 같은 규칙이어야 보정이 맞는다).
        import p14_llm_prompts as P14
        mi_ = pd.read_parquet(
            DATA / "model_inputs" / ("%s_%s_%s.parquet" % (world, info, partition)),
            columns=["case_id", "current_mode_label"])
        km = {str(c): (P14._mode("AV"), P14._mode(m))
              for c, m in zip(mi_.case_id, mi_.current_mode_label)}
        pairs = [km[str(r["case_id"])] for r in rows]
        sc, mass, free = candidate_scores_percase(
            model, tok, prompts, pairs, batch_size=lc["score_batch_size"],
            max_len=lc["max_seq_len"], return_extra=True)
    else:
        sc, mass, free = candidate_scores(model, tok, prompts, candidates=cand,
                                batch_size=lc["score_batch_size"],
                                max_len=lc["max_seq_len"], return_extra=True)
    out = pd.DataFrame({"case_id": [r["case_id"] for r in rows],
                        "respondent_id": [r["respondent_id"] for r in rows],
                        "q_av": sc[:, 0], "q_keep": sc[:, 1], "margin": margins(sc),
                        "p_av_raw": np.exp(sc[:, 0]), "p_keep_raw": np.exp(sc[:, 1]),
                        "candidate_mass": mass, "free_argmax": free})
    out.to_parquet(f, index=False)
    del model, base
    import torch; torch.cuda.empty_cache()
    return out


def fit_platt(margin, y, groups, n_splits=5):
    """margin -> 확률 변환을 development 에서 적합한다. respondent-grouped CV 로 성능만 본다.

    **기울기를 음수로 두지 않는다.** 점수 순서가 진실과 반대이면 제약 없는 Platt 은 기울기를
    음수로 잡아 예측을 통째로 뒤집는다. 그러면 방향조차 틀린 모형이 표에서는 멀쩡해 보인다.
    실측(2026-08-06): 학습 안 된 0.8B 제로샷이 test BASE 에서 AUC 0.387 이었다 - 0.5 미만,
    즉 순서가 반대다. 그대로 뒤집으면 AUC 0.613 인 척하게 된다.

    그래서 기울기를 0 이상으로 제약한다. 뒤집힌 점수는 기울기가 0 이 되어 **모두에게 기저율**
    을 주는 예측이 된다 - "쓸 만한 신호가 없다" 가 정확히 표현된다.

    전에는 여기서 SystemExit 으로 멈췄는데, 그건 과했다. 이건 오류가 아니라 결과이고,
    멈추면 보고를 못 한다. 대신 `orientation_flipped` 로 기록하고 크게 찍는다.
    """
    m = np.asarray(margin, float).reshape(-1, 1)
    y = np.asarray(y)
    base = float(np.mean(y))
    b0 = float(np.log(base / (1 - base))) if 0 < base < 1 else 0.0

    def fit(mm, yy):
        """기울기 >= 0 로 제약한 Platt. 음수로 가려 하면 기울기 0(=기저율)으로 눌러 앉힌다."""
        lr = LogisticRegression().fit(mm, yy)
        a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        if a > 0:
            return a, b, a
        bb = float(np.mean(yy))
        return 0.0, (float(np.log(bb / (1 - bb))) if 0 < bb < 1 else 0.0), a

    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(m, y, groups):
        a, b, _ = fit(m[tr], y[tr])
        oof[te] = 1.0 / (1.0 + np.exp(-(a * m[te, 0] + b)))
    a, b, a_raw = fit(m, y)

    ll = float(-np.mean(y * np.log(np.clip(oof, 1e-9, 1)) +
                        (1 - y) * np.log(np.clip(1 - oof, 1e-9, 1))))
    # margin 이 개인을 구분하는가. **보정 계수(a,b)만으로는 이걸 못 본다.**
    # 점수에 정보가 없으면 확률이 모두 기저율 근처로 가는데, 그러면 정확도는 다수 클래스
    # 비율(약 0.81)로 좋아 보이지만 실제로는 아무도 구분하지 못한 것이다.
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, np.asarray(margin))) if len(np.unique(y)) > 1 else np.nan
    flipped = bool(a_raw <= 0)
    if flipped:
        print("     [!] 점수 방향이 뒤집혀 있다 (제약 없는 기울기 %.4f, AUC %.4f). "
              "기울기를 0 으로 눌러 모두에게 기저율 %.4f 를 준다." % (a_raw, auc, base),
              flush=True)
    return {"a": a, "b": b, "slope_unconstrained": a_raw,
            "orientation_flipped": flipped,
            "oof_log_loss": ll, "n": int(len(y)),
            "margin_auc": auc, "base_rate": base,
            "margin_informative": bool(auc == auc and auc >= 0.60)}


def _labels_for(bank, world, info, src=None):
    lab = world_labels(world, info, "development", src)
    y = bank.case_id.map(lab)
    if y.isna().any():
        raise SystemExit("%s/%s 라벨 결측 %d 건 — 뱅크와 jsonl 의 case_id 가 어긋난다"
                         % (world, info, int(y.isna().sum())))
    return y.astype(int).to_numpy()


def apply_variant(cfg, profile=None, objective=None, run_tag=None, tau=None):
    """학습 때와 **같은 규칙으로** 어댑터 폴더 이름이 나오도록 cfg 를 맞춘다."""
    if objective:
        cfg["sft"]["objective"] = objective
    if profile:
        prof = (cfg.get("sft_profiles") or {}).get(profile)
        if prof is None:
            raise SystemExit("모르는 프로파일 %r — configs 의 sft_profiles 에 없다" % profile)
        cfg["sft"].update(prof)
        cfg["sft"]["_profile"] = profile
    if run_tag:
        cfg["sft"]["_run_tag"] = run_tag
    if tau is not None:
        # **학습 때 쓴 값과 맞춰야 한다.** 어댑터 폴더 이름에 tau 가 들어가므로,
        # config 가 그 사이 바뀌었으면 채점이 엉뚱한 폴더를 찾는다. 실제로 2026-08-09 에
        # 2B(tau=0 으로 학습)를 config 가 tau=1 인 상태로 채점하려다 못 찾았다.
        cfg["sft"].setdefault("loss_composite", {})["logit_adjust_tau"] = float(tau)
    return cfg


def calibrate_one(family, adapter, world="HUMAN", info="retro", seed=42, force=False,
                  dev_file=None, answer_prefix=None, candidates=None, name=None):
    """어댑터 **하나**를 development 로 채점하고 Platt 을 적합한다.

    격자를 도는 run() 과 달리 경로를 직접 받는다. 선행 AAAI 코드가 만든 어댑터는
    우리 폴더 규칙(seed42_목적함수_프로파일...) 밖에 있어서 격자로는 닿지 않는다.

    보정기는 development 에서만 적합한다. 설계(configs 의 calibration.fit_on)가 그렇고,
    test 를 보고 임계값을 고르면 그 숫자는 held-out 이 아니게 된다.
    """
    ad = Path(adapter)
    if not (ad / "adapter_config.json").exists():
        raise SystemExit("어댑터가 없다: %s" % ad)
    tag = name or ad.name
    print("\n=== Phase 17. development 채점 + Platt 보정 ===")
    print("  모델 %s / 어댑터 %s" % (family, tag), flush=True)
    ensure_dirs(CAL, REPORTS)
    sb = score_bank(family, world, info, "development", "sft_%s" % tag, ad, force,
                    src=dev_file, answer_prefix=answer_prefix, candidates=candidates)
    y = _labels_for(sb, world, info, dev_file)
    cal = fit_platt(sb.margin, y, sb.respondent_id)
    p = CAL / ("sft_%s_%s_%s_%s.json" % (family, world, info, tag))
    write_json(p, {"state": "sft", "family": family, "info": info, "world": world,
                   "seed": seed, "adapter": tag,
                   "dev_file": Path(dev_file).name if dev_file else None,
                   "answer_prefix": answer_prefix, "candidates": candidates, **cal})
    print("  [ok] %s" % p.name)
    print("      a=%.4f  b=%.4f  oof_log_loss=%.4f  margin_AUC=%.4f%s"
          % (cal["a"], cal["b"], cal["oof_log_loss"], cal["margin_auc"],
             "" if cal["margin_informative"] else "   <- 무정보 경고"))
    print("      경계가 놓이는 margin = -b/a = %+.4f  (여기 위가 확률 0.5 이상)"
          % (-cal["b"] / cal["a"] if cal["a"] else float("nan")))
    return p


def run(force=False, stages=None, family=None, profile=None, objective=None,
        run_tag=None, tau=None, answer_prefix=None, style="sft", dev_src=None):
    """격자를 돌며 보정기를 만든다.

    **answer_prefix 는 zero-shot 에만 넘긴다.** 9B zero-shot 은 채팅 서식이 사고 블록을
    열어 놓고 끝나서, 접두사 없이는 채점 자리가 답변 자리가 아니다(candidate_mass 0.0000).
    SFT 는 그 서식 그대로 학습했으므로 붙이지 않는다. 추론(p18)과 **같은 접두사**를 써야
    한다 - 보정기는 그 접두사로 만든 margin 위에 적합되기 때문이다.
    """
    print("\n=== Phase 17. development scoring + calibration ===")
    cfg = apply_variant(load_config(), profile, objective, run_tag, tau)
    ensure_dirs(CAL, REPORTS)
    fams = select_families(cfg, family)   # --family 로 카드별 담당을 고른다
    order = stages or cfg["grid"].get("execution_order", ["human", "synthetic"])
    made = []
    for fam in fams:
        for stage in order:
            g = cfg["grid"][stage]
            for world in g["worlds"]:
                for info in g["information_conditions"]:
                    # zero-shot: 뱅크는 세계 무관, 보정기는 세계별 라벨로 적합
                    # style 이 sft 가 아니면 그 양식으로 다시 렌더한 development
                    # jsonl(dev_src)이 필요하다 - 기본 jsonl 은 p14 가 sft 양식으로 만들었다.
                    zs = score_bank(fam, world, info, "development", "zeroshot", None, force,
                                    src=dev_src, answer_prefix=answer_prefix, style=style)
                    cal = fit_platt(zs.margin, _labels_for(zs, world, info), zs.respondent_id)
                    # 접두사가 다르면 margin 이 다른 값이라 보정기도 다른 것이다.
                    # 이름을 나눠 두지 않으면 접두사 없이 만든 보정기가 접두사 있는
                    # margin 에 조용히 씌워진다.
                    from llm_runtime import STYLE_TAG as _ST
                    p = CAL / ("zero_shot_%s_%s_%s%s.json"
                               % (fam, info, world,
                                  prefix_tag(answer_prefix) + _ST[style]))
                    write_json(p, {"state": "zeroshot", "family": fam, "info": info,
                                   "world": world, **cal}); made.append(p.name)
                    print("  [ok] zero-shot %s/%s/%s  a=%.4f b=%.4f oof_ll=%.4f "
                          "AUC=%.4f%s"
                          % (fam, world, info, cal["a"], cal["b"], cal["oof_log_loss"],
                             cal["margin_auc"],
                             "" if cal["margin_informative"] else "  <- 무정보 경고"),
                          flush=True)
                    for seed in g["llm_seeds"]:
                        ad = adapter_dir(fam, world, info, seed, cfg)
                        if not (ad / "adapter_config.json").exists():
                            print("  [skip] adapter 없음 %s/%s/%s/%s"
                                  % (fam, world, info, ad.name), flush=True)
                            continue
                        # **보정기 이름에 어댑터 폴더 이름을 넣는다.** 같은 셀을 목적함수만
                        # 바꿔 학습할 수 있으므로, seed 만으로 이름을 지으면 나중에 돌린
                        # 쪽이 앞의 보정기를 조용히 덮어쓴다.
                        sb = score_bank(fam, world, info, "development",
                                        "sft_%s" % ad.name, ad, force)
                        cal = fit_platt(sb.margin, _labels_for(sb, world, info),
                                        sb.respondent_id)
                        p = CAL / ("sft_%s_%s_%s_%s.json" % (fam, world, info, ad.name))
                        write_json(p, {"state": "sft", "family": fam, "info": info,
                                       "world": world, "seed": seed, "adapter": ad.name,
                                       **cal})
                        made.append(p.name)
                        print("  [ok] SFT %s/%s/%s/%s  a=%.4f oof_ll=%.4f "
                              "AUC=%.4f%s"
                              % (fam, world, info, ad.name, cal["a"], cal["oof_log_loss"],
                                 cal["margin_auc"],
                                 "" if cal["margin_informative"] else "  <- 무정보 경고"),
                              flush=True)
    write_json(REPORTS / "calibration.json", {"calibrators": made})
    print("  [ok] calibrator %d 개" % len(made))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", default=None, choices=["human", "synthetic"])
    ap.add_argument("--family", default=None,
                    help="이 모델군만 (예: qwen / qwen_small,qwen_mid). 생략하면 available 전부")
    ap.add_argument("--tau", type=float, default=None,
                    help="학습 때 쓴 logit_adjust_tau. 폴더 이름이 여기에 달려 있다")
    ap.add_argument("--run-tag", default=None, dest="run_tag",
                    help="학습 때 쓴 --run-tag. 폴더 이름이 여기에 달려 있다")
    ap.add_argument("--profile", default=None,
                    help="학습 때 쓴 프로파일. 어댑터 폴더 이름이 여기에 달려 있다")
    ap.add_argument("--objective", default=None,
                    choices=["token_ce", "margin_bce", "weighted_token_ce", "aaai_composite"],
                    help="학습 때 쓴 목적함수. 폴더 이름이 달라지므로 반드시 맞춰야 한다")
    ap.add_argument("--adapter", default=None,
                    help="이 어댑터 하나만 development 로 채점하고 보정한다. 폴더 규칙 밖에 "
                         "있는 어댑터(선행 AAAI 코드 출력)를 가리킬 때 쓴다")
    ap.add_argument("--name", default=None, help="보정기 파일에 쓸 이름. 생략하면 폴더 이름")
    ap.add_argument("--dev-file", default=None, dest="dev_file",
                    help="development 프롬프트 jsonl. 어댑터가 학습된 판과 같아야 한다")
    ap.add_argument("--answer-prefix", default=None, dest="answer_prefix",
                    help="답변의 고정된 앞부분. 이름: nothink (9B zero-shot 용 - 서식이 "
                         "<think> 를 열어 놓고 끝나 채점 자리가 답변 자리가 아니다), "
                         "aaai (선행 AAAI 형식). 추론(p18)과 **같은 값**을 줘야 한다")
    ap.add_argument("--candidates", default=None,
                    help="쉼표로 구분한 후보, 앞이 양성 (기본 AV,Keep). AAAI 형식은 AV,CURRENT")
    ap.add_argument("--prompt-style", default="sft", dest="style",
                    choices=["sft", "zeroshot", "pairwise", "noatt", "ab", "explained", "abswap", "respondent", "plain", "card", "fresh", "named", "neutral", "neutralswap", "shortnamed", "namedswap"],
                    help="zero-shot 프롬프트 양식 (D257 동결: 2B=zeroshot, 9B=ab). sft 가 "
                         "아니면 --dev-file 로 그 양식의 development jsonl 을 줘야 한다")
    ap.add_argument("--world", default="HUMAN")
    ap.add_argument("--info", default="retro")
    a = ap.parse_args()
    if a.adapter:
        calibrate_one(a.family or "qwen", a.adapter, world=a.world, info=a.info,
                      force=a.force, dev_file=a.dev_file, answer_prefix=a.answer_prefix,
                      candidates=a.candidates, name=a.name)
    else:
        run(force=a.force, stages=[a.stage] if a.stage else None, family=a.family,
            profile=a.profile, objective=a.objective, run_tag=a.run_tag, tau=a.tau,
            answer_prefix=a.answer_prefix, style=a.style, dev_src=a.dev_file)
