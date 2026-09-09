# -*- coding: utf-8 -*-
"""Phase 16. SFT 본학습 (Stage A 인간자료 -> Stage B 합성).

grid 는 configs/trc_experiment.yaml 의 grid 에서만 읽는다.
resume: adapter 폴더에 adapter_config.json 이 있으면 건너뛴다.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from common import DATA, ROOT, ensure_dirs, load_config, write_json
from llm_runtime import build_sft_dataset, load_base, lora_config, read_jsonl

ADP = ROOT / "artifacts" / "models" / "llm"


class StatusFile:
    """학습 상태를 **작은 json 파일에 직접 쓴다.** stdout 리디렉션에 기대지 않는다.

    두 번 당하고 만들었다.

    2026-08-07 20:48  로컬 PC 가 절전으로 들어가 SSH 파이프가 막혔다. 원격 파이썬이
      stdout 쓰기에서 멈췄고 결국 학습이 죽었다. 9B 와 0.8B 가 4초 차이로 같이 죽었고
      두 로그의 경과시간이 똑같이 "벽시계 7시간 동안 52분" 만 늘어난 것이 증거였다.
    2026-08-08 03:25  SSH 와 분리해 띄웠더니 이번엔 로그 파일이 0바이트로 보였다.
      Windows 가 다른 프로세스가 쓰기로 열어둔 파일의 크기를 갱신하지 않아서다.
      GPU 가 31.8GB 를 쓰고 있으니 살아 있는 건 알겠는데 **몇 step 인지 loss 가 얼마인지
      알 수 없었다.**

    둘 다 "출력을 파이프나 리디렉션으로 나른다" 는 구조에서 나왔다. 여기서는 매번 열고
    쓰고 닫는다. 1KB 미만이라 비용이 없고, 학습이 죽어도 마지막 상태가 파일에 남는다.
    """

    def __init__(self, path, tag):
        self.path, self.tag = Path(path), tag
        self.t0 = time.time()
        self._start_step = 0
        ensure_dirs(self.path.parent)

    def write(self, state, logs=None, extra=None):
        el = time.time() - self.t0
        done = int(getattr(state, "global_step", 0) or 0)
        total = int(getattr(state, "max_steps", 0) or 0)
        # **재개한 경우 done 에 이전 실행분이 포함된다.** 경과시간은 이번 실행분뿐이라
        # 그대로 나누면 s/step 이 과소평가된다. 이번 실행에서 진행한 step 으로 나눈다.
        run = max(done - self._start_step, 1)
        sps = el / run
        left = max(total - done, 0)
        d = {"tag": self.tag, "step": done, "max_steps": total,
             "epoch": round(float(getattr(state, "epoch", 0) or 0), 4),
             "elapsed_s": round(el), "s_per_step": round(sps, 2),
             "eta_h": round(sps * left / 3600, 2),
             "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            if torch.cuda.is_available():
                d["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
                d["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                d["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        except Exception:
            pass
        for k in ("loss", "grad_norm", "learning_rate"):
            if logs and k in logs:
                d[k] = logs[k]
        if extra:
            d.update(extra)
        try:
            self.path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass


def make_status_callback(base_cls, status):
    """HF TrainerCallback 을 상속해 상태 파일을 갱신한다."""

    class StatusCallback(base_cls):
        def on_train_begin(self, args, state, control, **kw):
            status._start_step = int(getattr(state, "global_step", 0) or 0)
            status.write(state, extra={"phase": "train_begin"})

        def on_log(self, args, state, control, logs=None, **kw):
            status.write(state, logs)

        def on_save(self, args, state, control, **kw):
            status.write(state, extra={"phase": "saved"})

        def on_train_end(self, args, state, control, **kw):
            status.write(state, extra={"phase": "done"})

    return StatusCallback()


def cells(cfg, stage, family=None):
    """돌 셀 목록. family 를 주면 그 모델군만 돈다.

    규모 사다리(0.8B/2B)를 설정에 추가하면서 필요해졌다. 지정하지 않으면 available 인
    모델군을 전부 도는데, 그러면 GV100 이 9B 를 돌리는 중에 소형까지 집어간다.
    카드마다 담당 모델군이 다르므로 명시적으로 고른다.
    """
    g = cfg["grid"][stage]
    fams = [f for f, v in cfg["llm"]["families"].items() if v.get("available")]
    if family:
        want = [x.strip() for x in str(family).split(",")]
        unknown = [x for x in want if x not in cfg["llm"]["families"]]
        if unknown:
            raise SystemExit("모르는 모델군 %s — 설정의 families 에 없다" % unknown)
        fams = [f for f in fams if f in want]
        if not fams:
            raise SystemExit("%s 는 available: false 다" % want)
    return [(f, w, i, s) for f in fams for w in g["worlds"]
            for i in g["information_conditions"] for s in g["llm_seeds"]]


def adapter_dir(family, world, info, seed, cfg, rnd=None):
    """어댑터 저장 위치. **목적함수·프로파일·회차가 다르면 폴더를 나눈다.**

    같은 셀(0.8B/HUMAN/retro)을 표준 SFT 로도, margin_bce 로도 학습할 수 있어야 한다.
    한 폴더를 쓰면 나중에 돌린 쪽이 앞의 것을 덮거나, adapter_config.json 이 있다고
    건너뛰어 아예 학습이 안 된다. 1 epoch 씩 끊어 이어 붙이는 회차도 마찬가지라서
    회차마다 폴더를 따로 둔다 - 그래야 r1/r2/r3 을 각각 채점해 어디서 멈출지 정한다.
    """
    obj = cfg["sft"].get("objective", "token_ce")
    tag = "seed%d" % seed
    if obj != "token_ce":
        tag += "_" + obj
    if cfg["sft"].get("_profile"):
        tag += "_" + str(cfg["sft"]["_profile"])
    # **손실 하이퍼파라미터도 이름에 넣는다.** 목적함수와 프로파일만으로 이름을 지으면,
    # tau 만 다른 두 실험이 같은 폴더를 가리킨다. 2026-08-08 22:45 에 실제로 그렇게 됐다 -
    # GPU1 이 tau=0 으로 돌고 있는 폴더에 GPU0 이 tau=1 로 붙어 checkpoint-100 을
    # 이어받았다. 15분 만에 115 step 이 찍혀서 알아챘고, 다음 저장 전에 죽여서 손상은 없었다.
    la = float((cfg["sft"].get("loss_composite") or {}).get("logit_adjust_tau", 0.0) or 0.0)
    if la:
        tag += "_la%g" % la
    if cfg["sft"].get("_run_tag"):
        tag += "_" + str(cfg["sft"]["_run_tag"])
    if rnd:
        tag += "_r%d" % int(rnd)
    return ADP / family / world / info / tag


def train_one(family, world, info, seed, cfg, smoke=False, rnd=None, init_adapter=None,
              train_file=None):
    out = adapter_dir(family, world, info, seed, cfg, rnd)
    if (out / "adapter_config.json").exists():
        print("  [skip] %s" % out.relative_to(ROOT)); return None
    # 2회차부터는 **앞 회차 어댑터에서 이어 학습한다.** 새 LoRA 를 얹으면 앞 회차가
    # 통째로 버려져 "1 epoch 씩 3번" 이 아니라 "1 epoch 을 3번 따로" 가 된다.
    prev = adapter_dir(family, world, info, seed, cfg, rnd - 1) if rnd and rnd > 1 else None
    # init_adapter: **회차 규칙과 무관하게 아무 어댑터에서나 시작한다.** --round 는 앞
    # 회차 폴더 이름(_r1)을 계산해서 찾는데, 3 epoch 를 한 번에 돌린 어댑터에는 그
    # 꼬리표가 없다. 그걸 _r1 로 개명하면 "1 epoch 짜리 1회차" 라는 틀린 이름이 남는다.
    # 경로를 직접 받으면 이름을 속이지 않고 이어 학습할 수 있다.
    if init_adapter:
        prev = Path(init_adapter)
    if prev is not None and not (prev / "adapter_config.json").exists():
        raise SystemExit("이어 학습할 어댑터에 adapter_config.json 이 없다: %s" % prev)
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    sft = cfg["sft"]; lc = cfg["llm"]
    # train_file: **프롬프트 판(version)을 명시적으로 고른다.** 기본 파일 이름은 world 와
    # info 로만 정해지는데, 2026-08-09 에 같은 HUMAN/retro 이름 아래에서 프롬프트를 다시
    # 만들어(2026년 거리대별 선택이유 추가) 파일을 덮었다. 그래서 "이유 없는 판으로 학습한
    # 어댑터" 를 "이유 있는 판" 으로 이어 학습하는 일이 조용히 일어난다. 그러면 epoch 을
    # 더 돌린 효과와 프롬프트를 바꾼 효과가 한 실험에 섞여 분리할 수 없다.
    src = Path(train_file) if train_file else (DATA / "llm" /
                                              ("%s_%s_development.jsonl" % (world, info)))
    if not src.exists():
        raise SystemExit("학습 파일이 없다: %s" % src)
    print("  [학습자료] %s" % src.name, flush=True)
    rows = read_jsonl(src, limit=lc["smoke"]["train_rows"] if smoke else None)
    tok, model = load_base(family, four_bit=lc["four_bit"])
    # **파이토치가 VRAM 밖으로 넘치지 못하게 상한을 건다** (2026-08-31).
    #
    # 윈도우(WDDM) 드라이버는 VRAM 이 차면 오류를 내지 않고 **시스템 RAM 으로 흘려보낸다**.
    # 그러면 죽지 않는 대신 PCIe 로 오가느라 느려지고, 파이토치 캐시는 그 자리를 계속
    # 붙들어 예약이 무한정 자란다. 실측(9B CV5, 2026-08-31):
    #
    #     alloc 11.95GB  peak 18.20GB   <- 실제로 필요한 양
    #     reserved 74.66 / 76.36 / 77.90GB   <- 32GB 카드에서 파이토치가 잡은 양
    #     시스템 RAM 1,024GB 중 889GB 사용   <- 세 학습이 흘려보낸 양
    #     스텝당 47초(정상) -> 118~177초
    #
    # set_per_process_memory_fraction 은 캐싱 할당자의 상한을 카드 용량의 비율로 못박는다.
    # 넘으면 시스템 RAM 으로 새는 대신 **OOM 을 낸다** - 그게 옳다. 실제 peak 가 18.2GB
    # 이므로 32GB 의 0.9(=28.8GB)는 넉넉하고, 이 상한에 걸릴 일이 없다.
    #
    # 학습 조건(배치·에폭·lr·시퀀스 길이)은 하나도 바꾸지 않는다. 메모리 관리만 바꾼다.
    import os as _os
    _frac = float(_os.environ.get("TRC_MEM_FRACTION", "0") or 0)
    if _frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_frac, torch.cuda.current_device())
        _tot = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("  [메모리상한] 카드의 %.0f%% = %.1fGB (초과 시 시스템 RAM 유출 대신 OOM)"
              % (_frac * 100, _frac * _tot), flush=True)
    # **4bit 모델에는 이 두 줄이 반드시 필요하다.**
    #
    # prepare_model_for_kbit_training  layernorm 을 fp32 로 올리고 gradient checkpointing 을
    #   모델 쪽에 제대로 붙인다. TrainingArguments 의 gradient_checkpointing=True 만으로는
    #   양자화 모델에서 붙지 않는다.
    # enable_input_require_grads       임베딩 출력이 grad 를 요구하게 만든다. 이게 없으면
    #   체크포인트 구간의 입력이 requires_grad=False 라 재계산이 일어나지 않고, 결국
    #   활성값을 전부 들고 있게 된다 - 즉 checkpointing 을 켠 값이 없어진다.
    #
    # 실측(RTX 3080, 0.8B): 이 두 줄이 있으면 peak 5.29GB, 없으면 학습 중 9.9GB 로
    # 카드가 꽉 차고 사용률이 43% 로 떨어졌다(메모리 대기). 샘플당 0.64초가 1.35초가 됐다.
    # 9B 를 GV100 에서 돌릴 때도 같은 차이가 난다.
    if lc["four_bit"] and bool(sft.get("gradient_checkpointing", True)):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if prev is not None:
        from peft import PeftModel
        # is_trainable=True 가 없으면 어댑터가 얼어서 학습이 0 으로 돈다.
        model = PeftModel.from_pretrained(model, str(prev), is_trainable=True)
        print("  [이어학습] %s 의 어댑터에서 시작" % prev.name, flush=True)
    else:
        model = get_peft_model(model, lora_config())
    if bool(sft.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    ds = build_sft_dataset(tok, rows, max_len=lc["max_seq_len"])
    per = int(lc["train_batch_size"])
    accum = max(1, int(sft["effective_batch_size"]) // per)
    ensure_dirs(out)
    args = TrainingArguments(
        output_dir=str(out / "_hf"), per_device_train_batch_size=per,
        gradient_accumulation_steps=accum, num_train_epochs=1 if smoke else int(sft["epochs"]),
        learning_rate=float(sft["learning_rate"]), lr_scheduler_type=sft["schedule"],
        warmup_ratio=float(sft["warmup_ratio"]),
        weight_decay=float(sft.get("weight_decay", 0.0)), logging_steps=25,
        save_strategy="steps", save_steps=int(sft.get("save_steps", 254)),
        save_total_limit=int(sft.get("save_total_limit", 2)),
        bf16=(lc.get("dtype", "bfloat16") == "bfloat16"),
        fp16=(lc.get("dtype", "bfloat16") == "float16"),
        report_to=[], seed=seed,
        **({"max_steps": int(sft["max_steps"])} if sft.get("max_steps") else {}),
        gradient_checkpointing=bool(sft.get("gradient_checkpointing", True)),
        optim=sft.get("optim", "paged_adamw_8bit"))
    coll = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    objective = sft.get("objective", "token_ce")
    if objective == "margin_bce":
        # AV/Keep 점수차를 직접 미는 목적함수. 표준 SFT 는 다수 클래스로 쏠린다.
        import margin_bce as MB
        cands = cfg["outcome"]["alternative_tokens"]
        ids = [tok(c, add_special_tokens=False).input_ids for c in cands]
        assert all(len(v) == 1 for v in ids), "후보가 1토큰이 아니면 margin 을 못 만든다: %s" % ids
        av_rate = float(np.mean([int(r["label"]) for r in rows]))
        w = float(sft.get("pos_weight", 1.0))
        assert MB.selftest(verbose=False),             "margin_bce 배치1 회귀 테스트 실패 - 가중치가 상쇄된다 (DECISIONS D-882)"
        print("  [objective] margin_bce  pos_weight=%.3f  학습자료 AV비율=%.4f  후보토큰 %s"
              % (w, av_rate, ids), flush=True)
        TrainerCls = MB.make_trainer_class(Trainer)
        tr = TrainerCls(model=model, args=args, train_dataset=ds, data_collator=coll,
                        av_id=ids[0][0], keep_id=ids[1][0], pos_weight=w, av_rate=av_rate)
    elif objective == "aaai_composite":
        # 선행 AAAI 프로젝트의 복합 손실을 그대로 재현한다. pos_weight 만 떼어 쓴
        # weighted_token_ce 가 2B/HOM 에서 AV F1 0.1987 에 그친 것이 계기다 (AAAI 는
        # 같은 가중치로 0.4549). 차이는 focal 과 Brier 가 함께 있었다는 점이다.
        import aaai_composite as AC
        cands = cfg["outcome"]["alternative_tokens"]
        ids = [tok(c, add_special_tokens=False).input_ids for c in cands]
        assert all(len(v) == 1 for v in ids), "후보가 1토큰이 아니다: %s" % ids
        av_rate = float(np.mean([int(r["label"]) for r in rows]))
        w = float(sft.get("pos_weight", 1.0))
        lc_w = sft.get("loss_composite") or {}
        assert AC.selftest(verbose=False), "aaai_composite 회귀 테스트 실패"
        if float(lc_w.get("logit_adjust_tau", 0.0)) and abs(w - 1.0) > 1e-9:
            print("  [warn] logit_adjust_tau 와 pos_weight 를 함께 켰다. 경계가 두 번 "
                  "움직인다 - pos_weight 를 1.0 으로 두는 것이 깨끗하다", flush=True)
        print("  [objective] aaai_composite  pos_weight=%.3f  AV비율=%.4f  "
              "wbce=%.2f focal=%.2f(gamma %.1f) brier=%.2f  binary=%.2f format=%.2f  "
              "logit_adjust_tau=%.2f"
              % (w, av_rate, lc_w.get("wbce_weight", 0.65), lc_w.get("focal_weight", 0.25),
                 lc_w.get("focal_gamma", 2.0), lc_w.get("brier_weight", 0.10),
                 lc_w.get("binary_weight", 0.90), lc_w.get("format_weight", 0.10),
                 lc_w.get("logit_adjust_tau", 0.0)), flush=True)
        TrainerCls = AC.make_trainer_class(Trainer)
        tr = TrainerCls(model=model, args=args, train_dataset=ds, data_collator=coll,
                        av_id=ids[0][0], keep_id=ids[1][0], pos_weight=w, av_rate=av_rate,
                        focal_gamma=lc_w.get("focal_gamma", 2.0),
                        wbce_w=lc_w.get("wbce_weight", 0.65),
                        focal_w=lc_w.get("focal_weight", 0.25),
                        brier_w=lc_w.get("brier_weight", 0.10),
                        binary_w=lc_w.get("binary_weight", 0.90),
                        format_w=lc_w.get("format_weight", 0.10),
                        logit_adjust_tau=lc_w.get("logit_adjust_tau", 0.0))
    elif objective == "weighted_token_ce":
        # 토큰 CE 는 그대로 두고 AV 응답 행에만 가중치를 건다. token_ce 의 다수쏠림과
        # margin_bce 의 "나머지 어휘를 안 건드림" 을 동시에 피하려는 절충이다.
        import weighted_token_ce as WT
        cands = cfg["outcome"]["alternative_tokens"]
        ids = [tok(c, add_special_tokens=False).input_ids for c in cands]
        assert all(len(v) == 1 for v in ids), "후보가 1토큰이 아니다: %s" % ids
        av_rate = float(np.mean([int(r["label"]) for r in rows]))
        w = float(sft.get("pos_weight", 1.0))
        assert WT.selftest(verbose=False), "weighted_token_ce 회귀 테스트 실패"
        print("  [objective] weighted_token_ce  pos_weight=%.3f  학습자료 AV비율=%.4f  "
              "정규화 분모=%.4f" % (w, av_rate, 1.0 + (w - 1.0) * av_rate), flush=True)
        TrainerCls = WT.make_trainer_class(Trainer)
        tr = TrainerCls(model=model, args=args, train_dataset=ds, data_collator=coll,
                        av_id=ids[0][0], pos_weight=w, av_rate=av_rate)
    else:
        tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=coll)
    # fp16 GradScaler 의 시작 배율을 낮춘다. **fp16 일 때만 의미가 있다.**
    # 기본 65536 에서 overflow 가 나면 그 step 의 갱신이 버려지고 배율이 절반이 된다.
    # 8 까지 내려오는 데 13 step 이 걸리고, 그 13 step 은 계산만 하고 학습은 0 이다.
    # AAAI 가 같은 GV100·fp16 에서 겪고 8 로 낮춘 값을 그대로 쓴다.
    from transformers import TrainerCallback
    _st = StatusFile(ROOT / "logs" / ("%s_%s_%s_%s.json" % (family, world, info, out.name)),
                     "%s/%s/%s/%s" % (family, world, info, out.name))
    tr.add_callback(make_status_callback(TrainerCallback, _st))
    print("  [status] %s" % _st.path.relative_to(ROOT), flush=True)
    _init = float(sft.get("fp16_initial_scale", 0) or 0)
    if _init and args.fp16:
        _sc = getattr(getattr(tr, "accelerator", None), "scaler", None)
        if _sc is None:
            print("  [warn] fp16 scaler 를 못 찾았다 - 초기 배율을 못 낮췄다", flush=True)
        else:
            _sc._init_scale = _init
            if getattr(_sc, "_scale", None) is not None:
                _sc._scale.fill_(_init)
            print("  [fp16] GradScaler 초기 배율 65536 -> %g" % _init, flush=True)
    # 중간 체크포인트가 있으면 이어서 학습한다. 어댑터 하나가 수십 시간이라
    # 처음부터 다시 하면 며칠을 잃는다.
    # 저장 도중에 프로세스가 죽으면 반쪽짜리 체크포인트 폴더가 남는다. 그걸 그대로
    # 고르면 Trainer 가 ValueError: Can't find a valid checkpoint 로 80 초 만에 죽고,
    # 워치독이 붙어 있으면 그 죽음이 무한 재시작이 된다. 2026-08-31 에 fold 3 이
    # 04:32 저장이 끊겨 README.md 하나만 남은 checkpoint-100 을 물고 15시간 30분을
    # 놀았다. 그래서 번호가 큰 것부터 내려오며 **쓸 수 있는 것**을 고른다.
    def _usable(d):
        # Trainer 가 재개에 실제로 요구하는 파일들. 어댑터 가중치는 safetensors 와
        # bin 두 형식이 다 가능하므로 둘 중 하나만 있으면 된다.
        w = any((d / n).exists() for n in
                ("adapter_model.safetensors", "adapter_model.bin",
                 "model.safetensors", "pytorch_model.bin"))
        return w and (d / "trainer_state.json").exists() and (d / "optimizer.pt").exists()

    ck = sorted((out / "_hf").glob("checkpoint-*"), key=lambda q: int(q.name.split("-")[-1]))
    resume = None
    if ck and sft.get("resume_from_checkpoint", True):
        for cand in reversed(ck):
            if _usable(cand):
                resume = str(cand)
                break
            print("  [resume] %s 는 불완전해서 건너뛴다 (저장 도중 중단된 것)"
                  % cand.name, flush=True)
    if resume:
        print("  [resume] %s 에서 이어서" % Path(resume).name, flush=True)
    elif ck:
        print("  [resume] 쓸 수 있는 체크포인트가 없다. 처음부터 학습한다.", flush=True)
    t0 = time.time(); res = tr.train(resume_from_checkpoint=resume); dt = time.time() - t0
    model.save_pretrained(str(out)); tok.save_pretrained(str(out))

    # **손실 이력을 어댑터 폴더에 따로 남긴다.**
    # 원래는 체크포인트 안의 trainer_state.json 에만 있었는데, save_total_limit 이 오래된
    # 체크포인트를 지우면 그 이력도 같이 사라진다. 실제로 254·508 번을 잃었다.
    # 학습곡선은 "언제부터 배웠나" 를 보는 유일한 기록이므로 체크포인트와 독립으로 둔다.
    hist = [h for h in tr.state.log_history if "loss" in h or "train_loss" in h]
    (out / "log_history.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        import pandas as _pd
        _pd.DataFrame(hist).to_csv(out / "log_history.csv", index=False)
    except Exception:
        pass

    # 환경도 같이 남긴다. 어느 카드에서 어떤 정밀도로 만든 어댑터인지 나중에 추적해야 한다
    # (GV100 은 float16, 로컬·3080 은 bfloat16 이다).
    env = {}
    try:
        env = {"gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        import transformers as _tf, peft as _pf
        env.update({"transformers": _tf.__version__, "peft": _pf.__version__})
        import fla as _fla                      # 선형어텐션 고속 커널 (있으면 2배 빠르다)
        env["fla"] = getattr(_fla, "__version__", "?")
    except Exception:
        env.setdefault("fla", "없음")
    write_json(out / "run_manifest.json",
               {"family": family, "world": world, "info": info, "seed": seed,
                "profile": sft.get("_profile"), "round": rnd,
                "continued_from": str(prev.relative_to(ROOT)) if prev else None,
                "learning_rate": float(sft["learning_rate"]), "schedule": sft["schedule"],
                "warmup_ratio": float(sft["warmup_ratio"]),
                "n_train": len(rows), "epochs": args.num_train_epochs,
                "optimizer_steps": int(res.global_step), "train_loss": float(res.training_loss),
                "seconds": round(dt, 1), "smoke": smoke,
                "objective": objective, "pos_weight": float(sft.get("pos_weight", 1.0)),
                # **tau 도 남긴다.** pos_weight 만 적혀 있고 tau 가 빠져 있어서, 2026-08-10 에
                # 2B 어댑터를 이어 학습하려다 원래 tau 를 기록에서 못 찾았다. 폴더 이름의
                # _la 꼬리표로 역추적해야 했다. tau 는 손실 자체를 바꾸므로 기록에 있어야 한다.
                "logit_adjust_tau": float((sft.get("loss_composite") or {})
                                          .get("logit_adjust_tau", 0.0) or 0.0),
                "init_adapter": str(init_adapter) if init_adapter else None,
                "train_file": src.name,
                "dtype": lc.get("dtype"), "four_bit": lc["four_bit"],
                "max_seq_len": lc["max_seq_len"],
                "effective_batch_size": int(sft["effective_batch_size"]),
                "save_steps": int(sft.get("save_steps", 254)),
                "checkpoints_kept": sorted(
                    int(q.name.split("-")[-1]) for q in (out / "_hf").glob("checkpoint-*")),
                "environment": env,
                "lora": {k: sft[k] for k in ("lora_rank", "lora_alpha", "lora_dropout")}})
    print("  [ok] %s  steps=%d loss=%.4f (%.0fs)"
          % (out.relative_to(ROOT), res.global_step, res.training_loss, dt))
    # **참조를 전부 끊고 순환참조까지 수거한다.**
    # Trainer <-> model <-> optimizer 가 서로를 참조해서 del 만으로는 안 풀린다.
    # 실측(2026-08-06, RTX 3080): 셀 하나가 끝난 뒤 같은 프로세스에서 다음 셀을 시작하니
    # 9.2GB 가 남아 있어 10.2GB 카드가 꽉 찼고, 샘플당 23초가 321초로 14배 느려졌다.
    # 그래도 프로세스 격리가 확실하므로 실행기는 셀마다 프로세스를 새로 띄운다.
    import gc
    del tr, res, ds, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("     (정리 후 GPU 점유 %.2fGB)" % (torch.cuda.memory_allocated() / 1e9), flush=True)
    return out


def run(stage="human", smoke=False, limit=None, gpu=None, only=None, family=None,
        objective=None, pos_weight=None, lr=None, steps=None, batch=None,
        profile=None, rnd=None, eff_batch=None, epochs=None, run_tag=None,
        init_adapter=None, tau=None, train_file=None):
    if gpu is not None:
        import torch
        torch.cuda.set_device(int(gpu))
    cfg = load_config()
    if profile:
        # 설정 묶음을 통째로 덮어쓴다. 값은 configs 에 있고 코드에는 없다.
        prof = (cfg.get("sft_profiles") or {}).get(profile)
        if prof is None:
            raise SystemExit("모르는 프로파일 %r — configs 의 sft_profiles 에 없다 (%s)"
                             % (profile, list((cfg.get("sft_profiles") or {}).keys())))
        cfg["sft"].update(prof)
        cfg["sft"]["_profile"] = profile
    if run_tag:
        cfg["sft"]["_run_tag"] = run_tag
    if epochs is not None:
        cfg["sft"]["epochs"] = int(epochs)
    if eff_batch is not None:
        # optimizer 가 보는 배치. **이건 학습 결과를 바꾼다** (--batch 와 다르다).
        # AAAI 는 5 인데, AV 비율이 18.8% 라 배치 5 에서는 갱신 세 번 중 한 번이
        # AV 를 한 건도 못 본다(35.2%). token_ce 는 다수 클래스로 쏠리는 목적함수라
        # 그 쏠림을 더 밀 수 있어 16 으로 올린다 (AV 0건 배치 3.6%).
        cfg["sft"]["effective_batch_size"] = int(eff_batch)
    if batch is not None:
        # **effective_batch_size 는 건드리지 않는다.** per-device 를 올리면 accum 이 그만큼
        # 줄어들어 optimizer 가 보는 배치는 그대로다 - 즉 학습 결과는 바뀌지 않고 처리량만
        # 오른다. 설정의 1 은 10GB 인 3080 기준이라, 32GB 인 GV100 에서는 낭비다.
        cfg["llm"]["train_batch_size"] = int(batch)
    if objective:
        cfg["sft"]["objective"] = objective
    if pos_weight is not None:
        cfg["sft"]["pos_weight"] = float(pos_weight)
    if tau is not None:
        # config 의 기본값은 1.0 이다. tau=0 으로 돌린 어댑터를 이어 학습할 때 그대로 두면
        # 손실이 조용히 바뀌어, "epoch 을 더 돌렸다" 와 "logit adjustment 를 켰다" 두 변화가
        # 한 실험에 섞인다. 어느 쪽이 숫자를 움직였는지 영영 못 가린다.
        cfg["sft"].setdefault("loss_composite", {})["logit_adjust_tau"] = float(tau)
    if lr is not None:
        cfg["sft"]["learning_rate"] = float(lr)
    if steps is not None:
        cfg["sft"]["max_steps"] = int(steps)
    # **덮어쓰기가 전부 끝난 뒤에 찍는다.** 프로파일 값을 먼저 찍으면 그 뒤의
    # --effective-batch 같은 덮어쓰기가 반영되지 않아, 로그에 5 라고 적혀 있는데
    # 실제로는 16 으로 도는 일이 생긴다. 실제로 2026-08-07 그렇게 찍혔다.
    _s = cfg["sft"]
    print("  [설정] profile=%s  lr=%s  schedule=%s  warmup=%s  effective_batch=%s"
          "  per_device=%s  epochs=%s  save_steps=%s/limit=%s  objective=%s"
          % (_s.get("_profile") or "없음", _s["learning_rate"], _s["schedule"],
             _s["warmup_ratio"], _s["effective_batch_size"], cfg["llm"]["train_batch_size"],
             _s["epochs"], _s.get("save_steps"), _s.get("save_total_limit"),
             _s.get("objective", "token_ce")))
    import math as _m
    _n = 8116
    print("  [예상] 1 epoch = ceil(%d/%s) = %d step" % (_n, _s["effective_batch_size"],
                                                       _m.ceil(_n / int(_s["effective_batch_size"]))))
    cs = cells(cfg, stage, family)
    if limit: cs = cs[:limit]
    print("\n=== Phase 16. SFT (%s) — %d cells ===" % (stage, len(cs)))
    miss = [f for f, v in cfg["llm"]["families"].items()
            if v.get("research", True) and not v.get("available")]
    if miss: print("  [warn] 로컬에 없는 모델군: %s — 이번 실행에서 제외" % miss)
    # only: 이 목록의 인덱스만 돈다. GPU 를 나눠 병렬로 돌릴 때 쓴다.
    #   GPU0 -> --only 0,2,4   GPU1 -> --only 1,3,5
    if only:
        keep = {int(x) for x in str(only).split(",")}
        cs = [c for k, c in enumerate(cs) if k in keep]
        print("  [only] %d개 셀: %s" % (len(cs), cs))
    if init_adapter and len(cs) != 1:
        raise SystemExit("--init-adapter 는 셀 하나에만 쓸 수 있다. 지금 %d개다: %s"
                         % (len(cs), cs))
    for f, w, i, s in cs:
        train_one(f, w, i, s, cfg, smoke=smoke, rnd=rnd, init_adapter=init_adapter,
                  train_file=train_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="human", choices=["human", "synthetic"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--only", default=None, help="쉼표로 구분한 셀 인덱스 (예: 0,2,4)")
    ap.add_argument("--lr", type=float, default=None,
                    help="학습률 덮어쓰기. margin_bce 는 기울기가 스칼라 하나에 집중돼 "
                         "토큰 교차엔트로피와 같은 값을 쓰면 발산한다")
    ap.add_argument("--steps", type=int, default=None, help="이 스텝만 돌고 멈춘다 (탐색용)")
    ap.add_argument("--objective", default=None, choices=["token_ce", "margin_bce", "weighted_token_ce", "aaai_composite"],
                    help="설정값을 덮어쓴다. margin_bce 는 별도 폴더에 저장된다")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="margin_bce 의 양성 가중치 (기본 설정값 2.233)")
    ap.add_argument("--family", default=None,
                    help="이 모델군만 (예: qwen / qwen_small). 생략하면 available 전부")
    ap.add_argument("--batch", type=int, default=None,
                    help="per-device 배치. effective_batch_size 는 그대로 두고 accum 만 "
                         "줄이므로 학습 결과는 안 바뀌고 처리량만 오른다 (GV100 32GB 용)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoch 수 덮어쓰기. aaai 프로파일은 1 이라 3 으로 돌리려면 필요하다")
    ap.add_argument("--effective-batch", type=int, default=None, dest="eff_batch",
                    help="optimizer 가 보는 배치. --batch(per-device)와 달리 학습 결과가 바뀐다")
    ap.add_argument("--run-tag", default=None, dest="run_tag",
                    help="어댑터 폴더에 붙일 임의 접미사. 같은 목적함수로 설정만 바꿔 "
                         "여러 판을 돌릴 때 폴더가 겹치지 않게 한다")
    ap.add_argument("--profile", default=None,
                    help="configs 의 sft_profiles 중 하나 (예: aaai). sft 값을 덮어쓴다")
    ap.add_argument("--round", type=int, default=None, dest="rnd",
                    help="1 epoch 씩 끊어 이어 붙일 때의 회차. 2 이상이면 앞 회차 "
                         "어댑터에서 이어 학습하고, 회차마다 폴더를 따로 만든다")
    ap.add_argument("--train-file", default=None, dest="train_file",
                    help="학습에 쓸 jsonl 을 직접 지정한다. 같은 world/info 이름 아래에서 "
                         "프롬프트를 다시 만들면 파일이 덮이므로, 어느 판으로 학습했는지 "
                         "이름만으로는 알 수 없다. 지정한 이름은 run_manifest 에 남는다")
    ap.add_argument("--tau", type=float, default=None,
                    help="loss_composite.logit_adjust_tau 덮어쓰기. 어댑터 폴더 이름에도 "
                         "붙는다 (tau=0 이면 안 붙는다)")
    ap.add_argument("--init-adapter", default=None, dest="init_adapter",
                    help="이 어댑터에서 이어 학습한다. --round 의 회차 이름 규칙을 쓰지 않고 "
                         "경로를 직접 받는다. 학습률 일정은 처음부터 새로 시작하므로 "
                         "결과는 '이어붙인 epoch' 이 아니라 warm restart 다")
    a = ap.parse_args()
    run(stage=a.stage, smoke=a.smoke, limit=a.limit, gpu=a.gpu, only=a.only,
        family=a.family, objective=a.objective, pos_weight=a.pos_weight,
        lr=a.lr, steps=a.steps, batch=a.batch, profile=a.profile, rnd=a.rnd,
        eff_batch=a.eff_batch, epochs=a.epochs, run_tag=a.run_tag,
        init_adapter=a.init_adapter, tau=a.tau, train_file=a.train_file)
