"""Bước 3.4 — Huấn luyện. Một lệnh cho mỗi thí nghiệm của Phần 4:

  A0  Transformer nhỏ huấn luyện từ đầu, chỉ Việt–Anh       (đo đóng góp của tiền huấn luyện)
  A   mBART-50 + LoRA, chỉ Việt–Anh                          (song ngữ)
  B   mBART-50 + LoRA, Việt–Anh + Pháp–Anh, lấy mẫu nhiệt độ (multilingual, vai trò A của Pháp–Anh)
  C   như B + mẫu kèm bằng chứng (để chạy bước dịch lại)      (thêm vai trò B của Pháp–Anh)

Giai đoạn 1: học chung theo nhiệt độ T. Giai đoạn 2 (chỉ khi có >1 ngôn ngữ): ngắn, Việt–Anh là chính
nhưng vẫn giữ ít Pháp–Anh và mẫu kèm bằng chứng. Dừng sớm theo chrF trên dev Việt–Anh.

Chạy:  python -m mtm.train --config configs/default.yaml --exp B [--temperature 2]
"""
import argparse
import os
import time
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .modeling import (build_lora_model, build_scratch_model, encode_sources, encode_targets, generate,
                       load_tokenizer, pad_batch, save_checkpoint, train_scratch_tokenizer)
from .sampling import fixed_probs, make_sampler, temperature_probs
from .utils import device, load_config, read_jsonl, save_json, set_seed, wpath


def load_rows(cfg, langs, split):
    rows = []
    for l in langs:
        rows += read_jsonl(wpath(cfg, "data", f"{l}_en_{split}.jsonl"))
    return rows


def featurize(tok, rows, cfg, use_evidence):
    m = cfg["model"]
    src = encode_sources(tok, [r["src"] for r in rows], [r["lang"] for r in rows],
                         [r.get("evidence") for r in rows] if use_evidence else None,
                         m["max_src_len"], m["max_evidence_len"])
    tgt = encode_targets(tok, [r["tgt"] for r in rows], m["max_tgt_len"])
    return [{"input_ids": s, "labels": t, "lang": r["lang"]} for s, t, r in zip(src, tgt, rows)]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        ids, mask = pad_batch([b["input_ids"] for b in batch], self.pad_id)
        labels, _ = pad_batch([b["labels"] for b in batch], -100)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def dev_chrf(model, tok, rows, n, max_new_tokens):
    import sacrebleu
    rows = rows[:n]
    hyps, _ = generate(model, tok, [r["src"] for r in rows], [r["lang"] for r in rows], num_beams=1,
                       max_new_tokens=max_new_tokens, batch_size=32, with_confidence=False)
    return sacrebleu.corpus_chrf(hyps, [[r["tgt"] for r in rows]]).score


def run_stage(name, model, tok, feats, probs, steps, lr, warmup, hp, cfg, dev_rows, ckpt_dir, meta, state):
    from transformers import get_linear_schedule_with_warmup
    from transformers.models.mbart.modeling_mbart import shift_tokens_right
    t = cfg["train"]
    bs, ga = hp["batch_size"], hp.get("grad_accum", 1)
    fp16 = t["fp16"] and torch.cuda.is_available()
    dl = DataLoader(feats, batch_size=bs, collate_fn=Collator(tok.pad_token_id), num_workers=2,
                    sampler=make_sampler([f["lang"] for f in feats], probs, steps * bs * ga, cfg["seed"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=hp.get("weight_decay", 0.0))
    sched = get_linear_schedule_with_warmup(opt, warmup, steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    dev, pad = next(model.parameters()).device, tok.pad_token_id
    print(f"[train] {name}: {steps} bước, lr={lr}, tỉ lệ ngôn ngữ={ {k: round(v, 3) for k, v in probs.items()} }")
    model.train()
    step, run_loss, t0, bad = 0, 0.0, time.time(), 0
    for i, batch in enumerate(dl):
        batch = {k: v.to(dev) for k, v in batch.items()}
        dec_in = shift_tokens_right(batch["labels"], pad)  # [en_XX] y </s> → </s> [en_XX] y
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           decoder_input_ids=dec_in, use_cache=False).logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), batch["labels"].view(-1),
                               ignore_index=-100, label_smoothing=t["label_smoothing"])
        scaler.scale(loss / ga).backward()
        run_loss += loss.item()
        if (i + 1) % ga:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f"[train] {name} bước {step}/{steps} · loss {run_loss / (50 * ga):.3f} · {time.time() - t0:.0f}s")
            run_loss = 0.0
        if step % t["eval_every"] == 0 or step == steps:
            score = dev_chrf(model, tok, dev_rows, t["eval_size"], cfg["decode"]["max_new_tokens"])
            torch.cuda.empty_cache()
            model.train()
            state["history"].append({"stage": name, "step": step, "dev_chrF": score})
            if score > state["best"]:
                state["best"], bad = score, 0
                save_checkpoint(model, tok, os.path.join(ckpt_dir, "best"), dict(meta, dev_chrF=score, stage=name, step=step))
                print(f"[train] {name} bước {step}: chrF dev {score:.2f}  ← tốt nhất, đã lưu")
            else:
                bad += 1
                print(f"[train] {name} bước {step}: chrF dev {score:.2f} (tốt nhất {state['best']:.2f}, chờ {bad}/{t['patience']})")
                if bad >= t["patience"]:
                    print(f"[train] {name}: dừng sớm")
                    break
        if step >= steps:
            break
    return state


def reload_best_lora(model, ckpt_dir):
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights
    set_peft_model_state_dict(model, load_peft_weights(os.path.join(ckpt_dir, "best")))


def train_experiment(cfg, exp_name, temperature=None):
    exp, t = cfg["experiments"][exp_name], cfg["train"]
    T = temperature if temperature is not None else exp.get("temperature", t["temperature"])
    run_name = exp_name if temperature is None else f"{exp_name}_T{temperature:g}"
    set_seed(cfg["seed"])
    langs = exp["langs"]
    rows = load_rows(cfg, langs, "train")
    dev_rows = read_jsonl(wpath(cfg, "data", "vi_en_dev.jsonl"))
    ckpt_dir = wpath(cfg, "checkpoints", run_name)

    if exp["type"] == "scratch":
        texts = [r["src"] for r in rows] + [r["tgt"] for r in rows]
        tok = train_scratch_tokenizer(texts, os.path.join(ckpt_dir, "tokenizer"), cfg["scratch"]["spm_vocab"])
        model = build_scratch_model(cfg, tok)
        hp = cfg["scratch"]
    else:
        tok = load_tokenizer(cfg["model"]["base"])
        model = build_lora_model(cfg)
        hp = t
    model.to(device())

    use_ev = bool(exp.get("augment"))
    if use_ev:  # mẫu kèm bằng chứng cho bước dịch lại
        from .augment import build_augmented
        from .memory import FrEnMemory, LabseEncoder
        from .rerank import Reranker
        memory = FrEnMemory.load(wpath(cfg, "memory"))
        enc, rr = LabseEncoder(cfg["retrieval"]["encoder"]), Reranker(cfg["retrieval"]["reranker"])
        rows = build_augmented(rows, memory, enc, rr, cfg, wpath(cfg, "data", "augmented_train.jsonl"), cfg["seed"])
        del enc, rr
        torch.cuda.empty_cache()

    feats = featurize(tok, rows, cfg, use_ev)
    sizes = Counter(r["lang"] for r in rows)
    meta = {"exp": run_name, "type": exp["type"], "langs": langs, "temperature": T, "augment": use_ev,
            "base": cfg["model"]["base"] if exp["type"] == "lora" else None}
    state = {"best": -1.0, "history": []}

    steps1 = hp.get("steps", t["stage1_steps"])
    run_stage("giai_doan_1", model, tok, feats, temperature_probs(dict(sizes), T), steps1,
              hp["lr"], hp["warmup"], hp, cfg, dev_rows, ckpt_dir, meta, state)

    if len(sizes) > 1 and t["stage2_steps"] > 0 and exp["type"] == "lora":
        reload_best_lora(model, ckpt_dir)
        run_stage("giai_doan_2", model, tok, feats, fixed_probs(dict(sizes), t["stage2_vi_share"]),
                  t["stage2_steps"], t["stage2_lr"], max(1, t["stage2_steps"] // 10), hp, cfg, dev_rows,
                  ckpt_dir, meta, state)

    save_json(os.path.join(ckpt_dir, "history.json"), state)
    print(f"[train] {run_name}: chrF dev tốt nhất = {state['best']:.2f} → {os.path.join(ckpt_dir, 'best')}")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--exp", required=True, help="A0 | A | B | C")
    ap.add_argument("--temperature", type=float, default=None, help="ghi đè T (vd 1, 2, 5) để quét nhiệt độ")
    a = ap.parse_args()
    train_experiment(load_config(a.config), a.exp, a.temperature)
