"""Xuất mô hình C + kho + ngưỡng + kết quả thử nghiệm thành một gói cho app demo (chạy CPU).

  python -m mtm.export --config configs/default.yaml --exp C

Tạo  <workdir>/export/nhom3_mt.zip  (chỉ giữ file zip) gồm:
  model/          mBART-50 đã gộp LoRA (fp16, ~1,2 GB) + tokenizer
  memory/         pairs.json (100K cặp Pháp–Anh) + emb.npy (vector LaBSE câu Pháp, fp16)
  rarity.json     từ điển tần suất cho độ hiếm R
  thresholds.json τ, ρ, δ đã chọn trên dev (theo từng ngôn ngữ)
  results/        kết quả test A0/A/B/C, truy xuất, chọn ngưỡng, ví dụ câu khó
  meta.json
Cài vào app demo (thư mục gốc repo):  python tai_mo_hinh.py work/export/nhom3_mt.zip
"""
import argparse
import glob
import os
import shutil

import numpy as np

from .utils import load_json, read_jsonl, save_json, wpath


def pick_examples(cfg, run, n_good=12, n_bad=4):
    """Ví dụ câu đi nhánh khó có dùng bằng chứng: câu được sửa tốt nhất và vài câu bị sửa hỏng (trung thực)."""
    import sacrebleu
    p = wpath(cfg, "results", "outputs", f"{run}_vi_test.jsonl")
    if not os.path.exists(p):
        return []
    rows = [r for r in read_jsonl(p) if r.get("evidence_used")]
    for r in rows:
        r["gain"] = sacrebleu.sentence_chrf(r["output"], [r["ref"]]).score - sacrebleu.sentence_chrf(r["draft"], [r["ref"]]).score
    rows.sort(key=lambda r: -r["gain"])
    keep = rows[:n_good] + [r for r in rows[::-1][:n_bad] if r["gain"] < 0]
    fields = ("src", "lang", "ref", "draft", "output", "C", "R", "evidence", "ev_score", "gain", "candidates")
    return [{k: r.get(k) for k in fields} for r in keep]


def export(cfg, run="C", fp16=True):
    import torch
    from .memory import FrEnMemory
    from .modeling import load_checkpoint

    out = wpath(cfg, "export", "nhom3_mt")
    model, tok, meta = load_checkpoint(cfg, wpath(cfg, "checkpoints", run, "best"), merge=True)
    model = model.to("cpu")
    if fp16:
        model = model.to(torch.float16)
    model.save_pretrained(os.path.join(out, "model"), safe_serialization=True)
    tok.save_pretrained(os.path.join(out, "model"))
    print("[export] đã lưu mô hình")

    mem = FrEnMemory.load(wpath(cfg, "memory"))
    os.makedirs(os.path.join(out, "memory"), exist_ok=True)
    np.save(os.path.join(out, "memory", "emb.npy"), mem.index.reconstruct_n(0, mem.index.ntotal).astype("float16"))
    shutil.copy(wpath(cfg, "memory", "pairs.json"), os.path.join(out, "memory", "pairs.json"))

    from .evaluate import get_rarity
    get_rarity(cfg)
    shutil.copy(wpath(cfg, "results", "rarity.json"), os.path.join(out, "rarity.json"))
    th = wpath(cfg, "checkpoints", run, "thresholds.json")
    if os.path.exists(th):
        shutil.copy(th, os.path.join(out, "thresholds.json"))
    else:
        print("[export] CẢNH BÁO: chưa có thresholds.json — chạy  python -m mtm.evaluate --calibrate --exp C")

    res = os.path.join(out, "results")
    os.makedirs(res, exist_ok=True)
    for p in glob.glob(wpath(cfg, "results", "*.json")) + glob.glob(wpath(cfg, "results", "summary.md")):
        if not p.endswith("rarity.json"):
            shutil.copy(p, res)
    save_json(os.path.join(res, "examples.json"), pick_examples(cfg, run))

    try:
        import underthesea  # noqa: F401
        rar_tok = "underthesea"
    except ImportError:
        rar_tok = "syllable"
    d, m = cfg["decode"], cfg["model"]
    save_json(os.path.join(out, "meta.json"), {
        "run": run, "base": cfg["model"]["base"], "langs": meta.get("langs", ["vi", "fr"]),
        "num_beams": d["num_beams"], "max_new_tokens": d["max_new_tokens"],
        "max_src_len": m["max_src_len"], "max_evidence_len": m["max_evidence_len"],
        "encoder": cfg["retrieval"]["encoder"], "reranker": cfg["retrieval"]["reranker"],
        "top_k": cfg["retrieval"]["top_k"], "rarity_tokenizer": rar_tok, "dev_chrF": meta.get("dev_chrF")})
    zip_path = shutil.make_archive(wpath(cfg, "export", "nhom3_mt"), "zip", root_dir=wpath(cfg, "export"), base_dir="nhom3_mt")
    shutil.rmtree(out, ignore_errors=True)  # chỉ giữ file zip (bớt ~1,4 GB)
    print(f"[export] xong: {zip_path} ({os.path.getsize(zip_path) / 1e9:.2f} GB)")
    return zip_path


if __name__ == "__main__":
    from .utils import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--exp", default="C")
    a = ap.parse_args()
    export(load_config(a.config), a.exp)
