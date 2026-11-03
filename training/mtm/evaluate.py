"""Phần 4 — Chọn ngưỡng trên dev, đánh giá trên test, tổng hợp bảng kết quả.

  python -m mtm.evaluate --config configs/default.yaml --calibrate --exp C   # chọn τ, ρ, δ trên dev
  python -m mtm.evaluate --config configs/default.yaml --exp A0              # đánh giá test (A0, A, B, C)
  python -m mtm.evaluate --config configs/default.yaml --retrieval           # Recall@1/5 của truy xuất
  python -m mtm.evaluate --config configs/default.yaml --summary             # bảng kết quả cho slide
Thêm --comet để tính COMET (cần cài unbabel-comet).

Chỉ số: BLEU, chrF (sacrebleu), COMET (wmt22-comet-da), AUROC của C và R trong việc phát hiện bản nháp kém,
Recall@5 của LaBSE + FAISS, Recall@1 trước/sau rerank.
"""
import argparse
import glob
import os

import numpy as np

from .gate import Rarity, auroc_bad_draft, is_hard, select_thresholds
from .utils import load_config, load_json, read_jsonl, save_json, wpath, write_jsonl


# ---------------------------------------------------------------- chỉ số
class Comet:
    def __init__(self, name="Unbabel/wmt22-comet-da"):
        from comet import download_model, load_from_checkpoint
        self.m = load_from_checkpoint(download_model(name))

    def segments(self, srcs, hyps, refs):
        import torch
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = self.m.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0, progress_bar=False)
        return [float(x) for x in out.scores]


def corpus_metrics(srcs, hyps, refs, comet=None):
    import sacrebleu
    m = {"BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
         "chrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}
    if comet:
        m["COMET"] = round(float(np.mean(comet.segments(srcs, hyps, refs))), 4)
    return m


def sentence_util(srcs, hyps, refs, comet=None):
    if comet:
        return comet.segments(srcs, hyps, refs)
    import sacrebleu
    return [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)]


def apply_gate(res, th):
    """Từ kết quả chế độ 'always', suy ra đầu ra của hệ thống có cổng với ngưỡng th."""
    out, hard, used = [], [], []
    for r in res:
        h = is_hard(r["C"], r["R"], th["tau"], th["rho"])
        u = h and r["refined"] is not None and r["ev_score"] is not None and r["ev_score"] >= th["delta"]
        out.append(r["refined"] if u else r["draft"]); hard.append(h); used.append(u)
    return out, hard, used


def get_rarity(cfg):
    p = wpath(cfg, "results", "rarity.json")
    if os.path.exists(p):
        return Rarity.from_json(load_json(p))
    rows = read_jsonl(wpath(cfg, "data", "vi_en_train.jsonl")) + read_jsonl(wpath(cfg, "data", "fr_en_train.jsonl"))
    rar = Rarity.build(rows, cfg["gate"]["rare_freq"])
    save_json(p, rar.to_json())
    return rar


def load_translator(cfg, run, retrieval):
    from .pipeline import Translator, load_system
    from .modeling import load_checkpoint
    ckpt = wpath(cfg, "checkpoints", run, "best")
    if retrieval:
        tr, meta = load_system(cfg, ckpt, with_retrieval=True)
        tr.rarity = get_rarity(cfg)
        return tr, meta
    model, tok, meta = load_checkpoint(cfg, ckpt)
    return Translator(model, tok, cfg), meta


# ---------------------------------------------------------------- chọn ngưỡng trên dev
def calibrate(cfg, run="C", use_comet=False):
    tr, meta = load_translator(cfg, run, retrieval=True)
    comet = Comet() if use_comet else None
    g, th_all, diag = cfg["gate"], {}, {}
    for lang in meta["langs"]:
        p = wpath(cfg, "data", f"{lang}_en_dev.jsonl")
        if not os.path.exists(p):
            continue
        rows = read_jsonl(p)
        srcs, refs = [r["src"] for r in rows], [r["tgt"] for r in rows]
        res = tr.translate(srcs, [lang] * len(rows), mode="always")
        ud = sentence_util(srcs, [r["draft"] for r in res], refs, comet)
        ur = sentence_util(srcs, [r["refined"] or r["draft"] for r in res], refs, comet)
        ev = [r["ev_score"] if r["ev_score"] is not None else -1.0 for r in res]
        th = select_thresholds([r["C"] for r in res], [r["R"] for r in res], ev, ud, ur,
                               g["tau_grid"], g["rho_grid"], g["delta_grid"])
        th_all[lang] = th
        diag[lang] = {"AUROC_C": auroc_bad_draft(ud, [1 - r["C"] for r in res]),
                      "AUROC_R": auroc_bad_draft(ud, [r["R"] for r in res])}
        print(f"[calibrate] {lang}: {th} · {diag[lang]}")
    save_json(wpath(cfg, "checkpoints", run, "thresholds.json"), th_all)
    save_json(wpath(cfg, "results", f"{run}_dev_calibration.json"), {"thresholds": th_all, "diagnostics": diag})
    return th_all


# ---------------------------------------------------------------- đánh giá test
def evaluate(cfg, run, split="test", use_comet=False):
    meta = load_json(wpath(cfg, "checkpoints", run, "best", "meta.json"))
    sys_c = bool(meta.get("augment"))
    if sys_c and not os.path.exists(wpath(cfg, "checkpoints", run, "thresholds.json")):
        calibrate(cfg, run, use_comet)
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    tr, meta = load_translator(cfg, run, retrieval=sys_c)
    comet = Comet() if use_comet else None
    report = {"run": run, "meta": meta, "split": split, "results": {}}
    for lang in meta["langs"]:
        rows = read_jsonl(wpath(cfg, "data", f"{lang}_en_{split}.jsonl"))
        srcs, refs = [r["src"] for r in rows], [r["tgt"] for r in rows]
        res = tr.translate(srcs, [lang] * len(rows), mode="always" if sys_c else "draft")
        drafts = [r["draft"] for r in res]
        report["results"][f"{lang}/draft"] = corpus_metrics(srcs, drafts, refs, comet)
        if sys_c:
            th = tr.thresholds(lang)
            full, hard, used = apply_gate(res, th)
            m = corpus_metrics(srcs, full, refs, comet)
            m.update({"hard_frac": round(float(np.mean(hard)), 3), "evidence_frac": round(float(np.mean(used)), 3)})
            report["results"][f"{lang}/full"] = m
            always = [r["refined"] or r["draft"] for r in res]
            report["results"][f"{lang}/always"] = corpus_metrics(srcs, always, refs, comet)
            ud = sentence_util(srcs, drafts, refs, comet)
            report["results"][f"{lang}/gate"] = {"AUROC_C": auroc_bad_draft(ud, [1 - r["C"] for r in res]),
                                                 "AUROC_R": auroc_bad_draft(ud, [r["R"] for r in res])}
            for r, y, h, u in zip(res, full, hard, used):
                r.update(output=y, hard=h, evidence_used=u)
        for r, ref in zip(res, refs):
            r["ref"] = ref
        write_jsonl(wpath(cfg, "results", "outputs", f"{run}_{lang}_{split}.jsonl"), res)
        print(f"[eval] {run} {lang}: { {k: v for k, v in report['results'].items() if k.startswith(lang)} }")
    save_json(wpath(cfg, "results", f"{run}_{split}.json"), report)
    return report


def retrieval_eval(cfg):
    from .memory import FrEnMemory, LabseEncoder
    from .rerank import Reranker
    rows = read_jsonl(wpath(cfg, "data", "retrieval_eval.jsonl"))
    if not rows:  # dữ liệu nhỏ (chạy thử) có thể không có câu Việt nào trùng câu Anh trong kho
        out = {"n": 0, "note": "không có câu để đo Recall"}
        save_json(wpath(cfg, "results", "retrieval.json"), out)
        print("[retrieval] bỏ qua: không có câu để đo")
        return out
    mem = FrEnMemory.load(wpath(cfg, "memory"))
    enc, rr = LabseEncoder(cfg["retrieval"]["encoder"]), Reranker(cfg["retrieval"]["reranker"])
    k = cfg["retrieval"]["top_k"]
    srcs = [r["src"] for r in rows]
    gold = [mem.ids_with_en(r["tgt"]) | set(r["gold_mem_ids"]) for r in rows]
    hits = mem.search(enc.encode(srcs), k=k)
    picks = rr.pick(srcs, [[mem.en[j] for j, _ in h] for h in hits])
    r1 = np.mean([bool(h) and h[0][0] in g for h, g in zip(hits, gold)])
    rk = np.mean([any(j in g for j, _ in h) for h, g in zip(hits, gold)])
    rr1 = np.mean([p[0] is not None and h[p[0]][0] in g for h, p, g in zip(hits, picks, gold)])
    out = {"n": len(rows), "Recall@1_LaBSE": round(float(r1), 4), f"Recall@{k}_LaBSE": round(float(rk), 4),
           "Recall@1_after_rerank": round(float(rr1), 4)}
    save_json(wpath(cfg, "results", "retrieval.json"), out)
    print("[retrieval]", out)
    return out


def summary(cfg):
    lines = ["| Thí nghiệm | Chế độ | VI BLEU | VI chrF | VI COMET | FR BLEU | FR chrF | % câu khó (VI) | % dùng bằng chứng (VI) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(wpath(cfg, "results", "*_test.json"))):
        rep = load_json(p)
        res = rep["results"]
        for mode in ("draft", "full", "always"):
            vi = res.get(f"vi/{mode}")
            if not vi:
                continue
            fr = res.get(f"fr/{mode}", {})
            lines.append(f"| {rep['run']} | {mode} | {vi['BLEU']} | {vi['chrF']} | {vi.get('COMET', '–')} | "
                         f"{fr.get('BLEU', '–')} | {fr.get('chrF', '–')} | {vi.get('hard_frac', '–')} | {vi.get('evidence_frac', '–')} |")
    table = "\n".join(lines)
    with open(wpath(cfg, "results", "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--comet", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.calibrate:
        calibrate(cfg, a.exp or "C", a.comet)
    elif a.retrieval:
        retrieval_eval(cfg)
    elif a.summary:
        summary(cfg)
    else:
        evaluate(cfg, a.exp, a.split, a.comet)
