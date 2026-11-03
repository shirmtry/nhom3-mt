"""Bước 3.7 — Dữ liệu huấn luyện cho bước dịch lại (Neural Fuzzy Repair mở rộng xuyên ngôn ngữ).

Mỗi mẫu huấn luyện (cả câu Việt và câu Pháp) được gán cố định một kiểu:
  50% thường   : [lang] x </s>
  35% đúng     : [lang] x </s> e* </s>   (e* = câu Anh tìm được trong kho, đã rerank)
  15% ngẫu nhiên: [lang] x </s> e_rand </s> (dạy mô hình bỏ qua bằng chứng sai)
Khi tìm bằng chứng cho một câu huấn luyện, loại chính cặp đó và mọi cặp có câu Anh trùng đáp án,
để mô hình không học cách chép đáp án.
"""
import os
import random

from .utils import read_jsonl, write_jsonl


def build_augmented(rows, memory, encoder, reranker, cfg, cache_path=None, seed=0):
    if cache_path and os.path.exists(cache_path):
        return read_jsonl(cache_path)
    a, k = cfg["augment"], cfg["retrieval"]["top_k"]
    rng = random.Random(seed)
    modes = rng.choices(["plain", "correct", "random"], weights=[a["p_plain"], a["p_correct"], a["p_random"]], k=len(rows))
    out = [dict(r, evidence=None, ev_mode="plain") for r in rows]

    corr = [i for i, m in enumerate(modes) if m == "correct"]
    if corr:
        emb = encoder.encode([rows[i]["src"] for i in corr], progress=True)
        exclude = []
        for i in corr:
            ex = memory.ids_with_en(rows[i]["tgt"])
            if "mem_id" in rows[i]:
                ex = ex | {rows[i]["mem_id"]}
            exclude.append(ex)
        hits = memory.search(emb, k=k, exclude=exclude)
        picks = reranker.pick([rows[i]["src"] for i in corr], [[memory.en[j] for j, _ in h] for h in hits])
        for i, h, (j, s, _) in zip(corr, hits, picks):
            if j is not None and s >= a["min_rerank_score"]:
                out[i]["evidence"], out[i]["ev_mode"], out[i]["ev_score"] = memory.en[h[j][0]], "correct", s

    n = len(memory.en)
    for i, m in enumerate(modes):
        if m == "random":
            ex = memory.ids_with_en(rows[i]["tgt"]) | ({rows[i]["mem_id"]} if "mem_id" in rows[i] else set())
            j = rng.randrange(n)
            while j in ex:
                j = rng.randrange(n)
            out[i]["evidence"], out[i]["ev_mode"] = memory.en[j], "random"

    if cache_path:
        write_jsonl(cache_path, out)
    got = {m: sum(1 for r in out if r["ev_mode"] == m) for m in ("plain", "correct", "random")}
    print(f"[augment] {got} (mẫu 'đúng' có điểm rerank thấp được đổi thành 'thường')")
    return out
