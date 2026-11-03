"""Bước 3.6 (tầng 2) — Reranker bge-reranker-v2-m3 (cross-encoder đa ngôn ngữ).

Đọc cả cặp (câu nguồn x, câu Anh ứng viên) rồi chấm điểm 0..1. Chọn ứng viên cao nhất làm e*;
nếu điểm < δ thì bỏ qua (giữ bản nháp).
"""
import numpy as np
import torch


class Reranker:
    def __init__(self, name="BAAI/bge-reranker-v2-m3", device=None, max_length=256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .utils import device as dv
        self.device = device or dv()
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, queries, passages, batch_size=64):
        scores = []
        for i in range(0, len(queries), batch_size):
            enc = self.tok(queries[i:i + batch_size], passages[i:i + batch_size], padding=True,
                           truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.view(-1).float()
            scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0, dtype="float32")

    def pick(self, queries, cand_lists, batch_size=64):
        """cand_lists[i]: list câu Anh ứng viên của câu i. Trả về list[(vị trí tốt nhất, điểm, mọi điểm)]."""
        flat_q, flat_p, owner = [], [], []
        for i, (q, cands) in enumerate(zip(queries, cand_lists)):
            for c in cands:
                flat_q.append(q); flat_p.append(c); owner.append(i)
        s = self.score(flat_q, flat_p, batch_size)
        per = [[] for _ in queries]
        for i, v in zip(owner, s):
            per[i].append(float(v))
        out = []
        for sc in per:
            if not sc:
                out.append((None, 0.0, []))
            else:
                j = int(np.argmax(sc))
                out.append((j, sc[j], sc))
        return out
