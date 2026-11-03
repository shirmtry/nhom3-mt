"""Bước 3.6 (tầng 1) — Kho Pháp–Anh: LaBSE + FAISS.

Kho = 100K cặp Pháp–Anh dùng để huấn luyện (vai trò B). Khóa tìm kiếm là câu Pháp.
- Câu Việt đầu vào → tìm xuyên ngôn ngữ (Việt ↔ Pháp)
- Câu Pháp đầu vào → tìm cùng ngôn ngữ (đúng dạng Neural Fuzzy Repair gốc)
Không dịch qua tiếng Pháp: chỉ lấy câu Anh đi kèm làm bằng chứng.

Chạy:  python -m mtm.memory --config configs/default.yaml
"""
import argparse
import os

import numpy as np

from .utils import load_config, norm_key, read_jsonl, save_json, load_json, wpath


class LabseEncoder:
    def __init__(self, name="sentence-transformers/LaBSE", device=None):
        from sentence_transformers import SentenceTransformer
        from .utils import device as dv
        self.model = SentenceTransformer(name, device=device or dv())

    def encode(self, texts, batch_size=128, progress=False):
        e = self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=progress)
        return e.astype("float32")


class FrEnMemory:
    def __init__(self, fr, en, index):
        self.fr, self.en, self.index = fr, en, index
        self.en_key = {}
        for i, t in enumerate(en):
            self.en_key.setdefault(norm_key(t), []).append(i)

    @classmethod
    def build(cls, rows, encoder):
        import faiss
        fr = [r["src"] for r in rows]
        en = [r["tgt"] for r in rows]
        emb = encoder.encode(fr, progress=True)
        index = faiss.IndexFlatIP(emb.shape[1])  # vector đã chuẩn hóa → tích vô hướng = cosine
        index.add(emb)
        return cls(fr, en, index)

    def save(self, folder):
        import faiss
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        save_json(os.path.join(folder, "pairs.json"), {"fr": self.fr, "en": self.en})

    @classmethod
    def load(cls, folder):
        import faiss
        p = load_json(os.path.join(folder, "pairs.json"))
        return cls(p["fr"], p["en"], faiss.read_index(os.path.join(folder, "index.faiss")))

    def search(self, query_emb, k=5, exclude=None):
        """query_emb: (n, d). exclude[i]: tập mem_id cần bỏ cho câu i (chính nó / trùng đáp án).
        Trả về list[list[(mem_id, cosine)]]."""
        extra = 0 if exclude is None else max((len(e) for e in exclude), default=0)
        D, I = self.index.search(np.ascontiguousarray(query_emb, dtype="float32"), k + extra)
        out = []
        for i in range(len(I)):
            ex = exclude[i] if exclude is not None else ()
            hits = [(int(j), float(d)) for j, d in zip(I[i], D[i]) if j >= 0 and int(j) not in ex]
            out.append(hits[:k])
        return out

    def ids_with_en(self, text):
        return set(self.en_key.get(norm_key(text), []))


def build_memory(cfg, encoder=None):
    rows = read_jsonl(wpath(cfg, "data", "fr_en_train.jsonl"))
    encoder = encoder or LabseEncoder(cfg["retrieval"]["encoder"])
    mem = FrEnMemory.build(rows, encoder)
    mem.save(wpath(cfg, "memory"))
    print(f"[memory] đã lưu {len(rows)} cặp vào {wpath(cfg, 'memory')}")
    return mem


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    build_memory(load_config(ap.parse_args().config))
