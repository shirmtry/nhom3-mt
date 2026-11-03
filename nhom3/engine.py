"""Bộ dịch Nhóm 3 chạy trên CPU: Multilingual Transformer Việt/Pháp → Anh với dữ liệu song ngữ hạn chế.

Một câu đi qua 6 bước (kiến trúc ở slide 3.0):
    ① tiền xử lý → ② Transformer dịch nháp ŷ + độ tự tin C → ③ cổng (C, R)
       câu dễ : y = ŷ, không tra kho
       câu khó: ④ LaBSE tìm Top-5 trong kho Pháp–Anh → ⑤ reranker chọn e* (điểm < δ: giữ ŷ)
                → ⑥ cùng Transformer dịch lại với đầu vào x + e*  →  y = ŷ′

Dữ liệu mô hình: thư mục models/nhom3_mt/ (giải nén từ nhom3_mt.zip, xem tai_mo_hinh.py).
"""
import json
import os
import re
import threading
import time
import unicodedata
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("NHOM3_DIR", os.path.join(ROOT, "models", "nhom3_mt"))
LANG_CODE = {"vi": "vi_VN", "fr": "fr_XX", "en": "en_XX"}
NEVER_HARD = {"tau": 0.0, "rho": 1.01, "delta": 1.01}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?…])\s+")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nfc(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def split_sentences(text, max_sentences=8):
    parts = [p.strip() for p in _SENT.split(nfc(text)) if p.strip()]
    if len(parts) > max_sentences:  # gộp phần dư vào câu cuối để không mất chữ
        parts = parts[: max_sentences - 1] + [" ".join(parts[max_sentences - 1:])]
    return parts


def has_underthesea():
    try:
        import underthesea  # noqa: F401
        return True
    except ImportError:
        return False


def words(text, lang, use_underthesea=True):
    """Tách từ — chỉ dùng cho độ hiếm R, giống hệt cách tạo rarity.json lúc huấn luyện.
    Tiếng Việt: underthesea gộp các âm tiết đi chung thành 1 từ ('bàn phím', 'sinh viên')."""
    text = text.lower()
    if lang == "vi" and use_underthesea:
        try:
            from underthesea import word_tokenize
            return [t for t in word_tokenize(text) if _WORD.search(t)]
        except ImportError:
            pass
    return _WORD.findall(text)


def is_hard(C, R, th):
    return (C < th["tau"]) or (R > th["rho"])


class Translator:
    """Nạp lười: Transformer khi có yêu cầu đầu tiên; kho + LaBSE + reranker khi gặp câu khó đầu tiên
    (hoặc khi gọi warmup)."""

    def __init__(self, folder=MODEL_DIR):
        self.folder = folder
        self.meta = load_json(os.path.join(folder, "meta.json"), {}) or {}
        self.thresholds = load_json(os.path.join(folder, "thresholds.json"), {}) or {}
        self.model = self.tok = None
        self.encoder = self.reranker = self.rr_tok = None
        self.mem_fr = self.mem_en = self.mem_emb = None
        self.freq, self.k, self.use_underthesea = {}, 3, True
        self.state = "chưa nạp"
        self._lock = threading.RLock()

    # ------------------------------------------------------------ trạng thái
    def available(self):
        return os.path.exists(os.path.join(self.folder, "model", "config.json"))

    def status(self):
        return {"available": self.available(), "state": self.state,
                "core_loaded": self.model is not None, "retrieval_loaded": self.reranker is not None,
                "calibrated": bool(self.thresholds), "meta": self.meta,
                "memory_size": len(self.mem_en) if self.mem_en is not None else None}

    def th(self, lang):
        return self.thresholds.get(lang) or self.thresholds.get("vi") or NEVER_HARD

    # ------------------------------------------------------------ nạp mô hình
    def ensure_core(self):
        with self._lock:
            if self.model is not None:
                return
            if not self.available():
                raise RuntimeError("Chưa có mô hình: chạy  python tai_mo_hinh.py  (xem README)")
            from transformers import AutoTokenizer, MBartForConditionalGeneration
            self.state = "đang nạp Transformer…"
            path = os.path.join(self.folder, "model")
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = MBartForConditionalGeneration.from_pretrained(path).float().eval()  # file fp16 → CPU fp32
            rar = load_json(os.path.join(self.folder, "rarity.json"), {"k": 3, "freq": {}})
            self.freq = {l: Counter(c) for l, c in rar["freq"].items()}
            self.k = rar["k"]
            self.use_underthesea = self.meta.get("rarity_tokenizer", "underthesea") == "underthesea"
            self.state = "sẵn sàng (Transformer)"

    def ensure_retrieval(self):
        with self._lock:
            if self.reranker is not None:
                return
            self.ensure_core()
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.state = "đang nạp kho Pháp–Anh + LaBSE + reranker…"
            pairs = load_json(os.path.join(self.folder, "memory", "pairs.json"))
            self.mem_fr, self.mem_en = pairs["fr"], pairs["en"]
            self.mem_emb = np.load(os.path.join(self.folder, "memory", "emb.npy")).astype("float32")
            self.encoder = SentenceTransformer(self.meta.get("encoder", "sentence-transformers/LaBSE"), device="cpu")
            name = self.meta.get("reranker", "BAAI/bge-reranker-v2-m3")
            self.rr_tok = AutoTokenizer.from_pretrained(name)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(name).float().eval()
            self.state = "sẵn sàng"

    # ------------------------------------------------------------ các bước
    def encode(self, text, lang, evidence=None):
        self.tok.src_lang = LANG_CODE[lang]
        ids = list(self.tok(text, truncation=True, max_length=self.meta.get("max_src_len", 128))["input_ids"])
        if evidence:  # [lang] x </s> e* </s>
            ev = self.tok(evidence, add_special_tokens=False, truncation=True,
                          max_length=self.meta.get("max_evidence_len", 96))["input_ids"]
            ids = ids + list(ev) + [self.tok.eos_token_id]
        return ids

    def generate(self, ids, with_conf=True):
        import torch
        with torch.inference_mode():
            x = torch.tensor([ids])
            m = torch.ones_like(x)
            beams = int(os.environ.get("NHOM3_BEAMS", self.meta.get("num_beams", 4)))
            seqs = self.model.generate(input_ids=x, attention_mask=m, num_beams=beams,
                                       max_new_tokens=self.meta.get("max_new_tokens", 160),
                                       forced_bos_token_id=self.tok.convert_tokens_to_ids("en_XX"),
                                       decoder_start_token_id=self.tok.eos_token_id)
            if not torch.is_tensor(seqs):
                seqs = seqs.sequences
            text = self.tok.decode(seqs[0], skip_special_tokens=True).strip()
            if not with_conf:
                return text, None
            # C = exp(trung bình log p của các token đã sinh, bỏ token en_XX bị ép) — giống hệt lúc chọn ngưỡng
            dec_in, tgt = seqs[:, :-1], seqs[:, 1:]
            logits = self.model(input_ids=x, attention_mask=m, decoder_input_ids=dec_in).logits.float()
            lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            valid = tgt.ne(self.tok.pad_token_id)
            valid[:, 0] = False
            C = float(torch.exp(lp.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp(min=1))[0])
            return text, C

    def rarity(self, text, lang):
        ws = words(text, lang, self.use_underthesea)
        if not ws:
            return 0.0, [], []
        f = self.freq.get(lang, Counter())
        rare = [w for w in ws if f[w] < self.k]
        return len(rare) / len(ws), rare, ws

    def retrieve(self, text, k):
        q = self.encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        scores = self.mem_emb @ q
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def rerank(self, query, cands):
        import torch
        enc = self.rr_tok([query] * len(cands), cands, padding=True, truncation=True, max_length=256,
                          return_tensors="pt")
        with torch.inference_mode():
            return torch.sigmoid(self.reranker(**enc).logits.view(-1).float()).tolist()

    # ------------------------------------------------------------ một câu đi qua hệ thống
    def translate_sentence(self, x, lang, mode="full"):
        T = {}
        t0 = time.perf_counter()
        ids = self.encode(x, lang)
        pieces = self.tok.convert_ids_to_tokens(ids)
        T["preprocess"] = time.perf_counter() - t0                                          # ①
        t0 = time.perf_counter()
        draft, C = self.generate(ids)                                                        # ②
        T["draft"] = time.perf_counter() - t0
        R, rare_words, seg = self.rarity(x, lang)                                            # ③
        seg_tool = "underthesea" if lang == "vi" and self.use_underthesea and has_underthesea() else "theo khoảng trắng"
        th = self.th(lang)
        hard = True if mode == "always" else (False if mode == "draft" else is_hard(C, R, th))
        step = {"src": x, "lang": lang, "lang_code": LANG_CODE[lang], "pieces": pieces[:40], "n_pieces": len(pieces),
                "words": seg[:40], "seg_tool": seg_tool, "draft": draft, "C": C, "R": R,
                "rare_words": rare_words[:10], "thresholds": th,
                "reasons": {"C_low": C < th["tau"], "R_high": R > th["rho"]}, "hard": hard,
                "candidates": [], "evidence": None, "ev_score": None, "evidence_used": False,
                "refined": None, "output": draft}
        if hard:
            self.ensure_retrieval()
            t0 = time.perf_counter()
            hits = self.retrieve(x, self.meta.get("top_k", 5))                              # ④
            T["retrieve"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            rr = self.rerank(x, [self.mem_en[i] for i, _ in hits])                           # ⑤
            T["rerank"] = time.perf_counter() - t0
            best = int(np.argmax(rr))
            step["candidates"] = [{"fr": self.mem_fr[i], "en": self.mem_en[i], "cosine": round(c, 3),
                                   "rerank": round(s, 3), "chosen": j == best}
                                  for j, ((i, c), s) in enumerate(zip(hits, rr))]
            step["evidence"], step["ev_score"] = self.mem_en[hits[best][0]], rr[best]
            if mode == "always" or rr[best] >= th["delta"]:
                t0 = time.perf_counter()
                refined, _ = self.generate(self.encode(x, lang, step["evidence"]), with_conf=False)  # ⑥
                T["refine"] = time.perf_counter() - t0
                step.update(refined=refined, output=refined, evidence_used=True)
        step["timings_ms"] = {k: round(v * 1000) for k, v in T.items()}
        return step

    def translate(self, text, lang, mode="full"):
        self.ensure_core()
        with self._lock:
            steps = [self.translate_sentence(s, lang, mode) for s in split_sentences(text)]
        return {"translation": " ".join(s["output"] for s in steps), "sentences": steps}
