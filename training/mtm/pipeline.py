"""Toàn bộ hệ thống lúc dịch (slide 3.0), từ trên xuống:

  ① tiền xử lý → ② Transformer dịch nháp ŷ + C → ③ cổng (C, R)
       câu dễ  : y = ŷ                                  (không tra kho)
       câu khó : ④ LaBSE + FAISS Top-5 → ⑤ rerank → e* (điểm < δ: giữ ŷ) → ⑥ cùng Transformer(x + e*) → y = ŷ′

mode = "full"   : có cổng (hệ thống đề xuất)
       "draft"  : chỉ ①②  (giống mô hình A/B)
       "always" : tra kho cho mọi câu (ablation: bỏ cổng)
"""
from .gate import Rarity, is_hard
from .modeling import generate
from .utils import nfc


class Translator:
    def __init__(self, model, tok, cfg, memory=None, encoder=None, reranker=None, rarity=None, thresholds=None):
        self.model, self.tok, self.cfg = model, tok, cfg
        self.memory, self.encoder, self.reranker, self.rarity = memory, encoder, reranker, rarity
        self.th = thresholds or {}
        d = cfg["decode"]
        self.gen_kw = dict(num_beams=d["num_beams"], max_new_tokens=d["max_new_tokens"], batch_size=d["batch_size"],
                           max_src=cfg["model"]["max_src_len"], max_ev=cfg["model"]["max_evidence_len"])

    def thresholds(self, lang):
        return self.th.get(lang) or self.th.get("vi") or {"tau": 0.0, "rho": 1.01, "delta": 1.01}

    def translate(self, texts, langs, mode="full"):
        texts = [nfc(t) for t in texts]                                              # ①
        drafts, C = generate(self.model, self.tok, texts, langs, **self.gen_kw)     # ②
        res = []
        for x, l, y, c in zip(texts, langs, drafts, C):
            R = self.rarity.score(x, l) if self.rarity else 0.0
            th = self.thresholds(l)
            hard = True if mode == "always" else (False if mode == "draft" else is_hard(c, R, th["tau"], th["rho"]))  # ③
            res.append({"src": x, "lang": l, "draft": y, "C": c, "R": R, "hard": hard, "candidates": [],
                        "evidence": None, "ev_score": None, "refined": None, "output": y})
        hard_idx = [i for i, r in enumerate(res) if r["hard"]]
        if not hard_idx or self.memory is None:
            return res
        q = [res[i]["src"] for i in hard_idx]
        hits = self.memory.search(self.encoder.encode(q), k=self.cfg["retrieval"]["top_k"])           # ④
        picks = self.reranker.pick(q, [[self.memory.en[j] for j, _ in h] for h in hits])               # ⑤
        ref_i, ref_ev = [], []
        for i, h, (j, s, allsc) in zip(hard_idx, hits, picks):
            r = res[i]
            r["candidates"] = [{"fr": self.memory.fr[m], "en": self.memory.en[m], "cosine": cs, "rerank": rs}
                               for (m, cs), rs in zip(h, allsc)]
            if j is None:
                continue
            r["evidence"], r["ev_score"] = self.memory.en[h[j][0]], s
            if mode == "always" or s >= self.thresholds(r["lang"])["delta"]:
                ref_i.append(i); ref_ev.append(r["evidence"])
        if ref_i:
            refined, _ = generate(self.model, self.tok, [res[i]["src"] for i in ref_i],                # ⑥
                                  [res[i]["lang"] for i in ref_i], evidences=ref_ev, with_confidence=False, **self.gen_kw)
            for i, y2 in zip(ref_i, refined):
                res[i]["refined"] = y2
                res[i]["output"] = y2
        return res


def load_system(cfg, ckpt_dir, with_retrieval=True):
    """Nạp mô hình C cùng kho, bộ mã hóa, reranker, từ điển độ hiếm và ngưỡng đã chọn trên dev."""
    import os
    from .modeling import load_checkpoint
    from .utils import load_json, read_jsonl, wpath
    model, tok, meta = load_checkpoint(cfg, ckpt_dir)
    memory = encoder = reranker = None
    if with_retrieval:
        from .memory import FrEnMemory, LabseEncoder
        from .rerank import Reranker
        memory = FrEnMemory.load(wpath(cfg, "memory"))
        encoder, reranker = LabseEncoder(cfg["retrieval"]["encoder"]), Reranker(cfg["retrieval"]["reranker"])
    rp = wpath(cfg, "results", "rarity.json")
    if os.path.exists(rp):
        rarity = Rarity.from_json(load_json(rp))
    else:
        rows = read_jsonl(wpath(cfg, "data", "vi_en_train.jsonl")) + read_jsonl(wpath(cfg, "data", "fr_en_train.jsonl"))
        rarity = Rarity.build(rows, cfg["gate"]["rare_freq"])
    tp = os.path.join(ckpt_dir, "..", "thresholds.json")
    th = load_json(tp) if os.path.exists(tp) else None
    return Translator(model, tok, cfg, memory, encoder, reranker, rarity, th), meta
