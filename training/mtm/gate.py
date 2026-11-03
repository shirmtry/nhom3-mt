"""Bước 3.5 — Cổng quyết định.

C  = độ tin cậy của bản nháp (tính trong modeling.generate)
R  = tỉ lệ từ có tần suất < k trong dữ liệu huấn luyện CÙNG ngôn ngữ nguồn
     (câu Việt: tách từ bằng underthesea, so với 15K Việt–Anh; câu Pháp: tách theo từ, so với 100K Pháp–Anh)
Khó/hiếm nếu  C < τ  hoặc  R > ρ.  τ, ρ (và ngưỡng bằng chứng δ) chọn trên dev.
"""
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)  # chỉ lấy từ có chữ cái


def words(text, lang):
    """Tách từ để đếm tần suất. Tiếng Việt dùng underthesea nếu có (gộp 'sinh viên' thành 1 từ)."""
    text = text.lower()
    if lang == "vi":
        try:
            from underthesea import word_tokenize
            toks = word_tokenize(text)
            return [t for t in toks if _WORD.search(t)]
        except ImportError:
            pass
    return _WORD.findall(text)


class Rarity:
    def __init__(self, freq, k=3):
        self.freq, self.k = freq, k

    @classmethod
    def build(cls, rows, k=3):
        freq = {}
        for r in rows:
            freq.setdefault(r["lang"], Counter()).update(words(r["src"], r["lang"]))
        return cls(freq, k)

    def score(self, text, lang):
        ws = words(text, lang)
        if not ws:
            return 0.0
        f = self.freq.get(lang, Counter())
        return sum(1 for w in ws if f[w] < self.k) / len(ws)

    def to_json(self):
        return {"k": self.k, "freq": {l: dict(c) for l, c in self.freq.items()}}

    @classmethod
    def from_json(cls, d):
        return cls({l: Counter(c) for l, c in d["freq"].items()}, d["k"])


def is_hard(C, R, tau, rho):
    return (C < tau) or (R > rho)


def select_thresholds(C, R, ev_score, u_draft, u_refined, taus, rhos, deltas):
    """Chọn (τ, ρ, δ) trên dev để điểm trung bình của bản dịch cuối cao nhất.

    u_draft[i], u_refined[i]: điểm câu (chrF hoặc COMET) của bản nháp và bản dịch lại câu i
    ev_score[i]: điểm reranker của bằng chứng tốt nhất (dưới δ → không dùng bằng chứng)
    Hòa điểm thì ưu tiên ít câu đi nhánh khó hơn (rẻ hơn).
    """
    C, R, ev = map(np.asarray, (C, R, ev_score))
    ud, ur = np.asarray(u_draft, dtype=float), np.asarray(u_refined, dtype=float)
    best = None
    for tau in taus:
        for rho in rhos:
            hard = (C < tau) | (R > rho)
            for delta in deltas:
                use = hard & (ev >= delta)
                util = float(np.where(use, ur, ud).mean())
                key = (round(util, 6), -float(use.mean()))
                if best is None or key > best[0]:
                    best = (key, {"tau": float(tau), "rho": float(rho), "delta": float(delta),
                                  "dev_util": util, "hard_frac": float(hard.mean()),
                                  "evidence_frac": float(use.mean()),
                                  "draft_only_util": float(ud.mean()), "always_util": float(ur.mean())})
    return best[1]


def auroc_bad_draft(u_draft, signal, bad_quantile=0.25):
    """AUROC của tín hiệu trong việc phát hiện bản nháp kém (25% điểm thấp nhất)."""
    from sklearn.metrics import roc_auc_score
    u = np.asarray(u_draft, dtype=float)
    y = (u <= np.quantile(u, bad_quantile)).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(signal, dtype=float)))
