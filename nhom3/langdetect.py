"""Nhận diện nhanh câu là tiếng Việt, tiếng Pháp hay tiếng Anh (đủ cho ô "Tự động phát hiện").

Dựa vào chữ cái đặc trưng và các từ rất thường gặp của mỗi ngôn ngữ — không cần thư viện ngoài.
"""
import re
import unicodedata

# chữ chỉ tiếng Việt có (tiếng Pháp không dùng): ă đ ơ ư và các nguyên âm có dấu nặng/hỏi/ngã
_VI_ONLY = set("ăđơư" "ạảãặẳẵậẩẫẹẻẽệểễịỉĩọỏõộổỗợởỡụủũựửữỵỷỹ" "ằắầấềếồốờớừứỳ")
_FR_ONLY = set("çœæëïüÿîû")
_VI_WORDS = {"tôi", "bạn", "là", "và", "của", "không", "có", "những", "các", "một", "được", "cho", "này", "người",
             "với", "đã", "sẽ", "rất", "chúng", "em", "anh", "chị", "nó", "đây", "đó", "thì", "mà", "gì", "nào",
             "toi", "ban", "khong", "nhung", "duoc", "nguoi", "voi", "chung"}
_FR_WORDS = {"le", "les", "des", "une", "un", "est", "et", "je", "vous", "nous", "pas", "pour", "dans", "que", "qui",
             "sur", "avec", "ce", "cette", "il", "elle", "du", "au", "aux", "ne", "mon", "ma", "mes", "très", "merci",
             "bonjour", "suis", "sont", "être", "avoir", "c'est", "j'ai", "mais", "où", "notre", "votre", "leur",
             "la", "avons", "avez", "ont", "sommes", "êtes", "fait", "plus", "comme", "tout", "bien", "aussi", "en",
             "ou", "donc", "car", "peu", "alors", "sans", "chez", "ces", "ses", "son", "sa", "leurs", "nos", "vos",
             "quand", "comment", "pourquoi", "parce", "ici", "oui", "non", "beaucoup", "aujourd'hui", "faire"}
_FR_ELISION = re.compile(r"^(?:l|d|j|qu|c|n|s|m|t)'[^\W\d_]+$", re.UNICODE)
_EN_WORDS = {"the", "is", "are", "and", "you", "i", "to", "of", "in", "it", "that", "this", "what", "with", "for",
             "not", "be", "have", "my", "we", "they", "was", "were", "am", "do", "does", "can", "will", "hello"}
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def detect(text):
    """Trả về (mã ngôn ngữ, độ chắc chắn 0..1). Mã: 'vi' | 'fr' | 'en' | 'unknown'."""
    t = unicodedata.normalize("NFC", text or "").lower()
    toks = _TOKEN.findall(t)
    if not toks:
        return "unknown", 0.0
    score = {
        "vi": 3 * sum(ch in _VI_ONLY for ch in t) + 2 * sum(w in _VI_WORDS for w in toks),
        "fr": 3 * sum(ch in _FR_ONLY for ch in t) + 2 * sum(w in _FR_WORDS or bool(_FR_ELISION.match(w)) for w in toks),
        "en": 2 * sum(w in _EN_WORDS for w in toks),
    }
    # dấu thanh kiểu Việt (huyền/sắc trên a, e, o…) có cả ở tiếng Pháp → chỉ cộng nhẹ cho tiếng Việt
    score["vi"] += sum(ch in "àáèéìíòóùúâêô" for ch in t)
    lang = max(score, key=lambda l: (score[l], l == "vi"))  # hòa điểm → ưu tiên tiếng Việt
    total = sum(score.values())
    if score[lang] == 0:
        return "unknown", 0.0
    return lang, round(score[lang] / total, 2)
