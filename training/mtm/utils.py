"""Tiện ích chung: cấu hình, đọc/ghi jsonl, chuẩn hóa văn bản, seed."""
import json
import os
import random
import re
import unicodedata

import numpy as np
import yaml

# Mã ngôn ngữ của mBART-50 (slide 3.2)
LANG_CODE = {"vi": "vi_VN", "fr": "fr_XX", "en": "en_XX"}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["workdir"], exist_ok=True)
    return cfg


def wpath(cfg, *parts):
    """Đường dẫn bên trong thư mục làm việc (workdir)."""
    p = os.path.join(cfg["workdir"], *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def nfc(text):
    """Chuẩn hóa Unicode NFC + gom khoảng trắng (slide 3.1, bước 1)."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_key(text):
    """Khóa so khớp trùng lặp: thường hóa, bỏ dấu câu, gom khoảng trắng."""
    t = _PUNCT.sub(" ", nfc(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
