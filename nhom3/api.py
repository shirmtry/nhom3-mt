"""Các API của app (Flask Blueprint)."""
import os
import re
import threading
import time

from .engine import Translator, load_json
from .langdetect import detect

LANG_NAME = {"vi": "tiếng Việt", "fr": "tiếng Pháp", "en": "tiếng Anh", "unknown": "không rõ"}


def _norm_tokens(s):
    """Chuẩn hóa để so câu Anh: chữ thường, gộp "do n't" → "don't", bỏ dấu câu."""
    s = re.sub(r"\s+n't\b", "n't", (s or "").lower())
    s = re.sub(r"\s+'(s|re|ve|ll|d|m)\b", r"'\1", s)
    return re.findall(r"[a-z0-9']+", s)


def near_duplicate(a, b, thr=0.8):
    """Hai câu Anh gần trùng nhau (Jaccard trên tập từ ≥ thr) — dùng để đánh dấu bằng chứng ≈ đáp án."""
    x, y = set(_norm_tokens(a)), set(_norm_tokens(b))
    if not x or not y:
        return False
    return len(x & y) / len(x | y) >= thr


def create_blueprint(system=None):
    from flask import Blueprint, jsonify, request

    bp = Blueprint("nhom3", __name__)
    sysm = system or Translator()

    @bp.route("/api/status")
    def status():
        return jsonify(sysm.status())

    @bp.route("/api/warmup", methods=["POST"])
    def warmup():
        def run():
            try:
                sysm.ensure_core()
                sysm.ensure_retrieval()
            except Exception as e:  # noqa: BLE001
                sysm.state = f"lỗi khi nạp: {e}"
        if sysm.available() and sysm.reranker is None and not sysm.state.startswith("đang"):
            sysm.state = "đang nạp…"
            threading.Thread(target=run, daemon=True).start()
        return jsonify(sysm.status())

    @bp.route("/api/translate", methods=["POST"])
    def translate():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        source = (data.get("source") or "auto").lower()
        mode = data.get("mode") if data.get("mode") in ("full", "draft", "always") else "full"
        if not text:
            return jsonify({"error": "Hãy nhập câu cần dịch"}), 400
        if len(text) > 2000:
            return jsonify({"error": "Văn bản quá dài (tối đa 2000 ký tự)"}), 400
        detected = None
        if source not in ("vi", "fr"):
            lang, conf = detect(text)
            detected = {"lang": lang, "confidence": conf}
            if lang not in ("vi", "fr"):
                return jsonify({"error": f"Mô hình chỉ dịch tiếng Việt hoặc tiếng Pháp sang tiếng Anh "
                                         f"(phát hiện: {LANG_NAME.get(lang, lang)}). "
                                         f"Nếu phát hiện sai, hãy chọn ngôn ngữ nguồn bằng tay."}), 400
            source = lang
        if not sysm.available():
            return jsonify({"error": "Chưa có mô hình: chạy  python tai_mo_hinh.py  (xem README)"}), 503
        t0 = time.perf_counter()
        try:
            out = sysm.translate(text, source, mode)
        except Exception as e:  # noqa: BLE001
            print(f"[lỗi] {e}")
            return jsonify({"error": f"Lỗi mô hình: {e}"}), 500
        out.update({"mode": mode, "original": text, "source": source, "detected": detected,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000)})
        print(f"[{mode}] {source}→en | {text[:50]!r} → {out['translation'][:50]!r} "
              f"| khó: {sum(s['hard'] for s in out['sentences'])}/{len(out['sentences'])}")
        return jsonify(out), 200

    @bp.route("/api/results")
    def results():
        rd = os.path.join(sysm.folder, "results")
        exps = [r for r in (load_json(os.path.join(rd, f"{run}_test.json")) for run in ("A0", "A", "B", "C")) if r]
        examples = load_json(os.path.join(rd, "examples.json"), []) or []
        seen, uniq = set(), []
        for e in examples:  # bỏ câu lặp, đánh dấu bằng chứng gần trùng đáp án (rò rỉ PhoMT ↔ TED2020)
            if e.get("src") in seen:
                continue
            seen.add(e.get("src"))
            e["leak"] = near_duplicate(e.get("evidence"), e.get("ref"))
            uniq.append(e)
        return jsonify({"available": bool(exps), "experiments": exps,
                        "retrieval": load_json(os.path.join(rd, "retrieval.json")),
                        "calibration": load_json(os.path.join(rd, "C_dev_calibration.json")),
                        "examples": uniq, "meta": sysm.meta})

    return bp
