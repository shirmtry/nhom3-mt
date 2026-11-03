"""Kiểm tra app bằng mô hình giả (không cần torch, không cần tải mô hình).

    python tests/test_app.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nhom3.api import near_duplicate  # noqa: E402
from nhom3.engine import NEVER_HARD, Translator, split_sentences  # noqa: E402
from nhom3.langdetect import detect  # noqa: E402


def make_translator(folder=None):
    d = folder or tempfile.mkdtemp()
    json.dump({"vi": {"tau": 0.7, "rho": 0.2, "delta": 0.5}}, open(os.path.join(d, "thresholds.json"), "w"))
    json.dump({"top_k": 3}, open(os.path.join(d, "meta.json"), "w"))
    s = Translator(d)

    class Tok:
        def convert_ids_to_tokens(self, ids):
            return ["vi_VN"] + [f"▁t{i}" for i in ids[1:]]
    s.tok, s.model = Tok(), object()
    s.ensure_core = lambda: None
    s.ensure_retrieval = lambda: None
    s.available = lambda: True
    s.mem_fr, s.mem_en = ["fr0", "fr1", "fr2", "fr3"], ["en0", "en1", "en2", "en3"]
    s.encode = lambda text, lang, evidence=None: [0, 1, 2] + ([9] if evidence else [])
    s.generate = lambda ids, with_conf=True: (("REFINED" if 9 in ids else "DRAFT"), (None if not with_conf else s._C))
    s.rarity = lambda text, lang: (0.0, [], ["bàn phím", "máy tính"])
    s.retrieve = lambda text, k: [(2, 0.9), (1, 0.8), (3, 0.7)][:k]
    s.rerank = lambda q, cands: s._rr
    s._C, s._rr = 0.95, [0.1, 0.1, 0.1]
    return s


# ---------------------------------------------------------------- bộ dịch
def test_easy_sentence_does_not_touch_memory():
    s = make_translator(); s._C, s._rr = 0.95, [0.9, 0.1, 0.1]
    r = s.translate_sentence("Tôi là sinh viên.", "vi")
    assert not r["hard"] and r["output"] == "DRAFT" and r["candidates"] == [] and "retrieve" not in r["timings_ms"]


def test_hard_sentence_uses_evidence():
    s = make_translator(); s._C, s._rr = 0.4, [0.3, 0.9, 0.2]
    r = s.translate_sentence("câu khó", "vi")
    assert r["hard"] and r["reasons"]["C_low"] and r["evidence"] == "en1" and r["output"] == "REFINED"
    assert [c["chosen"] for c in r["candidates"]] == [False, True, False]


def test_weak_evidence_keeps_draft():
    s = make_translator(); s._C, s._rr = 0.4, [0.3, 0.2, 0.1]
    r = s.translate_sentence("câu khó", "vi")
    assert r["hard"] and not r["evidence_used"] and r["output"] == "DRAFT"


def test_modes():
    s = make_translator(); s._C = 0.95
    assert s.translate_sentence("x", "fr", mode="always")["output"] == "REFINED"
    s._C = 0.1
    assert s.translate_sentence("x", "fr", mode="draft")["output"] == "DRAFT"
    assert s.th("fr") == s.th("vi")


def test_sentence_split():
    assert split_sentences("Câu một.  Câu hai! Câu ba?") == ["Câu một.", "Câu hai!", "Câu ba?"]
    assert len(split_sentences(" ".join(f"Câu {i}." for i in range(12)))) == 8


def test_unavailable_without_model():
    s = Translator(tempfile.mkdtemp())
    assert not s.available() and s.th("vi") == NEVER_HARD
    try:
        s.ensure_core()
        raise AssertionError("phải báo lỗi")
    except RuntimeError as e:
        assert "tai_mo_hinh.py" in str(e)


# ---------------------------------------------------------------- nhận diện ngôn ngữ, phát hiện gần trùng
def test_language_detection():
    cases = {"Tôi yêu gia đình của tôi.": "vi", "Hãy làm theo lời cha nhé.": "vi", "Cậu không khoẻ à?": "vi",
             "Je suis étudiant en intelligence artificielle.": "fr", "Nous avons mangé à la maison.": "fr",
             "L'homme est mortel.": "fr", "I am a student.": "en", "123 !!!": "unknown"}
    for text, lang in cases.items():
        assert detect(text)[0] == lang, (text, detect(text))


def test_near_duplicate():
    assert near_duplicate("Would n't it be awesome if we all had BLIS on our phones?",
                          "Wouldn't it be awesome if we all had BLIS on our phones?")
    assert not near_duplicate("He's absolutely right.", "I apologize, you are absolutely right.")


# ---------------------------------------------------------------- Flask
def test_flask_routes():
    try:
        import flask  # noqa: F401
    except ImportError:
        print("  (bỏ qua: chưa cài Flask)")
        return
    from app import create_app
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "results"))
    json.dump({"run": "A", "results": {"vi/draft": {"BLEU": 27.0, "chrF": 49.4}}}, open(os.path.join(d, "results", "A_test.json"), "w"))
    json.dump([{"src": "a", "evidence": "Wouldn't it", "ref": "Would n't it", "gain": 5},
               {"src": "a", "evidence": "x", "ref": "y", "gain": 1}], open(os.path.join(d, "results", "examples.json"), "w"))
    s = make_translator(d)
    c = create_app(s).test_client()
    assert c.get("/").status_code == 200 and "Nhóm 3" in c.get("/").get_data(as_text=True)
    assert c.get("/api/status").get_json()["calibrated"]
    r = c.post("/api/translate", json={"text": "Tôi là sinh viên. Câu hai.", "source": "auto", "mode": "full"})
    j = r.get_json()
    assert r.status_code == 200 and j["source"] == "vi" and j["detected"]["lang"] == "vi" and len(j["sentences"]) == 2
    r = c.post("/api/translate", json={"text": "I am a student.", "source": "auto"})
    assert r.status_code == 400 and "tiếng Anh" in r.get_json()["error"]
    assert c.post("/api/translate", json={"text": ""}).status_code == 400
    res = c.get("/api/results").get_json()
    assert res["available"] and len(res["examples"]) == 1 and res["examples"][0]["leak"] is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"Tất cả {len(tests)} kiểm tra đều đạt.")
