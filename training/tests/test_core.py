"""Kiểm tra các phần logic không cần GPU. Chạy:  python tests/test_core.py"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtm.data import basic_clean, drop_overlap, rows
from mtm.gate import Rarity, auroc_bad_draft, is_hard, select_thresholds
from mtm.sampling import example_weights, fixed_probs, temperature_probs
from mtm.utils import nfc, norm_key


def test_temperature_matches_slide():
    sizes = {"vi": 15000, "fr": 100000}
    got = {T: round(temperature_probs(sizes, T)["vi"] * 100) for T in (1, 2, 5)}
    assert got == {1: 13, 2: 28, 5: 41}, got  # đúng số trên slide 3.4


def test_example_weights_give_language_share():
    langs = ["vi"] * 3 + ["fr"] * 7
    probs = {"vi": 0.4, "fr": 0.6}
    w = example_weights(langs, probs)
    assert abs(sum(w[:3]) - 0.4) < 1e-9 and abs(sum(w[3:]) - 0.6) < 1e-9
    fp = fixed_probs({"vi": 1, "fr": 9}, 0.8)
    assert abs(fp["vi"] - 0.8) < 1e-9 and abs(fp["fr"] - 0.2) < 1e-9, fp
    assert fixed_probs({"vi": 5}, 0.8) == {"vi": 1.0}


def test_cleaning_steps():
    decomposed = "Tiếng Việt"  # dấu tách rời → NFC gộp lại
    assert nfc(decomposed) == "Tiếng Việt"
    pairs = [("Xin chào bạn.", "Hello friend."), ("Xin  chào bạn.", "Hello friend."),  # trùng sau chuẩn hóa
             ("", "Empty"), ("Ngắn", "This sentence is far far far too long to be a translation of it"),
             ("Tôi là sinh viên.", "I am a student.")]
    clean = basic_clean(pairs)
    assert clean == [("Xin chào bạn.", "Hello friend."), ("Tôi là sinh viên.", "I am a student.")], clean
    kept = drop_overlap(clean, banned_tgt={norm_key("I am a student")})
    assert kept == [("Xin chào bạn.", "Hello friend.")]
    r = rows(clean, "fr", "train", with_mem_id=True)
    assert r[1]["mem_id"] == 1 and r[1]["id"] == "fr-train-000001"


def test_rarity_per_language():
    train = [{"lang": "fr", "src": "le modèle apprend"}] * 3 + [{"lang": "fr", "src": "rare"}]
    rar = Rarity.build(train, k=3)
    assert rar.score("le modèle", "fr") == 0.0
    assert rar.score("le rare", "fr") == 0.5          # 'rare' xuất hiện 1 lần < 3
    assert rar.score("le modèle", "vi") == 1.0        # tiếng Việt có từ điển riêng
    assert Rarity.from_json(rar.to_json()).score("le rare", "fr") == 0.5


def test_gate_and_threshold_selection():
    assert is_hard(0.5, 0.0, 0.6, 0.2) and is_hard(0.9, 0.3, 0.6, 0.2) and not is_hard(0.9, 0.1, 0.6, 0.2)
    # 4 câu: 2 câu khó (C thấp) được lợi khi dịch lại; 2 câu dễ bị hại nếu dịch lại
    C, R = [0.4, 0.5, 0.9, 0.95], [0.0, 0.0, 0.0, 0.0]
    ev = [0.9, 0.8, 0.9, 0.9]
    ud, ur = [30, 30, 80, 80], [60, 50, 70, 70]
    th = select_thresholds(C, R, ev, ud, ur, taus=[0.0, 0.6, 1.0], rhos=[1.01], deltas=[0.0, 0.95])
    assert th["tau"] == 0.6 and th["hard_frac"] == 0.5 and abs(th["dev_util"] - 67.5) < 1e-9, th
    assert th["always_util"] == 62.5 and th["draft_only_util"] == 55.0


def test_auroc():
    u = [10, 20, 80, 90, 85, 95, 70, 88]
    conf = [0.3, 0.4, 0.9, 0.9, 0.85, 0.95, 0.8, 0.9]
    assert abs(auroc_bad_draft(u, [1 - c for c in conf]) - 1.0) < 1e-9


def test_pipeline_branching():
    """Kiểm tra luồng ①→⑥ bằng mô hình giả (không cần torch)."""
    calls = []

    def fake_generate(model, tok, texts, langs, evidences=None, with_confidence=True, **kw):
        calls.append(("refine" if evidences else "draft", list(texts)))
        if evidences:
            return [f"REFINED({t}|{e})" for t, e in zip(texts, evidences)], [None] * len(texts)
        conf = [0.95 if "dễ" in t else 0.3 for t in texts]
        return [f"DRAFT({t})" for t in texts], conf

    sys.modules["mtm.modeling"] = types.SimpleNamespace(generate=fake_generate)
    sys.modules.pop("mtm.pipeline", None)
    from mtm.pipeline import Translator
    from mtm.evaluate import apply_gate

    class Mem:
        fr, en = ["fr0", "fr1", "fr2"], ["en0", "en1", "en2"]
        def search(self, emb, k=5, exclude=None):
            return [[(1, 0.9), (2, 0.8)] for _ in emb]

    class Enc:
        def encode(self, texts):
            return [[0.0]] * len(texts)

    class RR:
        def pick(self, qs, cands):
            return [(0, 0.7, [0.7, 0.2]) if "khó" in q else (0, 0.1, [0.1, 0.05]) for q in qs]

    cfg = {"decode": {"num_beams": 5, "max_new_tokens": 10, "batch_size": 4},
           "model": {"max_src_len": 128, "max_evidence_len": 96}, "retrieval": {"top_k": 5}}
    th = {"vi": {"tau": 0.6, "rho": 1.01, "delta": 0.5}}
    tr = Translator(None, None, cfg, Mem(), Enc(), RR(), Rarity({}, 3), th)
    texts = ["câu dễ", "câu khó", "câu hiếm"]  # 'câu hiếm': C thấp nhưng bằng chứng yếu (< δ)
    res = tr.translate(texts, ["vi"] * 3, mode="full")
    assert [r["hard"] for r in res] == [False, True, True]
    assert res[0]["output"] == "DRAFT(câu dễ)" and res[0]["candidates"] == []      # câu dễ không tra kho
    assert res[1]["output"] == "REFINED(câu khó|en1)"                               # e* = câu Anh, không phải câu Pháp
    assert res[2]["output"] == "DRAFT(câu hiếm)" and res[2]["evidence"] == "en1"    # < δ → giữ bản nháp
    assert calls[-1] == ("refine", ["câu khó"])
    # chế độ always: mọi câu đều tra kho; apply_gate tái tạo đúng kết quả chế độ full
    res_all = tr.translate(texts, ["vi"] * 3, mode="always")
    out, hard, used = apply_gate(res_all, th["vi"])
    assert out == [r["output"] for r in res] and hard == [False, True, True] and used == [False, True, False]
    del sys.modules["mtm.modeling"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"Tất cả {len(tests)} kiểm tra đều đạt.")
