"""Chạy thử các bước không cần GPU với thành phần giả: dữ liệu (nhiều kiểu thư mục PhoMT), tạo mẫu bằng chứng,
bảng tổng hợp, chọn ví dụ khi xuất, và logic của notebook (chạy mới / chạy tiếp / chạy nền).
Chạy:  python tests/test_offline_full.py
"""
import json
import os
import random
import shutil
import sys
import tempfile
import types
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

from mtm import data as D
from mtm.utils import norm_key, read_jsonl, save_json


# ---------------------------------------------------------------- thành phần giả
class FakeTok:
    """Giả MBart50TokenizerFast: 1 từ = 1 token."""
    src_lang = "vi_VN"

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __call__(self, texts, add_special_tokens=True, **k):
        extra = 2 if add_special_tokens else 0
        return {"input_ids": [list(range(len(t.split()) + extra)) for t in texts]}


sys.modules["transformers"] = types.SimpleNamespace(MBart50TokenizerFast=FakeTok)

rng = random.Random(0)
WORDS = "dữ liệu mô hình dịch máy tiếng việt câu khó hiếm thường ngày học sinh trường lớp".split()
EWORDS = "data model machine translation vietnamese sentence hard rare usual day student school class".split()


def vi_sent(i):
    return " ".join(rng.choice(WORDS) for _ in range(6 + i % 7)) + f" số {i}."


def en_sent(i):
    return " ".join(rng.choice(EWORDS) for _ in range(6 + i % 7)) + f" number {i}."


def write_split(folder, split, pairs, naming="plain"):
    os.makedirs(folder, exist_ok=True)
    for lang, idx in (("vi", 0), ("en", 1)):
        name = {"plain": f"{split}.{lang}", "txt": f"{split}.{lang}.txt"}[naming]
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write("\n".join(p[idx] for p in pairs) + "\n")


def make_phomt(base, variant):
    train = [(vi_sent(i), en_sent(i)) for i in range(6000)]
    dev = [(vi_sent(10000 + i), en_sent(10000 + i)) for i in range(400)]
    test = [(vi_sent(20000 + i), en_sent(20000 + i)) for i in range(400)]
    train += [("Ngắn", "This one has a very very very long and strange english side for its tiny source")] * 5  # lệch độ dài
    train += [train[0]] * 3                                                                                        # trùng
    train += [(dev[0][0], "leaked dev source sentence number x")]                                                  # câu nguồn trùng dev
    if variant == "detok":
        for sp, p in (("train", train), ("dev", dev), ("test", test)):
            write_split(os.path.join(base, "PhoMT", "detokenization", sp), sp, p)
            write_split(os.path.join(base, "PhoMT", "tokenization", sp), sp, [(a + " _tok", b + " _tok") for a, b in p])
    elif variant == "zip":
        tmp = tempfile.mkdtemp()
        for sp, p in (("train", train), ("dev", dev), ("test", test)):
            write_split(os.path.join(tmp, "PhoMT", "detokenization", sp), sp, p)
        os.makedirs(os.path.join(base, "PhoMT"), exist_ok=True)
        with zipfile.ZipFile(os.path.join(base, "PhoMT", "PhoMT.zip"), "w") as z:
            for dp, _, fs in os.walk(tmp):
                for f in fs:
                    z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), tmp))
    elif variant == "txt":
        for sp, p in (("train", train), ("dev", dev), ("test", test)):
            write_split(os.path.join(base, "PhoMT", sp), sp, p, naming="txt")
    return train, dev, test


def fake_opus_factory(dev_en, train_en):
    def fake_opus(corpus, version, src, tgt, raw_dir):
        os.makedirs(os.path.join(raw_dir, corpus), exist_ok=True)
        out = [(f"phrase française {corpus} {i} " + " ".join(rng.choice(WORDS) for _ in range(5)), en_sent(50000 + i))
               for i in range(3000)]
        fr_long = lambda tag: f"{tag} " + " ".join(rng.choice(WORDS) for _ in range(8))
        out += [(fr_long(f"fuite {j}"), e) for j, e in enumerate(dev_en)]                  # câu Anh trùng kho dev/test gốc
        out += [(fr_long(f"triple {i}"), train_en[i]) for i in range(100, 400)]            # câu Anh trùng PhoMT train → Recall@5
        return out
    return fake_opus


def run_prepare(variant, broken_nc=False):
    work = tempfile.mkdtemp()
    cfg = {"workdir": work, "seed": 1, "model": {"base": "x"},
           "data": {"vi_source": "phomt", "phomt_dir": None, "vi_train_size": 1500, "vi_dev_size": 200, "vi_test_size": 200,
                    "fr_sources": [{"corpus": "TED2020", "version": "v1", "share": 0.6},
                                   {"corpus": "News-Commentary", "version": "v16", "share": 0.4}],
                    "fr_train_size": 3000, "fr_dev_size": 100, "fr_test_size": 200, "max_tokens": 128,
                    "retrieval_eval_size": 50, "keep_raw": False}}
    holder = {}

    def fake_fetch(raw_dir, token=None):
        holder["data"] = make_phomt(raw_dir, variant)
        return os.path.join(raw_dir, "PhoMT")

    train, dev, test = make_phomt(tempfile.mkdtemp(), "detok")  # để biết câu dev (cùng seed sinh khác nhau → dùng holder)
    D.fetch_phomt = fake_fetch
    base_opus = lambda *a: fake_opus_factory([p[1] for p in holder["data"][1] + holder["data"][2]], [p[1] for p in holder["data"][0]])(*a)
    if broken_nc:
        def base_opus2(corpus, version, *a):
            if corpus == "News-Commentary":
                raise ConnectionError(f"404 {version}")
            return base_opus(corpus, version, *a)
        D.opus_pairs = base_opus2
    else:
        D.opus_pairs = base_opus
    stats = D.prepare(cfg)
    return work, stats, holder["data"]


def check_prepare(variant, broken_nc=False):
    if broken_nc:
        good = D.opus_pairs
        def flaky(corpus, version, *a):
            if corpus == "News-Commentary":
                raise ConnectionError(f"404 {version}")
            return good(corpus, version, *a)
        D.opus_pairs = flaky
    work, stats, (train, dev, test) = run_prepare(variant, broken_nc)
    d = os.path.join(work, "data")
    rows = {n: read_jsonl(os.path.join(d, n + ".jsonl")) for n in
            ("vi_en_train", "vi_en_dev", "vi_en_test", "fr_en_train", "fr_en_dev", "fr_en_test", "retrieval_eval")}
    assert len(rows["vi_en_train"]) == 1500 and len(rows["vi_en_dev"]) == 200 and len(rows["vi_en_test"]) == 200, stats
    assert len(rows["fr_en_dev"]) == 100 and len(rows["fr_en_test"]) == 200
    assert 2900 <= len(rows["fr_en_train"]) <= 3000, stats
    assert all(not r["src"].endswith("_tok") for r in rows["vi_en_train"]), "phải ưu tiên bản detokenization"
    held_en = {norm_key(r["tgt"]) for r in rows["vi_en_dev"] + rows["vi_en_test"]}
    held_src = {norm_key(r["src"]) for r in rows["vi_en_dev"] + rows["vi_en_test"]}
    assert not any(norm_key(r["tgt"]) in held_en for r in rows["fr_en_train"] + rows["fr_en_dev"] + rows["fr_en_test"]), "rò rỉ đáp án dev/test vào kho"
    fuite = [r for r in rows["fr_en_train"] + rows["fr_en_dev"] + rows["fr_en_test"] if r["src"].startswith("fuite")]
    assert fuite and len(fuite) <= 800 - 400, "câu Anh của dev/test đã chọn phải bị loại khỏi Pháp–Anh"
    assert len(rows["retrieval_eval"]) > 0, "phải có câu để đo Recall@5"
    assert not any(norm_key(r["src"]) in held_src for r in rows["vi_en_train"]), "câu nguồn dev/test lọt vào train"
    keys = [(norm_key(r["src"]), norm_key(r["tgt"])) for r in rows["vi_en_train"]]
    assert len(keys) == len(set(keys)), "còn cặp trùng"
    assert all(r["mem_id"] == i for i, r in enumerate(rows["fr_en_train"])), "mem_id phải khớp vị trí trong kho"
    fr_tr = {norm_key(r["src"]) for r in rows["fr_en_train"]}
    assert not any(norm_key(r["src"]) in fr_tr for r in rows["fr_en_test"]), "câu test Pháp lọt vào kho"
    vi_tr = {norm_key(r["src"]) for r in rows["vi_en_train"]}
    for r in rows["retrieval_eval"]:
        g = rows["fr_en_train"][r["gold_mem_ids"][0]]
        assert norm_key(g["tgt"]) == norm_key(r["tgt"]) and norm_key(r["src"]) not in vi_tr
    assert not os.path.exists(os.path.join(work, "raw")), "dữ liệu thô phải được xóa"
    return stats


# ---------------------------------------------------------------- tạo mẫu bằng chứng
def check_augment():
    from mtm.augment import build_augmented
    from mtm.memory import FrEnMemory

    class Idx:
        def __init__(self, n):
            self.n = n
        def search(self, q, k):
            I = np.array([[(int(abs(v[0])) + j) % self.n for j in range(k)] for v in q])
            return np.ones_like(I, dtype="float32") * 0.8, I

    n = 50
    mem = FrEnMemory([f"fr{i}" for i in range(n)], [f"en {i}" for i in range(n)], Idx(n))
    rows = [{"lang": "fr", "src": f"fr{i}", "tgt": f"en {i}", "mem_id": i} for i in range(n)]
    rows += [{"lang": "vi", "src": f"vi{i}", "tgt": f"en {i}"} for i in range(n)]       # câu Anh trùng kho

    class Enc:
        def encode(self, texts, progress=False):
            return np.array([[float(t[2:])] for t in texts], dtype="float32")

    class RR:
        def pick(self, qs, cands):
            return [(0, 0.2 if q.endswith("7") else 0.9, [0.9] * len(c)) for q, c in zip(qs, cands)]

    cfg = {"augment": {"p_plain": 0.5, "p_correct": 0.35, "p_random": 0.15, "min_rerank_score": 0.3}, "retrieval": {"top_k": 5}}
    cache = os.path.join(tempfile.mkdtemp(), "aug.jsonl")
    out = build_augmented(rows, mem, Enc(), RR(), cfg, cache, seed=3)
    for r in out:
        if r["evidence"] is not None:
            assert r["evidence"] != r["tgt"], f"bằng chứng trùng đáp án: {r}"
    modes = [r["ev_mode"] for r in out]
    assert modes.count("correct") > 0 and modes.count("random") > 0 and modes.count("plain") > 0
    assert all(r["evidence"] is None for r in out if r["ev_mode"] == "plain")
    low = [r for r in out if r["src"].endswith("7") and r["ev_mode"] != "random"]
    assert all(r["ev_mode"] == "plain" for r in low), "điểm rerank thấp phải thành mẫu thường"
    again = build_augmented(rows, mem, None, None, cfg, cache, seed=3)  # đọc lại từ cache, không gọi encoder
    assert again == read_jsonl(cache)
    return {m: modes.count(m) for m in set(modes)}


# ---------------------------------------------------------------- bảng tổng hợp + ví dụ xuất
def check_summary_and_examples():
    from mtm import evaluate as E
    work = tempfile.mkdtemp()
    cfg = {"workdir": work}
    save_json(os.path.join(work, "results", "A_test.json"), {"run": "A", "results": {"vi/draft": {"BLEU": 20.1, "chrF": 45.2}}})
    save_json(os.path.join(work, "results", "C_test.json"), {"run": "C", "results": {
        "vi/draft": {"BLEU": 21, "chrF": 46}, "vi/full": {"BLEU": 22, "chrF": 47, "hard_frac": 0.2, "evidence_frac": 0.1},
        "fr/full": {"BLEU": 30, "chrF": 55}, "vi/always": {"BLEU": 20, "chrF": 44}}})
    table = E.summary(cfg)
    assert "| C | full | 22 | 47 | – | 30 | 55 | 0.2 | 0.1 |" in table and "| A | draft | 20.1 | 45.2 |" in table

    sys.modules["sacrebleu"] = types.SimpleNamespace(
        sentence_chrf=lambda h, refs: types.SimpleNamespace(score=100.0 if h == refs[0] else 10.0))
    from mtm import export as X
    outs = [{"src": f"s{i}", "lang": "vi", "ref": f"r{i}", "draft": "x", "output": f"r{i}" if i % 2 else "y",
             "C": 0.5, "R": 0.1, "evidence": "e", "ev_score": 0.8, "evidence_used": i % 3 != 0, "candidates": []} for i in range(12)]
    from mtm.utils import write_jsonl
    write_jsonl(os.path.join(work, "results", "outputs", "C_vi_test.jsonl"), outs)
    ex = X.pick_examples(cfg, "C", n_good=3, n_bad=2)
    assert ex[0]["gain"] == 90.0 and len(ex) >= 3 and all("candidates" in e for e in ex)
    del sys.modules["sacrebleu"]
    return len(ex)


# ---------------------------------------------------------------- logic của notebook
def notebook_cells():
    nb = json.load(open(os.path.join(ROOT, "notebooks", "TMG301_Nhom3_MultilingualMT.ipynb"), encoding="utf-8"))
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def py_only(src):
    return "\n".join(("pass  # " + l) if l.lstrip().startswith("!") else l for l in src.splitlines())


def check_notebook():
    cells = notebook_cells()
    for c in cells:  # mọi ô đều là Python hợp lệ (bỏ các dòng lệnh shell)
        compile(py_only(c), "<cell>", "exec")
    cfg_cell = next(c for c in cells if "QUICK_TEST =" in c)
    tok_cell = next(c for c in cells if "HF_TOKEN" in c and "getpass" in c)
    data_cell = next(c for c in cells if "mtm.data" in c)
    train_cell = next(c for c in cells if "mtm.train" in c and not c.lstrip().startswith("#"))
    eval_cell = next(c for c in cells if "--calibrate" in c)
    export_cell = next(c for c in cells if "mtm.export" in c)
    demo_cell = cells[-1]
    results = {}
    for quick, plan in ((True, "nhanh"), (False, "nhanh"), (False, "day_du")):
        tmp = tempfile.mkdtemp()
        proj = os.path.join(tmp, "proj")
        shutil.copytree(ROOT, proj, ignore=shutil.ignore_patterns("__pycache__", "*.zip"))
        cwd = os.getcwd()
        os.chdir(proj)
        try:
            import glob, subprocess, time
            ns = {"os": os, "glob": glob, "shutil": shutil, "time": time, "subprocess": subprocess,
                  "WORKDIR": os.path.join(tmp, "work"), "ROOT": tmp, "BATCH": False}
            src = cfg_cell.replace("QUICK_TEST = True", f"QUICK_TEST = {quick}").replace("PLAN = 'nhanh'", f"PLAN = '{plan}'")
            exec(py_only(src), ns)  # chạy thật cả tests/test_core.py qua run()
            from mtm.utils import load_config
            cfg = load_config("configs/run.yaml")
            assert cfg["workdir"].endswith("_test") == quick
            if plan == "nhanh" and not quick:
                assert cfg["train"]["stage1_steps"] == 4000 and cfg["decode"]["num_beams"] == 4
            cmds = []
            ns["run"] = lambda c: cmds.append(c)
            W = ns["W"]
            os.environ["HF_TOKEN"] = "hf_dummy"
            for c in (tok_cell, data_cell, train_cell, eval_cell):
                exec(py_only(c), ns)
            fresh = list(cmds)
            assert fresh[0].startswith("python -m mtm.data") and fresh[1].startswith("python -m mtm.memory")
            assert [c.split("--exp ")[-1] for c in fresh if "mtm.train" in c] == ["A0", "A", "B", "C"]
            assert fresh.index(next(c for c in fresh if "--calibrate" in c)) < fresh.index(next(c for c in fresh if "evaluate" in c and "--exp A0" in c))
            # chạy tiếp sau khi bị ngắt: đã có dữ liệu, kho, A0, A
            for p in (("data", "vi_en_train.jsonl"), ("memory", "index.faiss"), ("checkpoints", "A0", "history.json"),
                      ("checkpoints", "A", "history.json")):
                os.makedirs(os.path.join(W, *p[:-1]), exist_ok=True)
                open(os.path.join(W, *p), "w").close()
            cmds.clear()
            for c in (data_cell, train_cell):
                exec(py_only(c), ns)
            assert [c.split("--exp ")[-1] for c in cmds] == ["B", "C"], cmds
            # bước xuất in đúng đường dẫn
            os.makedirs(os.path.join(W, "export"), exist_ok=True)
            with open(os.path.join(W, "export", "nhom3_mt.zip"), "wb") as f:
                f.write(b"0" * 1000)
            exec(py_only(export_cell), ns)
            exec(py_only(demo_cell), ns)
            # chạy nền mà thiếu token → phải báo lỗi rõ ràng, không treo
            del os.environ["HF_TOKEN"]
            shutil.rmtree(os.path.join(W, "data"))
            ns["BATCH"] = True
            try:
                exec(py_only(tok_cell), ns)
                raise AssertionError("phải báo thiếu HF_TOKEN")
            except RuntimeError as e:
                assert "HF_TOKEN" in str(e)
            results[(quick, plan)] = len(fresh)
        finally:
            os.chdir(cwd)
    return results


if __name__ == "__main__":
    for v in ("detok", "zip", "txt"):
        print("PASS dữ liệu, PhoMT kiểu", v, check_prepare(v))
    print("PASS dữ liệu, News-Commentary hỏng → lấy bù từ TED2020", check_prepare("detok", broken_nc=True))
    print("PASS tạo mẫu bằng chứng", check_augment())
    print("PASS bảng tổng hợp + ví dụ xuất", check_summary_and_examples())
    print("PASS notebook (chạy mới / chạy tiếp / chạy nền)", check_notebook())
    print("Tất cả kiểm tra offline đều đạt.")
