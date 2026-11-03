"""Bước 3.1 — Chuẩn bị dữ liệu.

Kết quả (trong <workdir>/data):
  vi_en_{train,dev,test}.jsonl   50K / 1K / 1K cặp Việt–Anh (PhoMT)
  fr_en_{train,dev,test}.jsonl   300K / 500 / 500 cặp Pháp–Anh (OPUS TED2020 + News Commentary)
  retrieval_eval.jsonl           câu Việt có câu Anh trùng với một cặp trong kho → đo Recall@5

Quy trình lọc (đúng 5 bước trên slide 3.1):
  1 chuẩn hóa NFC · 2 bỏ cặp trùng · 3 bỏ câu > 128 token · 4 lọc tỉ lệ độ dài · 5 khử trùng với dev/test

Chạy:  python -m mtm.data --config configs/default.yaml
"""
import argparse
import glob
import os
import random
import tarfile
import zipfile

from .utils import load_config, nfc, norm_key, set_seed, write_jsonl, wpath

OPUS_URL = "https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/{pair}.txt.zip"
OPUS_FALLBACK = {"News-Commentary": ["v16", "v14", "v11", "v9.1"], "TED2020": ["v1"]}  # thử lần lượt nếu phiên bản trong cấu hình lỗi


# ---------------------------------------------------------------- tải dữ liệu
def download(url, dest):
    import requests
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[data] tải {url}")
    if os.path.exists(dest + ".part"):
        os.remove(dest + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(dest + ".part", dest)
    return dest


def extract_archives(folder):
    """Giải nén mọi .zip/.tar.gz/.tgz trong thư mục (một lần)."""
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        done = path + ".extracted"
        if os.path.exists(done):
            continue
        out = os.path.dirname(path)
        if path.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
        elif path.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(path) as t:
                t.extractall(out)
        else:
            continue
        open(done, "w").close()


def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def opus_pairs(corpus, version, src, tgt, raw_dir):
    """Tải một corpus OPUS dạng moses; trả về danh sách (câu src, câu tgt)."""
    pair = "-".join(sorted([src, tgt]))
    folder = os.path.join(raw_dir, f"{corpus}-{version}-{pair}")
    download(OPUS_URL.format(corpus=corpus, version=version, pair=pair), os.path.join(folder, f"{pair}.txt.zip"))
    extract_archives(folder)
    fs = glob.glob(os.path.join(folder, f"*.{pair}.{src}"))
    ft = glob.glob(os.path.join(folder, f"*.{pair}.{tgt}"))
    if not fs or not ft:
        raise FileNotFoundError(f"Không thấy file .{src}/.{tgt} trong {folder}")
    a, b = read_lines(fs[0]), read_lines(ft[0])
    assert len(a) == len(b), "hai phía không cùng số dòng"
    return list(zip(a, b))


def find_phomt(phomt_dir):
    """Tìm các file {train,dev,test}.{vi,en} trong thư mục PhoMT (ưu tiên bản detokenization)."""
    extract_archives(phomt_dir)
    found = {}
    for split in ("train", "dev", "test"):
        cands = {}
        for lang in ("vi", "en"):
            hits = []
            for pat in (f"{split}.{lang}", f"{split}.{lang}.txt", f"{split}_{lang}.txt"):
                hits += glob.glob(os.path.join(phomt_dir, "**", pat), recursive=True)
            hits.sort(key=lambda p: (0 if "detok" in p.lower() else 1, len(p)))
            if not hits:
                raise FileNotFoundError(f"Không thấy {split}.{lang} trong {phomt_dir} — đặt data.phomt_dir tới thư mục chứa file")
            cands[lang] = hits[0]
        found[split] = cands
    return found


def fetch_phomt(raw_dir, token=None):
    """Tải PhoMT từ Hugging Face (vinai/PhoMT, cần đăng nhập + chấp nhận điều khoản)."""
    from huggingface_hub import snapshot_download
    dest = os.path.join(raw_dir, "PhoMT")
    snapshot_download(repo_id="vinai/PhoMT", repo_type="dataset", local_dir=dest,
                      token=token or os.environ.get("HF_TOKEN"))
    return dest


# ---------------------------------------------------------------- lọc dữ liệu
def basic_clean(pairs, max_char_ratio=3.0, max_words=120):
    """Bước 1, 2 và lọc thô theo tỉ lệ ký tự. Giữ thứ tự gốc."""
    seen, out = set(), []
    for s, t in pairs:
        s, t = nfc(s), nfc(t)
        if not s or not t:
            continue
        ls, lt = len(s), len(t)
        if max(ls, lt) / max(1, min(ls, lt)) > max_char_ratio:
            continue
        if len(s.split()) > max_words or len(t.split()) > max_words:
            continue
        key = (norm_key(s), norm_key(t))
        if key in seen:
            continue
        seen.add(key)
        out.append((s, t))
    return out


def token_filter(pairs, tok, src_lang, max_tokens=128, max_ratio=2.5, batch=2000):
    """Bước 3 và 4: bỏ câu > max_tokens token và cặp có tỉ lệ token bất thường."""
    from .utils import LANG_CODE
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        tok.src_lang = LANG_CODE[src_lang]
        ls = [len(x) for x in tok([s for s, _ in chunk], add_special_tokens=False)["input_ids"]]
        tok.src_lang = LANG_CODE["en"]
        lt = [len(x) for x in tok([t for _, t in chunk], add_special_tokens=False)["input_ids"]]
        for (s, t), a, b in zip(chunk, ls, lt):
            if a + 2 > max_tokens or b + 2 > max_tokens:
                continue
            if max(a, b) / max(1, min(a, b)) > max_ratio:
                continue
            out.append((s, t))
    return out


def drop_overlap(pairs, banned_src=(), banned_tgt=()):
    """Bước 5: bỏ cặp có câu nguồn hoặc câu Anh trùng dev/test (chống rò rỉ đáp án)."""
    bs, bt = set(banned_src), set(banned_tgt)
    return [(s, t) for s, t in pairs if norm_key(s) not in bs and norm_key(t) not in bt]


def rows(pairs, lang, split, start=0, with_mem_id=False):
    out = []
    for i, (s, t) in enumerate(pairs):
        r = {"id": f"{lang}-{split}-{start + i:06d}", "lang": lang, "src": s, "tgt": t}
        if with_mem_id:
            r["mem_id"] = start + i
        out.append(r)
    return out


# ---------------------------------------------------------------- quy trình chính
def prepare(cfg):
    from transformers import MBart50TokenizerFast

    dc, seed = cfg["data"], cfg["seed"]
    set_seed(seed)
    rng = random.Random(seed)
    raw = wpath(cfg, "raw")
    tok = MBart50TokenizerFast.from_pretrained(cfg["model"]["base"])
    mt = dc["max_tokens"]

    # ---- Việt–Anh: nguồn chính PhoMT
    if dc["vi_source"] == "phomt":
        pdir = dc.get("phomt_dir") or fetch_phomt(raw)
        files = find_phomt(pdir)
        load = lambda sp: list(zip(read_lines(files[sp]["vi"]), read_lines(files[sp]["en"])))
        vi_dev_pool, vi_test_pool = load("dev"), load("test")
        vi_train_path = files["train"]
    else:  # dự phòng khi chưa có quyền PhoMT: OPUS TED2020 en-vi, tự chia dev/test
        allp = basic_clean(opus_pairs("TED2020", "v1", "vi", "en", raw))
        rng.shuffle(allp)
        n = 5000
        vi_dev_pool, vi_test_pool = allp[:n], allp[n:2 * n]
        tmp = os.path.join(raw, "ted2020_vi_train")
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "train.vi"), "w", encoding="utf-8") as fv, \
             open(os.path.join(tmp, "train.en"), "w", encoding="utf-8") as fe:
            for s, t in allp[2 * n:]:
                fv.write(s + "\n"); fe.write(t + "\n")
        vi_train_path = {"vi": os.path.join(tmp, "train.vi"), "en": os.path.join(tmp, "train.en")}

    def take(pool, n):
        pool = basic_clean(pool)
        rng.shuffle(pool)
        return token_filter(pool[: int(n * 1.5) + 200], tok, "vi", mt)[:n]

    vi_dev, vi_test = take(vi_dev_pool, dc["vi_dev_size"]), take(vi_test_pool, dc["vi_test_size"])
    held_src = {norm_key(s) for s, _ in vi_dev + vi_test}
    held_en = {norm_key(t) for _, t in vi_dev + vi_test}

    # ---- Pháp–Anh: TED2020 + News Commentary (cùng miền với PhoMT)
    # Tải từng nguồn; nguồn nào hỏng (sai phiên bản, máy chủ lỗi) thì bỏ qua và chia lại tỉ lệ cho nguồn còn lại.
    pools = []
    for src in dc["fr_sources"]:
        versions = [src["version"]] + [v for v in OPUS_FALLBACK.get(src["corpus"], []) if v != src["version"]]
        for v in versions:
            try:
                pools.append((src, opus_pairs(src["corpus"], v, "fr", "en", raw)))
                print(f"[data] đã tải {src['corpus']} {v}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[data] không tải được {src['corpus']} {v}: {e}")
        else:
            print(f"[data] CẢNH BÁO: bỏ qua {src['corpus']}, lấy bù từ nguồn khác")
    if not pools:
        raise RuntimeError("Không tải được nguồn Pháp–Anh nào từ OPUS — thử chạy lại sau ít phút")
    total_share = sum(src["share"] for src, _ in pools)
    fr_all = []
    for src, pairs in pools:
        p = drop_overlap(basic_clean(pairs), banned_tgt=held_en)
        rng.shuffle(p)
        need = int((dc["fr_train_size"] + dc["fr_dev_size"] + dc["fr_test_size"]) * src["share"] / total_share * 1.3)
        fr_all += token_filter(p[:need], tok, "fr", mt)
        print(f"[data] {src['corpus']}: giữ {len(fr_all)} cặp (cộng dồn)")
    # khử trùng giữa hai corpus
    fr_all = basic_clean(fr_all)
    rng.shuffle(fr_all)
    nd, nt, ntr = dc["fr_dev_size"], dc["fr_test_size"], dc["fr_train_size"]
    fr_dev, fr_test = fr_all[:nd], fr_all[nd:nd + nt]
    held_fr = {norm_key(s) for s, _ in fr_dev + fr_test} | {norm_key(t) for _, t in fr_dev + fr_test}
    fr_train = [(s, t) for s, t in fr_all[nd + nt:] if norm_key(s) not in held_fr and norm_key(t) not in held_fr][:ntr]
    if len(fr_train) < ntr:
        print(f"[data] CẢNH BÁO: chỉ có {len(fr_train)} cặp Pháp–Anh (< {ntr})")

    # ---- Việt–Anh train: lấy mẫu 15K từ PhoMT train (đọc theo chỉ số để tiết kiệm RAM)
    with open(vi_train_path["vi"], encoding="utf-8", errors="replace") as f:
        total = sum(1 for _ in f)
    want = set(rng.sample(range(total), min(total, dc["vi_train_size"] * 3)))
    mem_en = {norm_key(t): i for i, (_, t) in enumerate(fr_train)}
    cand, retr = [], []
    with open(vi_train_path["vi"], encoding="utf-8", errors="replace") as fv, \
         open(vi_train_path["en"], encoding="utf-8", errors="replace") as fe:
        for i, (s, t) in enumerate(zip(fv, fe)):
            s, t = nfc(s), nfc(t)
            if i in want:
                cand.append((s, t))
            else:
                k = norm_key(t)  # câu Việt có câu Anh trùng với kho → dữ liệu đo Recall@5
                if k in mem_en and len(retr) < dc["retrieval_eval_size"] * 5 and len(k.split()) >= 5:
                    retr.append({"src": s, "tgt": t, "gold_mem_ids": [mem_en[k]]})
    cand = drop_overlap(basic_clean(cand), banned_src=held_src, banned_tgt=held_en)
    rng.shuffle(cand)
    vi_train = token_filter(cand, tok, "vi", mt)[: dc["vi_train_size"]]
    rng.shuffle(retr)
    retr = [r for r in retr if norm_key(r["src"]) not in held_src][: dc["retrieval_eval_size"]]

    d = wpath(cfg, "data")
    write_jsonl(os.path.join(d, "vi_en_train.jsonl"), rows(vi_train, "vi", "train"))
    write_jsonl(os.path.join(d, "vi_en_dev.jsonl"), rows(vi_dev, "vi", "dev"))
    write_jsonl(os.path.join(d, "vi_en_test.jsonl"), rows(vi_test, "vi", "test"))
    write_jsonl(os.path.join(d, "fr_en_train.jsonl"), rows(fr_train, "fr", "train", with_mem_id=True))
    write_jsonl(os.path.join(d, "fr_en_dev.jsonl"), rows(fr_dev, "fr", "dev"))
    write_jsonl(os.path.join(d, "fr_en_test.jsonl"), rows(fr_test, "fr", "test"))
    write_jsonl(os.path.join(d, "retrieval_eval.jsonl"), retr)
    stats = {k: len(v) for k, v in dict(vi_train=vi_train, vi_dev=vi_dev, vi_test=vi_test, fr_train=fr_train,
                                         fr_dev=fr_dev, fr_test=fr_test, retrieval_eval=retr).items()}
    if not dc.get("keep_raw", False) and not dc.get("phomt_dir"):
        import shutil
        shutil.rmtree(raw, ignore_errors=True)  # chỉ giữ dữ liệu đã chọn (bớt 0,5–1 GB)
    print("[data] xong:", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    prepare(load_config(ap.parse_args().config))
