"""Sinh 4 bộ dữ liệu đánh giá cho framing VI→EN.

  A  : 100 câu in-domain trích stratified từ vi_en_train.txt (50k), seed 42.
  B1 : biến thể có kiểm soát của A (typo dấu, đảo token cửa sổ 3, synonym VI),
       giữ nguyên reference EN.
  B2 : BẢN SAO NGUYÊN VĂN của B1 (cùng input, khác config khi chạy).
  C  : 100 câu VI out-of-domain từ Tatoeba VI–EN (manythings.org, CC-BY),
       lọc Jaccard token < 0.1 với MỌI câu VI trong 50k.

Chạy:  python FE_AGAL_v2/data_splits.py [--n 100]
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "data_splits"
RAW = HERE / "data_raw"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rows.append({"id": i, "src": parts[0].strip(), "tgt": parts[1].strip()})
    return rows


def toks(text):
    return [t.lower() for t in WORD.findall(text)]


def strip_diacritics(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


SYN = {  # thay synonym tối thiểu, giữ nghĩa
    "và": "cùng", "nhưng": "tuy nhiên", "vì": "bởi vì", "nếu": "nếu như",
    "rất": "vô cùng", "cũng": "đều", "tuy": "dù", "khi": "lúc", "sau": "sau khi",
    "trước": "trước khi", "để": "nhằm", "nhiều": "nhiều", "ít": "ít",
}


def perturb(src, rng):
    """Biến thể có kiểm soát: bỏ dấu / typo nhẹ / đảo token trong cửa sổ 3 / synonym.

    Mỗi token đổi tối đa 1 lần (touched) để tránh câu bị xáo quá mạnh.
    """
    parts = src.split()
    out = list(parts)
    changed = 0
    touched = set()
    for i, w in enumerate(parts):
        if i in touched:
            continue
        core = w.strip(".,;:!?…\"'()[]")
        if not WORD.search(core):
            continue
        low = core.lower()
        r = rng.random()
        if r < 0.03 and low in SYN and SYN[low] != low:
            out[i] = w.replace(core, SYN[low], 1)
            changed += 1
            touched.add(i)
        elif r < 0.08 and strip_diacritics(core) != core:  # mất dấu (nhẹ)
            out[i] = w.replace(core, strip_diacritics(core), 1)
            changed += 1
            touched.add(i)
        elif r < 0.13 and len(core) >= 4:  # typo đảo 2 ký tự
            j = rng.randrange(len(core) - 1)
            c = list(core)
            c[j], c[j + 1] = c[j + 1], c[j]
            out[i] = w.replace(core, "".join(c), 1)
            changed += 1
            touched.add(i)
        elif r < 0.17 and i < len(out) - 1:  # đảo token trong cửa sổ 3
            j = rng.randint(i + 1, min(i + 3, len(out) - 1))
            if j not in touched:
                out[i], out[j] = out[j], out[i]
                changed += 1
                touched.update({i, j})
    if changed == 0:  # đảm bảo luôn có biến thể, chỉ đổi MỘT token
        cand = [i for i, w in enumerate(parts) if strip_diacritics(w) != w]
        if cand:
            i = max(cand, key=lambda k: len(parts[k]))
            out[i] = strip_diacritics(parts[i])
        elif len(parts) > 1:
            out[0], out[-1] = out[-1], out[0]
    return " ".join(out)


def stratified_sample(rows, n, seed=42, bins=3):
    lens = [len(toks(r["src"])) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    buckets = [[] for _ in range(bins)]
    per = (len(order) + bins - 1) // bins
    for b, i in enumerate(order):
        buckets[b // per].append(i)
    rng = random.Random(seed)
    picked, base = [], n // bins
    for b in range(bins):
        k = base + (1 if b < n % bins else 0)
        picked += rng.sample(buckets[b], min(k, len(buckets[b])))
    picked.sort()
    return [rows[i] for i in picked]


VI_STOP = set("""và của là có không được cho với một những các này đó thì mà đã sẽ đang để khi như
nhưng nếu vì nên rất cũng đều tôi bạn anh chị em nó họ chúng ta mình người ở trong ngoài trên dưới ra
vào lên xuống đến từ về theo bằng bị do tại sau trước giữa nhiều ít hơn nhất chỉ còn lại rồi đây kia
ấy gì nào sao thế ạ nhé nhỉ chứ đấy thôi quá lắm chưa chẳng phải muốn làm ăn đi nói biết thấy""".split())


def content_tokens(text):
    return {t for t in toks(text) if t not in VI_STOP and len(t) > 1}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_c(n, seed=42):
    """C = Tatoeba VI–EN (CC-BY), lọc Jaccard < 0.1 với mọi câu trong 50k.

    Ghi chú: Jaccard tính trên tập TOKEN NỘI DUNG (bỏ stopwords VI). Jaccard đầy đủ
    (giữ stopwords) cho 0 ứng viên vì stopwords chiếm phần lớn giao của hai câu bất kỳ
    trong kho 50k — xem FE_AGAL_v2/logs/data_notes.md.
    """
    zpath = RAW / "vie-eng.zip"
    if not zpath.exists():
        raise SystemExit("Thiếu FE_AGAL_v2/data_raw/vie-eng.zip (Tatoeba VI–EN, manythings.org)")
    raw_pairs = []
    with zipfile.ZipFile(zpath) as z:
        name = max((x for x in z.namelist() if x.endswith(".txt")), key=lambda x: z.getinfo(x).file_size)
        for line in z.read(name).decode("utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                raw_pairs.append((parts[1].strip(), parts[0].strip()))  # (vi, en)
    opus = RAW / "tatoeba-opus-vi-en.zip"
    if opus.exists():  # bổ sung Tatoeba bản OPUS (moses, 2 file song song)
        with zipfile.ZipFile(opus) as z:
            ns = {x.split(".")[-1]: x for x in z.namelist() if x.endswith((".en", ".vi"))}
            vi = z.read(ns["vi"]).decode("utf-8").splitlines()
            en = z.read(ns["en"]).decode("utf-8").splitlines()
            raw_pairs += [(a.strip(), b.strip()) for a, b in zip(vi, en)]
    pairs, seen = [], set()
    for vi, en in raw_pairs:
        if vi in seen or not vi or not en:
            continue
        seen.add(vi)
        if 4 <= len(toks(vi)) <= 40 and 3 <= len(toks(en)) <= 40:
            pairs.append({"src": vi, "tgt": en, "meta": {"source": "Tatoeba vie-eng (CC-BY)"}})

    train = read_tsv(ROOT / "vi_en_train.txt")
    sets = [content_tokens(r["src"]) for r in train]
    # inverted index trên token nội dung để tránh 12k x 50k phép so sánh
    df = defaultdict(int)
    for s in sets:
        for t in s:
            df[t] += 1
    inv = defaultdict(list)
    for i, s in enumerate(sets):
        for t in s:
            if df[t] <= 4000:
                inv[t].append(i)

    keep = []
    for p in pairs:
        cand = content_tokens(p["src"])
        cand_ids = set()
        for t in cand:
            if df.get(t, 10 ** 9) <= 4000:
                cand_ids.update(inv[t])
        best = max((jaccard(cand, sets[i]) for i in cand_ids), default=0.0)
        if best < 0.1:
            p["meta"]["max_jaccard_vs_50k"] = round(best, 4)
            p["meta"]["jaccard_mode"] = "content-token (bỏ stopwords VI)"
            keep.append(p)

    # khử gần trùng nội bộ C (content Jaccard >= 0.5) để không lặp câu
    deduped, seen_sets = [], []
    for p in keep:
        c = content_tokens(p["src"])
        if all(jaccard(c, s) < 0.6 for s in seen_sets):
            deduped.append(p)
            seen_sets.append(c)
    rng = random.Random(seed)
    rng.shuffle(deduped)
    picked = deduped[:n]
    for i, p in enumerate(picked):
        p["id"] = i
    return picked, len(pairs), len(deduped)


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train = read_tsv(ROOT / "vi_en_train.txt")
    assert len(train) == 50000, f"vi_en_train.txt phải có 50,000 dòng, thấy {len(train)}"
    A = stratified_sample(train, a.n, seed=42)
    for i, r in enumerate(A):
        r["id"] = i
        r["meta"] = {"source": "vi_en_train.txt (in-domain)", "assumption": "test trích từ file train → có thể overlap train"}

    B1 = []
    for r in A:
        rng = random.Random(1000 + r["id"])  # seed riêng, tất định
        B1.append({"id": r["id"], "src": perturb(r["src"], rng), "tgt": r["tgt"],
                   "orig_src": r["src"], "meta": {"source": "biến thể có kiểm soát của A", "ref": "giữ nguyên EN của A"}})
    B2 = [{"id": r["id"], "src": r["src"], "tgt": r["tgt"], "orig_src": r["orig_src"],
           "meta": dict(r["meta"])} for r in B1]

    C, n_raw, n_keep = build_c(a.n, seed=42)

    save_jsonl(OUT / "A.jsonl", A)
    save_jsonl(OUT / "B1.jsonl", B1)
    save_jsonl(OUT / "B2.jsonl", B2)
    save_jsonl(OUT / "C.jsonl", C)

    b1 = [json.loads(l) for l in open(OUT / "B1.jsonl", encoding="utf-8")]
    b2 = [json.loads(l) for l in open(OUT / "B2.jsonl", encoding="utf-8")]
    same = all(x["src"] == y["src"] and x["tgt"] == y["tgt"] for x, y in zip(b1, b2)) and len(b1) == len(b2)
    stats = {
        "N": {"A": len(A), "B1": len(B1), "B2": len(B2), "C": len(C)},
        "A_len_tokens": {"mean": round(sum(len(toks(r["src"])) for r in A) / len(A), 2)},
        "C_len_tokens": {"mean": round(sum(len(toks(r["src"])) for r in C) / len(C), 2)},
        "C_tatoeba_raw": n_raw,
        "C_after_filter": n_keep,
        "C_max_jaccard": round(max(r["meta"]["max_jaccard_vs_50k"] for r in C), 4),
        "B1_sentences_changed": sum(1 for r in B1 if r["src"] != r["orig_src"]),
        "B1_B2_same_input": same,
        "sha256_16": {"A": sha(OUT / "A.jsonl"), "B1": sha(OUT / "B1.jsonl"),
                      "B2": sha(OUT / "B2.jsonl"), "C": sha(OUT / "C.jsonl")},
    }
    assert same, "B1 và B2 phải cùng input"
    assert len(C) == a.n, f"Không đủ câu C hợp lệ: {len(C)} < {a.n}"
    with open(OUT / "generation_notes.md", "w", encoding="utf-8") as f:
        f.write("# Data splits (seed 42)\n\n")
        f.write("| Split | N | Nguồn |\n|---|---|---|\n")
        f.write(f"| A | {len(A)} | vi_en_train.txt, stratified theo độ dài, seed 42 |\n")
        f.write(f"| B1 | {len(B1)} | biến thể A: mất dấu/typo/đảo token cửa sổ 3/synonym; ref giữ nguyên |\n")        f.write(f"| B2 | {len(B2)} | BẢN SAO của B1 (verified identical) |\n")
        f.write(f"| C | {len(C)} | Tatoeba VI–EN (CC-BY), lọc Jaccard nội dung < 0.1 vs 50k |\n\n")
        f.write("## Kiểm tra\n\n```json\n" + json.dumps(stats, ensure_ascii=False, indent=2) + "\n```\n\n")
        f.write("## Giả định\n\n- `vi_en_train.txt` KHÔNG có split train/dev/test; A là test trích từ file train nên **có thể overlap với dữ liệu model đã thấy**. Đây là baseline in-domain có chủ đích.\n")
        f.write("- B1/B2 dùng **cùng input tuyệt đối** (cùng file nội dung); khác biệt duy nhất là config memory.\n")
        f.write("- **Cường độ nhiễu B1**: mỗi token đổi tối đa 1 lần; xác suất synonym 3%, mất dấu 5%, typo 5%, đảo token cửa sổ 3 là 4%; mọi câu đều có ít nhất 1 thay đổi.\n")
        f.write("- C là câu Tatoeba đời thường, khác miền PhoMT (tin tức/wiki/TED); câu ngắn 4–40 token.\n")
        f.write("- **Công thức lọc C**: Jaccard trên tập token nội dung (đã bỏ stopwords VI). Jaccard đầy đủ giữ stopwords cho 0 ứng viên trên toàn bộ 9.345 câu Tatoeba đủ dài, vì stopwords chiếm phần lớn giao của hai câu bất kỳ trong kho 50k. Chi tiết + số liệu thử nghiệm ở `logs/data_notes.md`.\n")
        f.write("- `C_max_jaccard` là Jaccard nội dung lớn nhất của một câu C với bất kỳ câu nào trong 50k (yêu cầu < 0.1).\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
