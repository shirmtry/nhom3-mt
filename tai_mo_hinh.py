r"""Tải mô hình Nhóm 3 (nhom3_mt.zip, ~1,6 GB) và giải nén vào models/nhom3_mt/.

    python tai_mo_hinh.py                              # tải từ GitHub Release của repo này
    python tai_mo_hinh.py D:\Downloads\nhom3_mt.zip    # dùng file zip có sẵn trên máy
    python tai_mo_hinh.py https://.../nhom3_mt.zip     # tải từ đường dẫn khác

Sau khi giải nén, script tải sẵn 2 mô hình phụ (LaBSE + bge-reranker-v2-m3, ~4 GB) để lúc demo
không phải chờ. Bỏ bước này bằng  --bo-qua-mo-hinh-phu
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
TARGET = os.path.join(MODELS, "nhom3_mt")
TAG, FILE = "v1.0", "nhom3_mt.zip"
DEFAULT_REPO = "triethuynh321/nhom3-mt"   # chỉ dùng khi không đọc được địa chỉ repo từ git


def repo_from_git():
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001  (máy không có git)
        return None
    m = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else None


def release_url():
    repo = os.environ.get("NHOM3_REPO") or repo_from_git() or DEFAULT_REPO
    return f"https://github.com/{repo}/releases/download/{TAG}/{FILE}"


def download(url, dest):
    print(f"Đang tải: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "nhom3-mt"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB  ({done * 100 // total}%)", end="", flush=True)
    print()


def extract(zip_path):
    print(f"Đang giải nén vào {MODELS} …")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names if n.startswith(("/", "\\")) or ".." in n.replace("\\", "/").split("/")]
        if bad or {n.replace("\\", "/").split("/")[0] for n in names} != {"nhom3_mt"}:
            sys.exit("File zip không đúng: bên trong phải là một thư mục nhom3_mt/ (xuất từ notebook, bước 6).")
        os.makedirs(MODELS, exist_ok=True)
        if os.path.isdir(TARGET):
            old = TARGET + "_cu"
            shutil.rmtree(old, ignore_errors=True)
            os.rename(TARGET, old)
            print(f"  Mô hình cũ được đổi tên thành {old}")
        z.extractall(MODELS)
    if not os.path.exists(os.path.join(TARGET, "model", "config.json")):
        sys.exit("Giải nén xong nhưng thiếu models/nhom3_mt/model/config.json")
    print(f"  Xong: {TARGET}")


def prefetch_helpers():
    meta = json.load(open(os.path.join(TARGET, "meta.json"), encoding="utf-8"))
    enc = meta.get("encoder", "sentence-transformers/LaBSE")
    rr = meta.get("reranker", "BAAI/bge-reranker-v2-m3")
    print(f"Đang tải sẵn mô hình phụ: {enc} và {rr} (~4 GB, chỉ lần đầu) …")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    SentenceTransformer(enc, device="cpu")
    AutoTokenizer.from_pretrained(rr)
    AutoModelForSequenceClassification.from_pretrained(rr)
    print("  Xong.")


def main():
    ap = argparse.ArgumentParser(description="Tải và giải nén mô hình Nhóm 3")
    ap.add_argument("nguon", nargs="?", help="đường dẫn file nhom3_mt.zip hoặc URL (mặc định: GitHub Release)")
    ap.add_argument("--bo-qua-mo-hinh-phu", action="store_true", help="không tải sẵn LaBSE + reranker")
    a = ap.parse_args()

    src = a.nguon or release_url()
    if os.path.isfile(src):
        extract(src)
    else:
        if "TEN_GITHUB_CUA_BAN" in src:
            sys.exit("Chưa biết địa chỉ repo. Chạy lệnh này trong thư mục đã git clone, "
                     "hoặc đưa đường dẫn file zip / URL: python tai_mo_hinh.py <nhom3_mt.zip>")
        tmp = os.path.join(tempfile.gettempdir(), FILE)
        try:
            download(src, tmp)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Không tải được ({e}). Kiểm tra mạng, hoặc tải tay file {FILE} rồi chạy: "
                     f"python tai_mo_hinh.py <đường dẫn file zip>")
        extract(tmp)
        os.remove(tmp)
    if not a.bo_qua_mo_hinh_phu:
        try:
            prefetch_helpers()
        except Exception as e:  # noqa: BLE001
            print(f"  Chưa tải được mô hình phụ ({e}). App sẽ tự tải ở lần dịch câu khó đầu tiên.")
    print("\nHoàn tất. Chạy app:  python app.py   rồi mở http://localhost:5000")


if __name__ == "__main__":
    main()
