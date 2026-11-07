"""App demo của Nhóm 3 — dịch tiếng Việt / tiếng Pháp sang tiếng Anh.

    python app.py                 # mở http://localhost:5000
    python app.py --port 8000     # đổi cổng
    python app.py --host 0.0.0.0  # cho máy khác trong cùng mạng truy cập
"""
import argparse
import os
import sys

# --- Cấu hình an toàn cho ML model (phải set TRƯỚC khi import torch) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Nếu có HF_TOKEN trong .env thì dùng để tránh rate-limit khi tải model
# os.environ.setdefault("HF_TOKEN", "...")  # bỏ comment và điền token nếu cần


def _check_model_files(model_dir: str) -> list:
    """Kiểm tra file model thật, không chỉ thư mục."""
    candidates = [
        os.path.join(model_dir, "model"),
        os.path.join(model_dir, "adapter_model.bin"),
        os.path.join(model_dir, "adapter_model.safetensors"),
        os.path.join(model_dir, "config.json"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def create_app(system=None):
    from flask import Flask, render_template
    from nhom3.api import create_blueprint

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(system))

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--threaded", action="store_true",
                    help="Bật threading (CHỈ dùng nếu model đã có lock nội bộ)")
    a = ap.parse_args()

    try:
        from nhom3.engine import MODEL_DIR
    except Exception as e:
        print(f"[FATAL] Không import được nhom3.engine: {e}", file=sys.stderr)
        sys.exit(1)

    found = _check_model_files(MODEL_DIR)
    print("=" * 64)
    print(" Nhóm 3 · Dịch Việt/Pháp → Anh")
    print(f" Mô hình : {MODEL_DIR}")
    if not found:
        print("  (CHƯA CÓ model — chạy: python tai_mo_hinh.py)")
    else:
        print(f"  Đã thấy: {[os.path.basename(p) for p in found]}")
    print(f" Mở      : http://localhost:{a.port}")
    print(f" Threaded : {a.threaded} (khuyến nghị False cho ML model)")
    print("=" * 64)

    app = create_app()

    try:
        app.run(
            host=a.host,
            port=a.port,
            debug=False,
            threaded=a.threaded,   # MẶC ĐỊNH False — tránh race condition
            use_reloader=False,
        )
    except OSError as e:
        print(f"[FATAL] Không mở được cổng {a.port}: {e}", file=sys.stderr)
        print(f"        Thử: python app.py --port {a.port + 1}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng server.")