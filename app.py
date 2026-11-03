"""App demo của Nhóm 3 — dịch tiếng Việt / tiếng Pháp sang tiếng Anh.

    python app.py                 # mở http://localhost:5000
    python app.py --port 8000     # đổi cổng
    python app.py --host 0.0.0.0  # cho máy khác trong cùng mạng truy cập
"""
import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
    a = ap.parse_args()
    from nhom3.engine import MODEL_DIR
    print("=" * 64)
    print(" Nhóm 3 · Dịch Việt/Pháp → Anh")
    print(f" Mô hình : {MODEL_DIR}" + ("" if os.path.exists(os.path.join(MODEL_DIR, "model")) else "  (CHƯA CÓ — chạy python tai_mo_hinh.py)"))
    print(f" Mở      : http://localhost:{a.port}")
    print("=" * 64)
    create_app().run(host=a.host, port=a.port, debug=False, threaded=True)
