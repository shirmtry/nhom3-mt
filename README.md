# Nhóm 3 · Dịch Việt/Pháp → Anh

**Multilingual Transformer với dữ liệu song ngữ hạn chế** — đồ án TMG301, Nhóm 3.

Một mô hình duy nhất (mBART-50 + LoRA) học chung **50K cặp Việt–Anh** và **300K cặp Pháp–Anh**, dịch tiếng Việt hoặc tiếng Pháp sang tiếng Anh. Với câu khó, hệ thống tra một kho Pháp–Anh để lấy câu Anh tương tự làm bằng chứng rồi dịch lại.

```
Câu Việt / Pháp ─► ① Tiền xử lý ─► ② Transformer dịch nháp ŷ + độ tự tin C ─► ③ Cổng (C, độ hiếm R)
     câu dễ : dùng luôn ŷ
     câu khó: ④ LaBSE tìm Top-5 trong kho Pháp–Anh ─► ⑤ reranker chọn e* (điểm < δ → giữ ŷ)
              ─► ⑥ cùng Transformer dịch lại với x + e* ─► ŷ′
```

App demo hiện từng bước trên cho mỗi câu, kèm tab **Kết quả thử nghiệm** và **Cách hoạt động**.

---

## Chạy app (cho người mới)

**Cần:** Python 3.10–3.12, Git, khoảng **8 GB dung lượng trống** (mô hình 1,6 GB + 2 mô hình phụ ~4 GB) và **RAM từ 12 GB** (app dùng khoảng 8 GB). Không cần GPU.

### Windows (CMD)

```bat
git clone https://github.com/triethuynh321/nhom3-mt.git
cd nhom3-mt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python tai_mo_hinh.py
python app.py
```

### macOS / Linux

```bash
git clone https://github.com/triethuynh321/nhom3-mt.git
cd nhom3-mt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python tai_mo_hinh.py
python app.py
```

Mở **http://localhost:5000**. Lần đầu nạp mô hình mất 1–2 phút (đèn trạng thái góc phải chuyển sang *Sẵn sàng*).

- `python tai_mo_hinh.py` tự tải `nhom3_mt.zip` từ mục **Releases** của repo, giải nén vào `models/nhom3_mt/`, rồi tải sẵn LaBSE và reranker.
- Đã có file zip trên máy thì dùng: `python tai_mo_hinh.py D:\Downloads\nhom3_mt.zip`
- Lần sau chỉ cần: `venv\Scripts\activate` rồi `python app.py`.
- Kiểm tra code (không cần mô hình): `python tests/test_app.py`

---

## Kết quả thử nghiệm

Tập test: 1.000 câu Việt–Anh (PhoMT) và 500 câu Pháp–Anh.

| Thí nghiệm | VI→EN BLEU | VI→EN chrF | FR→EN BLEU | FR→EN chrF |
|---|---|---|---|---|
| A0 · Transformer nhỏ học từ đầu | 6,17 | 25,89 | – | – |
| A · mBART-50 + LoRA, chỉ Việt–Anh | **26,98** | **49,43** | – | – |
| B · + Pháp–Anh học chung | 26,14 | 48,65 | 28,94 | 53,59 |
| C · chỉ dịch nháp | 25,75 | 48,62 | 28,70 | 53,70 |
| C · có cổng (hệ thống đề xuất) | 26,20 | 48,85 | 28,90 | 53,49 |
| C · luôn tra kho | 25,54 | 48,38 | 28,95 | 53,50 |

Truy xuất (500 câu): Recall@1 = 89,8% (LaBSE) → 90,4% (sau rerank), Recall@5 = 94%.

**Nhận xét trung thực**

- Tiền huấn luyện là yếu tố quyết định: BLEU từ 6,2 lên 27,0.
- Học chung Pháp–Anh **không** cải thiện Việt–Anh trong cấu hình này (B thấp hơn A 0,8 BLEU), nhưng một mô hình dịch được cả hai ngôn ngữ.
- Cổng tốt hơn "luôn tra kho" (+0,66 BLEU) và "chỉ dịch nháp" (+0,45 BLEU). Các chênh lệch dưới 1 BLEU có thể nằm trong sai số.
- Độ hiếm R không dự đoán được bản nháp kém (AUROC 0,42 với tiếng Việt). Độ tự tin C có ích hơn (AUROC 0,66–0,70).
- **Rò rỉ gần trùng:** PhoMT và TED2020 cùng lấy từ các bài nói TED. Bộ lọc chỉ loại câu Anh trùng *chính xác*, nên vài câu test vẫn có câu Anh gần trùng trong kho (ví dụ `do n't` / `don't`). Phần cải thiện của C vì vậy có thể bị đánh giá cao hơn thực tế. Tab Kết quả đánh dấu các ví dụ này là "bằng chứng ≈ đáp án".

---

## Cấu trúc thư mục

```
app.py              chạy web demo (Flask)
nhom3/engine.py     bộ dịch 6 bước (chạy CPU)
nhom3/api.py        các API: /api/translate, /api/status, /api/results
nhom3/langdetect.py nhận diện Việt / Pháp / Anh cho ô "Tự động"
templates/, static/ giao diện
tai_mo_hinh.py      tải và giải nén mô hình
tests/test_app.py   kiểm tra bằng mô hình giả
training/           code huấn luyện + notebook Kaggle/Colab (xem training/README.md)
models/             nơi chứa mô hình (không đưa vào git)
```

## Phát hành mô hình (dành cho chủ repo)

File mô hình quá nặng cho git (giới hạn 100 MB/file), nên để ở **GitHub Releases** (giới hạn 2 GB/file):

1. Trên trang repo: **Releases → Create a new release**.
2. **Tag:** `v1.0` (đúng tên này, vì `tai_mo_hinh.py` tìm theo tag `v1.0`).
3. Kéo thả file `nhom3_mt.zip` vào ô *Attach binaries*, rồi **Publish release**.

`tai_mo_hinh.py` tự đọc địa chỉ repo từ `git clone`, nên không cần sửa gì. Nếu chạy ngoài thư mục git, sửa `DEFAULT_REPO` trong file đó.

## Huấn luyện lại

Xem `training/README.md`. Notebook `training/notebooks/TMG301_Nhom3_MultilingualMT.ipynb` chạy toàn bộ (dữ liệu → 4 mô hình → chọn ngưỡng → đánh giá → xuất `nhom3_mt.zip`) trên 1 GPU T4 miễn phí của Kaggle, mất khoảng 3 giờ.

## Giấy phép

Mã nguồn: MIT (xem `LICENSE`). Mô hình được tinh chỉnh từ `facebook/mbart-large-50` (MIT) trên PhoMT, OPUS TED2020 và News-Commentary; mỗi bộ dữ liệu có điều khoản riêng, nên mô hình chỉ dùng cho học tập và nghiên cứu.
