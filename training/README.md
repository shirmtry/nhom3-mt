# Multilingual Transformer Việt/Pháp → Anh với dữ liệu song ngữ hạn chế

TMG301 · Nhóm 3. Mã nguồn hiện thực đúng kiến trúc ở Phần 3 (slide 3.0 – 3.8) và chạy các thí nghiệm cho Phần 4.

```
Câu Việt / Pháp ─► ① Tiền xử lý ─► ② Multilingual Transformer (mBART-50 + LoRA) ─► bản nháp ŷ, độ tin cậy C
                                        ▲ θ học chung Việt–Anh 50K + Pháp–Anh 300K (vai trò A)
   ③ Cổng: C ≥ τ và R ≤ ρ ─► câu dễ: y = ŷ  (không tra kho)
           ngược lại      ─► ④ LaBSE + FAISS Top-5 trong kho Pháp–Anh (vai trò B)
                             ⑤ reranker chọn câu Anh e* (điểm < δ → giữ ŷ)
                             ⑥ cùng Transformer dịch lại với x + e* ─► y = ŷ′
```

## Thành phần ↔ file

| Slide | Thành phần | File |
|---|---|---|
| 3.1 | Dữ liệu: PhoMT 50K, OPUS TED2020 + News Commentary 300K, 5 bước lọc, chống rò rỉ dev/test | `mtm/data.py` |
| 3.2 | SentencePiece 250K của mBART-50, mã `vi_VN` / `fr_XX` → `en_XX` | `mtm/modeling.py` |
| 3.3 | mbart-large-50 + LoRA r=16 trên q, v (≈ 2,36M tham số) | `mtm/modeling.py` |
| 3.4 | Lấy mẫu nhiệt độ p ∝ q^(1/T), 2 giai đoạn, dừng sớm theo chrF dev | `mtm/sampling.py`, `mtm/train.py` |
| 3.5 | Cổng: độ tin cậy C, độ hiếm R theo ngôn ngữ nguồn, chọn τ, ρ, δ trên dev | `mtm/gate.py`, `mtm/evaluate.py` |
| 3.6 | Kho Pháp–Anh LaBSE + FAISS, reranker bge-reranker-v2-m3 | `mtm/memory.py`, `mtm/rerank.py` |
| 3.7 | Dịch lại kiểu Neural Fuzzy Repair, trộn 50/35/15 | `mtm/augment.py`, `mtm/pipeline.py` |
| Phần 4 | A0 / A / B / C, BLEU, chrF, COMET, AUROC, Recall@5, demo, xuất cho app | `mtm/evaluate.py`, `app.py`, `mtm/export.py` |

## Chạy

Cách dễ nhất: mở `notebooks/TMG301_Nhom3_MultilingualMT.ipynb` trên Colab hoặc Kaggle (GPU T4), tải `mt_multilingual.zip` lên, chạy từ trên xuống.

Hoặc từng lệnh:

```bash
pip install -r requirements.txt && pip install underthesea
python tests/test_core.py                                   # kiểm tra logic, không cần GPU
python -m mtm.data     --config configs/default.yaml        # 3.1
python -m mtm.memory   --config configs/default.yaml        # 3.6 kho Pháp–Anh
python -m mtm.train    --config configs/default.yaml --exp A0
python -m mtm.train    --config configs/default.yaml --exp A
python -m mtm.train    --config configs/default.yaml --exp B
python -m mtm.train    --config configs/default.yaml --exp C
python -m mtm.evaluate --config configs/default.yaml --calibrate --exp C   # τ, ρ, δ trên dev
python -m mtm.evaluate --config configs/default.yaml --exp C               # (lặp cho A0, A, B)
python -m mtm.evaluate --config configs/default.yaml --retrieval
python -m mtm.evaluate --config configs/default.yaml --summary             # bảng cho slide Phần 4
python app.py --config configs/default.yaml --exp C --share                # demo
python -m mtm.export   --config configs/default.yaml --exp C               # nhom3_mt.zip cho app demo
```

**PhoMT** cần đăng nhập Hugging Face và chấp nhận điều khoản tại huggingface.co/datasets/vinai/PhoMT (chỉ dùng cho nghiên cứu/học tập, không phát tán lại). Chưa được duyệt thì tạm đặt `data.vi_source: opus_ted2020`.

## Các thí nghiệm của Phần 4

| Mô hình | Khác biệt | Trả lời câu hỏi |
|---|---|---|
| A0 | Transformer nhỏ, huấn luyện từ đầu, chỉ Việt–Anh | Tiền huấn luyện đóng góp bao nhiêu? |
| A | mBART-50 + LoRA, chỉ Việt–Anh | Mốc song ngữ |
| B | + Pháp–Anh học chung (T = 5; tùy chọn quét T = 1, 2) | Pháp–Anh dạy mô hình được bao nhiêu (vai trò A)? |
| C · draft | mô hình C, chỉ dịch nháp | Học thêm mẫu bằng chứng có làm hỏng bản dịch thường? |
| C · full | có cổng + truy xuất + dịch lại | Hệ thống đề xuất |
| C · always | tra kho cho mọi câu | Cổng có cần không? |

Ngoài ra: AUROC của C và R (cổng có nhận ra bản nháp kém không), % câu đi nhánh khó, % câu thật sự dùng bằng chứng, Recall@1/@5 của truy xuất (trước và sau rerank), và Pháp → Anh làm chỉ số phụ.

Kết quả: `work/results/summary.md`, `work/results/*_test.json`, bản dịch từng câu trong `work/results/outputs/`.

## Dùng với app demo

`python -m mtm.export` tạo `work/export/nhom3_mt.zip` (mô hình C đã gộp LoRA, kho Pháp–Anh, ngưỡng, kết quả, ví dụ). Ở thư mục gốc của repo chạy `python tai_mo_hinh.py <đường dẫn>/nhom3_mt.zip` rồi `python app.py`: tab Dịch hiện 6 bước của mô hình, tab Kết quả thử nghiệm hiện bảng điểm.

## Ghi chú kỹ thuật

- Không dịch qua tiếng Pháp: câu Pháp chỉ là khóa tìm kiếm, bằng chứng luôn là câu Anh đi kèm.
- Khi tạo bằng chứng cho câu huấn luyện, loại chính cặp đó và mọi cặp có câu Anh trùng đáp án (tránh học chép).
- Kho Pháp–Anh không chứa câu Anh nào **trùng chính xác** với dev/test Việt–Anh (bước 5 của slide 3.1). Lưu ý: bước này so khớp chính xác nên vẫn sót các câu **gần trùng** (PhoMT viết `do n't`, TED2020 viết `don't`; hoặc lệch 1 từ). Muốn chặt hơn, thêm so khớp gần đúng (ví dụ Jaccard ≥ 0,8 trên tập từ) ở `drop_overlap`.
- Giai đoạn 2 vẫn giữ 20% Pháp–Anh và mẫu kèm bằng chứng, để mô hình không quên tiếng Pháp và cách dùng bằng chứng.
- Độ hiếm R tính theo dữ liệu của chính ngôn ngữ nguồn; ngưỡng τ, ρ, δ chọn riêng cho từng ngôn ngữ trên dev.
