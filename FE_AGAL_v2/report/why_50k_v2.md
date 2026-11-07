# Vì sao chọn 50.000 cặp VI–EN?

Tài liệu trả lời **đúng** câu hỏi của đề bài. Số liệu lấy từ chính repo (`vi_en_train.txt`) và log `FE_AGAL_v2/logs/why_50k_stats.json`; không suy đoán.

## 6.1. 50k là gì?

- `vi_en_train.txt`: **50.000 cặp Việt–Anh**, UTF-8, tab-separated, mỗi dòng `câu VI \t câu EN`, không header.
- Đây là **dataset CHÍNH** cho nhiệm vụ VI→EN (cấu hình `training/configs/default.yaml`: `vi_train_size: 50000`, `stage2_vi_share: 0.8`).
- 300.000 cặp `fr_en_train.txt` **không phải** dữ liệu chính cho VI→EN; nó chỉ là kho FR–EN auxiliary/memory.
- File **không có split train/dev/test**. Vì vậy mọi test set trích từ file này (kể cả kịch bản A của FE_AGAL_v2) **có thể overlap với dữ liệu model đã thấy khi train**. Đây là baseline "in-domain/nhìn thấy rồi", không phải test độc lập.

## 6.2. Vì sao 50k mà không phải 10k / 100k / full?

### (a) Độ phủ từ vựng — learning curve thực nghiệm trên chính file 50k

Lấy mẫu con ngẫu nhiên (seed 42) từ 50k và đo độ phủ trên toàn bộ vocab của 50k (vocab tính theo token chữ, 21.882 kiểu từ tiếng Việt):

| Cỡ mẫu | Vocab VI | % phủ vocab 50k | Tăng thêm so với mốc trước |
|---:|---:|---:|---:|
| 5.000 | 5.531 | 25,3% | – |
| 10.000 | 8.152 | 37,3% | +12,0 điểm |
| 20.000 | 12.428 | 56,8% | +19,5 điểm |
| 30.000 | 15.882 | 72,6% | +15,8 điểm |
| 40.000 | 19.037 | 87,0% | +14,4 điểm |
| **50.000** | **21.882** | **100%** | +13,0 điểm |

Diễn giải bằng số:
- **10k chỉ phủ 37,3%** vocab của 50k → hơn 6/10 kiểu từ của tập mục tiêu chưa từng xuất hiện khi train.
- **20k phủ 56,8%**, **30k phủ 72,6%**, **40k phủ 87,0%**.
- Đường cong vẫn tăng ~13 điểm ở 10k cuối → **50k là mức dữ liệu lớn nhất repo có**, và việc tăng tiếp vẫn có ích về vocab. Không có 100k/full để so; nếu có, câu hỏi "sao không 100k" trả lời thẳng: **repo không có dữ liệu VI–EN ngoài 50k**.

Đo bổ sung trên câu out-of-domain (C, Tatoeba đời thường): tỉ lệ token OOV so với vocab 50k là **1,37%**, so với vocab 5k là 2,05%, 10k là 1,54%. Nghĩa là 50k gần như phủ hết **bề mặt từ vựng** của cả miền đời thường; khoảng cách còn lại của C là **khác miền ngữ nghĩa/cú pháp**, không phải thiếu chữ.

### (b) Phân phối độ dài — 50k đại diện cho câu thực tế

| Thống kê | VI | EN |
|---|---:|---:|
| Trung bình token | 16,38 | 13,10 |
| Trung vị | 12 | 10 |
| p10 / p90 | 5 / 34 | – |

50k trải từ câu cụt 5 token đến câu dài 34+ token (p90), đủ để học cả câu ngắn lẫn câu dài. Tập C (Tatoeba) trung bình chỉ **5,84 token** — ngắn hơn nhiều, nên so sánh A↔C có confound độ dài (đã xử lý bằng phân tích length-matched trong `results/length_matched_AC.json`).

### (c) Từ vựng phổ biến — 50k nghiêng về văn viết/tin tức

Top-20 token VI trong 50k: và (13.727), có (13.113), là (12.452), của (11.368), một (11.030), không (10.494), tôi (9.810), được (7.851), bạn (7.117), trong (6.950), các (6.684), thể (6.523), đã (6.382), người (6.247), cho (5.872), với (5.770), anh (5.485), đó (5.385), sẽ (5.106), những (5.066).

Phân phối này đại diện cho **tin tức/wiki/TED** (PhoMT) hơn là hội thoại đời thường. Đó chính là lý do kịch bản C (Tatoeba) là phép thử domain shift hợp lý, và cũng là gợi ý mở rộng dữ liệu ở mục 6.3.

### (d) Chi phí tính toán

- Train theo config gốc: 50k VI–EN + 300k FR–EN trên 1 GPU T4 (~3 giờ cho toàn pipeline theo README). Bản đầu dùng 15k.
- Inference CPU (máy hiện tại): ~6–11 giây/câu với beam=4 tuỳ có refine hay không; 100 câu/kịch bản là mức khả thi cho đánh giá exploratory.
- Với 10k: nhanh hơn nhưng mất 62,7% vocab mục tiêu — đánh đổi không hợp lý khi 50k đã có sẵn và vừa 1 GPU T4.
- Với 100k/full: không có dữ liệu; nếu có, chi phí train tăng tuyến tính và pipeline hiện tại **sẵn sàng** (chỉ đổi đường dẫn data, không đổi code).

## 6.3. Kết luận

**50k là "sweet spot" vì:**
1. **Đây là toàn bộ dữ liệu VI–EN chất lượng cao của repo** (không có 100k/full); 300k FR–EN đã đóng vai trò auxiliary nên không cần tăng VI–EN bằng dữ liệu Pháp.
2. **Độ phủ vocab gấp 2,7 lần mốc 10k** (100% so với 37,3%) và gấp 1,76 lần mốc 20k (56,8%); đường cong chưa bão hoà ở 40k → 50k tận dụng tối đa dữ liệu sẵn có.
3. **Khớp cấu hình huấn luyện và kết quả công bố** (`meta.json: dev_chrF 45,48`; README: VI→EN BLEU 26,98 / chrF 49,43 trên 1.000 câu PhoMT), nên đánh giá trên 50k là đánh giá đúng hệ thống đang có.
4. **OOV của miền đời thường chỉ 1,37%** → nút thắt của domain shift không phải vocab mà là phân phối cú pháp/ngữ nghĩa; thêm dữ liệu cùng miền PhoMT sẽ ít giá trị hơn thêm dữ liệu hội thoại.

**Nếu mở rộng:** dựa trên failure analysis (kịch bản C và các câu tệ nhất), cần thêm dữ liệu **hội thoại/đời thường** (Tatoeba/OpenSubtitles) và **câu ngắn**, vì 50k nghiêng tin tức/TED và trung bình 16,4 token.

**Nếu có 100k VI–EN:** pipeline đã sẵn sàng — chỉ cần đặt file mới và trỏ `vi_en_train.txt`/`DATA_DIR` tương ứng, chạy lại `data_splits.py` và `run_all.py`; toàn bộ seed/split/metric giữ nguyên.
