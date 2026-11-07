# Ghi chú dữ liệu & quyết định lọc (FE_AGAL_v2)

## 1. Nguồn dữ liệu

| Split | File | N | Nguồn | Ghi chú |
|---|---|---:|---|---|
| A | `data_splits/A.jsonl` | 100 | `vi_en_train.txt` (50.000 cặp) | stratified theo độ dài, seed 42; có thể overlap train |
| B1 | `data_splits/B1.jsonl` | 100 | biến thể của A | mất dấu / typo / đảo token cửa sổ 3 / synonym; reference giữ nguyên |
| B2 | `data_splits/B2.jsonl` | 100 | **bản sao nguyên văn của B1** | SHA-256 16 ký tự: `a114cc5aa9a7281b` cho cả hai |
| C | `data_splits/C.jsonl` | 100 | Tatoeba VI–EN (manythings.org + OPUS v2026-07-08, CC-BY) | lọc Jaccard nội dung < 0,1 với mọi câu trong 50k |

A/B1/B2/C đều dùng seed 42. B1 có 100/100 câu thay đổi so với A. Mỗi token B1 đổi tối đa 1 lần; xác suất synonym 3%, mất dấu 5%, typo 5%, đảo token cửa sổ 3 là 4%; mọi câu có ít nhất 1 thay đổi. Độ tương đồng token trung bình A↔B1 (SequenceMatcher) = 0,805 (min 0,40).

## 2. Vì sao C dùng "Jaccard nội dung" thay vì Jaccard đầy đủ?

Đề bài yêu cầu: *Jaccard token overlap với bất kỳ câu nào trong 50k < 0,1*.

Thử nghiệm thực tế trên toàn bộ câu Tatoeba đủ dài (9.345 câu, 5–30 token):

| Cách tính Jaccard | Ứng viên đạt < 0,1 |
|---|---:|
| Đầy đủ (giữ stopwords) | **0 / 9.345** |
| Chỉ token nội dung (bỏ stopwords VI) | **76 / 9.345** (mở rộng 4–40 token: 117 sau khử trùng nội bộ) |

Lý do: với kho 50k câu, hai câu bất kỳ chia sẻ rất nhiều stopwords (`và, của, là, có, không, …`); các stopwords này chiếm phần lớn giao (intersection) và làm Jaccard đầy đủ luôn > 0,1. Ngưỡng đầy đủ vì vậy **không thể đạt** với bất kỳ câu Tatoeba nào.

Quyết định: dùng Jaccard trên **tập token nội dung** (bỏ danh sách stopwords VI, lowercase, bỏ token 1 ký tự) — vẫn là "Jaccard ở mức token", nhưng đo trùng lặp **ngữ nghĩa/từ vựng** thay vì trùng stopwords. Công thức:

```
J(A,B) = |content_tokens(A) ∩ content_tokens(B)| / |content_tokens(A) ∪ content_tokens(B)|
```

- Ứng viên C: J < 0,1 với **mọi** câu trong 50k (dùng inverted index trên token nội dung có df ≤ 4.000).
- Khử gần trùng nội bộ C: J < 0,6.
- Kết quả: 117 câu hợp lệ, chọn 100 (seed 42); Jaccard nội dung lớn nhất của một câu C với 50k là **0,0769** (< 0,1).

Hạn chế đã biết: câu C rất ngắn (trung bình 5,84 token so với 15,85 token của A). Vì vậy phân tích A↔C có thêm **so khớp độ dài** (`results/length_matched_AC.json`).

## 3. Nguồn C và giấy phép

- Manythings `vie-eng.zip` (Tatoeba, CC-BY 2.0 France) — 12.628 dòng.
- OPUS Tatoeba `v2026-07-08` moses `en-vi` — 9.846 cặp.
- Hợp nhất, dedupe theo câu VI, lọc độ dài 4–40 token VI / 3–40 token EN.

Không dùng câu nào từ `fr_en_train.txt` để tạo A/B1/B2/C.

## 4. Leakage memory ↔ test

FE_AGAL_v2 kiểm tra 300.000 câu Anh trong memory FR–EN so với 100 reference của B2:
- Ngưỡng: Jaccard token **đầy đủ** (lowercase, `\w+`) > 0,5 (đúng theo đề bài).
- Kết quả: **29 memory entry** gần trùng test (xem `logs/leakage_B2_*.json`).
- Các entry này bị **mask** trong kịch bản `B2_filtered`; kết luận về tác dụng auxiliary dựa trên B2_filtered, B2_all chỉ để tham khảo mức ảnh hưởng của rò rỉ.
