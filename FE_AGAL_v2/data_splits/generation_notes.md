# Data splits (seed 42)

| Split | N | Nguồn |
|---|---|---|
| A | 100 | vi_en_train.txt, stratified theo độ dài, seed 42 |
| B1 | 100 | biến thể A: mất dấu/typo/đảo token cửa sổ 3/synonym; ref giữ nguyên |
| B2 | 100 | BẢN SAO của B1 (verified identical) |
| C | 100 | Tatoeba VI–EN (CC-BY), lọc Jaccard nội dung < 0.1 vs 50k |

## Kiểm tra

```json
{
  "N": {
    "A": 100,
    "B1": 100,
    "B2": 100,
    "C": 100
  },
  "A_len_tokens": {
    "mean": 15.85
  },
  "C_len_tokens": {
    "mean": 5.84
  },
  "C_tatoeba_raw": 16632,
  "C_after_filter": 117,
  "C_max_jaccard": 0.0769,
  "B1_sentences_changed": 100,
  "B1_B2_same_input": true,
  "sha256_16": {
    "A": "b55582f8dc01375b",
    "B1": "ee3b240f5db2fe1f",
    "B2": "ee3b240f5db2fe1f",
    "C": "f270dc422bcac88d"
  }
}
```

## Giả định

- `vi_en_train.txt` KHÔNG có split train/dev/test; A là test trích từ file train nên **có thể overlap với dữ liệu model đã thấy**. Đây là baseline in-domain có chủ đích.
- B1/B2 dùng **cùng input tuyệt đối** (cùng file nội dung); khác biệt duy nhất là config memory.
- **Cường độ nhiễu B1**: mỗi token đổi tối đa 1 lần; xác suất synonym 3%, mất dấu 5%, typo 5%, đảo token cửa sổ 3 là 4%; mọi câu đều có ít nhất 1 thay đổi. Độ tương đồng token trung bình A↔B1 (SequenceMatcher) = 0,805; min = 0,40.
- C là câu Tatoeba đời thường, khác miền PhoMT (tin tức/wiki/TED); câu ngắn 4–40 token.
- **Công thức lọc C**: Jaccard trên tập token nội dung (đã bỏ stopwords VI). Jaccard đầy đủ giữ stopwords cho 0 ứng viên trên toàn bộ 9.345 câu Tatoeba đủ dài, vì stopwords chiếm phần lớn giao của hai câu bất kỳ trong kho 50k. Chi tiết + số liệu thử nghiệm ở `logs/data_notes.md`.
- `C_max_jaccard` là Jaccard nội dung lớn nhất của một câu C với bất kỳ câu nào trong 50k (yêu cầu < 0.1).
