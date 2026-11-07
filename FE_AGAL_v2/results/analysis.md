# Phân tích kết quả FE_AGAL_v2 (VI→EN là nhiệm vụ chính)

## Bảng so sánh

| Scenario | BLEU | chrF | chrF++ | METEOR | TER | EM | TokenF1 | DictCov | Latency (ms) | LengthRatio |
|---|---|---|---|---|---|---|---|---|---|---|
| A · in-domain (aux ON) | 30.14 | 51.12 | 40.13 | 43.52 | 60.19 | 3.00 | 0.51 | 98.06 | 6944 | 1.041 |
| B1 · semi (aux OFF) | 20.00 | 43.83 | 31.65 | 34.15 | 73.30 | 0.00 | 0.43 | 97.15 | 5651 | 1.124 |
| B2_all · semi (aux ON) | 20.28 | 44.14 | 31.82 | 33.99 | 73.46 | 1.00 | 0.43 | 97.21 | 8002 | 1.127 |
| B2_filtered · semi (aux ON, no leak) | 19.86 | 44.09 | 31.76 | 34.04 | 72.92 | 1.00 | 0.43 | 97.15 | 6815 | 1.128 |
| C · out-of-domain (aux ON) | 44.79 | 64.53 | 54.80 | 56.74 | 45.45 | 21.00 | 0.66 | 98.25 | 4030 | 1.014 |


## B1 vs B2 (paired bootstrap, cùng input)

- **chrF**: Δ(B2−B1) = +0.415 · 95% CI [-0.351, +1.246] · p = 0.2952 · thắng/hoà/thua = 11/78/11 (n=100)
- **BLEU**: Δ(B2−B1) = -0.052 · 95% CI [-0.960, +0.872] · p = 0.9032 · thắng/hoà/thua = 9/81/10 (n=100)
- Evidence dùng: B1 0% vs B2 31%; câu 'hard': B1 0% vs B2 99%

- B1 vs B2_filtered (chrF): Δ = +0.308, p = 0.4412

## Leakage check
- Memory gần trùng test (Jaccard > 0.5): **29** câu trên 100 test.

## Trả lời 5 câu hỏi

**Q1. A (in-domain) mạnh/yếu thế nào?** BLEU corpus 30.14, chrF 51.12 trên 100 câu. README gốc báo cáo VI→EN BLEU 26,98 / chrF 49,43 trên 1.000 câu PhoMT — cùng mức, khác cỡ mẫu. Lưu ý A trích từ chính file train nên có thể đã thấy dữ liệu.

**Q2. Thêm biến thể (B1 vs A) giảm bao nhiêu?** ΔBLEU -10.14, ΔchrF -7.29. Biến thể gồm mất dấu/typo/đảo token/synonym — mức giảm phản ánh độ bền với nhiễu bề mặt.

**Q3. FR–EN auxiliary có giúp VI→EN không?** không khác biệt có ý nghĩa (p ≥ 0.05): ΔchrF = +0.415, p = 0.2952. Đây là can thiệp thực nghiệm duy nhất khác giữa B1/B2.

**Q4. Out-of-domain (C) giảm bao nhiêu so với A?** ΔBLEU +14.65, ΔchrF +13.42. C là câu Tatoeba ngắn; so khớp độ dài: 60 câu A có ref ≤ 12 token đạt BLEU 25.35 / chrF 47.73 vs C 44.7912 / 64.5333. Độ dài khác nhau là confound, cần đọc thận trọng.

**Q5. Failure modes phổ biến (10 câu tệ nhất/kịch bản).**

- A: mistranslation (lệch từ vựng) (6), hallucination / thêm ý (3), paraphrase xa / thiếu ý (1)
- B1: mistranslation (lệch từ vựng) (6), hallucination / thêm ý (3), VI token còn sót (1)
- B2: mistranslation (lệch từ vựng) (6), hallucination / thêm ý (3), VI token còn sót (1)
- B2_filtered: mistranslation (lệch từ vựng) (6), hallucination / thêm ý (3), VI token còn sót (1)
- C: mistranslation (lệch từ vựng) (8), missing word / rút gọn (1), hallucination / thêm ý (1)

Ví dụ tệ nhất mỗi kịch bản nằm trong `qualitative.md` và `failure_analysis.json`.
