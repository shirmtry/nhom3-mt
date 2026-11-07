# Self-review checklist — FE_AGAL_v2

Ngày kiểm: 2026-11-07. Mỗi mục ghi rõ **cách kiểm** và **bằng chứng**; không đánh dấu PASS nếu không có artefact.

## A. Ràng buộc bắt buộc (Phần 0)

| # | Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|---|
| A1 | Không sửa `nhom3/engine.py` | PASS | `git diff --stat -- nhom3/engine.py` → rỗng |
| A2 | Không sửa `nhom3/api.py` | PASS | `git diff --stat -- nhom3/api.py` → rỗng |
| A3 | Không sửa `training/mtm/*` | PASS | `git diff --stat -- training/mtm` → rỗng |
| A4 | Không sửa `models/nhom3_mt/*` | PASS | `git diff --stat -- models/nhom3_mt` → rỗng |
| A5 | Không sửa `app.py` (root) | PASS | `git diff --stat -- app.py` → rỗng |
| A6 | Chỉ import/gọi engine | PASS | `run_all.py` dùng `from nhom3.engine import Translator`; không monkey-patch core, chỉ override `translator.retrieve` tạm thời cho B2_filtered rồi khôi phục |
| A7 | Tạo `FE_AGAL_v2/` song song, không xóa `FE_AGAL/` | PASS | Cả hai folder tồn tại; `FE_AGAL/` không bị đụng |
| A8 | Số liệu viết lại từ đầu | PASS | Mọi `metrics.json` sinh bởi `run_all.py` ngày 2026-11-07; không copy từ `FE_AGAL/results` |
| A9 | Model thực sự hỗ trợ VI→EN trước khi tiếp tục | PASS | `logs/00_recon_v2.md` Q1–Q6; dry-run và full run dịch đúng VI→EN |

**Lệnh kiểm tra tổng:**
```bash
git diff --stat -- nhom3/engine.py nhom3/api.py training/mtm models/nhom3_mt app.py
# kết quả: rỗng (0 dòng)
```

## B. Recon & dữ liệu

| # | Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|---|
| B1 | Trả lời Q1–Q6 bằng code line cụ thể | PASS | `logs/00_recon_v2.md` |
| B2 | Xác nhận 50k dòng, format TSV VI–EN | PASS | `data_splits.py::read_tsv` assert 50.000; `why_50k_v2.md` §6.1 |
| B3 | Xác nhận FR–EN chỉ là auxiliary | PASS | `report/architecture.md` §1.3; `engine.py:204-226` |
| B4 | Xác nhận có cơ chế bật/tắt | PASS | `mode="draft"` (tắt) vs `"full"` (production); `architecture.md` §1.3 |
| B5 | Không phân biệt VI/FR tự động | PASS | Đã ghi trong recon + Limitations #7: caller phải truyền `lang="vi"` |
| B6 | A/B1/B2/C đủ 100 câu | PASS | `wc -l data_splits/*.jsonl` → 100 mỗi file |
| B7 | B1 ≡ B2 tuyệt đối | PASS | `cmp` rỗng; SHA-256 16 ký tự `ee3b240f5db2fe1f` cho cả hai |
| B8 | C lọc Jaccard < 0,1 và ghi sai lệch | PASS | `C_max_jaccard = 0,0769`; sai lệch full-token → content-token ghi tại `logs/data_notes.md` |
| B9 | Seed cố định | PASS | seed 42 trong `data_splits.py`, `run_all.py`, bootstrap |
| B10 | Kịch bản D đánh dấu không áp dụng | PASS | `FINAL_REPORT_v2.md` §3 và Limitations #12 — pipeline không có VI→FR→EN |

## C. Chạy & metric

| # | Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|---|
| C1 | Dry-run 5 câu trước full run | PASS | `logs/run_20261107_182315.log:1` và `..._182408.log:1` (`dry=True · n=5`) |
| C2 | Load qua `nhom3.engine`, không batch giả | PASS | `run_all.py` khởi tạo `Translator()`, gọi `translate_sentence`/`translate`; engine không hỗ trợ batch → chạy tuần tự, đã ghi limitation |
| C3 | Có log từng câu | PASS | `logs/run_20261107_183251.log` in tiến độ 10 câu/lần; `full_run_console.log` |
| C4 | Đủ 5 kịch bản A/B1/B2/B2_filtered/C | PASS | Mỗi folder có `hypotheses.jsonl`, `per_sample.csv`, `metrics.json` |
| C5 | Metric bắt buộc: BLEU, chrF, chrF++, METEOR, TER, EM, TokenP/R/F1, DictCov, latency, throughput, length ratio | PASS | `metrics.json` từng scenario chứa đủ; corpus BLEU/TER/chrF riêng |
| C6 | Không dùng COMET/BERTScore | PASS | Không import/sử dụng; đã xóa `metrics/comet.py`, `metrics/bertscore.py`; figure chỉ 3 loại không liên quan |
| C7 | DictionaryCoverage có công thức ghi rõ | PASS | `FINAL_REPORT_v2.md` §4 + `run_all.py`; = % token hypothesis thuộc vocab EN của memory |
| C8 | Test metric pass | PASS | `pytest metrics/test_metrics.py -q` → **6 passed** |
| C9 | Môi trường được ghi lại | PASS | `logs/env.txt` (Python 3.14.7, torch 2.15.0+cpu, sacrebleu 2.6.0...) |

## D. Phân tích

| # | Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|---|
| D1 | Bảng so sánh có **Scenario** là cột đầu | PASS | `results/comparison.md` header bắt đầu `| Scenario | BLEU |...` |
| D2 | Paired bootstrap B1 vs B2 có p-value | PASS | `results/bootstrap_B1_B2.json`: chrF p=0,2952; BLEU p=0,9032; 10.000 lần, seed 42 |
| D3 | Kiểm tra rò rỉ B2_all vs B2_filtered | PASS | 29 memory entry / 13 câu test bị mask; B2 vs B2_filtered: 98 hoà, 2 thay đổi, p=0,274 |
| D4 | Kết luận dùng B2_filtered | PASS | `FINAL_REPORT_v2.md` §5; B1 vs B2_filtered p=0,4412 |
| D5 | Trả lời 5 câu hỏi phân tích | PASS | `results/analysis.md` có đúng 5 mục Q1–Q5 |
| D6 | 5 tốt nhất + 5 tệ nhất mỗi kịch bản | PASS | `results/qualitative.md` có 10 mục (5 kịch bản × 2) |
| D7 | Failure modes phân loại | PASS | `results/failure_analysis.json` (mistranslation, hallucination, VI token, paraphrase, missing word) |
| D8 | Hình không có COMET/BERTScore | PASS | `results/figures/`: bar metrics, boxplot chrF, histogram latency |
| D9 | Xử lý confound độ dài A vs C | PASS | `results/length_matched_AC.json`: 60 câu A ngắn vs C |
| D10 | Hiệu năng (latency/throughput) | PASS | `metrics.json`: 4,0–8,0 s/câu; 0,125–0,248 câu/s |

## E. Báo cáo

| # | Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|---|
| E1 | `report/architecture.md` có Mermaid + giải thích FR–EN auxiliary | PASS | 2 diagram Mermaid; §1.3 giải thích retrieval/rerank/refine, không pivot |
| E2 | `report/why_50k_v2.md` trung thực, có số liệu | PASS | Vocab 21.882; phủ 25,3%→100%; OOV C 1,37%; không có 100k |
| E3 | `report/FINAL_REPORT_v2.md` đủ 10 phần | PASS | Executive Summary → Limitations → Recommendations + phụ lục |
| E4 | Executive Summary đúng 10 dòng | PASS | §1 có 10 mục đánh số |
| E5 | Limitations trung thực | PASS | 13 mục, gồm N=100, A overlap train, B1 synthetic, leakage, C ngắn/nhãn sai, custom chrF, CPU no-batch, no COMET, D N/A |
| E6 | Khuyến nghị 3–5+ | PASS | §10 có 6 khuyến nghị hành động |
| E7 | Framing đúng: VI→EN chính, FR–EN phụ | PASS | Toàn bộ báo cáo; không gọi FR–EN là nhiệm vụ thứ hai |

## F. Những điểm KHÔNG đạt / cần lưu ý

Không có mục bắt buộc nào FAIL. Các giới hạn đã được ghi minh bạch trong báo cáo:

1. **N=100 là exploratory** — không đủ power cho hiệu ứng < 1 chrF; p > 0,05 không đồng nghĩa "không có tác động".
2. **A trích từ train** — không phải test độc lập (file 50k không có split).
3. **B1 là nhiễu tổng hợp** — chưa có human evaluation / nhiễu thật.
4. **C có 2/100 câu Tatoeba sai ngôn ngữ** — đã nêu trong Limitations #5 và §6.
5. **Lọc C dùng Jaccard nội dung thay vì full-token** — bắt buộc vì full-token cho 0 ứng viên; đã ghi `logs/data_notes.md`.
6. **chrF custom khác sacrebleu** — chỉ so tương đối trong cùng implementation.
7. **Kịch bản D không chạy được** — pipeline không hỗ trợ VI→FR→EN.

## G. Kết luận self-review

**Tất cả mục bắt buộc PASS.** Artefact đầy đủ, số liệu tái lập được từ `run_all.py` + `analyze.py`, core không bị sửa, framing VI→EN đúng. Báo cáo đã nêu rõ giới hạn và không overclaim.
