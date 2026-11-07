# FE_AGAL_v2 — Recon log (framing đúng: VI→EN là nhiệm vụ chính)

Ngày: 2026-11-07. Người thực hiện: AI agent (senior ML engineer / evaluator).
Mục đích: khảo sát code core **trước khi** viết pipeline, trả lời 6 câu Q1–Q6.
Mọi kết luận dưới đây đọc trực tiếp từ repo, kèm vị trí code.

## 0. Trạng thái code core (ràng buộc [0.1])

`git diff --stat -- nhom3/engine.py nhom3/api.py training/mtm models/nhom3_mt app.py` → **rỗng** tại thời điểm bắt đầu.
Pipeline v2 chỉ `import` `nhom3.engine.Translator`; không sửa file core nào.

## 1. Các file đã khảo sát

| File | Vai trò |
|---|---|
| `nhom3/engine.py` | Bộ dịch runtime 6 bước (tiền xử lý → draft → gate → retrieval → rerank → refine) |
| `nhom3/api.py` | Flask blueprint: `/api/translate`, `/api/status`, `/api/warmup`, `/api/results` |
| `nhom3/langdetect.py` | Heuristic detect vi/fr/en cho ô "Tự động" của API |
| `models/nhom3_mt/meta.json` | Meta model: base, langs, beams, encoder, reranker, top_k |
| `models/nhom3_mt/thresholds.json` | Ngưỡng gate riêng cho `vi` và `fr` |
| `models/nhom3_mt/memory/pairs.json` | Kho câu FR–EN (300.000 cặp) + `emb.npy` (embedding LaBSE của câu FR) |
| `training/configs/default.yaml` | Config huấn luyện (data, model, train, retrieval, gate, experiments) |
| `training/mtm/memory.py` | Code build kho FR–EN từ `fr_en_train.jsonl` |
| `training/mtm/evaluate.py` | Evaluator gốc của training (BLEU/chrF/COMET, vi + fr) |
| `vi_en_train.txt` | 50.000 cặp VI–EN (dataset chính) |
| `fr_en_train.txt` | 300.000 cặp FR–EN (chỉ là nguồn auxiliary/memory) |
| `README.md` | Xác nhận kết quả chính thức VI→EN (BLEU ~26–27 trên 1.000 câu PhoMT) |

## 2. Chi tiết kỹ thuật quan trọng

### 2.1. `nhom3/engine.py`
- `LANG_CODE = {"vi": "vi_VN", "fr": "fr_XX", "en": "en_XX"}` (dòng 23).
- `Translator.translate(text, lang, mode="full")` (dòng 232) — **có tham số `lang` nguồn**; caller truyền `"vi"` hoặc `"fr"`.
- `encode(text, lang, evidence=None)` (dòng 136–143): đặt `self.tok.src_lang = LANG_CODE[lang]` → mBART nhận **language token nguồn** (`vi_VN` / `fr_XX`); nếu có evidence thì nối `evidence + </s>` vào input.
- `generate(...)` (dòng 145–167): luôn ép đích `forced_bos_token_id = en_XX` → **đầu ra luôn là tiếng Anh**.
- `translate_sentence(...)` (dòng 192–230):
  - `draft, C = generate(ids)` — dịch nháp VI→EN.
  - `R = rarity(x, lang)` — độ hiếm token nguồn (dùng `rarity.json`).
  - `hard = is_hard(C, R, thresholds[lang])` nếu `mode="full"`; `mode="draft"` → luôn `hard=False` (không đụng memory); `mode="always"` → luôn `hard=True` và luôn dùng evidence.
  - Nếu `hard`: `retrieve(x, top_k=5)` bằng LaBSE (cross-lingual), query là **câu nguồn (Việt hoặc Pháp)**, kho là 300k câu **Pháp**; `rerank` chọn câu **Anh** tốt nhất; nếu `rr[best] >= delta[lang]` (hoặc mode always) → `generate(encode(x, lang, evidence))` để dịch lại.
- `ensure_retrieval()` (dòng 118–133): nạp `memory/pairs.json` (`fr`, `en`), `memory/emb.npy`, LaBSE encoder, reranker `BAAI/bge-reranker-v2-m3`.

### 2.2. `nhom3/api.py`
- `POST /api/translate` nhận `{text, source: "vi"|"fr"|"auto", mode: "full"|"draft"|"always"}`; trả `translation`, `sentences[]` (kèm `draft`, `C`, `R`, `hard`, `candidates`, `evidence`, `evidence_used`), `elapsed_ms`.
- Nếu `source="auto"` thì dùng `langdetect.detect`; chỉ chấp nhận vi/fr; từ chối nếu là ngôn ngữ khác.
- Endpoint không chứa logic dịch riêng — gọi `Translator.translate`.

### 2.3. `models/nhom3_mt/meta.json`
```json
{"base": "facebook/mbart-large-50", "langs": ["vi", "fr"],
 "num_beams": 4, "max_new_tokens": 160, "max_src_len": 128, "max_evidence_len": 96,
 "encoder": "sentence-transformers/LaBSE", "reranker": "BAAI/bge-reranker-v2-m3",
 "top_k": 5, "rarity_tokenizer": "underthesea", "dev_chrF": 45.48}
```
→ Model là **bilingual nguồn VI + FR → đích EN**, không phải FR-only.

### 2.4. `models/nhom3_mt/thresholds.json`
- `vi`: τ=0.7, ρ=0.25, δ=0.8, hard_frac dev = 0.944, evidence_frac dev = 0.285.
- `fr`: τ=0.6, ρ=0.1, δ=0.0, hard_frac dev = 0.642, evidence_frac dev = 0.642.
→ Với câu Việt, ~94% câu bị coi là "khó" → có tra kho; ~28,5% thực sự dùng evidence để dịch lại.

### 2.5. `models/nhom3_mt/memory/pairs.json`
- Dict 2 khóa `fr` (300.000) và `en` (300.000), không phải VI–EN.
- Nội dung trùng với `fr_en_train.txt` (dòng đầu khớp nguyên văn) — xem `training/mtm/memory.py:75-81`.
- `emb.npy` là embedding LaBSE **của câu FR** (kích thước 300.000 × 384).

### 2.6. `training/configs/default.yaml`
- Không có config dictionary / `use_fr_en_bridge` / flag bật-tắt auxiliary.
- Có mục `retrieval` (LaBSE + reranker + top_k=5) và `gate` (τ, ρ, δ grid).
- `experiments`: A (LoRA, langs [vi]), B (languages [vi, fr]), C (multilingual + augment).
- `stage2_vi_share: 0.8` — giai đoạn 2 lấy Việt–Anh là chính, 20% Pháp–Anh.

### 2.7. `training/mtm/evaluate.py`
- `evaluate()` lặp `for lang in meta["langs"]` — nếu model có `vi` thì **có** đánh giá VI→EN (`lang="vi"`), không riêng một hàm "VI→EN" độc lập.
- Đây là evaluator của training pipeline (cần checkpoint/workdir), không dùng trực tiếp cho FE_AGAL_v2; FE_AGAL_v2 dùng `nhom3.engine` (runtime đã export).

## 3. Trả lời 6 câu hỏi bắt buộc

**Q1. Model dịch VI→EN bằng cách nào? (direct hay pivot qua FR?)**
**Direct.** mBART-50 + LoRA decode trực tiếp từ câu Việt sang tiếng Anh: `src_lang=vi_VN`, forced BOS `en_XX` (`engine.py:23,137,153`). **Không có pivot VI→FR→EN**. FR–EN memory chỉ là *bằng chứng* (evidence) được nối vào input để dịch lại, không phải bước dịch trung gian. README gốc cũng ghi "Không dịch qua tiếng Pháp: chỉ lấy câu Anh đi kèm làm bằng chứng" (`training/mtm/memory.py:6`).

**Q2. Khi cho input tiếng Việt, model dùng memory FR–EN thế nào?**
Chỉ khi câu bị gate là "khó" (`C < τ` hoặc `R > ρ`) **và** `mode != "draft"`:
1. LaBSE encode câu Việt → so cosine với embedding 300k câu Pháp (cross-lingual retrieval);
2. lấy top-5 câu Pháp, reranker chấm cặp (câu Việt, câu Anh tương ứng) → chọn câu Anh điểm cao nhất;
3. nếu `score >= δ` (hoặc mode `always`) → nối câu Anh đó vào encoder input `[vi_VN] x </s> e* </s>` rồi decode lại ra tiếng Anh (`engine.py:211-228`).
Với câu dễ hoặc `mode="draft"`: memory **không được gọi**.

**Q3. Có flag/parameter bật/tắt FR–EN auxiliary không? Tên là gì?**
Không có flag tên `use_fr_en_bridge`. **Cơ chế bật/tắt có sẵn là tham số `mode` của `Translator.translate`**:
- `mode="draft"` → **tắt** auxiliary (không bao giờ tra memory);
- `mode="full"` → **bật** auxiliary theo gate (cấu hình production);
- `mode="always"` → bật cưỡng bức cho mọi câu (dùng cho chẩn đoán).
Vì vậy FE_AGAL_v2 định nghĩa B1 = `draft`, B2 = `full` (mặc định production). Ngoài ra, để kiểm soát rò rỉ, FE_AGAL_v2 thêm biến thể **B2_filtered**: giữ nguyên `mode="full"` nhưng mask các memory index có câu Anh gần trùng test (Jaccard token > 0.5) trước bước top-k (override `retrieve` ở phía evaluator, không sửa core). Lý do: không có tham số chính thức để loại trừ memory; mask index là cách can thiệp tối thiểu, không thay đổi model/cổng.

**Q4. File 50k `vi_en_train.txt` có cột gì? Format ra sao? Số dòng chính xác?**
- **50.000 dòng** (đếm `wc -l` = 50000).
- UTF-8, **tab-separated**, **không header**: `câu tiếng Việt \t câu tiếng Anh`.
- Mỗi dòng gồm đúng 2 trường (source VI, target EN).
- **Không có split train/dev/test** → mọi test set trích từ file này có thể **overlap với dữ liệu train**; FE_AGAL_v2 ghi rõ giả định này và coi kịch bản A là "in-domain, có thể đã thấy khi train".

**Q5. Model có biết phân biệt VI vs FR không?**
**Có**, qua **language token nguồn** của mBART: `tok.src_lang = vi_VN` / `fr_XX` (`engine.py:23,137`). Đích luôn `en_XX`. Engine **không tự detect**; caller phải truyền `lang` đúng. API có heuristic `langdetect.detect` chỉ khi `source="auto"`. Không có prefix ngôn ngữ trong text; nếu truyền sai `lang`, model sẽ dịch sai hướng.

**Q6. Memory `pairs.json` là gì so với training data? Có trùng lặp không?**
- Là **kho 300.000 cặp FR–EN**, lấy từ chính `fr_en_train.txt` (TED2020 + News-Commentary), dùng làm bằng chứng retrieval lúc inference.
- **Không phải** dữ liệu VI–EN và **không phải** test set. Khác hoàn toàn `vi_en_train.txt` (50k).
- Về trùng lặp: README gốc cảnh báo PhoMT (VI–EN) và TED2020 (FR–EN memory) **cùng lấy từ các bài TED**, nên câu Anh trong memory có thể gần trùng đáp án test VI–EN (lọc cũ chỉ loại trùng chính xác). FE_AGAL_v2 sẽ **đo lại bằng Jaccard token > 0.5** và báo cáo B2_all vs B2_filtered.

## 4. Kết luận recon

✅ Model **hỗ trợ VI→EN thật** (không phải chỉ FR→EN): `langs=["vi","fr"]`, source token `vi_VN`, README có kết quả VI→EN BLEU ~26–27 trên 1.000 câu PhoMT, và benchmark nhanh dịch đúng câu Việt sang Anh.

→ **Tiếp tục** pipeline v2 với framing: nhiệm vụ chính VI→EN, FR–EN memory là auxiliary có thể bật/tắt bằng `mode`.

### Quyết định kỹ thuật của evaluator (ghi rõ để tái lập)
1. **A**: 100 câu (CPU chậm, xem log benchmark) trích stratified từ 50k, seed 42, `mode="full"`.
2. **B1/B2**: cùng 100 câu biến thể (B1.jsonl ≡ B2.jsonl), B1 `mode="draft"`, B2 `mode="full"`; thêm **B2_filtered** (mask memory trùng test) làm căn cứ kết luận.
3. **C**: 100 câu VI từ Tatoeba VI–EN (manythings.org, CC-BY), lọc Jaccard token < 0.1 với mọi câu trong 50k, `mode="full"`.
4. **D**: không chạy — model chỉ decode đích `en_XX`, không hỗ trợ VI→FR → "not applicable".
5. **DictionaryCoverage** định nghĩa lại: `% token của hypothesis EN nằm trong từ vựng EN của kho FR–EN` (công thức ghi trong `run_all.py` và report). Kèm diagnostic `EvidenceOverlap` = `|tokens(hyp) ∩ tokens(evidence)| / |tokens(hyp)|` (0 nếu không dùng evidence).
