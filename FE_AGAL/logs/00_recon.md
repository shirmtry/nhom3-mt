# FE_AGAL reconnaissance log

| Path | Description | Format | Size |
|---|---|---|---|
| `app.py` | Flask entry point | Python | 3053 bytes |
| `nhom3/engine.py` | Core Translator and French/English inference implementation | Python | ~11.7 KB |
| `nhom3/api.py` | Existing `/api/translate` endpoint and public `Translator` integration | Python | ~10.1 KB |
| `tai_mo_hinh.py` | Downloads/extracts the bundled model | Python | ~5.3 KB |
| `fr_en_train.txt` | French-English parallel corpus | UTF-8 tab-separated text | 300,000 lines, 74,268,705 bytes |
| `vi_en_train.txt` | Vietnamese-English parallel corpus | UTF-8 tab-separated text | 50,000 lines, 8,589,158 bytes |
| `models/nhom3_mt/model/model.safetensors` | Exported mBART translation model | safetensors | 1,222,319,244 bytes |
| `models/nhom3_mt/model/sentencepiece.bpe.model` | SentencePiece tokenizer | binary model | 5,060,506 bytes |
| `models/nhom3_mt/memory/pairs.json` | French-English retrieval evidence / dictionary fallback | JSON | 78,507,370 bytes |
| `training/mtm/evaluate.py` | Existing corpus BLEU, chrF, optional COMET evaluator | Python | present |
| `training/README.md` | Training workflow and evaluation invocation | Markdown | present |

## Decisions and assumptions

- The requested 50k corpus is `vi_en_train.txt` (50,000 Vietnamese-English pairs), not the French-English corpus. The French-English data has 300,000 pairs.
- No dedicated French-English dictionary file was found. `models/nhom3_mt/memory/pairs.json` is used as the dictionary fallback by extracting aligned whole words from its `fr`/`en` pairs; this is a loggable, reproducible proxy and not a second model/API.
- Existing `Translator.translate(text, "fr", mode=...)` is the inference API imported by FE_AGAL.
- Existing evaluation in `training/mtm/evaluate.py` covers corpus BLEU/chrF and optional COMET, but it is not a complete scenario-level BLEU/chrF/TER/METEOR/statistics pipeline; FE_AGAL implements missing wrappers.
- `fr_en_train.txt` appears UTF-8 and tab-delimited, despite Windows console rendering replacement characters in some probes. FE_AGAL reads it as UTF-8.
