# So sánh kịch bản (nhiệm vụ chính: VI→EN)

BLEU/TER: corpus-level sacrebleu. chrF/chrF++: custom (trung bình câu). METEOR: nltk ×100. DictCov: % token hypothesis nằm trong vocab EN của kho FR–EN.

| Scenario | BLEU | chrF | chrF++ | METEOR | TER | EM | TokenF1 | DictCov | Latency (ms) | LengthRatio |
|---|---|---|---|---|---|---|---|---|---|---|
| A · in-domain (aux ON) | 30.14 | 51.12 | 40.13 | 43.52 | 60.19 | 3.00 | 0.51 | 98.06 | 6944 | 1.041 |
| B1 · semi (aux OFF) | 20.00 | 43.83 | 31.65 | 34.15 | 73.30 | 0.00 | 0.43 | 97.15 | 5651 | 1.124 |
| B2_all · semi (aux ON) | 20.28 | 44.14 | 31.82 | 33.99 | 73.46 | 1.00 | 0.43 | 97.21 | 8002 | 1.127 |
| B2_filtered · semi (aux ON, no leak) | 19.86 | 44.09 | 31.76 | 34.04 | 72.92 | 1.00 | 0.43 | 97.15 | 6815 | 1.128 |
| C · out-of-domain (aux ON) | 44.79 | 64.53 | 54.80 | 56.74 | 45.45 | 21.00 | 0.66 | 98.25 | 4030 | 1.014 |
