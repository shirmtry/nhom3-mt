# Data split notes

- A uses a length-stratified sample from `fr_en_train.txt`, seed 42, excluded via unique-source sampling.
- B uses held-out 300k corpus sources with controlled reversal/noise/punctuation; identical for B1/B2.
- C is a small synthetic out-of-domain set because no public OOD set is present locally. Jaccard overlap must be remeasured before calling it representative.
