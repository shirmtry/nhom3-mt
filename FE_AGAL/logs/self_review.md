# FE_AGAL self-review

- [x] Reconnaissance log exists at `logs/00_recon.md`.
- [x] Scenario YAML loads successfully in the runner.
- [x] `run_all.py --dry-run` completed with the original model.
- [x] Full/reduced run produced A, B1, B2, and C output directories.
- [x] Metric unit tests pass: 6 tests.
- [x] `analyze.py` produced comparison CSV/Markdown and figures.
- [x] `FINAL_REPORT.md` and `why_50k.md` exist.
- [x] Timestamped run logs and frozen `logs/env.txt` exist.
- [x] No original source/model/data file was modified.

## Compute constraint
A full configured run was reduced to 50/50/20 because CPU generation exceeded the available execution window. A contained 48 unique-source pairs because three sampled rows collided under unique-source filtering.

## Dependency caveat
COMET was unavailable and no cached BERTScore model was reachable offline; these columns use zero fallback as logged.

## Result counts
- A_in_domain: 48
- B1_semi_no_dict: 50
- B2_semi_with_dict: 50
- C_out_of_domain: 20
