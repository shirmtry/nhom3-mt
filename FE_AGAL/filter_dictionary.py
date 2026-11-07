import json
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEED_TEXT = "FE_AGAL B2 dictionary filter"


def tokens(text):
    import re
    return set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower(), flags=re.ASCII))


def jaccard(a, b):
    a, b = tokens(a), tokens(b)
    return len(a & b) / max(len(a | b), 1)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(filename=HERE / "logs" / f"dictionary_filter_{timestamp}.log",
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pairs = json.loads((ROOT / "models" / "nhom3_mt" / "memory" / "pairs.json").read_text(encoding="utf-8"))
    test_rows = [json.loads(line) for line in (HERE / "data_splits" / "B.jsonl").read_text(encoding="utf-8").splitlines() if line]
    test_texts = [row["src"] for row in test_rows]
    original = list(zip(pairs["fr"], pairs["en"]))
    kept, removed = [], []
    for fr, en in original:
        if max(jaccard(fr, src) for src in test_texts) > 0.5:
            removed.append([fr, en])
        else:
            kept.append([fr, en])
    output = {"fr": [row[0] for row in kept], "en": [row[1] for row in kept],
              "source": "models/nhom3_mt/memory/pairs.json", "filter_note": "Removed sentence pairs with Jaccard > 0.5 to any B input",
              "original_count": len(original), "removed_count": len(removed), "kept_count": len(kept)}
    path = HERE / "data_splits" / "dictionary_filtered.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Original=%d removed=%d kept=%d output=%s", len(original), len(removed), len(kept), path)
    print(f"Original={len(original)} removed={len(removed)} kept={len(kept)}")


if __name__ == "__main__":
    main()
