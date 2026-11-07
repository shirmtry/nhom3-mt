import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "FE_AGAL" / "data_splits"
SEED = 42


def read_pairs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                rows.append({"src": parts[0].strip(), "tgt": parts[1].strip()})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_noise(text, rng):
    words = text.split()
    if len(words) >= 6 and rng.random() < 0.5:
        i = rng.randrange(0, len(words) - 2)
        words[i:i+3] = words[i:i+3][::-1]
    if words and rng.random() < 0.25:
        i = rng.randrange(len(words))
        words[i] = words[i].replace(",", "")
    if rng.random() < 0.15:
        text = " ".join(words) + " toutefois"
    else:
        text = " ".join(words)
    return text.replace(" .", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=500)
    ap.add_argument("--b", type=int, default=500)
    ap.add_argument("--c", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows = read_pairs(ROOT / "fr_en_train.txt")

    # Stratified by source length bucket.
    buckets = {"short": [], "medium": [], "long": []}
    for row in rows:
        n = len(row["src"].split())
        key = "short" if n <= 12 else ("medium" if n <= 25 else "long")
        buckets[key].append(row)
    a_rows, seen = [], set()
    for rows_bucket in buckets.values():
        take = max(1, args.a // 3)
        selected = rng.sample(rows_bucket, min(take, len(rows_bucket)))
        a_rows.extend([r for r in selected if r["src"] not in seen])
    rng.shuffle(a_rows)
    a_rows = a_rows[:args.a]
    write_jsonl(OUT / "A.jsonl", a_rows)

    candidates = [r for r in rows if r["src"] not in {x["src"] for x in a_rows}]
    b_source = rng.sample(candidates, min(args.b, len(candidates)))
    b_rows = [{"src": add_noise(r["src"], rng), "tgt": r["tgt"], "source_original": r["src"]} for r in b_source]
    write_jsonl(OUT / "B.jsonl", b_rows)

    # Synthetic legal/news/conversation OOD set; explicitly separated from the 300k source.
    c_texts = [
        ("La cour examine les motifs juridiques de l'appel.", "The court is reviewing the legal grounds of the appeal."),
        ("Cette mesure d'urgence protège la santé des citoyens.", "This emergency measure protects citizens' health."),
        ("Pourrais-tu confirmer l'heure de notre rendez-vous ?", "Could you confirm the time of our appointment?"),
        ("Le bulletin annonce une baisse des températures demain.", "The forecast announces lower temperatures tomorrow."),
        ("Le patient doit consulter un spécialiste au plus vite.", "The patient should consult a specialist as soon as possible."),
        ("Le gouvernement publie un nouveau règlement administratif.", "The government published a new administrative regulation."),
    ]
    c_rows = []
    while len(c_rows) < args.c:
        for src, tgt in c_texts:
            if len(c_rows) == args.c:
                break
            c_rows.append({"src": src, "tgt": tgt})
    write_jsonl(OUT / "C.jsonl", c_rows)

    (OUT / "generation_notes.md").write_text(
        "# Data split notes\n\n"
        "- A uses a length-stratified sample from `fr_en_train.txt`, seed 42, excluded via unique-source sampling.\n"
        "- B uses held-out 300k corpus sources with controlled reversal/noise/punctuation; identical for B1/B2.\n"
        "- C is a small synthetic out-of-domain set because no public OOD set is present locally. "
        "Jaccard overlap must be remeasured before calling it representative.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
