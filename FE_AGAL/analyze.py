import csv
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_csv(name):
    path = RESULTS / name / "per_sample.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def qualitative(name, rows):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r.get("COMET", 0) or 0) + float(r.get("chrF", 0) or 0))
    worst, best = rows[:10], sorted(rows, key=lambda r: float(r.get("COMET", 0) or 0) + float(r.get("chrF", 0) or 0), reverse=True)[:10]
    lines = ["# Qualitative review", "", "## 10 worst", ""]
    for r in worst:
        lines.append(f"- FR: {r.get('src','')}\\n  EN ref: {r.get('ref','')}\\n  Hyp: {r.get('hyp','')}")
    lines += ["", "## 10 best", ""]
    for r in best:
        lines.append(f"- FR: {r.get('src','')}\\n  EN ref: {r.get('ref','')}\\n  Hyp: {r.get('hyp','')}")
    lines += ["", "## Failure mode hypotheses", "",
              "- Missing word: content tokens absent from the hypothesis.",
              "- Hallucination: extra unsupported English tokens.",
              "- Word order: meaning mostly preserved but modifier/adverb order differs.",
              "- Dictionary miss: rare domain word not represented in the memory fallback.",
              "- Noise sensitivity: controlled mutation propagates translation errors in B."]
    (RESULTS / name / "qualitative.md").write_text("\n".join(lines), encoding="utf-8")


def paired_test(a, b):
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return None


def main():
    scenarios = ["A_in_domain", "B1_semi_no_dict", "B2_semi_with_dict", "C_out_of_domain"]
    summary_rows = []
    for name in scenarios:
        p = RESULTS / name / "metrics.json"
        if not p.exists():
            continue
        metrics = json.loads(p.read_text(encoding="utf-8"))
        summary_rows.append({"scenario": name, **{k: round(v["mean"], 4) for k, v in metrics.items()}})
        qualitative(name, load_csv(name))
    if not summary_rows:
        print("No results found; run FE_AGAL/run_all.py first.")
        return

    csv_path = RESULTS / "comparison.csv"
    preferred = ["BLEU", "chrF", "chrF++", "METEOR", "TER", "EM", "TokenPrecision",
                 "TokenRecall", "TokenF1", "DictionaryCoverage", "LatencyMs", "LengthRatio", "Throughput"]
    keys = ["scenario"] + [key for key in preferred if any(key in row for row in summary_rows)]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["# Scenario comparison", "", "| Scenario | " + " | ".join(k for k in keys if k != "scenario") + " |",
             "|---|" + "|".join("---" for _ in keys if _ != "scenario") + "|"]
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    (RESULTS / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    b1, b2 = load_csv("B1_semi_no_dict"), load_csv("B2_semi_with_dict")
    if b1 and b2:
        impact = {}
        for key in ["BLEU", "chrF", "METEOR", "LatencyMs", "DictionaryCoverage"]:
            x = np.array([float(r[key]) for r in b1])
            y = np.array([float(r[key]) for r in b2])
            n = min(len(x), len(y))
            impact[key] = {"delta": float(y[:n].mean()-x[:n].mean()), "p_value": paired_test(y[:n]-x[:n], np.zeros(n))}
        (RESULTS / "dictionary_impact.json").write_text(json.dumps(impact, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        labels = [r["scenario"] for r in summary_rows]
        for metric in ["BLEU", "chrF", "METEOR", "TER", "DictionaryCoverage"]:
            if metric not in summary_rows[0]:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar([x[:12] for x in labels], [r[metric] for r in summary_rows])
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=.3)
            fig.tight_layout()
            fig.savefig(RESULTS / "figures" / f"{metric.lower()}_bar.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        print(f"Plot generation unavailable: {exc}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    main()
