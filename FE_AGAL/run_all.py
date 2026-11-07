import argparse
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "metrics"))

from nhom3.engine import Translator
from bleu import bleu
from chrf import chrf, chrf_plus
from meteor import meteor
from ter import ter
from latency import exact_match, token_f1
from dict_coverage import dictionary_coverage, load_memory_dictionary
from statistics import summarize
from comet import comet_scores
from bertscore import bertscore_scores


def setup_logger(timestamp):
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("FE_AGAL")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir / f"run_{timestamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_rows(rows, translator, use_dictionary, logger):
    dictionary = load_memory_dictionary(ROOT / "models" / "nhom3_mt" / "memory" / "pairs.json")
    results, latencies = [], []
    t_all = time.perf_counter()
    mode = "full" if use_dictionary else "draft"
    for row in rows:
        t0 = time.perf_counter()
        out = translator.translate(row["src"], "fr", mode=mode)
        elapsed = (time.perf_counter() - t0) * 1000
        hypothesis = out["translation"]
        latencies.append(elapsed)
        bleus = bleu(hypothesis, row["tgt"])
        precision, recall, f1 = token_f1(hypothesis, row["tgt"])
        results.append({
            "src": row["src"], "ref": row["tgt"], "hyp": hypothesis, "reference": row["tgt"],
            "BLEU": bleus, "chrF": chrf(hypothesis, row["tgt"]),
            "chrF++": chrf_plus(hypothesis, row["tgt"]), "METEOR": meteor(hypothesis, row["tgt"]),
            "TER": ter(hypothesis, row["tgt"]), "COMET": 0.0, "BERTScore": 0.0,
            "EM": exact_match(hypothesis, row["tgt"]), "TokenPrecision": precision,
            "TokenRecall": recall, "TokenF1": f1,
            "DictionaryCoverage": dictionary_coverage(hypothesis, dictionary),
            "LatencyMs": elapsed, "LengthRatio": len(hypothesis.split()) / max(len(row["tgt"].split()), 1),
        })
    elapsed_all = time.perf_counter() - t_all
    sources = [r["src"] for r in results]
    refs = [r["ref"] for r in results]
    hyps = [r["hyp"] for r in results]
    comet_values = comet_scores(sources, hyps, refs)
    if comet_values is None:
        logger.warning("COMET unavailable; using BERTScore fallback/zero when unavailable")
        comet_values = bertscore_scores(hyps, refs)
        if comet_values is None:
            comet_values = [0.0] * len(results)
    for row, comet_value in zip(results, comet_values):
        row["COMET"] = comet_value
    logger.info("Translated %d sentences in %.3fs", len(rows), elapsed_all)
    return results


def write_outputs(scenario, rows, result_rows):
    out_dir = HERE / "results" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hypotheses.jsonl", "w", encoding="utf-8") as f:
        for src, row in zip(rows, result_rows):
            f.write(json.dumps({"src": src["src"], "hyp": row["hyp"], "ref": row["ref"]}, ensure_ascii=False) + "\n")
    with open(out_dir / "per_sample.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    metrics = {}
    throughput = len(result_rows) / (sum(r["LatencyMs"] for r in result_rows) / 1000)
    for key in ["BLEU", "chrF", "chrF++", "METEOR", "TER", "COMET", "BERTScore", "EM",
                "TokenPrecision", "TokenRecall", "TokenF1", "DictionaryCoverage", "LatencyMs", "LengthRatio"]:
        metrics[key] = summarize([r[key] for r in result_rows])
    metrics["Throughput"] = {"mean": throughput, "std": 0.0, "ci95_low": throughput, "ci95_high": throughput}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(timestamp)
    random.seed(42)
    np.random.seed(42)
    logger.info("FE_AGAL run seed=42 timestamp=%s dry_run=%s", timestamp, args.dry_run)

    config = yaml.safe_load((HERE / "test_scenarios.yaml").read_text(encoding="utf-8"))
    if not (ROOT / "models" / "nhom3_mt" / "model" / "config.json").exists():
        logger.error("Original model not found")
        raise SystemExit(1)
    translator = Translator()
    for scenario, cfg in config["scenarios"].items():
        input_path = ROOT / cfg["input_path"].replace("/", "\\") if "\\" in cfg["input_path"] else ROOT / cfg["input_path"]
        rows = read_jsonl(input_path)
        if args.dry_run:
            rows = rows[:5]
        logger.info("Scenario %s samples=%d dictionary=%s", scenario, len(rows), cfg["use_dictionary"])
        result_rows = evaluate_rows(rows, translator, cfg["use_dictionary"], logger)
        write_outputs(scenario, rows, result_rows)
        logger.info("Saved scenario %s", scenario)


if __name__ == "__main__":
    main()
