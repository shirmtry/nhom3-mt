"""Chạy 4 kịch bản đánh giá VI→EN (A, B1, B2, C) + B2_filtered.

  python FE_AGAL_v2/run_all.py                 # full (100 câu/kịch bản)
  python FE_AGAL_v2/run_all.py --dry-run       # 5 câu/kịch bản, lưu vào results_dryrun/

Dịch qua đúng API runtime `nhom3.engine.Translator.translate(text, lang="vi", mode=...)`:
  - B1: mode="draft"  → TẮT FR–EN auxiliary (không tra memory)
  - A/B2/C: mode="full" → BẬT auxiliary theo cổng (production)
  - B2_filtered: mode="full" + mask các memory index có câu EN gần trùng test

Định nghĩa DictionaryCoverage (v2): 100 * |token(hyp_EN) ∩ vocab_EN(kho FR–EN)| / |token(hyp_EN)|.
  - vocab_EN = tập token (lowercase) xuất hiện trong trường "en" của memory/pairs.json.
Diagnostic thêm: EvidenceOverlap = 100 * |token(hyp) ∩ token(evidence)| / |token(hyp)| (0 nếu không dùng evidence).
"""
import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sacrebleu
import yaml

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "metrics"))

from nhom3.engine import Translator  # noqa: E402
from chrf import chrf, chrf_plus  # noqa: E402

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokens(text):
    return [t.lower() for t in WORD.findall(text or "")]


def setup_logger(timestamp):
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("FE_AGAL_v2")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    h = logging.FileHandler(HERE / "logs" / f"run_{timestamp}.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(sh)
    return lg


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load_memory_en_and_vocab():
    p = json.loads((ROOT / "models" / "nhom3_mt" / "memory" / "pairs.json").read_text(encoding="utf-8"))
    vocab = set()
    for s in p["en"]:
        vocab.update(tokens(s))
    return p["en"], vocab


def find_leaked_memory(mem_en, refs, thr=0.5):
    """Trả về list index memory có câu EN gần trùng bất kỳ reference test nào (Jaccard > thr)."""
    ref_sets = [set(tokens(r)) for r in refs]
    inv = {}
    for i, s in enumerate(ref_sets):
        for t in s:
            inv.setdefault(t, []).append(i)
    blocked, hits = [], []
    for j, en in enumerate(mem_en):
        ts = set(tokens(en))
        cand = set()
        for t in ts:
            if t in inv:
                cand.update(inv[t])
        best, best_i = 0.0, -1
        for i in cand:
            ja = jaccard(ts, ref_sets[i])
            if ja > best:
                best, best_i = ja, i
        if best > thr:
            blocked.append(j)
            if len(hits) < 20:
                hits.append({"mem_index": j, "ref_index": best_i, "jaccard": round(best, 3),
                             "mem_en": en[:160], "ref": refs[best_i][:160]})
    return blocked, hits


def masked_retrieve(translator, blocked):
    blocked = np.array(sorted(blocked), dtype=np.int64)
    orig = translator.retrieve

    def retrieve(text, k):
        q = translator.encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        scores = translator.mem_emb @ q
        if len(blocked):
            scores[blocked] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    return retrieve, orig


def sentence_metrics(hyp, ref, vocab):
    ht = tokens(hyp)
    ev = max(len(ht), 1)
    p, r, f1 = _token_prf(ht, tokens(ref))
    return {
        "BLEU": round(sacrebleu.sentence_bleu(hyp, [ref]).score, 4),
        "chrF": round(chrf(hyp, ref), 4),
        "chrF++": round(chrf_plus(hyp, ref), 4),
        "METEOR": round(100 * _meteor(hyp, ref), 4),
        "TER": round(sacrebleu.sentence_ter(hyp, [ref]).score, 4),
        "EM": 100.0 if hyp.strip().lower() == ref.strip().lower() else 0.0,
        "TokenPrecision": round(p, 4),
        "TokenRecall": round(r, 4),
        "TokenF1": round(f1, 4),
        "DictionaryCoverage": round(100 * sum(t in vocab for t in ht) / ev, 4),
        "LengthRatio": round(len(ht) / max(len(tokens(ref)), 1), 4),
    }


def _token_prf(ht, rt):
    hs, rs = set(ht), set(rt)
    if not hs and not rs:
        return 1.0, 1.0, 1.0
    ov = len(hs & rs)
    p = ov / max(len(hs), 1)
    r = ov / max(len(rs), 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


_METEOR = None


def _meteor(hyp, ref):
    global _METEOR
    if _METEOR is None:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        for res in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{res}" if res == "wordnet" else f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        _METEOR = meteor_score
    return _METEOR([ref.split()], hyp.split())


def evidence_overlap(hyp, evidence):
    if not evidence:
        return 0.0
    ht, et = set(tokens(hyp)), set(tokens(evidence))
    return round(100 * len(ht & et) / max(len(ht), 1), 4)


def summary(values):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(x):
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(1000)])
    return {"mean": round(float(np.mean(x)), 4),
            "std": round(float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, 4),
            "ci95_low": round(float(np.quantile(means, 0.025)), 4),
            "ci95_high": round(float(np.quantile(means, 0.975)), 4)}


METRIC_KEYS = ["BLEU", "chrF", "chrF++", "METEOR", "TER", "EM", "TokenPrecision", "TokenRecall",
               "TokenF1", "DictionaryCoverage", "EvidenceOverlap", "LatencyMs", "LengthRatio"]


def run_scenario(name, cfg, rows, translator, vocab, logger, out_root):
    mode = cfg["memory_mode"]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s | n=%d | mode=%s | aux=%s ===", name, len(rows), mode, cfg.get("fr_en_auxiliary"))
    results, lap = [], []
    wall0 = time.perf_counter()
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        out = translator.translate(r["src"], "vi", mode=mode)
        ms = (time.perf_counter() - t0) * 1000
        lap.append(ms)
        steps = out["sentences"]
        evidence = next((s["evidence"] for s in steps if s.get("evidence")), None)
        ev_score = max([s["ev_score"] for s in steps if s.get("ev_score") is not None], default=None)
        rec = {
            "id": r["id"], "src": r["src"], "ref": r["tgt"], "hyp": out["translation"],
            "draft": " ".join(s["draft"] for s in steps),
            "evidence": evidence,
            "evidence_used": any(s["evidence_used"] for s in steps),
            "hard": any(s["hard"] for s in steps),
            "C": min(s["C"] for s in steps),
            "R": max(s["R"] for s in steps),
            "ev_score": round(ev_score, 4) if ev_score is not None else None,
            "n_sentences": len(steps),
            "LatencyMs": round(ms, 2),
        }
        rec.update(sentence_metrics(rec["hyp"], rec["ref"], vocab))
        rec["EvidenceOverlap"] = evidence_overlap(rec["hyp"], evidence)
        results.append(rec)
        if (i + 1) % 10 == 0:
            logger.info("  %s %d/%d · %.1fs/câu · hard=%d ev_used=%d", name, i + 1, len(rows),
                        np.mean(lap[-10:]) / 1000, sum(x["hard"] for x in results), sum(x["evidence_used"] for x in results))
    wall = time.perf_counter() - wall0
    with open(out_dir / "hypotheses.jsonl", "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps({k: x[k] for k in ("id", "src", "ref", "hyp", "draft", "evidence",
                                                  "evidence_used", "hard", "C", "R", "ev_score", "LatencyMs")},
                               ensure_ascii=False) + "\n")
    with open(out_dir / "per_sample.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    metrics = {k: summary([x[k] for x in results]) for k in METRIC_KEYS if k in results[0]}
    hyps = [x["hyp"] for x in results]
    refs = [x["ref"] for x in results]
    metrics["corpus"] = {
        "BLEU_sacrebleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 4),
        "TER_sacrebleu": round(sacrebleu.corpus_ter(hyps, [refs]).score, 4),
        "chrF_sacrebleu": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 4),
    }
    metrics.update({
        "n": len(results), "scenario": name, "config": cfg, "wall_seconds": round(wall, 2),
        "Throughput_sent_per_s": round(len(results) / wall, 4) if wall > 0 else 0.0,
        "hard_frac": round(sum(x["hard"] for x in results) / len(results), 4),
        "evidence_frac": round(sum(x["evidence_used"] for x in results) / len(results), 4),
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("%s done: wall=%.1fs · BLEU(corpus)=%.2f · chrF=%.2f · hard=%.3f · ev_used=%.3f",
                name, wall, metrics["corpus"]["BLEU_sacrebleu"], metrics["chrF"]["mean"],
                metrics["hard_frac"], metrics["evidence_frac"])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="vd: A,B1,B2")
    a = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(ts)
    cfg_all = yaml.safe_load((HERE / "test_scenarios.yaml").read_text(encoding="utf-8"))
    scenarios = cfg_all["scenarios"]
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        scenarios = {k: v for k, v in scenarios.items() if k in keep}
    out_root = HERE / ("results_dryrun" if a.dry_run else "results")
    n = 5 if a.dry_run else (a.limit or cfg_all.get("n_per_scenario", 100))
    logger.info("FE_AGAL_v2 start · dry=%s · n=%d · out=%s", a.dry_run, n, out_root)

    mem_en, vocab = load_memory_en_and_vocab()
    logger.info("memory EN=%d · vocab_EN=%d", len(mem_en), len(vocab))

    # leakage check cho B2/B2_filtered
    leak_info = {"threshold": 0.5, "n_blocked": 0, "blocked": [], "examples": []}
    if "B2" in scenarios:
        rows_b2 = read_jsonl(ROOT / scenarios["B2"]["input"])
        if a.dry_run:
            rows_b2 = rows_b2[:n]
        t0 = time.perf_counter()
        blocked, hits = find_leaked_memory(mem_en, [r["tgt"] for r in rows_b2])
        leak_info.update({"n_blocked": len(blocked), "blocked": blocked, "examples": hits,
                          "n_test": len(rows_b2), "seconds": round(time.perf_counter() - t0, 2)})
        logger.info("leakage check: %d/%d memory câu gần trùng test (Jaccard>0.5)", len(blocked), len(mem_en))
        (HERE / "logs" / f"leakage_B2_{ts}.json").write_text(
            json.dumps(leak_info, ensure_ascii=False, indent=2), encoding="utf-8")

    translator = Translator()
    t0 = time.perf_counter()
    translator.ensure_core()
    logger.info("core loaded in %.1fs", time.perf_counter() - t0)

    ran = {}
    for name in ["A", "B1", "B2", "B2_filtered", "C"]:
        if name not in scenarios:
            continue
        cfg = scenarios[name]
        if name == "B2_filtered" and not leak_info["n_blocked"]:
            logger.info("B2_filtered: 0 leakage → copy B2 (identical by definition)")
            continue
        rows = read_jsonl(ROOT / cfg["input"])
        if a.dry_run or a.limit:
            rows = rows[:n]
        orig = None
        if name == "B2_filtered":
            patched, orig = masked_retrieve(translator, set(leak_info["blocked"]))
            translator.retrieve = patched
        try:
            ran[name] = run_scenario(name, cfg, rows, translator, vocab, logger, out_root)
        finally:
            if orig is not None:
                translator.retrieve = orig
    if "B2_filtered" not in ran and "B2" in ran:
        import shutil
        src, dst = out_root / "B2", out_root / "B2_filtered"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "hypotheses.jsonl", dst / "hypotheses.jsonl")
        shutil.copy(src / "per_sample.csv", dst / "per_sample.csv")
        m = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        m["scenario"] = "B2_filtered"
        m["note"] = "0 memory leakage → giống hệt B2"
        (dst / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ALL DONE in %.1fs", time.perf_counter() - t0)
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "hard_frac", "evidence_frac", "Throughput_sent_per_s")}
                      for k, v in ran.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
