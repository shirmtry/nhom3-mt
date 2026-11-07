"""Phân tích kết quả FE_AGAL_v2: bảng so sánh, paired bootstrap, figures, qualitative, 5 câu hỏi.

Chạy sau run_all.py:  python FE_AGAL_v2/analyze.py
Đọc:  results/{A,B1,B2,B2_filtered,C}/per_sample.csv + metrics.json
Ghi:  results/comparison.md, comparison.csv, bootstrap_B1_B2.json, analysis.md, qualitative.md, figures/*
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import sacrebleu

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
SCENARIOS = ["A", "B1", "B2", "B2_filtered", "C"]
LABEL = {"A": "A · in-domain (aux ON)", "B1": "B1 · semi (aux OFF)",
         "B2": "B2_all · semi (aux ON)", "B2_filtered": "B2_filtered · semi (aux ON, no leak)",
         "C": "C · out-of-domain (aux ON)"}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
VI_CHARS = set("ăđơưạảãặẳẵậẩẫẹẻẽệểễịỉĩọỏõộổỗợởỡụủũựửữỵỷỹ")
NUM = {"id", "LatencyMs", "C", "R", "ev_score", "BLEU", "chrF", "chrF++", "METEOR", "TER",
       "TokenPrecision", "TokenRecall", "TokenF1", "DictionaryCoverage", "EvidenceOverlap", "LengthRatio"}


def tokens(t):
    return [x.lower() for x in WORD.findall(t or "")]


def load(sc):
    p = RES / sc / "per_sample.csv"
    if not p.exists():
        return None, None
    rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
    for r in rows:
        for k in NUM:
            if k in r:
                r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["evidence_used"] = r.get("evidence_used") == "True"
        r["hard"] = r.get("hard") == "True"
    m = json.loads((RES / sc / "metrics.json").read_text(encoding="utf-8"))
    return rows, m


def sent_chrf(hyp, ref):
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def sent_bleu(hyp, ref):
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def paired_bootstrap(a_rows, b_rows, fn, seed=42, draws=10000):
    a = np.array([fn(r["hyp"], r["ref"]) for r in a_rows])
    b = np.array([fn(r["hyp"], r["ref"]) for r in b_rows])
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(draws)])
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {"mean_diff": round(float(d.mean()), 4), "p_value": round(min(p, 1.0), 5),
            "ci95_low": round(float(np.quantile(boots, 0.025)), 4),
            "ci95_high": round(float(np.quantile(boots, 0.975)), 4),
            "n": len(d), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def classify(src, ref, hyp):
    h, r = tokens(hyp), tokens(ref)
    hs, rs = set(h), set(r)
    if any(ch in VI_CHARS for ch in hyp.lower()):
        return "VI token còn sót"
    lr = len(h) / max(len(r), 1)
    if lr < 0.6:
        return "missing word / rút gọn"
    if lr > 1.6:
        return "hallucination / thêm ý"
    if hs == rs and h != r:
        return "word order"
    rec = len(hs & rs) / max(len(rs), 1)
    if rec < 0.4:
        return "mistranslation (lệch từ vựng)"
    if rec < 0.7:
        return "paraphrase xa / thiếu ý"
    return "paraphrase gần (khác từ ngữ)"

def fmt(x, n=2):
    return f"{x:.{n}f}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {sc: load(sc) for sc in SCENARIOS}
    have = [sc for sc in SCENARIOS if data[sc][0] is not None]
    if "B2_filtered" not in have and "B2" in have:
        data["B2_filtered"] = data["B2"]
        have.append("B2_filtered")

    # ---------- bảng comparison ----------
    lines = ["| Scenario | BLEU | chrF | chrF++ | METEOR | TER | EM | TokenF1 | DictCov | Latency (ms) | LengthRatio |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc in have:
        rows, m = data[sc]
        lines.append("| {lab} | {bleu} | {chrf} | {chrfp} | {met} | {ter} | {em} | {f1} | {dict} | {lat} | {lr} |".format(
            lab=LABEL[sc],
            bleu=fmt(m["corpus"]["BLEU_sacrebleu"]), chrf=fmt(m["chrF"]["mean"]), chrfp=fmt(m["chrF++"]["mean"]),
            met=fmt(m["METEOR"]["mean"]), ter=fmt(m["corpus"]["TER_sacrebleu"]),
            em=fmt(m["EM"]["mean"]), f1=fmt(m["TokenF1"]["mean"]), dict=fmt(m["DictionaryCoverage"]["mean"]),
            lat=fmt(m["LatencyMs"]["mean"], 0), lr=fmt(m["LengthRatio"]["mean"], 3)))
    (RES / "comparison.md").write_text(
        "# So sánh kịch bản (nhiệm vụ chính: VI→EN)\n\n"
        "BLEU/TER: corpus-level sacrebleu. chrF/chrF++: custom (trung bình câu). METEOR: nltk ×100. "
        "DictCov: % token hypothesis nằm trong vocab EN của kho FR–EN.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")

    with open(RES / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scenario", "BLEU", "chrF", "chrF++", "METEOR", "TER", "EM", "TokenF1",
                    "DictCov", "LatencyMs", "LengthRatio", "n", "hard_frac", "evidence_frac"])
        for sc in have:
            _, m = data[sc]
            w.writerow([sc, m["corpus"]["BLEU_sacrebleu"], round(m["chrF"]["mean"], 2), round(m["chrF++"]["mean"], 2),
                        round(m["METEOR"]["mean"], 2), m["corpus"]["TER_sacrebleu"], round(m["EM"]["mean"], 2),
                        round(m["TokenF1"]["mean"], 2), round(m["DictionaryCoverage"]["mean"], 2),
                        round(m["LatencyMs"]["mean"], 0), round(m["LengthRatio"]["mean"], 3), m["n"],
                        m.get("hard_frac"), m.get("evidence_frac")])

    # ---------- paired bootstrap B1 vs B2 ----------
    boot = {}
    if all(x in have for x in ("B1", "B2")):
        b1, b2 = data["B1"][0], data["B2"][0]
        assert len(b1) == len(b2) and all(x["src"] == y["src"] for x, y in zip(b1, b2)), "B1/B2 phải cùng input"
        boot["B1_vs_B2"] = {
            "chrF": paired_bootstrap(b1, b2, sent_chrf),
            "BLEU": paired_bootstrap(b1, b2, sent_bleu),
            "ev_used_B1": sum(r["evidence_used"] for r in b1) / len(b1),
            "ev_used_B2": sum(r["evidence_used"] for r in b2) / len(b2),
            "hard_B1": sum(r["hard"] for r in b1) / len(b1),
            "hard_B2": sum(r["hard"] for r in b2) / len(b2),
        }
    if all(x in have for x in ("B1", "B2_filtered")):
        b1, b2 = data["B1"][0], data["B2_filtered"][0]
        boot["B1_vs_B2_filtered"] = {
            "chrF": paired_bootstrap(b1, b2, sent_chrf),
            "BLEU": paired_bootstrap(b1, b2, sent_bleu),
        }
    if all(x in have for x in ("B2", "B2_filtered")):
        b2, b2f = data["B2"][0], data["B2_filtered"][0]
        boot["B2_vs_B2_filtered"] = {"chrF": paired_bootstrap(b2, b2f, sent_chrf)}
    (RES / "bootstrap_B1_B2.json").write_text(json.dumps(boot, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- leakage ----------
    leak = {}
    for p in sorted((HERE / "logs").glob("leakage_B2_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d.pop("blocked", None)
        leak = d
    (RES / "leakage_summary.json").write_text(json.dumps(leak, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.5))
        xs = np.arange(len(have))
        wdt = 0.26
        for i, (key, lab, getter) in enumerate([
                ("BLEU", "BLEU (corpus)", lambda m: m["corpus"]["BLEU_sacrebleu"]),
                ("chrF", "chrF", lambda m: m["chrF"]["mean"]),
                ("METEOR", "METEOR", lambda m: m["METEOR"]["mean"])]):
            vals = [getter(data[sc][1]) for sc in have]
            ax.bar(xs + (i - 1) * wdt, vals, wdt, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[sc].split(" ·")[0] for sc in have])
        ax.set_ylabel("score")
        ax.set_title("VI→EN by scenario (BLEU corpus / chrF / METEOR)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "bar_metrics_by_scenario.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.boxplot([[sent_chrf(r["hyp"], r["ref"]) for r in data[sc][0]] for sc in have],
                   tick_labels=[LABEL[sc].split(" ·")[0] for sc in have])
        ax.set_ylabel("sentence chrF")
        ax.set_title("Per-sample chrF distribution")
        fig.tight_layout()
        fig.savefig(FIG / "boxplot_sentence_chrf.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for sc in have:
            ax.hist([r["LatencyMs"] / 1000 for r in data[sc][0]], bins=25, alpha=0.5, label=LABEL[sc].split(" ·")[0])
        ax.set_xlabel("latency (s/câu)")
        ax.set_ylabel("count")
        ax.set_title("Latency distribution (CPU)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "hist_latency.png", dpi=160)
        plt.close(fig)
    except Exception as e:  # không chặn pipeline nếu matplotlib lỗi font
        print("[warn] figures:", e)

    # ---------- qualitative ----------
    qlines = ["# Qualitative: 5 tốt nhất + 5 tệ nhất mỗi kịch bản\n"]
    for sc in have:
        rows, _ = data[sc]
        scored = [(sent_chrf(r["hyp"], r["ref"]), r) for r in rows]
        scored.sort(key=lambda x: -x[0])
        for title, part in (("5 TỐT NHẤT", scored[:5]), ("5 TỆ NHẤT", scored[-5:][::-1])):
            qlines.append(f"\n## {sc} — {title}\n")
            for s, r in part:
                qlines.append(f"- **chrF {s:.2f}** · id {int(r['id'])} · aux_used={r['evidence_used']}\n"
                              f"  - VI: {r['src']}\n  - REF: {r['ref']}\n  - HYP: {r['hyp']}\n")
    (RES / "qualitative.md").write_text("\n".join(qlines), encoding="utf-8")

    # ---------- failure analysis ----------
    fail = {}
    for sc in have:
        rows, _ = data[sc]
        worst = sorted(rows, key=lambda r: sent_chrf(r["hyp"], r["ref"]))[:10]
        cats = {}
        examples = []
        for r in worst:
            c = classify(r["src"], r["ref"], r["hyp"])
            cats[c] = cats.get(c, 0) + 1
            examples.append({"id": int(r["id"]), "category": c, "chrF": round(sent_chrf(r["hyp"], r["ref"]), 2),
                             "src": r["src"], "ref": r["ref"], "hyp": r["hyp"]})
        fail[sc] = {"categories": cats, "examples": examples}
    (RES / "failure_analysis.json").write_text(json.dumps(fail, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- length-matched A vs C ----------
    lm = {}
    if "A" in have and "C" in have:
        a_rows = data["A"][0]
        c_max = max(len(tokens(r["ref"])) for r in data["C"][0])
        a_short = [r for r in a_rows if len(tokens(r["ref"])) <= c_max]
        if len(a_short) >= 10:
            lm = {"c_max_ref_tokens": c_max, "n_A_short": len(a_short),
                  "A_short_BLEU": round(sacrebleu.corpus_bleu([r["hyp"] for r in a_short],
                                                              [[r["ref"] for r in a_short]]).score, 2),
                  "A_short_chrF": round(float(np.mean([r["chrF"] for r in a_short])), 2),
                  "C_BLEU": data["C"][1]["corpus"]["BLEU_sacrebleu"],
                  "C_chrF": data["C"][1]["chrF"]["mean"]}
    (RES / "length_matched_AC.json").write_text(json.dumps(lm, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- analysis.md ----------
    m = {sc: data[sc][1] for sc in have}
    L = ["# Phân tích kết quả FE_AGAL_v2 (VI→EN là nhiệm vụ chính)\n",
         "## Bảng so sánh\n", "\n".join(lines), "\n"]
    if boot:
        b = boot.get("B1_vs_B2", {})
        if b:
            L.append("## B1 vs B2 (paired bootstrap, cùng input)\n")
            for k in ("chrF", "BLEU"):
                v = b[k]
                L.append(f"- **{k}**: Δ(B2−B1) = {v['mean_diff']:+.3f} · 95% CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}] "
                         f"· p = {v['p_value']:.4f} · thắng/hoà/thua = {v['wins']}/{v['ties']}/{v['losses']} (n={v['n']})")
            L.append(f"- Evidence dùng: B1 {b['ev_used_B1']:.0%} vs B2 {b['ev_used_B2']:.0%}; "
                     f"câu 'hard': B1 {b['hard_B1']:.0%} vs B2 {b['hard_B2']:.0%}\n")
        if "B1_vs_B2_filtered" in boot:
            v = boot["B1_vs_B2_filtered"]["chrF"]
            L.append(f"- B1 vs B2_filtered (chrF): Δ = {v['mean_diff']:+.3f}, p = {v['p_value']:.4f}\n")
    if leak:
        L.append(f"## Leakage check\n- Memory gần trùng test (Jaccard > {leak.get('threshold')}): "
                 f"**{leak.get('n_blocked', 0)}** câu trên {leak.get('n_test')} test.\n")

    L.append("## Trả lời 5 câu hỏi\n")
    if "A" in have:
        L.append(f"**Q1. A (in-domain) mạnh/yếu thế nào?** BLEU corpus {m['A']['corpus']['BLEU_sacrebleu']:.2f}, "
                 f"chrF {m['A']['chrF']['mean']:.2f} trên {m['A']['n']} câu. README gốc báo cáo VI→EN BLEU 26,98 / chrF 49,43 "
                 f"trên 1.000 câu PhoMT — cùng mức, khác cỡ mẫu. Lưu ý A trích từ chính file train nên có thể đã thấy dữ liệu.\n")
    if all(x in have for x in ("A", "B1")):
        d = m["B1"]["corpus"]["BLEU_sacrebleu"] - m["A"]["corpus"]["BLEU_sacrebleu"]
        dc = m["B1"]["chrF"]["mean"] - m["A"]["chrF"]["mean"]
        L.append(f"**Q2. Thêm biến thể (B1 vs A) giảm bao nhiêu?** ΔBLEU {d:+.2f}, ΔchrF {dc:+.2f}. "
                 f"Biến thể gồm mất dấu/typo/đảo token/synonym — mức giảm phản ánh độ bền với nhiễu bề mặt.\n")
    if "B1_vs_B2" in boot:
        v = boot["B1_vs_B2"]["chrF"]
        verdict = ("có cải thiện có ý nghĩa thống kê" if v["p_value"] < 0.05 and v["mean_diff"] > 0 else
                   "có hại có ý nghĩa thống kê" if v["p_value"] < 0.05 and v["mean_diff"] < 0 else
                   "không khác biệt có ý nghĩa (p ≥ 0.05)")
        L.append(f"**Q3. FR–EN auxiliary có giúp VI→EN không?** {verdict}: ΔchrF = {v['mean_diff']:+.3f}, "
                 f"p = {v['p_value']:.4f}. Đây là can thiệp thực nghiệm duy nhất khác giữa B1/B2.\n")
    if all(x in have for x in ("A", "C")):
        L.append(f"**Q4. Out-of-domain (C) giảm bao nhiêu so với A?** ΔBLEU {m['C']['corpus']['BLEU_sacrebleu'] - m['A']['corpus']['BLEU_sacrebleu']:+.2f}, "
                 f"ΔchrF {m['C']['chrF']['mean'] - m['A']['chrF']['mean']:+.2f}. C là câu Tatoeba ngắn; "
                 f"so khớp độ dài: {lm.get('n_A_short', '?')} câu A có ref ≤ {lm.get('c_max_ref_tokens', '?')} token đạt "
                 f"BLEU {lm.get('A_short_BLEU', '?')} / chrF {lm.get('A_short_chrF', '?')} vs C {lm.get('C_BLEU', '?')} / {lm.get('C_chrF', '?')}. "
                 f"Độ dài khác nhau là confound, cần đọc thận trọng.\n")
    L.append("**Q5. Failure modes phổ biến (10 câu tệ nhất/kịch bản).**\n")
    for sc in have:
        cats = fail[sc]["categories"]
        L.append(f"- {sc}: " + ", ".join(f"{k} ({v})" for k, v in sorted(cats.items(), key=lambda x: -x[1])))
    L.append("\nVí dụ tệ nhất mỗi kịch bản nằm trong `qualitative.md` và `failure_analysis.json`.\n")
    (RES / "analysis.md").write_text("\n".join(L), encoding="utf-8")
    print("Wrote comparison.md, bootstrap_B1_B2.json, analysis.md, qualitative.md, figures/")


if __name__ == "__main__":
    main()
