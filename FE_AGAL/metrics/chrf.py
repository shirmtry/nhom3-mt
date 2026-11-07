import re


def _ngrams(text, n):
    padded = " ".join(text.lower().split())
    padded = "^" + padded + "$"
    return set(padded[i:i+n] for i in range(len(padded)-n+1))


def chrf(hypothesis, reference, beta=2, min_n=1, max_n=6):
    def ngram_counts(text):
        from collections import Counter
        text = " ".join(text.lower().split())
        counts = Counter()
        padded = " " + text + " "
        for n in range(min_n, max_n + 1):
            counts.update(padded[i:i+n] for i in range(len(padded) - n + 1))
        return counts

    hc, rc = ngram_counts(hypothesis), ngram_counts(reference)
    overlap = sum(min(v, rc[k]) for k, v in hc.items())
    precision = overlap / max(sum(hc.values()), 1)
    recall = overlap / max(sum(rc.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 100 * (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)


def chrf_plus(hypothesis, reference):
    # chrF++ augments character F-score with space-delimited word bigrams.
    def bigrams(text):
        w = re.findall(r"\S+", text.lower())
        return set(zip(w, w[1:]))
    sh, sr = set(hypothesis.lower().replace(" ", "")), set(reference.lower().replace(" ", ""))
    overlap = len(sh & sr)
    bh, br = bigrams(hypothesis), bigrams(reference)
    bo = len(bh & br)
    char_f = overlap / max((len(sh)+len(sr))/2, 1)
    word_f = bo / max((len(bh)+len(br))/2, 1)
    return 100 * (0.5 * char_f + 0.5 * word_f)
