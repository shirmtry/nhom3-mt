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
    # chrF++ combines character n-grams (1-6) with word bigram F-scores.
    def ngram_f1(h, r, n):
        from collections import Counter
        hc = Counter(h[i:i+n] for i in range(len(h)-n+1))
        rc = Counter(r[i:i+n] for i in range(len(r)-n+1))
        overlap = sum(min(v, rc[k]) for k, v in hc.items())
        precision = overlap / max(sum(hc.values()), 1)
        recall = overlap / max(sum(rc.values()), 1)
        return 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)

    hyp_words = re.findall(r"\w+", hypothesis.lower())
    ref_words = re.findall(r"\w+", reference.lower())
    hyp_chars = " " + " ".join(hypothesis.lower().split()) + " "
    ref_chars = " " + " ".join(reference.lower().split()) + " "
    char_f = sum(ngram_f1(hyp_chars, ref_chars, n) for n in range(1, 7)) / 6
    hyp_word_bigrams = tuple(zip(hyp_words, hyp_words[1:]))
    ref_word_bigrams = tuple(zip(ref_words, ref_words[1:]))
    word_f = ngram_f1(hyp_word_bigrams, ref_word_bigrams, 1)
    return 100 * (char_f + word_f) / 2
