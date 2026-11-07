import re


def latency_summary(milliseconds):
    import numpy as np
    x = np.asarray(milliseconds, dtype=float)
    return float(x.mean()) if len(x) else 0.0


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def exact_match(hypothesis, reference):
    return 100 if hypothesis.strip().lower() == reference.strip().lower() else 0


def token_f1(hypothesis, reference):
    h, r = set(_tokens(hypothesis)), set(_tokens(reference))
    if not h and not r:
        return 1.0, 1.0, 1.0
    overlap = len(h & r)
    precision = overlap / max(len(h), 1)
    recall = overlap / max(len(r), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1
