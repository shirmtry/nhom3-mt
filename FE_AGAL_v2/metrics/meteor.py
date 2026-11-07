import re
from collections import Counter


def meteor(hypothesis, reference, beta=1.0):
    h = re.findall(r"\w+", hypothesis.lower())
    r = re.findall(r"\w+", reference.lower())
    hc, rc = Counter(h), Counter(r)
    matched = sum(min(v, rc[x]) for x, v in hc.items())
    if not matched:
        return 0.0
    p = matched / max(len(h), 1)
    rr = matched / max(len(r), 1)
    return 100 * p * rr / max(p * beta + rr, 1e-12)
