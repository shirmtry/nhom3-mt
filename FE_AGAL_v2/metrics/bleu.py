import math
import re
from collections import Counter


def _tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower(), flags=re.ASCII)


def bleu(hypothesis, reference):
    h, r = _tokens(hypothesis), _tokens(reference)
    if not h:
        return 0.0
    max_order = 4
    precisions = []
    actual_order = min(max_order, len(h), len(r))
    for n in range(1, actual_order + 1):
        hc = Counter(tuple(h[i:i+n]) for i in range(len(h)-n+1))
        rc = Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
        overlap = sum(min(v, rc[g]) for g, v in hc.items())
        total = max(sum(hc.values()), 1)
        precisions.append(overlap / total if overlap else 1e-12)
    geo = math.exp(sum(math.log(p) for p in precisions) / actual_order)
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / max(len(h), 1))
    return 100 * bp * geo
