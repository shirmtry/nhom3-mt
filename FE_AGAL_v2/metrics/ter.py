from collections import Counter


def ter(hypothesis, reference):
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not r:
        return 0.0
    # Bounded token edit distance is the standard TER fallback implementation.
    prev = list(range(len(h)+1))
    for i, ref_word in enumerate(r, 1):
        cur = [i]
        for j, hyp_word in enumerate(h, 1):
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+(ref_word != hyp_word)))
        prev = cur
    return 100 * prev[-1] / len(r)
