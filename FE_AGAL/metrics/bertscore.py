def bertscore_available():
    try:
        import bert_score  # noqa: F401
        return True
    except Exception:
        return False


def bertscore_scores(hypotheses, references):
    if not bertscore_available():
        return None
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from bert_score import score
        _, _, f1 = score(hypotheses, references, lang="en", model_type=None)
        return [float(x) for x in f1.tolist()]
    except Exception:
        return None
