def comet_available():
    try:
        import sys
        metric_dir = __file__.rsplit("\\", 1)[0]
        sys.path = [p for p in sys.path if str(p).replace("/", "\\") != metric_dir]
        import comet  # noqa: F401
        return hasattr(comet, "download_model") and hasattr(comet, "load_from_checkpoint")
    except Exception:
        return False


def comet_scores(sources, hypotheses, references):
    if not comet_available():
        return None
    from comet import download_model, load_from_checkpoint
    import torch
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0,
                           progress_bar=False)
    return [float(x) for x in output.scores]
