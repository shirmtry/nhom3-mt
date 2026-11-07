import numpy as np


def summarize(values):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(x):
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    ci = bootstrap_ci(x.tolist())
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "ci95_low": ci[0], "ci95_high": ci[1]}


def bootstrap_ci(values, seed=42, draws=1000):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    if not len(x):
        return 0.0, 0.0
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(draws)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
