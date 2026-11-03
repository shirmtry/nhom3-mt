"""Bước 3.4 — Lấy mẫu theo nhiệt độ: p_l ∝ q_l^(1/T) (Arivazhagan et al., 2019).

Với 15K Việt–Anh và 100K Pháp–Anh: T=1 → 13% câu Việt, T=2 → 28%, T=5 → 41%.
Mỗi batch trộn cả hai ngôn ngữ (Johnson et al., 2017).
"""


def temperature_probs(sizes, T):
    """sizes: {lang: số cặp}. Trả về {lang: xác suất chọn ngôn ngữ đó}."""
    total = sum(sizes.values())
    w = {l: (n / total) ** (1.0 / T) for l, n in sizes.items() if n > 0}
    z = sum(w.values())
    return {l: v / z for l, v in w.items()}


def fixed_probs(sizes, vi_share):
    """Giai đoạn 2: cố định tỉ lệ câu Việt (vd 0.8), phần còn lại cho tiếng Pháp."""
    langs = [l for l, n in sizes.items() if n > 0]
    if len(langs) == 1:
        return {langs[0]: 1.0}
    return {l: (vi_share if l == "vi" else (1 - vi_share) / (len(langs) - 1)) for l in langs}


def example_weights(langs, probs):
    """Trọng số từng mẫu sao cho xác suất rơi vào ngôn ngữ l đúng bằng probs[l]."""
    count = {}
    for l in langs:
        count[l] = count.get(l, 0) + 1
    return [probs.get(l, 0.0) / count[l] for l in langs]


def make_sampler(langs, probs, num_samples, seed=0):
    import torch
    from torch.utils.data import WeightedRandomSampler
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(example_weights(langs, probs), num_samples=num_samples, replacement=True, generator=g)
