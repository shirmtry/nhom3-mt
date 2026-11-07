import pytest

try:
    from .bleu import bleu
    from .chrf import chrf, chrf_plus
    from .meteor import meteor
    from .ter import ter
    from .latency import exact_match, token_f1
    from .dict_coverage import dictionary_coverage
    from .statistics import summarize
except ImportError:
    from bleu import bleu
    from chrf import chrf, chrf_plus
    from meteor import meteor
    from ter import ter
    from latency import exact_match, token_f1
    from dict_coverage import dictionary_coverage
    from statistics import summarize


def test_bleu_perfect():
    assert bleu("the cat sat", "the cat sat") == pytest.approx(100, abs=1e-6)


def test_chrf_and_chrf_plus():
    assert chrf("hello", "hello") > 0
    assert chrf_plus("hello world", "hello world") > 0


def test_meteor_and_ter():
    assert meteor("hello world", "hello world") > 0
    assert ter("hello world", "hello world") == 0


def test_em_and_token_f1():
    assert exact_match("Hello", "hello") == 100
    assert token_f1("hello world", "hello world")[2] == 1


def test_dictionary_coverage():
    assert dictionary_coverage("hello world", {"hello", "world"}) == 100


def test_summary():
    result = summarize([1, 2, 3])
    assert result["mean"] == 2
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]
