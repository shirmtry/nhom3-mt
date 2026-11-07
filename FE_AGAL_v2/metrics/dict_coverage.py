import json
import re


def dictionary_coverage(hypothesis, dictionary_entries=None):
    if not dictionary_entries:
        return 0.0
    tokens = re.findall(r"\w+", hypothesis.lower())
    return 100 * sum(t in dictionary_entries for t in tokens) / max(len(tokens), 1)


def load_memory_dictionary(memory_json):
    pairs = json.loads(open(memory_json, encoding="utf-8").read())
    return {re.sub(r"[^a-z0-9]+", "", x.lower()) for x in pairs["en"] for x in x.split()}
