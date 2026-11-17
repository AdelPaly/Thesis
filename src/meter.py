"""What one item cost: counters the runner resets before an item and reads after.

The pipeline's clients add to them where the cost is paid: an LLM request, its
tokens, an answer replayed from the response cache, a Wikidata request. The
runner writes the totals on the item's record as `cost`, next to the seconds
the item took, so price and time can be compared across configurations.

A cached LLM answer adds to `llm_cache_hits` and to nothing else: the cache
keeps the text of the answer, not its tokens or the date it was fetched. A
record with cache hits and no calls was therefore not paid for in this run.
"""
from __future__ import annotations

from collections import Counter

_counts: Counter = Counter()


def reset() -> None:
    _counts.clear()


def add(key: str, n: float = 1) -> None:
    if n:
        _counts[key] += n


def read() -> dict:
    return {key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in sorted(_counts.items())}
