"""Character n-gram floor (tier 0).

Every training place gets a profile: the counts of the character n-grams
(lengths `ngram_min`..`ngram_max`) over all of its training texts. A post is
scored against every profile by cosine overlap of the n-gram counts, and the
best-matching place is the prediction.

No toponym input and no gazetteer: the model sees only surface form. It is the
floor that says whether surface statistics alone carry the signal, rather than
the entity clue.

Training texts are the dev items of the dataset being run. An item that is
itself a training item is scored with its own text removed from its place's
profile, so a dev item never matches against itself.

Pure Python; no dependency beyond the standard library.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def ngrams(text: str, n_min: int, n_max: int) -> Counter:
    """Counts of the character n-grams of `text`, lower-cased."""
    text = str(text).lower()
    counts = Counter()
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            counts[text[i:i + n]] += 1
    return counts


def _dot(a: Counter, b: Counter) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b[k] for k, v in a.items() if k in b))


def _norm(c: Counter) -> float:
    return math.sqrt(sum(v * v for v in c.values()))


class CharNgramModel:
    """Place profiles built from training rows.

    `rows` are dicts with `item_id`, `text`, `qid` and the place fields that go
    into a prediction (`country`, `continent`, `lat`, `lon`, `location_name`).
    """

    def __init__(self, rows: list[dict], n_min: int, n_max: int):
        self.n_min, self.n_max = n_min, n_max
        self.profiles: dict[str, Counter] = defaultdict(Counter)
        self.places: dict[str, dict] = {}
        self.items: dict[str, tuple[str, Counter]] = {}
        for row in rows:
            grams = ngrams(row["text"], n_min, n_max)
            self.profiles[row["qid"]].update(grams)
            self.places.setdefault(row["qid"], row)
            self.items[row["item_id"]] = (row["qid"], grams)
        self.norms = {q: _norm(p) for q, p in self.profiles.items()}

    def rank(self, text: str, item_id: str | None = None) -> list[tuple[str, float]]:
        """[(qid, score), ...] best first. `item_id` names the item being
        scored, so its own text is left out of its place's profile."""
        query = ngrams(text, self.n_min, self.n_max)
        q_norm = _norm(query)
        own_place, own = self.items.get(item_id, (None, None))
        scored = []
        for qid, profile in self.profiles.items():
            if qid == own_place:
                profile = profile - own
                p_norm = _norm(profile)
            else:
                p_norm = self.norms[qid]
            if not p_norm or not q_norm:
                continue
            scored.append((qid, _dot(query, profile) / (q_norm * p_norm)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored
