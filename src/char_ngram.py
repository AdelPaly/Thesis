"""Character n-gram ranking (tier 3, the lexical floor).

The candidate places and the text that represents each one are dense
retrieval's own: the same pool and the same indexed fields (src/dense_index.py,
build_catalogue). A post and every place text are turned into counts of
character n-grams (lengths `ngram_min`..`ngram_max`, lower-cased), each count
is weighted by the n-gram's inverse document frequency over the place texts,
and places are ranked by the cosine of those weighted vectors.

The weighting is what makes this lexical overlap rather than a measure of
how English a text is: with raw counts, the short n-grams of ordinary English
("the ", "e ") dominate every vector, long generic descriptions win for every
post, and a post opening with "Tripoli" ranks Tripoli 3324th. An n-gram that
occurs in no place text cannot match and is left out of the query.

Nothing is learned: no training texts, no encoder, no toponym input, no
gazetteer lookup at query time. Against dense retrieval it separates what
surface overlap between a post and a place description carries from what a
learned encoder adds.

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


class CharNgramIndex:
    """An inverted index of idf-weighted n-gram counts over a list of texts."""

    def __init__(self, texts: list[str], n_min: int, n_max: int):
        self.n_min, self.n_max = n_min, n_max
        counted = [ngrams(text, n_min, n_max) for text in texts]
        df = Counter()
        for grams in counted:
            df.update(grams.keys())
        n = len(texts)
        self.idf = {gram: math.log(n / d) for gram, d in df.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.norms = []
        for i, grams in enumerate(counted):
            weights = {g: c * self.idf[g] for g, c in grams.items()}
            for gram, w in weights.items():
                if w:
                    self.postings[gram].append((i, w))
            self.norms.append(math.sqrt(sum(w * w for w in weights.values())))

    def rank(self, text: str, top_k: int) -> list[tuple[int, float]]:
        """[(text index, cosine), ...], best first, at most `top_k`."""
        query = {g: c * self.idf[g] for g, c in ngrams(text, self.n_min, self.n_max).items()
                 if g in self.idf}
        q_norm = math.sqrt(sum(w * w for w in query.values()))
        if not q_norm:
            return []
        dots: dict[int, float] = defaultdict(float)
        for gram, q_weight in query.items():
            for i, weight in self.postings.get(gram, ()):
                dots[i] += q_weight * weight
        scored = [(i, dot / (q_norm * self.norms[i]))
                  for i, dot in dots.items() if self.norms[i]]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:top_k]
