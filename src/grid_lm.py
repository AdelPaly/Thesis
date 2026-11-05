"""Grid language-model geolocation (tier 3).

The earth is cut into a regular grid of `cell_degrees` x `cell_degrees`
latitude/longitude cells, and each cell gets a unigram language model. A post
is scored against every cell by the KL divergence from the post's word
distribution to the cell's, and cells are taken best first.

What goes into a cell: the dev texts whose gold place falls in it, and the
indexed text of every candidate place that falls in it. The candidate places
and their texts are dense retrieval's own (src/dense_index.py,
build_catalogue). The dev texts are what the method learns from; the place
texts are what lets a cell exist wherever a candidate place is. With dev
texts alone, a 1-degree cell holds a test place for 1 of 94 test places in
social_media_clean, so almost no test place could ever be predicted.

A cell is a region and the evaluation is by place, so a winning cell is
resolved to its candidate places, each ranked by the KL divergence from the
post to that place's own text model. That is a lookup within the cell, not a
centroid and not a distance.

Smoothing. A cell or place text is small, so most words are unseen. Each
model keeps a Good-Turing estimate of its unseen mass, N1 / (N + 1) (N1 =
words seen once, at least 1, N = tokens), and spreads that mass over the words
it has not seen in proportion to their global probability. The global
distribution is add-one smoothed over the vocabulary, so a word seen nowhere
still has a probability. There is no tuned parameter.

A dev item is scored with its own words removed from its cell.

No gazetteer at query time, no mention detection, no LLM. Pure Python; no
dependency beyond the standard library.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def cell_of(lat: float, lon: float, degrees: float) -> tuple[int, int]:
    """The grid cell holding a coordinate."""
    return (math.floor((lat + 90.0) / degrees), math.floor((lon + 180.0) / degrees))


class GridLM:
    """Cell and place language models.

    `rows` are dev items: dicts with `item_id`, `text`, `lat`, `lon`.
    `places` maps QID -> place record with `lat` and `lon`; `place_texts`
    maps QID -> the place's indexed text. Records without a coordinate are
    skipped.
    """

    def __init__(self, rows: list[dict], places: dict[str, dict],
                 place_texts: dict[str, str], cell_degrees: float):
        self.degrees = cell_degrees
        self.cells: dict[tuple, Counter] = defaultdict(Counter)
        self.cell_places: dict[tuple, list[str]] = defaultdict(list)
        self.place_words: dict[str, Counter] = {}
        self.items: dict[str, tuple[tuple, Counter]] = {}

        for qid, place in places.items():
            if place.get("lat") is None or place.get("lon") is None:
                continue
            cell = cell_of(place["lat"], place["lon"], cell_degrees)
            words = Counter(tokens(place_texts.get(qid, "")))
            self.cells[cell].update(words)
            self.cell_places[cell].append(qid)
            self.place_words[qid] = words
        for row in rows:
            if row.get("lat") is None or row.get("lon") is None:
                continue
            cell = cell_of(row["lat"], row["lon"], cell_degrees)
            words = Counter(tokens(row["text"]))
            self.cells[cell].update(words)
            self.items[row["item_id"]] = (cell, words)

        self.global_counts = Counter()
        for counts in self.cells.values():
            self.global_counts.update(counts)
        self._global_total = sum(self.global_counts.values())
        self._vocab = len(self.global_counts)

    def p_global(self, word: str) -> float:
        return ((self.global_counts.get(word, 0) + 1)
                / (self._global_total + self._vocab + 1))

    def _neg_kl(self, p_doc: dict[str, float], counts: Counter) -> float | None:
        """-KL(post || model of `counts`), or None for an empty model."""
        n = sum(counts.values())
        if n == 0:
            return None
        n1 = max(1, sum(1 for c in counts.values() if c == 1))
        unseen_mass = n1 / (n + 1)
        seen_global = sum(self.p_global(w) for w in counts)
        unseen_global = max(1.0 - seen_global, 1e-12)
        kl = 0.0
        for w, p in p_doc.items():
            c = counts.get(w, 0)
            if c:
                log_q = math.log((1.0 - unseen_mass) * c / n)
            else:
                log_q = math.log(unseen_mass * self.p_global(w) / unseen_global)
            kl += p * (math.log(p) - log_q)
        return -kl

    def rank(self, text: str, item_id: str | None = None,
             top_k: int = 10) -> list[tuple[str, float]]:
        """[(qid, score), ...] best first: cells in order of fit, and within a
        cell its places in order of fit. A place's score is its cell's.
        `item_id` names the item being scored, so its own words are left out
        of its cell."""
        doc = Counter(tokens(text))
        total = sum(doc.values())
        if not total:
            return []
        p_doc = {w: c / total for w, c in doc.items()}
        own_cell, own = self.items.get(item_id, (None, None))

        cells = []
        for cell, counts in self.cells.items():
            if not self.cell_places.get(cell):
                continue
            if cell == own_cell:
                counts = counts - own
            score = self._neg_kl(p_doc, counts)
            if score is not None:
                cells.append((cell, score))
        cells.sort(key=lambda t: (-t[1], t[0]))

        out = []
        for cell, score in cells:
            within = []
            for qid in self.cell_places[cell]:
                fit = self._neg_kl(p_doc, self.place_words[qid])
                within.append((qid, fit if fit is not None else float("-inf")))
            within.sort(key=lambda t: (-t[1], t[0]))
            out.extend((qid, score) for qid, _ in within)
            if len(out) >= top_k:
                break
        return out[:top_k]
