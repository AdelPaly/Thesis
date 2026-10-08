"""Grid language-model geolocation (tier 3).

The earth is cut into a regular grid of `cell_degrees` x `cell_degrees`
latitude/longitude cells. Every training text is assigned to the cell that
holds its gold place, and each cell gets a unigram language model of the words
of its training texts. A post is scored against every cell by the KL
divergence from the post's word distribution to the cell's, and the cell with
the smallest divergence wins.

No gazetteer, no mention detection, no LLM: words in, cell out.

Smoothing. A cell is small, so most of its words are unseen. Each cell keeps a
Good-Turing estimate of its unseen mass, N1 / (N + 1) (N1 = words seen once
in the cell, at least 1, N = tokens in the cell), and spreads that mass over the words it
has not seen in proportion to their global probability. The global
distribution is add-one smoothed over the training vocabulary, so a word never
seen in training still has a probability. There is no tuned parameter.

A cell is a region, not a place, and the evaluation is by place: a cell's
prediction is its most frequent training place.

Training texts are the dev items of the dataset being run. An item that is
itself a training item is scored with its own words removed from its cell.

Pure Python; no dependency beyond the standard library.
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
    """Cell language models built from training rows.

    `rows` are dicts with `item_id`, `text`, `qid`, `lat`, `lon` and the place
    fields that go into a prediction. Rows without a coordinate are skipped.
    """

    def __init__(self, rows: list[dict], cell_degrees: float):
        self.degrees = cell_degrees
        self.cells: dict[tuple, Counter] = defaultdict(Counter)
        self.cell_places: dict[tuple, Counter] = defaultdict(Counter)
        self.places: dict[str, dict] = {}
        self.items: dict[str, tuple[tuple, Counter]] = {}
        self.item_places: dict[str, str] = {}
        for row in rows:
            if row.get("lat") is None or row.get("lon") is None:
                continue
            cell = cell_of(row["lat"], row["lon"], cell_degrees)
            words = Counter(tokens(row["text"]))
            self.cells[cell].update(words)
            self.cell_places[cell][row["qid"]] += 1
            self.places.setdefault(row["qid"], row)
            self.items[row["item_id"]] = (cell, words)
            self.item_places[row["item_id"]] = row["qid"]

        self.global_counts = Counter()
        for counts in self.cells.values():
            self.global_counts.update(counts)
        self._global_total = sum(self.global_counts.values())
        self._vocab = len(self.global_counts)

    def p_global(self, word: str) -> float:
        return ((self.global_counts.get(word, 0) + 1)
                / (self._global_total + self._vocab + 1))

    def _scorer(self, counts: Counter):
        """log P_cell(word) for one cell's counts."""
        n = sum(counts.values())
        if n == 0:
            return None
        n1 = max(1, sum(1 for c in counts.values() if c == 1))
        unseen_mass = n1 / (n + 1)
        seen_global = sum(self.p_global(w) for w in counts)
        unseen_global = max(1.0 - seen_global, 1e-12)

        def log_p(word: str) -> float:
            c = counts.get(word, 0)
            if c:
                return math.log((1.0 - unseen_mass) * c / n)
            return math.log(unseen_mass * self.p_global(word) / unseen_global)
        return log_p

    def representative(self, cell: tuple, item_id: str | None = None) -> str | None:
        """The cell's most frequent training place (ties broken by QID), with
        the scored item itself not counted."""
        places = Counter(self.cell_places[cell])
        if item_id in self.item_places and self.items[item_id][0] == cell:
            places[self.item_places[item_id]] -= 1
        places = +places
        return min(places, key=lambda q: (-places[q], q)) if places else None

    def rank(self, text: str, item_id: str | None = None) -> list[tuple[tuple, float]]:
        """[(cell, -KL(post || cell)), ...] best first. `item_id` names the item
        being scored, so its own words are left out of its cell."""
        doc = Counter(tokens(text))
        total = sum(doc.values())
        if not total:
            return []
        p_doc = {w: c / total for w, c in doc.items()}
        own_cell, own = self.items.get(item_id, (None, None))
        scored = []
        for cell, counts in self.cells.items():
            if cell == own_cell:
                counts = counts - own
            log_p = self._scorer(counts)
            if log_p is None:
                continue
            kl = sum(p * (math.log(p) - log_p(w)) for w, p in p_doc.items())
            scored.append((cell, -kl))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored
