"""
Tests for the repeated-clue drop and the condition balancing in finalise.py.

Run from the repo root:

    python -m unittest discover generation/synthetic/tests

No network or API. The repeated-clue test reads the committed plan
(output/item_plan.jsonl) read-only; the balancing tests use made-up posts.
"""
from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

SYN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYN))

import finalise  # noqa: E402

PLAN_PATH = SYN / "output" / "item_plan.jsonl"
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]


def entry(item_id, condition, continent):
    item = {"item_id": item_id, "condition": condition, "place_continent": continent}
    return item, {}, "text"


def make_kept(sizes):
    """sizes: {condition: {continent: count}} -> list of kept entries."""
    kept = []
    for condition, by_continent in sizes.items():
        for continent, count in by_continent.items():
            kept += [entry(f"{condition}-{continent}-{i}", condition, continent)
                     for i in range(count)]
    return kept


class RepeatedClueTests(unittest.TestCase):
    def test_detects_repeat(self):
        item = {"clues": [{"clue_qid": "Q1"}, {"clue_qid": "Q2"}, {"clue_qid": "Q1"}]}
        self.assertEqual(finalise.repeated_clue(item), "Q1")

    def test_distinct_and_missing_clues_pass(self):
        self.assertIsNone(finalise.repeated_clue({"clues": [{"clue_qid": "Q1"}, {"clue_qid": "Q2"}]}))
        self.assertIsNone(finalise.repeated_clue({"clues": []}))
        self.assertIsNone(finalise.repeated_clue({}))

    def test_verdict_drops_repeat_even_before_generation(self):
        item = {"item_id": "x", "condition": "A3", "place_qid": "Q9", "place_name": "P",
                "clues": [{"clue_qid": "Q1", "clue_label": "a"}, {"clue_qid": "Q1", "clue_label": "a"}]}
        status, reason = finalise.verdict_for(item, None, {})
        self.assertEqual(status, "repeated_clue")
        self.assertIn("Q1", reason)

    @unittest.skipUnless(PLAN_PATH.exists(), "committed plan not present")
    def test_committed_plan_has_19_repeats_all_a3(self):
        items = [json.loads(line) for line in PLAN_PATH.read_text(encoding="utf-8").splitlines() if line]
        bad = [i for i in items if finalise.repeated_clue(i)]
        self.assertEqual(len(items), 984)
        self.assertEqual(len(bad), 19)
        self.assertEqual({i["condition"] for i in bad}, {"A3"})


class AllocateEvenTests(unittest.TestCase):
    def test_even_when_supply_allows(self):
        self.assertEqual(finalise.allocate_even({"a": 10, "b": 10, "c": 10}, 9), {"a": 3, "b": 3, "c": 3})

    def test_short_group_hands_its_share_to_the_rest(self):
        self.assertEqual(finalise.allocate_even({"a": 2, "b": 10, "c": 10}, 12), {"a": 2, "b": 5, "c": 5})

    def test_remainder_goes_in_sorted_order(self):
        self.assertEqual(finalise.allocate_even({"a": 5, "b": 5, "c": 5}, 4), {"a": 2, "b": 1, "c": 1})

    def test_never_exceeds_supply(self):
        take = finalise.allocate_even({"a": 1, "b": 2}, 10)
        self.assertEqual(take, {"a": 1, "b": 2})

    def test_zero_and_empty(self):
        self.assertEqual(finalise.allocate_even({"a": 3}, 0), {"a": 0})
        self.assertEqual(finalise.allocate_even({}, 5), {})


class BalanceTests(unittest.TestCase):
    def setUp(self):
        spread = lambda n: {c: n // 6 for c in CONTINENTS}   # noqa: E731
        self.kept = make_kept({"A1": spread(120), "A3": spread(60), "B": spread(36)})

    def test_every_condition_cut_to_the_smallest(self):
        selected, excluded, n = finalise.balance(self.kept, seed=1)
        self.assertEqual(n, 36)
        self.assertEqual(Counter(e[0]["condition"] for e in selected), {"A1": 36, "A3": 36, "B": 36})
        self.assertEqual(len(selected) + len(excluded), len(self.kept))

    def test_continents_even_within_each_condition(self):
        selected, _, _ = finalise.balance(self.kept, seed=1)
        for condition in ("A1", "A3", "B"):
            counts = Counter(e[0]["place_continent"] for e in selected if e[0]["condition"] == condition)
            self.assertEqual(set(counts.values()), {6}, condition)

    def test_deterministic_and_seed_dependent(self):
        first = {e[0]["item_id"] for e in finalise.balance(self.kept, seed=7)[0]}
        again = {e[0]["item_id"] for e in finalise.balance(list(reversed(self.kept)), seed=7)[0]}
        other = {e[0]["item_id"] for e in finalise.balance(self.kept, seed=8)[0]}
        self.assertEqual(first, again)           # input order does not matter
        self.assertNotEqual(first, other)

    def test_uneven_continents_fill_from_the_rest(self):
        kept = make_kept({"A1": {"Africa": 2, "Asia": 50, "Europe": 50},
                          "B": {"Africa": 30, "Asia": 30, "Europe": 30}})
        selected, _, n = finalise.balance(kept, seed=1)
        self.assertEqual(n, 90)
        a1 = Counter(e[0]["place_continent"] for e in selected if e[0]["condition"] == "A1")
        self.assertEqual(a1, {"Africa": 2, "Asia": 44, "Europe": 44})

    def test_per_condition_override_and_limit(self):
        _, _, n = finalise.balance(self.kept, seed=1, per_condition=12)
        self.assertEqual(n, 12)
        with self.assertRaises(ValueError):
            finalise.balance(self.kept, seed=1, per_condition=37)

    def test_report_marks_cut_posts(self):
        selected, excluded, n = finalise.balance(self.kept, seed=1)
        report = [{"item_id": e[0]["item_id"], "status": "kept", "reason": ""} for e in self.kept]
        finalise.apply_balanced_out(report, excluded, n, 1)
        statuses = Counter(r["status"] for r in report)
        self.assertEqual(statuses["balanced_out"], len(excluded))
        self.assertEqual(statuses["kept"], len(selected))


if __name__ == "__main__":
    unittest.main()
