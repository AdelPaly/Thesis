"""
Table-driven tests for the pure logic in generation/synthetic.

Run from the repo root:

    python -m unittest discover generation/synthetic/tests

Nothing here touches the network, the API, or the real output/ state/ data/
files: the random stream tests use temporary files.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SYN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYN))

import build_exemplars as bex  # noqa: E402
import finalise                # noqa: E402
import generate as gen         # noqa: E402
import plan                    # noqa: E402


def make_stream(values, low, high, cursor=0):
    """A RandomStream over a temporary file of integers."""
    tmp = Path(tempfile.mkdtemp())
    stream_path = tmp / "stream.txt"
    stream_path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")
    state = tmp / "state.json"
    if cursor:
        state.write_text(json.dumps({"position": cursor}), encoding="utf-8")
    return plan.RandomStream(stream_path, state, low=low, high=high)


class RandomStreamBelow(unittest.TestCase):
    def test_rejection_sampling_table(self):
        # (values, low, high, n, expected index, expected end position)
        # span 10, n 4: limit is 8, so offsets 8 and 9 are rejected.
        table = [
            ([0], 0, 9, 4, 0, 1),
            ([7], 0, 9, 4, 3, 1),
            ([8, 9, 6], 0, 9, 4, 2, 3),      # two rejections, then 6 % 4
            ([9, 8, 7], 0, 9, 4, 3, 3),
            ([13], 10, 19, 4, 3, 1),         # offset by low: 13 - 10 = 3
            ([18, 19, 11], 10, 19, 4, 1, 3),  # 18, 19 rejected; 11 - 10 = 1
            ([9], 0, 9, 5, 4, 1),            # n divides the span: nothing rejected
            ([5], 0, 9, 1, 0, 1),            # n == 1 always 0
            ([9], 0, 9, 10, 9, 1),           # n == span
        ]
        for values, low, high, n, expected, end in table:
            with self.subTest(values=values, low=low, high=high, n=n):
                stream = make_stream(values + [low], low, high)
                self.assertEqual(stream.below(n), expected)
                self.assertEqual(stream.position, end)

    def test_uniform_over_accepted_range(self):
        # Offsets 0..7 (below the limit of 8) map to each index of 4 equally often.
        counts = [0] * 4
        for v in range(8):
            counts[make_stream([v], 0, 9).below(4)] += 1
        self.assertEqual(counts, [2, 2, 2, 2])

    def test_n_larger_than_span_is_refused(self):
        stream = make_stream([1, 2], 0, 9)
        with self.assertRaises(SystemExit):
            stream.below(11)

    def test_non_positive_n(self):
        stream = make_stream([1, 2], 0, 9)
        for n in (0, -3):
            with self.subTest(n=n), self.assertRaises(ValueError):
                stream.below(n)

    def test_exhaustion_is_refused(self):
        stream = make_stream([9], 0, 9)   # rejected for n=4, then nothing left
        with self.assertRaises(SystemExit):
            stream.below(4)

    def test_out_of_range_value_is_refused_at_load(self):
        with self.assertRaises(SystemExit):
            make_stream([1, 99], 0, 9)

    def test_cursor_resumes_and_saves_atomically(self):
        stream = make_stream([1, 2, 3, 4], 0, 9, cursor=2)
        self.assertEqual(stream.below(10), 3)
        stream.save_cursor()
        self.assertEqual(json.loads(stream.state_path.read_text())["position"], 3)
        self.assertFalse(stream.state_path.with_name("state.json.tmp").exists())


class ExtractJsonObject(unittest.TestCase):
    def test_table(self):
        table = [
            ('{"a": 1}', '{"a": 1}'),
            ('Sure, here:\n```json\n{"a": 1}\n```\nbye', '{"a": 1}'),
            ('{"a": {"b": 2}} trailing {"c": 3}', '{"a": {"b": 2}}'),
            ('{"a": "x } y"}', '{"a": "x } y"}'),           # brace inside a string
            ('{"a": "quote \\" } still"}', '{"a": "quote \\" } still"}'),  # escaped quote
            ('{"a": 1', None),                               # unbalanced
            ("no braces", None),
            ("", None),
        ]
        for text, expected in table:
            with self.subTest(text=text):
                self.assertEqual(gen.extract_json_object(text), expected)


class ParseResponse(unittest.TestCase):
    @staticmethod
    def raw(**overrides):
        fields = {name: "x" for name in gen.RESPONSE_SCHEMA["required"]}
        fields.update({"post_text": "a post", "entities_referenced": [],
                       "naming_form_used": None})
        fields.update(overrides)
        return json.dumps(fields)

    def test_valid(self):
        parsed, error = gen.parse_response(self.raw())
        self.assertIsNone(error)
        self.assertEqual(parsed["post_text"], "a post")
        self.assertIsNone(parsed["naming_form_used"])

    def test_null_string_forms_become_none(self):
        # The literal "null"/"none"/"" must not be scored as a place called null.
        for value in ("null", "None", "NULL", "  ", ""):
            with self.subTest(value=value):
                parsed, error = gen.parse_response(self.raw(naming_form_used=value))
                self.assertIsNone(error)
                self.assertIsNone(parsed["naming_form_used"])

    def test_real_name_is_kept(self):
        parsed, _ = gen.parse_response(self.raw(naming_form_used="Young"))
        self.assertEqual(parsed["naming_form_used"], "Young")

    def test_wrapped_in_chatter(self):
        parsed, error = gen.parse_response("Here you go:\n" + self.raw() + "\nEnjoy")
        self.assertIsNone(error)
        self.assertEqual(parsed["post_text"], "a post")

    def test_failures(self):
        table = [
            (None, "content was null"),
            ("   ", "content was empty"),
            ("no json here", "no balanced JSON object"),
            ('{"post_text": }', "not valid JSON"),
            ("{}", "missing fields"),
            (self.raw(post_text="  "), "post_text was empty"),
            (self.raw(post_text=5), "post_text was empty or not a string"),
        ]
        for raw, fragment in table:
            with self.subTest(raw=raw):
                parsed, error = gen.parse_response(raw)
                self.assertIsNone(parsed)
                self.assertIn(fragment, error)

    def test_non_list_entities_become_empty(self):
        parsed, _ = gen.parse_response(self.raw(entities_referenced="oops"))
        self.assertEqual(parsed["entities_referenced"], [])


class ContainsWordAndLabelLeak(unittest.TestCase):
    def test_contains_word_table(self):
        table = [
            ("I love Panama", "panama", True),
            ("Visiting PANAMÁ today", "Panama", True),       # accents and case
            ("Panamanian hats", "Panama", False),            # whole word only
            ("the Young Turks", "Young", True),
            ("youngster", "Young", False),
            ("hello", "", False),                            # empty needle
            ("hello", "   ", False),
            ("a (film) b", "(film)", True),                  # punctuation inside the needle
            ("x.y", "x", True),
        ]
        for haystack, needle, expected in table:
            with self.subTest(haystack=haystack, needle=needle):
                self.assertEqual(plan.contains_word(haystack, needle), expected)

    def test_underscore_alias_still_works(self):
        self.assertIs(plan._contains_word, plan.contains_word)
        self.assertIs(plan._label_leak, plan.label_leak)
        self.assertEqual(plan._fold("É"), plan.fold("É"))

    def test_label_leak_table(self):
        table = [
            ("Tuvalu Association of Sports", ["Funafuti"], ["Tuvalu"], [], "country"),
            ("Rock-cut tombs of Qaw el-Kebir", ["Tjebu", "Qaw el-Kebir"], [], [], "place name"),
            ("Some Band", ["Young"], ["Uruguay"], ["Rio Negro"], None),
            ("Rio Negro Rowing Club", ["Young"], ["Uruguay"], ["Rio Negro"], "region"),
        ]
        for label, forms, country, regions, fragment in table:
            with self.subTest(label=label):
                got = plan.label_leak(label, forms, country, regions)
                if fragment is None:
                    self.assertIsNone(got)
                else:
                    self.assertIn(fragment, got)

    def test_label_leak_default_regions(self):
        self.assertIsNone(plan.label_leak("Anything", ["x"], ["y"]))

    def test_is_qualified(self):
        self.assertTrue(plan._is_qualified("Las Cruces, NM", []))
        self.assertTrue(plan._is_qualified("Guatemala City", ["Guatemala"]))
        self.assertFalse(plan._is_qualified("Young", ["Uruguay"]))

    def test_distractor_clues_memoised_and_reset(self):
        donor = plan.Place("Q2", "Donor", "city", "X", "Asia", 1, 1, 1,
                           single_clues=[{"clue_label": "Foo Bar"},
                                         {"clue_label": "Uruguay Post"}])
        place = plan.Place("Q1", "Young", "town", "Uruguay", "South America", 1, 1, 1,
                           name_forms=["Young"], country_forms=["Uruguay"])
        first = place.distractor_clues(donor)
        self.assertEqual([c["clue_label"] for c in first], ["Foo Bar"])
        self.assertIs(place.distractor_clues(donor), first)   # cached
        place.country_forms = []
        place.reset_caches()
        self.assertEqual(len(place.distractor_clues(donor)), 2)


class PlanHelpers(unittest.TestCase):
    RULES = plan.ClueRules()

    @staticmethod
    def clue(label="C", leaks=False, toponym=False, label_leak=None, pa=1, qid="Q9"):
        return {"clue_qid": qid, "clue_label": label, "property": "P1",
                "leaks_place_name": leaks, "is_toponym": toponym,
                "label_leak": label_leak, "place_ambiguity": pa}

    def test_clue_excluded_because_table(self):
        types = {"Q9": {"categories": ["village", "island"]}}
        table = [
            (self.clue(), None),
            (self.clue(leaks=True), "leaks the place name (leak_strict)"),
            (self.clue(toponym=True), "is itself a place name (village, island)"),
            (self.clue(label_leak="label contains X"), "label contains X"),
        ]
        for clue, expected in table:
            with self.subTest(clue=clue):
                self.assertEqual(plan.clue_excluded_because(clue, types, self.RULES), expected)
        off = plan.ClueRules(exclude_leaks=False, exclude_toponyms=False,
                             exclude_label_leaks=False)
        self.assertIsNone(plan.clue_excluded_because(self.clue(leaks=True, toponym=True), types, off))

    def test_filter_forms(self):
        place = plan.Place("Q1", "Young", "town", "Uruguay", "South America", 1, 1, 1)
        forms = [
            {"text": "Young", "kind": "exact", "source": "label"},
            {"text": "Young, Rio Negro", "kind": "ambiguous_alias", "source": "alt"},
            {"text": "Yg", "kind": "ambiguous_alias", "source": "alt", "shared_by": ["Q5"]},
            {"text": "Younger", "kind": "exact", "source": "alt"},
        ]
        held = plan.filter_forms("Q1", place, forms, ["Uruguay"], self.RULES)
        self.assertEqual([f["text"] for f in place.surface_forms], ["Young", "Yg"])
        self.assertEqual([h["text"] for h in held], ["Young, Rio Negro", "Younger"])
        self.assertTrue(all(h["record"] == "form_excluded" for h in held))

    def test_ceiling_source(self):
        label_forms = [{"source": "label", "kind": "exact"}]
        self.assertEqual(plan.ceiling_source_of(label_forms, self.RULES), "label")
        self.assertEqual(plan.ceiling_source_of([], self.RULES), "derived")
        self.assertIsNone(plan.ceiling_source_of(label_forms,
                                                 plan.ClueRules(ceiling_plain_name=False)))

    def test_atomic_write_jsonl(self):
        path = Path(tempfile.mkdtemp()) / "sub" / "out.jsonl"
        n = plan.write_jsonl(path, [{"a": 1}, {"b": "é"}])
        self.assertEqual(n, 2)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"a": 1}\n{"b": "é"}\n')
        self.assertFalse(path.with_name("out.jsonl.tmp").exists())


class LatestRows(unittest.TestCase):
    def test_last_row_wins_and_ids_required(self):
        rows = [
            {"item_id": "a", "n": 1},
            {"item_id": "b", "n": 2},
            {"item_id": "a", "n": 3},
            {"n": 4},                      # no item_id: ignored
            {"item_id": "", "n": 5},       # empty id: ignored
        ]
        latest = gen.latest_rows(rows)
        self.assertEqual(set(latest), {"a", "b"})
        self.assertEqual(latest["a"]["n"], 3)

    def test_empty(self):
        self.assertEqual(gen.latest_rows([]), {})


class FinaliseCheck(unittest.TestCase):
    NAMES = {"forms": ["Yg Town"], "country": ["Uruguay", "Uruguayan"],
             "regions": ["Rio Negro"]}

    @staticmethod
    def item(condition, surface=None, clues=(), noise=None):
        return {"condition": condition, "place_name": "Young",
                "surface_form": {"text": surface},
                "clues": [{"clue_label": c} for c in clues],
                "noise_entity": {"clue_label": noise} if noise else None}

    def test_table(self):
        names = self.NAMES
        table = [
            # no-name conditions
            (self.item("A1", clues=["Esteban Conde"]), "Esteban Conde got a coffee", None),
            (self.item("A1", clues=["Esteban Conde"]), "Esteban Conde in Young", ("names_place", 'wrote "Young"')),
            (self.item("A1", clues=["Esteban Conde"]), "Esteban Conde in Yg Town", ("names_place", 'wrote "Yg Town"')),
            (self.item("A3", clues=["X"]), "X is Uruguayan", ("names_place", None)),
            (self.item("A3", clues=["X"]), "X by the Rio Negro", ("names_place", None)),
            (self.item("A1", clues=["Esteban Conde"]), "a post about nothing", ("clue_missing", '"Esteban Conde" is missing')),
            (self.item("A1", clues=["Film (film)"]), "I watched Film last night", None),  # qualifier stripped
            # ambiguous conditions
            (self.item("B", surface="Young"), "Young is lovely", None),
            (self.item("B", surface="Young"), "It is lovely", ("wrong_toponym", 'planned name "Young" is missing')),
            (self.item("B", surface=None), "anything", ("wrong_toponym", 'planned name "None" is missing')),
            (self.item("B", surface="Young"), "Young, Uruguay", ("wrong_toponym", None)),
            (self.item("B", surface="Young"), "Young near the Rio Negro", ("wrong_toponym", None)),
            (self.item("B", surface="Young"), "Young aka Yg Town", ("wrong_toponym", 'also wrote "Yg Town"')),
            (self.item("AB", surface="Young", clues=["Bob"]), "Young has Bob", None),
            (self.item("NOISE_B", surface="Young", noise="Zed"), "Young and Zed", None),
            (self.item("NOISE_B", surface="Young", noise="Zed"), "Young only", ("noise_missing", '"Zed" is missing')),
            # CEILING
            (self.item("CEILING", surface="Young"), "Young!", None),
            (self.item("CEILING", surface="Young"), "nothing", ("missing_name", '"Young" is missing')),
            (self.item("CEILING", surface=None), "visit Young", None),
        ]
        for item, text, expected in table:
            with self.subTest(condition=item["condition"], text=text):
                got = finalise.check(item, text, names)
                if expected is None:
                    self.assertIsNone(got)
                else:
                    status, reason = expected
                    self.assertEqual(got[0], status)
                    if reason:
                        self.assertEqual(got[1], reason)

    def test_surface_containing_other_form_is_not_double_counted(self):
        # The planned name "Young, Rio Negro" contains the place name and the
        # region; neither counts as "another name".
        names = {"forms": [], "country": [], "regions": ["Rio Negro"]}
        item = self.item("B", surface="Young, Rio Negro")
        self.assertIsNone(finalise.check(item, "Back in Young, Rio Negro", names))

    def test_verdict_for(self):
        item = {**self.item("CEILING", surface="Young"), "place_qid": "Q1"}
        aliases = {"Q1": self.NAMES}
        table = [
            (None, ("not_generated", "")),
            ({"error": "boom"}, ("error", "boom")),
            ({"post_text": "   "}, ("empty", "")),
            ({"post_text": "Young!"}, None),
        ]
        for row, expected in table:
            with self.subTest(row=row):
                self.assertEqual(finalise.verdict_for(item, row, aliases), expected)

    def test_error_reason_is_truncated(self):
        item = {**self.item("CEILING", surface="Young"), "place_qid": "Q1"}
        status, reason = finalise.verdict_for(item, {"error": "e" * 500}, {})
        self.assertEqual(status, "error")
        self.assertEqual(len(reason), finalise.ERROR_REASON_CHARS)

    def test_no_coordinates_drop(self):
        a = ({"place_qid": "Q1", "item_id": "i1"}, {}, "t1")
        b = ({"place_qid": "Q2", "item_id": "i2"}, {}, "t2")
        kept, dropped = finalise.split_by_coordinates([a, b], {"Q1": (1.0, 2.0)})
        self.assertEqual(kept, [a])
        self.assertEqual([d["item_id"] for d in dropped], ["i2"])
        report = [{"item_id": "i1", "place_qid": "Q1", "status": "kept", "reason": ""},
                  {"item_id": "i2", "place_qid": "Q2", "status": "kept", "reason": ""}]
        finalise.apply_no_coordinates(report, dropped)
        self.assertEqual([r["status"] for r in report], ["kept", "no_coordinates"])
        self.assertIn("Q2", report[1]["reason"])

    def test_csv_roundtrip_matches_csv_writer_defaults(self):
        text = finalise.csv_text(["a", "b"], [{"a": 1, "b": "x,y"}])
        self.assertEqual(text, 'a,b\r\n1,"x,y"\r\n')


class BuildExemplarsLogic(unittest.TestCase):
    def test_text_filter_table(self):
        # Avoid the spam filter's own behaviour: only clear-cut cases.
        table = [
            ("just had a really nice walk this morning and the sun was out", True),
            ("too short", False),
            ("check this out https://example.com it is a really good one", False),
            ("what a lovely day for a walk @someone in the park today", False),
            ("had a nice walk with Alice in the park today and it was good", False),  # proper noun
            ("I think it is going to rain later so I will take a coat", True),
            ("x " * 70, False),                                       # too long
        ]
        for text, expected in table:
            with self.subTest(text=text[:40]):
                self.assertEqual(bex.passes_text_filter(text), expected)

    def test_acceptable_and_pool(self):
        names = {"a", "reply_in_thread", "photo_caption"}
        cand = {"uri": "u1", "text": "t", "is_reply": False, "has_images": False}
        ok = {"register": "a", "names_anything": False}
        self.assertEqual(bex.acceptable(cand, ok, names), "a")
        self.assertIsNone(bex.acceptable(cand, {**ok, "names_anything": True}, names))
        self.assertIsNone(bex.acceptable(cand, {**ok, "register": "zzz"}, names))
        self.assertIsNone(bex.acceptable(cand, {**ok, "register": "reply_in_thread"}, names))
        self.assertIsNone(bex.acceptable(cand, {**ok, "register": "photo_caption"}, names))

        pool = bex.ExemplarPool({"a"}, per_register=1, batch_size=5)
        pool.take(cand, ok)
        pool.take({**cand, "uri": "u2"}, ok)        # over the cap: ignored
        self.assertEqual(len(pool.chosen["a"]), 1)
        self.assertTrue(pool.full())

    def test_pool_batches_registers_furthest_behind(self):
        pool = bex.ExemplarPool({"personal_update"}, per_register=2, batch_size=2)
        cands = [{"uri": f"u{i}", "text": "plain", "is_reply": False,
                  "has_images": False, "created_at": None} for i in range(3)]
        pool.add_candidates(cands, {})
        first = pool.next_batch()
        self.assertEqual(len(first), 2)
        self.assertEqual(len(pool.next_batch()), 1)
        self.assertEqual(pool.next_batch(), [])

    def test_parse_labels(self):
        batch = [{"uri": "u1"}, {"uri": "u2"}]
        raw = json.dumps({"labels": [{"n": 1, "register": "a", "names_anything": False},
                                     {"n": 9, "register": "b"}]})
        self.assertEqual(bex.parse_labels(raw, batch),
                         {"u1": {"register": "a", "names_anything": False}})
        with self.assertRaises(json.JSONDecodeError):
            bex.parse_labels("{", batch)

    def test_exemplar_lines_empty_chosen_is_empty(self):
        self.assertEqual(bex.exemplar_lines({}), "")


if __name__ == "__main__":
    unittest.main()
