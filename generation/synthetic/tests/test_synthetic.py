"""
Table-driven tests for the pure logic in generation/synthetic.

Run from the repo root:

    python -m unittest discover generation/synthetic/tests

Nothing here touches the network, the API, or the real output/ state/ data/
files: the random stream tests use temporary files.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from collections import Counter
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

    def test_starts_word_table(self):
        table = [
            ("Huronian glaciation", "Huron", True),
            ("Odinarchaeum yellowstonii", "YELL", True),
            ("Kagulu of Buganda", "Uganda", False),          # inside a word
            ("Hangötidningen", "Hango", True),               # accents
            ("hello", "", False),
        ]
        for haystack, needle, expected in table:
            with self.subTest(haystack=haystack, needle=needle):
                self.assertEqual(plan.starts_word(haystack, needle), expected)

    def test_label_leak_table(self):
        table = [
            ("Tuvalu Association of Sports", ["Funafuti"], ["Tuvalu"], [], "country"),
            ("Rock-cut tombs of Qaw el-Kebir", ["Tjebu", "Qaw el-Kebir"], [], [], "place name"),
            ("Some Band", ["Young"], ["Uruguay"], ["Rio Negro"], None),
            ("Rio Negro Rowing Club", ["Young"], ["Uruguay"], ["Rio Negro"], "region"),
            # a place form starting a longer word
            ("Huronian glaciation", ["Lake Huron", "Huron"], [], [], "word starting"),
            ("Métropole lémanique", ["Lake Geneva", "Léman"], [], [], "word starting"),
            # ... but not inside one, not a short form, not a country form
            ("Kagulu of Buganda", ["Lake Victoria"], ["Uganda"], [], None),
            ("Elysian Fields", ["Ely"], [], [], None),
            ("Panamanian hats", ["Colón"], ["Panama"], [], None),
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
    def clue(label="C", toponym=False, label_leak=None, pa=1, qid="Q9"):
        return {"clue_qid": qid, "clue_label": label, "property": "P1",
                "is_toponym": toponym,
                "label_leak": label_leak, "place_ambiguity": pa}

    def test_clue_excluded_because_table(self):
        types = {"Q9": {"categories": ["village", "island"]}}
        table = [
            (self.clue(), None),
            (self.clue(toponym=True), "is itself a place name (village, island)"),
            (self.clue(label_leak="label contains X"), "label contains X"),
        ]
        for clue, expected in table:
            with self.subTest(clue=clue):
                self.assertEqual(plan.clue_excluded_because(clue, types, self.RULES), expected)
        off = plan.ClueRules(exclude_toponyms=False, exclude_label_leaks=False)
        self.assertIsNone(plan.clue_excluded_because(self.clue(toponym=True, label_leak="label contains X"), types, off))

    def test_select_clues_skips_set_naming_one_entity_twice(self):
        place = plan.Place(qid="Q1", name="X", place_type="town", country="C",
                           continent="Europe", places_sharing_name=1, sitelinks=0,
                           wikidata_links=0)
        by_key = {(q, p): {**self.clue(label=q, qid=q), "property": p}
                  for q, p in [("Q2", "P19"), ("Q3", "P19"), ("Q3", "P20"), ("Q4", "P19")]}
        ref = lambda q, p: {"clue_qid": q, "property": p}  # noqa: E731
        best_sets = [
            {"resolves": True, "clues": [ref("Q2", "P19"), ref("Q3", "P19"), ref("Q3", "P20")]},
            {"resolves": True, "clues": [ref("Q2", "P19"), ref("Q3", "P19"), ref("Q4", "P19")]},
        ]
        plan.select_clues("Q1", place, best_sets, by_key, {}, self.RULES)
        self.assertEqual([s["source_set_index"] for s in place.triple_sets], [1])

    def test_target_slate_spreads_before_repeating(self):
        def place(qid, continent):
            return plan.Place(qid=qid, name=qid, place_type="town", country="C",
                              continent=continent, places_sharing_name=1,
                              sitelinks=0, wikidata_links=0)
        places = {f"Q{i}": place(f"Q{i}", "Africa" if i < 6 else "Asia") for i in range(10)}
        feasible = {"B": sorted(places), "A3": ["Q0", "Q1", "Q2"]}
        stream = make_stream(range(1, 1001), 1, 1000)
        slate = plan.target_slate(places, feasible, ["A3", "B"], 7, 6, stream)
        per = lambda c: Counter(q for q, cond, _ in slate if cond == c)  # noqa: E731
        # B: 10 places, 7 items -> 7 distinct places, split 4/3 over continents
        self.assertEqual(sum(per("B").values()), 7)
        self.assertEqual(max(per("B").values()), 1)
        by_cont = Counter(places[q].continent for q in per("B"))
        self.assertEqual(sorted(by_cont.values()), [3, 4])
        # A3: 3 places, 7 items -> 3, 2, 2 (every place twice before any thrice)
        self.assertEqual(sorted(per("A3").values()), [2, 2, 3])
        # replicate indices are 0.. per place
        self.assertEqual(sorted(r for q, c, r in slate if c == "A3" and q == "Q0")[0], 0)

    def test_target_slate_follows_type_targets(self):
        def place(qid, place_type):
            return plan.Place(qid=qid, name=qid, place_type=place_type, country="C",
                              continent="Europe", places_sharing_name=1,
                              sitelinks=0, wikidata_links=0)
        places = {f"T{i}": place(f"T{i}", "town") for i in range(10)}
        places |= {f"L{i}": place(f"L{i}", "lake") for i in range(4)}
        places |= {f"B{i}": place(f"B{i}", "bay") for i in range(4)}
        cfg = {"type_groups": {"bay": "coast"},
               "type_targets": {"town": 0.5, "lake": 0.25, "coast": 0.25}}
        stream = make_stream(range(1, 1001), 1, 1000)
        groups = lambda slate: Counter(plan.type_group(places[q], cfg) for q, _, _ in slate)  # noqa: E731
        slate = plan.target_slate(places, {"A1": sorted(places)}, ["A1"], 8, 1, stream, cfg)
        self.assertEqual(groups(slate), Counter(town=4, lake=2, coast=2))
        # 16 items but only 4 lakes and 4 bays: towns take the rest, not short
        slate = plan.target_slate(places, {"A1": sorted(places)}, ["A1"], 16, 1, stream, cfg)
        self.assertEqual(groups(slate), Counter(town=8, lake=4, coast=4))

    def test_target_slate_takes_preferred_places_first(self):
        places = {f"Q{i}": plan.Place(f"Q{i}", f"Q{i}", "town", "C", "Europe", 1, 50, 0)
                  for i in range(10)}
        prefer = {"Q2", "Q5", "Q7"}
        slate = plan.target_slate(places, {"B": sorted(places)}, ["B"], 5, 1,
                                  make_stream(range(1, 1001), 1, 1000), prefer=prefer)
        chosen = {q for q, _, _ in slate}
        self.assertEqual(len(chosen), 5)
        self.assertTrue(prefer <= chosen)

    def test_country_balance_spreads_over_countries(self):
        places = {f"U{i}": plan.Place(f"U{i}", f"U{i}", "town", "UK", "Europe", 1, 50, 0)
                  for i in range(10)}
        places |= {f"B{i}": plan.Place(f"B{i}", f"B{i}", "town", "Belgium", "Europe", 1, 50, 0)
                   for i in range(2)}
        places["E0"] = plan.Place("E0", "E0", "town", "Estonia", "Europe", 1, 50, 0)
        slate = plan.target_slate(places, {"A1": sorted(places)}, ["A1"], 5, 1,
                                  make_stream(range(1, 1001), 1, 1000),
                                  {"balance_countries": True})
        got = Counter(places[q].country for q, _, _ in slate)
        self.assertEqual(got, Counter(UK=2, Belgium=2, Estonia=1))
        # With a cap of 3 per country, the lake group gives way to towns elsewhere
        places |= {f"L{i}": plan.Place(f"L{i}", f"L{i}", "lake", "UK", "Europe", 1, 50, 0)
                   for i in range(6)}
        cfg = {"balance_countries": True, "max_places_per_country": 3,
               "type_targets": {"town": 0.5, "lake": 0.5}}
        slate = plan.target_slate(places, {"A1": sorted(places)}, ["A1"], 6, 1,
                                  make_stream(range(1, 1001), 1, 1000), cfg)
        got = Counter(places[q].country for q, _, _ in slate)
        self.assertEqual(got, Counter(UK=3, Belgium=2, Estonia=1))

    def test_group_quotas_hand_shortfall_on_by_target(self):
        q = plan.group_quotas({"town": 100, "lake": 2, "park": 100},
                              {"town": 0.5, "lake": 0.25, "park": 0.25}, 20)
        self.assertEqual(q, {"town": 12, "lake": 2, "park": 6})

    def test_core_conditions_share_the_same_places(self):
        def mk(qid):
            p = plan.Place(qid, qid, "town", f"C{qid}", "Europe", 1, 50, 0)
            p.single_clues = [self.clue(label=f"C{qid}", qid=f"C{qid}")]
            p.triple_sets = [{"source_set_index": 0, "clues": [self.clue()] * 3}]
            p.label_form = {"text": qid, "kind": "exact"}
            return p
        places = {f"Q{i}": mk(f"Q{i}") for i in range(12)}
        config = {"plan": {"replicates": {"A1": 1, "A3": 1, "EXPLICIT": 1},
                           "cover_all_places": ["EXPLICIT"], "max_replicates_per_pair": 1,
                           "core_conditions": ["A1", "A3"], "cover_used_places_only": True,
                           "surplus_factor": 1.0},
                  "finalise": {"per_condition": 5}}
        slate, _ = plan.build_slate(places, config, make_stream(range(1, 1001), 1, 1000))
        per = lambda c: {q for q, cond, _ in slate if cond == c}  # noqa: E731
        self.assertEqual(len(per("A1")), 5)
        self.assertEqual(per("A1"), per("A3"))
        self.assertEqual(per("EXPLICIT"), per("A1"))

    def test_b_only_place_is_feasible_wherever_its_clues_allow(self):
        place = plan.Place(qid="Q1", name="London", place_type="town", country="Canada",
                           continent="North America", places_sharing_name=9,
                           sitelinks=86, wikidata_links=0, b_only=True)
        place.surface_forms = [{"text": "London", "kind": "ambiguous_alias"},
                               {"text": "London, Canada", "kind": "exact"}]
        place.label_form = {"text": "London", "kind": "ambiguous_alias"}
        ok_no_clue, reason_no_clue = plan.feasible(place, "B", {"Q1": place})
        self.assertFalse(ok_no_clue)
        self.assertIn("single-clue", reason_no_clue)
        place.single_clues = [self.clue()]
        ok_b, _ = plan.feasible(place, "B", {"Q1": place})
        ok_explicit, _ = plan.feasible(place, "EXPLICIT", {"Q1": place})
        ok_a1, _ = plan.feasible(place, "A1", {"Q1": place})
        ok_a3, reason = plan.feasible(place, "A3", {"Q1": place})
        self.assertTrue(ok_b)
        self.assertTrue(ok_explicit)
        self.assertTrue(ok_a1)
        self.assertFalse(ok_a3)
        self.assertIn("size-3", reason)

    def test_min_sitelinks_floor_spares_older_places(self):
        def mk(qid, sitelinks, added):
            return plan.Place(qid, qid, "town", "C", "Europe", 1, sitelinks, 0, added=added)
        places = {"Q1": mk("Q1", 10, ""), "Q2": mk("Q2", 10, "2026-10-27"),
                  "Q3": mk("Q3", 45, "2026-10-27")}
        cfg = {"min_sitelinks": 30, "min_sitelinks_from": "2026-10-27"}
        dropped = plan.drop_little_known(places, cfg)
        self.assertEqual([r["place_qid"] for r in dropped], ["Q2"])
        self.assertEqual(sorted(places), ["Q1", "Q3"])
        self.assertEqual(plan.drop_little_known(places, {}), [])
        places["Q4"] = mk("Q4", 400, "")
        dropped = plan.drop_little_known(places, {"max_sitelinks": 300})
        self.assertEqual([r["place_qid"] for r in dropped], ["Q4"])

    def test_unknown_clues_are_held_back(self):
        place = plan.Place("Q1", "X", "town", "C", "Europe", 1, 50, 0)
        known, unknown = self.clue(label="K", qid="QK"), self.clue(label="U", qid="QU")
        place.single_clues = [known, unknown]
        place.triple_sets = [{"source_set_index": 0, "clues": [unknown, known, unknown]},
                             {"source_set_index": 1, "clues": [unknown] * 3}]
        path = Path(tempfile.mkdtemp()) / "clues.jsonl"
        path.write_text('{"clue_qid": "QK", "clue_wikipedia_articles": 3}\n'
                        '{"clue_qid": "QU", "clue_wikipedia_articles": 0}\n', encoding="utf-8")
        records = plan.drop_unknown_clues({"Q1": place}, path,
                                          {"min_clue_wikipedia_articles": 1})
        self.assertEqual([c["clue_qid"] for c in place.single_clues], ["QK"])
        self.assertEqual([s["source_set_index"] for s in place.triple_sets], [0])
        self.assertEqual(records[0]["three_clue_sets_held_back"], 1)

    def test_noise_donor_is_any_other_place_outside_the_country(self):
        def mk(qid, continent, country):
            p = plan.Place(qid, f"P{qid}", "town", country, continent, 1, 1, 1)
            p.single_clues = [self.clue(label=f"C{qid}", qid=f"C{qid}")]
            return p
        a, same_country, same_continent, far = (mk("Q1", "Europe", "France"), mk("Q2", "Europe", "France"),
                                                mk("Q3", "Europe", "Spain"), mk("Q4", "Asia", "Japan"))
        places = {p.qid: p for p in (a, same_country, same_continent, far)}
        self.assertEqual(sorted(plan.noise_donors(a, places)), ["Q3", "Q4"])   # never itself or its country
        self.assertEqual([plan.donor_relation(a, p) for p in (same_country, same_continent, far)],
                         ["same_country", "same_continent", "other_continent"])
        a.clues = [{"clue_qid": "CQ3"}]          # a clue of the item's own place is no distractor
        a.reset_caches()
        self.assertEqual(sorted(plan.noise_donors(a, places)), ["Q4"])

    def test_cover_all_places_plans_every_feasible_place(self):
        places = {}
        for qid, continent in (("Q1", "Europe"), ("Q2", "Europe"), ("Q3", "Asia")):
            p = plan.Place(qid, qid, "town", "X", continent, 1, 1, 1)
            p.label_form = {"text": qid, "kind": "exact"}
            places[qid] = p
        config = {"plan": {"replicates": {"EXPLICIT": 1}, "cover_all_places": ["EXPLICIT"],
                           "surplus_factor": 1.0}, "finalise": {"per_condition": 2}}
        slate, _ = plan.build_slate(places, config, None)
        self.assertEqual(sorted(slate), [("Q1", "EXPLICIT", 0), ("Q2", "EXPLICIT", 0), ("Q3", "EXPLICIT", 0)])

    def test_b_conditions_carry_one_clue_and_explicit_none(self):
        specs = plan.CONDITION_SPECS
        self.assertEqual([specs[c]["n_clues"] for c in ("B", "NOISE_B", "EXPLICIT")], [1, 1, 0])

    def test_per_condition_target(self):
        cfg = {"plan": {"surplus_factor": 1.3}, "finalise": {"per_condition": 150}}
        self.assertEqual(plan.per_condition_target(cfg), 195)
        self.assertIsNone(plan.per_condition_target({"plan": {}, "finalise": {"per_condition": None}}))
        self.assertIsNone(plan.per_condition_target({"plan": {}}))

    def test_filter_forms(self):
        place = plan.Place("Q1", "Young", "town", "Uruguay", "South America", 1, 1, 1)
        forms = [
            {"text": "Young", "kind": "exact", "source": "label"},
            {"text": "Young, Rio Negro", "kind": "ambiguous_alias", "source": "alt"},
            {"text": "Yg", "kind": "ambiguous_alias", "source": "alt", "shared_by": ["Q5"]},
            {"text": "Younger", "kind": "exact", "source": "alt"},
        ]
        held = plan.filter_forms("Q1", place, forms, ["Uruguay"], self.RULES)
        self.assertEqual([f["text"] for f in place.surface_forms], ["Young", "Yg", "Younger"])
        self.assertEqual([h["text"] for h in held], ["Young, Rio Negro"])
        self.assertTrue(all(h["record"] == "form_excluded" for h in held))
        self.assertEqual(place.label_form["text"], "Young")

    def test_explicit_uses_the_bare_label_and_records_ambiguity(self):
        place = plan.Place("Q1", "Lagos", "town", "Nigeria", "Africa", 24, 189, 1)
        stream = None   # EXPLICIT draws nothing
        place.label_form = {"text": "Lagos", "kind": "ambiguous_alias", "shared_by": 24,
                            "top_namesake": {"qid": "Q2", "sitelinks": 58}}
        form = plan.choose_surface_form(place, "EXPLICIT", stream)
        self.assertEqual((form["kind"], form["text"], form["name_ambiguous"]),
                         ("exact", "Lagos", True))
        self.assertEqual(form["top_namesake"]["sitelinks"], 58)
        place.label_form = {"text": "Jask", "kind": "exact", "shared_by": 1}
        self.assertFalse(plan.choose_surface_form(place, "EXPLICIT", stream)["name_ambiguous"])

    def test_atomic_write_jsonl(self):
        path = Path(tempfile.mkdtemp()) / "sub" / "out.jsonl"
        n = plan.write_jsonl(path, [{"a": 1}, {"b": "é"}])
        self.assertEqual(n, 2)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"a": 1}\n{"b": "é"}\n')
        self.assertFalse(path.with_name("out.jsonl.tmp").exists())


class PromptConstraint(unittest.TestCase):
    PROMPT = plan.load_config(SYN / "config.yaml")["prompt"]

    @staticmethod
    def item(kind, surface=None, place="Lake Baikal", country="Russia"):
        return {"place_name": place, "place_country": country,
                "surface_form": {"kind": kind, "text": surface}}

    def test_quoted_list(self):
        self.assertEqual(gen.quoted_list([]), "")
        self.assertEqual(gen.quoted_list(["A"]), '"A"')
        self.assertEqual(gen.quoted_list(["A", "B", "C"]), '"A", "B" or "C"')

    def test_ambiguous_form_that_is_the_place_name_is_not_also_banned(self):
        text = gen.build_constraint(self.item("ambiguous_alias", "Lake Baikal"), self.PROMPT)
        self.assertIn('Do not write "Russia" anywhere', text)
        self.assertNotIn('write "Lake Baikal" or', text)

    def test_ambiguous_alias_still_bans_the_place_name(self):
        text = gen.build_constraint(
            self.item("ambiguous_alias", "Baikal Lake", place="Lake Baikal"), self.PROMPT)
        self.assertIn('Do not write "Lake Baikal" or "Russia" anywhere', text)

    def test_several_countries_are_listed_one_by_one(self):
        text = gen.build_constraint(
            self.item("none", place="Dead Sea", country="Israel; Jordan; Palestine"), self.PROMPT)
        self.assertIn('no "Dead Sea", no "Israel", "Jordan" or "Palestine",', text)
        self.assertNotIn(";", text)

    def test_scale_line_is_in_every_themed_prompt(self):
        self.assertIn("never by name", self.PROMPT["scale_line"])
        self.assertIn("abroad", self.PROMPT["scale_line"])

    def test_river_clue_name_is_allowed_only_when_present(self):
        item = self.item("none", place="Oxford", country="United Kingdom")
        self.assertNotIn("exception", gen.build_constraint(item, self.PROMPT))
        item["clues"] = [{"clue_label": "River Cherwell", "is_toponym": True},
                         {"clue_label": "Ada Lovelace", "is_toponym": False}]
        text = gen.build_constraint(item, self.PROMPT)
        self.assertIn('name the river "River Cherwell"', text)
        self.assertNotIn("Ada Lovelace", text)

    def test_only_river_toponyms_pass_the_clue_rules(self):
        rules = plan.ClueRules(toponym_clue_categories=frozenset({"watercourse"}))
        self.assertTrue(rules.allows_toponym(["watercourse"]))
        self.assertTrue(rules.allows_toponym(["body_of_water", "watercourse"]))
        self.assertFalse(rules.allows_toponym(["body_of_water"]))
        self.assertFalse(rules.allows_toponym(["settlement", "watercourse"]))
        self.assertFalse(plan.ClueRules().allows_toponym(["watercourse"]))

    def test_clue_line_uses_plain_relation_words(self):
        clue = {"clue_label": "Ali Hassan Mwinyi", "property": "P19",
                "property_label": "place of birth", "entity_type": "person"}
        self.assertEqual(gen.describe_clue(clue, self.PROMPT).strip(),
                         '- "Ali Hassan Mwinyi" (a person), born here')
        unknown = {**clue, "property": "P9999", "property_label": "some property"}
        self.assertIn("some property", gen.describe_clue(unknown, self.PROMPT))

    def test_untyped_property_gets_no_type_note(self):
        clue = {"clue_label": "Battle of X", "property": "P276",
                "property_label": "location", "entity_type": "place_or_org"}
        self.assertNotIn("(a place or organisation)", gen.describe_clue(clue, self.PROMPT))
        clue = {**clue, "property": "P159", "property_label": "headquarters location"}
        self.assertIn("(a place or organisation)", gen.describe_clue(clue, self.PROMPT))


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
            (self.item("B", surface="Young", clues=["Esteban Conde"]), "Esteban Conde is in Young", None),
            (self.item("B", surface="Young", clues=["Esteban Conde"]), "Young is lovely", ("clue_missing", '"Esteban Conde" is missing')),
            (self.item("NOISE_B", surface="Young", clues=["Esteban Conde"], noise="Zed"), "Esteban Conde and Zed in Young", None),
            (self.item("NOISE_B", surface="Young", clues=["Esteban Conde"], noise="Zed"), "Esteban Conde in Young", ("noise_missing", '"Zed" is missing')),
            # EXPLICIT: the bare name, nothing that adds context
            (self.item("EXPLICIT", surface="Young"), "Young!", None),
            (self.item("EXPLICIT", surface="Young"), "nothing", ("missing_name", 'planned name "Young" is missing')),
            (self.item("EXPLICIT", surface=None), "visit Young", None),
            (self.item("EXPLICIT", surface="Young"), "Young, Uruguay", ("wrong_toponym", None)),
            (self.item("EXPLICIT", surface="Young"), "Young by the Rio Negro", ("wrong_toponym", None)),
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

    def test_surname_drops_numbering_and_qualifiers(self):
        self.assertEqual(finalise.surname(
            "Percival Fredrickstein von Musel Klossowski de Rolo III"), "Rolo")
        self.assertEqual(finalise.surname("Louis Jones Jr."), "Jones")
        self.assertEqual(finalise.surname("Pelé (footballer)"), "Pelé")

    def test_mention_forms(self):
        person = {"clue_label": "Percival Fredrickstein von Musel Klossowski de Rolo III",
                  "is_human": True, "aliases": []}
        org = {"clue_label": "Abeid Amani Karume International Airport",
               "is_human": False, "aliases": ["Zanzibar Airport"]}
        table = [
            (person, "saw Percy Rolo at the market", "Rolo"),
            (person, "saw Percy at the market", None),          # nickname needs the surname
            (org, "flight out of Zanzibar Airport", "Zanzibar Airport"),
            (org, "flight out of Karume airport", None),         # not a label or alias
            ({**person, "is_human": False}, "saw Percy Rolo", None),  # surname rule is for people only
        ]
        for clue, text, expected in table:
            with self.subTest(text=text):
                self.assertEqual(finalise.mention_form(text, clue), expected)

    def test_surface_containing_other_form_is_not_double_counted(self):
        # The planned name "Young, Rio Negro" contains the place name and the
        # region; neither counts as "another name".
        names = {"forms": [], "country": [], "regions": ["Rio Negro"]}
        item = self.item("B", surface="Young, Rio Negro")
        self.assertIsNone(finalise.check(item, "Back in Young, Rio Negro", names))

    def test_verdict_for(self):
        item = {**self.item("EXPLICIT", surface="Young"), "place_qid": "Q1"}
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
        item = {**self.item("EXPLICIT", surface="Young"), "place_qid": "Q1"}
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

    def test_environment_check(self):
        env = finalise.environment_pattern(["pollut", "water (quality|level)", "melt"])
        themed = {**self.item("B", surface="Young"), "env_theme": "air pollution"}
        plain = self.item("B", surface="Young")
        table = [
            (themed, "Young smells of pollution again", None),
            (themed, "Young water levels are way down", None),
            (themed, "Young is lovely today", ("no_environment", None)),
            (themed, "A smelt dinner in Young", ("no_environment", None)),   # "melt" only at a word start
            (plain, "Young is lovely today", None),                          # no theme, no check
        ]
        for item, text, expected in table:
            with self.subTest(text=text):
                got = finalise.check(item, text, self.NAMES, env)
                if expected is None:
                    self.assertIsNone(got)
                else:
                    self.assertEqual(got[0], expected[0])
        # off when there are no terms
        self.assertIsNone(finalise.environment_pattern([]))
        self.assertIsNone(finalise.check(themed, "Young is lovely today", self.NAMES, None))


class WordLimits(unittest.TestCase):
    ITEM = {"condition": "EXPLICIT", "place_name": "Young", "surface_form": {"text": "Young"},
            "clues": [], "noise_entity": None}
    NAMES = {"forms": [], "country": [], "regions": []}

    def test_limits_from_config(self):
        self.assertEqual(finalise.word_limits({"min_words": 7, "max_words": 30}), (7, 30))
        self.assertIsNone(finalise.word_limits({"min_words": 7}))
        self.assertIsNone(finalise.word_limits(None))
        real = plan.load_config(SYN / "config.yaml")
        self.assertEqual(finalise.word_limits(real["finalise"]), (7, 30))

    def test_word_count_check(self):
        six = "Young is so dry this week"
        seven = "Young is so dry this week again"
        thirty = " ".join(["Young"] + ["word"] * 29)
        thirty_one = thirty + " more"
        table = [(six, "length"), (seven, None), (thirty, None), (thirty_one, "length")]
        for text, expected in table:
            with self.subTest(words=len(text.split())):
                got = finalise.check(self.ITEM, text, self.NAMES, None, (7, 30))
                self.assertEqual(got[0] if got else None, expected)
        # off without limits
        self.assertIsNone(finalise.check(self.ITEM, six, self.NAMES, None, None))


class EnvironmentThemes(unittest.TestCase):
    CONFIG = plan.load_config(SYN / "config.yaml")

    def test_places_read_their_themes(self):
        tmp = Path(tempfile.mkdtemp()) / "places.csv"
        tmp.write_text("place_name,type,country,continent,wikidata_id,env_themes\n"
                       "Lake X,lake,Y,Europe,Q1,falling water levels; algal blooms\n"
                       "Town Z,town,Y,Europe,Q2,\n", encoding="utf-8")
        places = plan.load_places(tmp)
        self.assertEqual(places["Q1"].env_themes, ["falling water levels", "algal blooms"])
        self.assertEqual(places["Q2"].env_themes, [])

    def test_prompt_carries_the_theme_only_when_the_item_has_one(self):
        item = {"item_id": "abc", "register": "personal_update", "length_band": "7_to_30_words",
                "place_name": "Lake X", "place_country": "Y", "clues": [], "slots": {},
                "surface_form": {"kind": "exact", "text": "Lake X"}}
        without = gen.build_messages(item, self.CONFIG, {})[-1]["content"]
        self.assertNotIn("What the post is about", without)
        with_theme = gen.build_messages({**item, "env_theme": "falling water levels"},
                                        self.CONFIG, {})[-1]["content"]
        self.assertIn("What the post is about: falling water levels.", with_theme)
        # after the naming constraint, before the style block
        self.assertLess(with_theme.index("Name the place plainly"),
                        with_theme.index("What the post is about"))
        self.assertLess(with_theme.index("What the post is about"),
                        with_theme.index("Kind of post"))

    def test_every_length_band_has_wording(self):
        # The registers' bands (or the default) must all be defined in the prompt.
        bands = self.CONFIG["prompt"]["length_bands"]
        for reg in self.CONFIG["registers"]:
            for band in reg.get("length_bands", plan.DEFAULT_LENGTH_BANDS):
                self.assertIn(band, bands)
        self.assertIn("7 and 30 words", bands["7_to_30_words"])

    def test_item_id_depends_on_the_theme(self):
        payload = {"place_qid": "Q1", "condition": "EXPLICIT", "clues": [],
                   "surface_form": {"text": "X"}, "noise_entity": None,
                   "register": "personal_update", "length_band": "short", "slots": {}}
        a = plan.make_item_id({**payload, "env_theme": "flooding"}, 0)
        b = plan.make_item_id({**payload, "env_theme": "drought"}, 0)
        self.assertNotEqual(a, b)


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

    def test_topical_labels(self):
        batch = [{"uri": "u1"}, {"uri": "u2"}]
        raw = json.dumps({"labels": [
            {"n": 1, "register": "a", "names_anything": False, "subtopic": "debunking"},
            {"n": 2, "register": "a", "names_anything": False}]})
        labels = bex.parse_labels(raw, batch, topical=True)
        self.assertEqual(labels["u1"]["subtopic"], "debunking")
        self.assertEqual(labels["u2"]["subtopic"], "none")

        cand = {"uri": "u1", "text": "t", "is_reply": False, "has_images": False,
                "created_at": None}
        names = {"a"}
        self.assertEqual(bex.acceptable(cand, labels["u1"], names), "a")
        self.assertIsNone(bex.acceptable(cand, labels["u2"], names))   # off topic

        pool = bex.ExemplarPool(names, per_register=2, batch_size=5)
        pool.take(cand, labels["u1"])
        row = json.loads(bex.exemplar_lines(pool.chosen, "environment"))
        self.assertEqual((row["topic"], row["subtopic"]), ("environment", "debunking"))

    def test_topic_prompt_and_keywords(self):
        topic = bex.TOPICS["environment"]
        prompt = bex.label_prompt([{"name": "a", "description": "d"}],
                                  [{"text": "hot again"}], topic)
        self.assertIn('"subtopic"', prompt)
        self.assertIn("experience", prompt)
        self.assertNotIn('"subtopic"', bex.label_prompt([{"name": "a", "description": "d"}],
                                                        [{"text": "x"}]))
        rx = re.compile(topic["keywords"], re.I)
        self.assertTrue(rx.search("the heatwave is back"))
        self.assertFalse(rx.search("lovely sunny day"))


if __name__ == "__main__":
    unittest.main()
