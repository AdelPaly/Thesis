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
import judge                   # noqa: E402
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
        donor = mk_place("Q2", "Donor", country="X", clues=[clue("Foo Bar", "QA"),
                                                            clue("Uruguay Post", "QB")])
        place = mk_place("Q1", "Young", country="Uruguay", country_forms=["Uruguay"])
        first = place.distractor_clues(donor)
        self.assertEqual([c["clue_label"] for c in first], ["Foo Bar"])
        self.assertIs(place.distractor_clues(donor), first)   # cached
        place.country_forms = []
        place.reset_caches()
        self.assertEqual(len(place.distractor_clues(donor)), 2)


def clue(label="C", qid="Q9", toponym=False):
    return {"clue_qid": qid, "clue_label": label, "property": "P19", "is_toponym": toponym,
            "label_leak": None}


def mk_place(qid, name=None, country="C", continent="EU", a="", b="", explicit=True,
             clues=(), b_clues=None, rival=None, country_forms=()):
    clues = list(clues)
    return plan.Place(qid=qid, name=name or qid, place_type="settlement", sub="settlement",
                      country=country, country_qid=f"C-{country}", continent=continent,
                      a_pool=a, b_pool=b, explicit=explicit, clues=clues,
                      b_clues=clues if b_clues is None else b_clues,
                      rival=rival or {}, name_forms=[name or qid],
                      country_forms=list(country_forms))


class PlanHelpers(unittest.TestCase):
    CONFIG = {"plan": {"conditions": ["A1", "A3", "B", "EXPLICIT", "NOISE_A1", "NOISE_B"]}}
    RIVAL = {"qid": "QR", "label": "Young", "sitelinks": 80, "km": 900}

    def test_conditions_follow_the_places_csv_pools(self):
        three = [clue(f"C{i}", f"QC{i}") for i in range(3)]
        places = {
            "Q1": mk_place("Q1", a="plan", b="reserve", clues=three, rival=self.RIVAL),
            "Q2": mk_place("Q2", country="D", a="reserve",
                           clues=[clue(f"D{i}", f"QD{i}") for i in range(3)]),
            "Q3": mk_place("Q3", country="E"),                       # EXPLICIT only (infra): nothing now
            "Q4": mk_place("Q4", country="F"),                       # in no set: nothing
        }
        places["Q3"].place_type = "infra"
        slate, rejections = plan.build_slate(places, self.CONFIG)
        got = {(q, c): pool for q, c, pool in slate}
        self.assertEqual(got[("Q1", "EXPLICIT")], "plan")
        self.assertEqual(got[("Q2", "EXPLICIT")], "reserve")         # reserve place: reserve item
        self.assertFalse([c for q, c in got if q == "Q4"])
        self.assertEqual(got[("Q1", "A1")], "plan")
        self.assertEqual(got[("Q1", "B")], "reserve")
        self.assertEqual(got[("Q2", "A3")], "reserve")
        self.assertNotIn(("Q2", "B"), got)
        self.assertEqual([c for q, c in got if q == "Q3"], [])          # no A or B item to compare with
        self.assertEqual(rejections, [])

    def test_explicit_copies_the_draws_of_its_b_item(self):
        class Cycle:                                   # a different answer every draw
            position = 0
            def pick(self, options): self.position += 1; return options[self.position % len(options)]
            def below(self, n): self.position += 1; return self.position % n
        config = plan.load_config(SYN / "config.yaml")
        places = {"Q1": mk_place("Q1", a="plan", b="plan", clues=[clue("C0", "QC0")], rival=self.RIVAL),
                  "Q2": mk_place("Q2", country="D", a="plan", clues=[clue(f"D{i}", f"QD{i}") for i in range(3)])}
        for p in places.values():
            p.env_themes, p.env_themes_positive = ["flooding", "drought", "smog"], ["a repair cafe", "bike lanes"]
        items, _ = plan.plan_items(places, {**config, "plan": {**config["plan"], "conditions": ["A1", "B", "EXPLICIT"]}},
                                   Cycle(), [(10, 1), (12, 1), (14, 1)])
        by = {(i["place_qid"], i["condition"]): i for i in items}
        keys = ("register", "target_len", "slots", "env_theme", "valence", "scale")
        for qid, twin in (("Q1", "B"), ("Q2", "A1")):           # B when the place has one, else A1
            explicit = by[qid, "EXPLICIT"]
            self.assertEqual(explicit["paired_with"], by[qid, twin]["item_id"])
            self.assertEqual({k: explicit[k] for k in keys}, {k: by[qid, twin][k] for k in keys})
        self.assertNotEqual({k: by["Q1", "A1"][k] for k in keys}, {k: by["Q1", "B"][k] for k in keys})

    def test_infeasible_pairs_are_logged_not_repaired(self):
        rival = self.RIVAL
        places = {
            "Q1": mk_place("Q1", a="plan", clues=[clue()]),             # one clue: no A3
            "Q2": mk_place("Q2", country="D", b="plan", clues=[clue("X", "QX")], b_clues=[],
                           rival=rival),                              # clue linked to rival
            "Q3": mk_place("Q3", "Las Cruces, NM", country="E", b="plan",
                           clues=[clue("Y", "QY")], rival=rival),       # qualified name
        }
        slate, rejections = plan.build_slate(places, self.CONFIG)
        why = {(r["place_qid"], r["condition"]): r["reason"] for r in rejections}
        self.assertIn("needs 3", why[("Q1", "A3")])
        self.assertIn("rival", why[("Q2", "B")])
        self.assertIn("qualifier", why[("Q3", "B")])
        self.assertIn(("Q1", "A1", "plan"), slate)

    def test_clue_choice_is_fixed(self):
        clues = [clue(f"C{i}", f"QC{i}") for i in range(4)]
        place = mk_place("Q1", clues=clues, b_clues=clues[2:])
        self.assertEqual(plan.choose_clues(place, "A1"), clues[:1])
        self.assertEqual(plan.choose_clues(place, "NOISE_A1"), clues[:1])
        self.assertEqual(plan.choose_clues(place, "A3"), clues[:3])
        self.assertEqual(plan.choose_clues(place, "B"), clues[2:3])
        self.assertEqual(plan.choose_clues(place, "EXPLICIT"), [])

    def test_surface_forms(self):
        place = mk_place("Q1", "Lagos", rival={"qid": "Q2", "label": "Lagos", "sitelinks": 58, "km": 4000})
        place.places_sharing_name = 24
        for condition, kind in (("B", "ambiguous_alias"), ("EXPLICIT", "exact")):
            form = plan.choose_surface_form(place, condition)
            self.assertEqual((form["kind"], form["text"], form["name_ambiguous"], form["shared_by"]),
                             (kind, "Lagos", True, 24))
            self.assertEqual(form["top_namesake"]["sitelinks"], 58)
        self.assertEqual(plan.choose_surface_form(place, "A1")["kind"], "none")
        unique = mk_place("Q3", "Jask")
        form = plan.choose_surface_form(unique, "EXPLICIT")
        self.assertFalse(form["name_ambiguous"])
        self.assertIsNone(form["top_namesake"])

    def test_noise_donor_is_any_other_place_outside_the_country(self):
        def mk(qid, continent, country):
            return mk_place(qid, f"P{qid}", country=country, continent=continent,
                            clues=[clue(f"C{qid}", f"C{qid}")])
        a, same_country, same_continent, far = (mk("Q1", "EU", "France"), mk("Q2", "EU", "France"),
                                                mk("Q3", "EU", "Spain"), mk("Q4", "AS", "Japan"))
        places = {p.qid: p for p in (a, same_country, same_continent, far)}
        self.assertEqual(sorted(plan.noise_donors(a, places)), ["Q3", "Q4"])   # never itself or its country
        self.assertEqual([plan.donor_relation(a, p) for p in (same_country, same_continent, far)],
                         ["same_country", "same_continent", "other_continent"])
        a.clues = [{"clue_qid": "CQ3"}]          # a clue of the item's own place is no distractor
        a.reset_caches()
        self.assertEqual(sorted(plan.noise_donors(a, places)), ["Q4"])

    def test_b_conditions_carry_one_clue_and_explicit_none(self):
        specs = plan.CONDITION_SPECS
        self.assertEqual([specs[c]["n_clues"] for c in ("B", "NOISE_B", "EXPLICIT")], [1, 1, 0])

    def test_load_places_joins_csv_and_inputs(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "places.csv").write_text(
            "i,label,type,sub,c,continent,split,lat,lon,sl,pbin,n_fam,cbin,hooks,clues,old,"
            "rival,rival_sl,rival_km,rbin,rank,n_names,A,B,EXPLICIT\n"
            "Q1,Young,settlement,settlement,Q77,SA,dev,-32.7,-57.6,40,mid,2,>=10,,[],,"
            "Q5,90,8000,rival_leads,2,3,plan,reserve,yes\n", encoding="utf-8")
        rec = {"place_qid": "Q1", "name_forms": ["Young", "Young City"], "country": "Uruguay",
               "country_forms": ["Uruguay", "Uruguayan"], "region_forms": ["Río Negro"],
               "clues": [{"clue_qid": "QA", "clue_label": "Esteban Conde", "property": "P19",
                          "family": "people", "sitelinks": 12, "is_human": True},
                         {"clue_qid": "QB", "clue_label": "Club Young", "property": "P159",
                          "family": "org", "sitelinks": 4}],
               "b_clues": ["QB"], "rival_label": "Young", "env_themes": ["flooding"]}
        (tmp / "inputs.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        p = plan.load_places(tmp / "places.csv", tmp / "inputs.jsonl")["Q1"]
        self.assertEqual((p.country, p.country_qid, p.split, p.a_pool, p.b_pool, p.explicit),
                         ("Uruguay", "Q77", "dev", "plan", "reserve", True))
        self.assertEqual(p.rival, {"qid": "Q5", "label": "Young", "sitelinks": 90, "km": 8000})
        self.assertEqual(p.places_sharing_name, 3)
        self.assertEqual([c["clue_qid"] for c in p.b_clues], ["QB"])
        self.assertTrue(p.clues[0]["is_human"])
        self.assertEqual(p.env_themes, ["flooding"])
        rec["place_qid"] = "Q9"
        (tmp / "inputs.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            plan.load_places(tmp / "places.csv", tmp / "inputs.jsonl")

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

    def test_every_scale_has_wording_and_global_names_no_place(self):
        config = plan.load_config(SYN / "config.yaml")
        self.assertEqual(set(config["plan"]["scales"]), set(self.PROMPT["scale_lines"]))
        self.assertIn("do not name any other place", self.PROMPT["scale_lines"]["global"])
        self.assertIn("never by name", self.PROMPT["claim_rules"])
        self.assertIn("other", self.PROMPT["role_hints"])

    def test_place_name_clues_are_allowed_only_when_present(self):
        item = self.item("none", place="Oxford", country="United Kingdom")
        self.assertNotIn("exception", gen.build_constraint(item, self.PROMPT))
        item["clues"] = [{"clue_label": "River Cherwell", "is_toponym": True},
                         {"clue_label": "Cherwell District", "is_toponym": True},
                         {"clue_label": "Ada Lovelace", "is_toponym": False}]
        text = gen.build_constraint(item, self.PROMPT)
        self.assertIn('you may write "River Cherwell" or "Cherwell District"', text)
        self.assertNotIn("Ada Lovelace", text)

    def test_clue_line_uses_plain_relation_words(self):
        clue = {"clue_label": "Lake Kivu", "property": "P206",
                "property_label": "located next to body of water", "entity_type": "place_or_org"}
        line = gen.describe_clue(clue, self.PROMPT).strip()
        self.assertTrue(line.startswith('- "Lake Kivu" (a place or organisation). Its link to the place, for you '
                                        'only, not for the post: lies on its shore or banks.'))
        self.assertIn("The post must make clear that it is here", line)       # the anchor, on every clue
        unknown = {**clue, "property": "P9999", "property_label": "some property"}
        self.assertIn("some property", gen.describe_clue(unknown, self.PROMPT))

    def test_people_come_with_what_they_are_known_for_not_their_link(self):
        clue = {"clue_label": "Ali Hassan Mwinyi", "property": "P19", "entity_type": "person",
                "is_human": True, "is_living": False, "description": "politician"}
        line = gen.describe_clue(clue, self.PROMPT)
        self.assertTrue(line.strip().startswith('- "Ali Hassan Mwinyi" (a person from history), known as: politician.'))
        self.assertIn("acting on this place or its people", line)
        self.assertNotIn("born", line)
        # no description: the type note's wording stands in
        self.assertIn("known as: a living person",
                      gen.describe_clue({**clue, "is_living": True, "description": ""}, self.PROMPT))

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
        self.assertEqual(finalise.word_limits({"min_words": 7, "max_words": 30}), {"default": (7, 30)})
        self.assertEqual(finalise.word_limits({"min_words": 7, "max_words": 30,
                                               "words_by_condition": {"A3": [15, 43]}}),
                         {"default": (7, 30), "A3": (15, 43)})
        self.assertIsNone(finalise.word_limits({"min_words": 7}))
        self.assertIsNone(finalise.word_limits(None))
        real = plan.load_config(SYN / "config.yaml")
        dist = json.loads((SYN / real["paths"]["length_dist"]).read_text(encoding="utf-8"))
        limits = finalise.word_limits(real["finalise"])
        self.assertEqual(limits["default"], (dist["q1"], dist["q3"]))     # 8-28, the quartiles
        self.assertEqual(limits["A3"], (dist["median"], dist["p90"]))     # 15-43, the longer posts
        # the planner draws targets from the same ranges
        self.assertEqual(plan.length_range(real, "A1"), limits["default"])
        self.assertEqual(plan.length_range(real, "A3"), limits["A3"])

    def test_condition_range_applies(self):
        twenty_nine = " ".join(["word"] * 29)
        limits = {"default": (8, 28), "A3": (15, 43)}
        got = finalise.check({**self.ITEM, "condition": "A1", "surface_form": {}}, twenty_nine,
                             self.NAMES, None, limits)
        self.assertEqual(got[0], "length")
        self.assertIsNone(finalise.check({**self.ITEM, "condition": "A3", "surface_form": {}},
                                         twenty_nine, self.NAMES, None, limits))

    def test_word_count_check(self):
        six = "Young is so dry this week"
        seven = "Young is so dry this week again"
        thirty = " ".join(["Young"] + ["word"] * 29)
        thirty_one = thirty + " more"
        table = [(six, "length"), (seven, None), (thirty, None), (thirty_one, "length")]
        for text, expected in table:
            with self.subTest(words=len(text.split())):
                got = finalise.check(self.ITEM, text, self.NAMES, None, {"default": (7, 30)})
                self.assertEqual(got[0] if got else None, expected)
        # off without limits
        self.assertIsNone(finalise.check(self.ITEM, six, self.NAMES, None, None))


class EnvironmentThemes(unittest.TestCase):
    CONFIG = plan.load_config(SYN / "config.yaml")

    def test_prompt_carries_the_theme_only_when_the_item_has_one(self):
        item = {"item_id": "abc", "register": "personal_update", "target_len": 12,
                "place_name": "Lake X", "place_country": "Y", "clues": [], "slots": {},
                "surface_form": {"kind": "exact", "text": "Lake X"}}
        without = gen.build_messages(item, self.CONFIG, {})[-1]["content"]
        self.assertNotIn("What the post is about", without)
        with_theme = gen.build_messages({**item, "env_theme": "falling water levels"},
                                        self.CONFIG, {})[-1]["content"]
        self.assertIn("What the post is about: start from falling water levels,", with_theme)
        # after the naming constraint, before the style block
        self.assertLess(with_theme.index("Name the place plainly"),
                        with_theme.index("What the post is about"))
        self.assertLess(with_theme.index("What the post is about"),
                        with_theme.index("Kind of post"))

    def test_length_line_uses_the_target_and_old_items_keep_their_band(self):
        item = {"item_id": "abc", "register": "personal_update", "place_name": "Lake X",
                "place_country": "Y", "clues": [], "slots": {},
                "surface_form": {"kind": "exact", "text": "Lake X"}}
        new = gen.build_messages({**item, "target_len": 12}, self.CONFIG, {})[-1]["content"]
        self.assertIn("Length: about 12 words", new)
        old = gen.build_messages({**item, "length_band": "7_to_30_words"}, self.CONFIG, {})[-1]["content"]
        self.assertIn("between 7 and 30 words", old)

    def test_scale_role_and_rules_follow_the_theme(self):
        item = {"item_id": "abc", "register": "personal_update", "place_name": "X",
                "place_country": "Y", "slots": {}, "target_len": 12, "env_theme": "flooding",
                "scale": "local", "role": "company;other", "surface_form": {"kind": "none"},
                "clues": [{"clue_label": "Acme", "property": "P159", "role": "company"},
                          {"clue_label": "Ann Lee", "property": "P19", "role": "politician",
                           "is_human": True}]}
        text = gen.build_messages(item, self.CONFIG, {})[-1]["content"]
        hints = self.CONFIG["prompt"]["role_hints"]
        self.assertIn(self.CONFIG["prompt"]["scale_lines"]["local"].strip(), text)
        self.assertIn(f'- "Acme": {hints["company"]}', text)          # one hint per name
        self.assertIn(f'- "Ann Lee": {hints["politician"]}', text)
        self.assertIn("Do not invent a specific accident", text)
        self.assertLess(text.index("What the post is about"), text.index("Keep it local"))

    def test_target_length_follows_the_counts(self):
        class Fixed:
            def __init__(self, draws): self.draws = iter(draws)
            def below(self, n): return next(self.draws)
        lengths = [(3, 2), (4, 5), (9, 1)]   # 8 posts: draws 0-1 -> 3, 2-6 -> 4, 7 -> 9
        got = [plan.choose_target_len(lengths, Fixed([d])) for d in (0, 1, 2, 6, 7)]
        self.assertEqual(got, [3, 3, 4, 4, 9])

    def test_item_id_depends_on_the_theme_and_scale(self):
        payload = {"place_qid": "Q1", "condition": "EXPLICIT", "clues": [],
                   "surface_form": {"text": "X"}, "noise_entity": None,
                   "register": "personal_update", "target_len": 12, "slots": {}}
        a = plan.make_item_id({**payload, "env_theme": "flooding"}, 0)
        b = plan.make_item_id({**payload, "env_theme": "drought"}, 0)
        c = plan.make_item_id({**payload, "env_theme": "drought", "scale": "global"}, 0)
        d = plan.make_item_id({**payload, "env_theme": "drought", "scale": "global",
                               "valence": "positive"}, 0)
        self.assertEqual(len({a, b, c, d}), 4)

    def test_every_valence_has_wording_and_follows_the_theme(self):
        config = plan.load_config(SYN / "config.yaml")
        self.assertEqual(set(config["plan"]["valences"]), set(self.CONFIG["prompt"]["valence_lines"]))
        item = {"item_id": "abc", "register": "personal_update", "place_name": "X",
                "place_country": "Y", "slots": {}, "target_len": 12, "clues": [],
                "env_theme": "a repair cafe", "valence": "positive", "surface_form": {"kind": "none"}}
        text = gen.build_messages(item, self.CONFIG, {})[-1]["content"]
        line = self.CONFIG["prompt"]["valence_lines"]["positive"].strip()
        self.assertIn(line, text)
        self.assertLess(text.index("What the post is about"), text.index(line))

    def test_registers_fit_the_valence(self):
        class First:
            def pick(self, options): return options[0]
        config = {"registers": [{"name": "local_complaint", "valences": ["negative"]},
                                {"name": "personal_update"}]}
        self.assertEqual(plan.choose_register(config, First(), "positive")["name"], "personal_update")
        self.assertEqual(plan.choose_register(config, First(), "negative")["name"], "local_complaint")
        with self.assertRaises(SystemExit):
            plan.choose_register({"registers": config["registers"][:1]}, First(), "positive")

    def test_positive_items_take_the_positive_themes(self):
        place = plan.Place(qid="Q1", name="X", place_type="settlement", sub="settlement", country="Y",
                           country_qid="Q2", continent="EU", split="test", sitelinks=10, lat=0, lon=0,
                           env_themes=["flooding"], env_themes_positive=["a repair cafe"])
        self.assertEqual(plan.themes_for_valence(place, "positive"), ["a repair cafe"])
        self.assertEqual(plan.themes_for_valence(place, "negative"), ["flooding"])
        with self.assertRaises(SystemExit):
            plan.themes_for_valence(plan.Place(qid="Q1", name="X", place_type="settlement", sub="settlement",
                                               country="Y", country_qid="Q2", continent="EU"), "positive")


class JudgeAndRetries(unittest.TestCase):
    ITEM = {"item_id": "abc", "condition": "A3", "place_qid": "Q1", "place_name": "X",
            "clues": [{"clue_qid": "Q10", "clue_label": "Ann Lee", "role": "politician"},
                      {"clue_qid": "Q11", "clue_label": "Red River", "role": ""}],
            "spare_clues": [{"clue_qid": "Q12", "clue_label": "Bo Kay", "role": "athlete"}],
            "surface_form": {"kind": "none"}, "noise_entity": None}
    OK = {"decorative_names": [], "imagined_reaction": False, "unlisted_people": [],
          "unlisted_places": [], "environmental_claim": True, "coherent": True, "reason": "ok"}

    def test_verdict_order_and_allowed_names(self):
        self.assertIsNone(judge.verdict(self.ITEM, self.OK))
        # a listed name reported as "unlisted" is ignored
        self.assertIsNone(judge.verdict(self.ITEM, {**self.OK, "unlisted_places": ["Red River"]}))
        both = {**self.OK, "unlisted_people": ["Jean Dupont"], "decorative_names": ["Ann Lee"]}
        self.assertEqual(judge.verdict(self.ITEM, both)[0], "judge_person")   # leaks first
        self.assertEqual(judge.verdict(self.ITEM, {**self.OK, "decorative_names": ["Ann Lee"]})[0],
                         "judge_decorative")
        self.assertEqual(judge.verdict(self.ITEM, {**self.OK, "environmental_claim": False})[0],
                         "judge_no_environment")

    def test_unanchored_counts_only_planned_clues(self):
        self.assertEqual(judge.verdict(self.ITEM, {**self.OK, "unanchored_names": ["Ann Lee"]})[0],
                         "judge_unanchored")
        noisy = {**self.ITEM, "noise_entity": {"clue_qid": "Q99", "clue_label": "Far Corp"},
                 "surface_form": {"kind": "ambiguous_alias", "text": "Young"}}
        # the distractor and the place name need no anchor
        self.assertIsNone(judge.verdict(noisy, {**self.OK, "unanchored_names": ["Far Corp", "Young"]}))
        self.assertIn("unanchored_names", judge.JUDGE_SCHEMA["required"])

    def test_invented_setting_is_a_leak(self):
        both = {**self.OK, "invented_setting": ["concrete plant"], "decorative_names": ["Ann Lee"]}
        self.assertEqual(judge.verdict(self.ITEM, both)[0], "judge_setting")      # before decorative
        self.assertIsNone(judge.verdict(self.ITEM, {**self.OK, "invented_setting": [" "]}))
        topic = judge.judge_messages({**self.ITEM, "env_theme": "falling water levels"}, "x",
                                     plan.load_config(SYN / "config.yaml"))[-1]["content"]
        self.assertIn("Topic the writer was given: falling water levels", topic)

    def test_decorative_clue_is_swapped_for_the_next_spare(self):
        work = json.loads(json.dumps(self.ITEM))
        swaps = gen.swap_decorative(work, ["Ann Lee"])
        self.assertEqual([c["clue_label"] for c in work["clues"]], ["Bo Kay", "Red River"])
        self.assertEqual(swaps[0]["from_qid"], "Q10")
        self.assertEqual(work["spare_clues"], [])
        self.assertEqual(gen.swap_decorative(work, ["Red River"]), [])   # no spare left

    def test_retry_seeds_differ_and_try_one_keeps_the_item_seed(self):
        self.assertEqual(gen.try_seed("abcdef0123456789", 1), gen.seed_for_item("abcdef0123456789"))
        self.assertEqual(len({gen.try_seed("abcdef0123456789", n) for n in (1, 2, 3)}), 3)

    def test_reserve_matches_condition_continent_type_and_bin(self):
        def it(q, pool, cont="EU", typ="natural", pbin="mid"):
            return {"place_qid": q, "pool": pool, "condition": "A1", "place_continent": cont,
                    "place_type": typ, "pbin": pbin}
        plan_items = [it("Q1", "plan"), it("Q2", "reserve", cont="AS"), it("Q3", "reserve", pbin="low"),
                      it("Q4", "reserve"), it("Q5", "reserve", typ="island")]
        self.assertEqual(gen.find_reserve(it("Q1", "plan"), plan_items, set()), "Q4")
        self.assertIsNone(gen.find_reserve(it("Q1", "plan"), plan_items, {"Q4"}))

    def test_swap_takes_the_whole_set_and_the_reserve_explicit(self):
        def it(q, cond, pool):
            return {"item_id": f"{q}{cond}", "place_qid": q, "place_name": q, "condition": cond,
                    "pool": pool, "place_continent": "EU", "place_type": "natural", "pbin": "mid"}
        plan_items = [it("Q1", c, "plan") for c in ("A1", "A3", "NOISE_A1", "EXPLICIT")] + \
                     [it("Q9", c, "reserve") for c in ("A1", "A3", "NOISE_A1", "EXPLICIT")]
        rows = [{"accepted": False, "final_status": "judge_incoherent", "_item": plan_items[0]},
                {"accepted": True, "_item": plan_items[1]}]
        todo, swaps = gen.plan_swaps(rows, plan_items, [plan_items[0], plan_items[1]], [])
        self.assertEqual(swaps[0]["replacement_qid"], "Q9")
        self.assertEqual(swaps[0]["conditions"], ["A1", "A3"])          # only what ran for Q1
        self.assertEqual(sorted(t["condition"] for t in todo), ["A1", "A3", "EXPLICIT"])
        self.assertTrue(all(t["swapped_from"] == "Q1" for t in todo))
        # already swapped: nothing more
        self.assertEqual(gen.plan_swaps(rows, plan_items, [plan_items[0]], swaps)[1], [])

    def test_spread_sample_covers_groups(self):
        items = [{"item_id": f"{c}{k}", "condition": c, "place_continent": cont, "place_type": "x"}
                 for c in ("A1", "B") for k, cont in enumerate(["EU"] * 5 + ["AS"] * 5)]
        got = gen.spread_sample(items, 4)
        self.assertEqual(len(got), 8)
        self.assertEqual({i["place_continent"] for i in got if i["condition"] == "A1"}, {"EU", "AS"})

    def test_spare_clues_follow_the_chosen_ones(self):
        p = mk_place("Q1", clues=[clue(f"C{k}", qid=f"C{k}") for k in range(5)])
        self.assertEqual([c["clue_qid"] for c in plan.spare_clues(p, "A1")], ["C1", "C2", "C3", "C4"])
        self.assertEqual([c["clue_qid"] for c in plan.spare_clues(p, "A3")], ["C3", "C4"])
        self.assertEqual(plan.spare_clues(p, "EXPLICIT"), [])

    def test_failed_row_keeps_its_reason_and_swapped_clues_count(self):
        item = {**self.ITEM, "condition": "EXPLICIT", "surface_form": {"text": "X"}, "clues": []}
        row = {"accepted": False, "final_status": "judge_incoherent", "final_reason": "jump",
               "post_text": "X is dry"}
        self.assertEqual(finalise.verdict_for(item, row, {})[0], "judge_incoherent")
        used = finalise.used_item(self.ITEM, {"clues_used": [{"clue_qid": "Q12", "clue_label": "Bo Kay"}]})
        self.assertEqual(used["clues"][0]["clue_qid"], "Q12")


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
