"""
The rules added after smoke13 (2026-11-02): one line, no emoji or hashtags,
no unplanned names, at most two anchors, anchors rejected on Wikidata, the
new judge verdicts, and the retry prompt carrying the rejection.

    python -m unittest discover generation/synthetic/tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SYN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYN))

import finalise                # noqa: E402
import generate as gen         # noqa: E402
import judge                   # noqa: E402
import validate_anchors as va  # noqa: E402

NAMES = {"forms": [], "country": [], "regions": []}


def item(condition="A", clues=(), noise=None, surface=None, place="Ansan", theme=None):
    return {"condition": condition, "place_qid": "Q1", "place_name": place, "clues": list(clues),
            "noise_entity": noise, "surface_form": {"text": surface} if surface else {}, "env_theme": theme}


def clue(label, qid="Q9", human=False, description=""):
    return {"clue_label": label, "clue_qid": qid, "is_human": human, "description": description,
            "entity_type": "person" if human else "place_or_org"}


class Format(unittest.TestCase):
    def setUp(self):
        finalise._anchor_verdicts = {}          # no anchor_checks.jsonl in the tests

    def status(self, text, it=None):
        got = finalise.check(it or item(clues=[clue("Tomáš Klus", human=True)]), text, NAMES)
        return got[0] if got else None

    def test_emoji_hashtag_and_line_break_fail(self):
        self.assertEqual(self.status("Tomáš Klus fans cleaning the river with us 🌍"), "format")
        self.assertEqual(self.status("Tomáš Klus fans cleaning the river #green"), "format")
        self.assertEqual(self.status("Tomáš Klus fans out\ncleaning the river"), "format")

    def test_plain_line_passes(self):
        self.assertIsNone(self.status("Tomáš Klus fans out cleaning the river again"))

    def test_unplanned_person_fails(self):
        it = item("EXPLICIT", place="Abington Township")
        got = finalise.check(it, "If you're in Abington Township, join the solar co-op. Chris Smith got it moving.",
                             NAMES)
        self.assertEqual(got[0], "unplanned_entity")
        self.assertIn("Smith", got[1])          # "Chris" opens a sentence; the judge also lists unlisted people

    def test_sentence_starts_and_days_are_not_names(self):
        self.assertIsNone(self.status("Out again Monday. Tomáš Klus fans cleaning the river with us"))

    def test_more_than_two_anchors_fail(self):
        it = item("A", clues=[clue("A", "Q2"), clue("B", "Q3"), clue("C", "Q4")])
        self.assertEqual(finalise.check(it, "A B C", NAMES)[0], "too_many_anchors")

    def test_rejected_anchor_fails(self):
        finalise._anchor_verdicts = {("Q1", "Q44256", False): ("elsewhere", "also filed under Goyang")}
        it = item(clues=[clue("Goyang Zaicro FC", "Q44256")])
        self.assertEqual(finalise.check(it, "Goyang Zaicro FC fans picking up litter", NAMES)[0], "anchor_invalid")


class AnchorRules(unittest.TestCase):
    def facts(self, **kw):
        return {"Q9": {"classes": set(), "dated": False, "dissolved": False, "located": set(),
                       "occurrence": False, **kw}}

    def test_past_event_and_elsewhere_and_alias(self):
        c = clue("2000 Summer Olympics")
        self.assertEqual(va.judge_anchor(c, "Q1", self.facts(occurrence=True, dated=True), set(), [], False)[0],
                         "past_event")
        c = clue("Goyang Zaicro FC")
        located = {("Q1", "Ansan"), ("Q5", "Goyang")}
        self.assertEqual(va.judge_anchor(c, "Q1", self.facts(located=located), set(), [], False)[0], "elsewhere")
        self.assertIsNone(va.judge_anchor(c, "Q1", self.facts(located=located), {("Q5", "Q1")}, [], False))
        c = {**clue("2000 Summer Olympics"), "aliases": ["Sydney 2000"]}
        self.assertEqual(va.judge_anchor(c, "Q1", self.facts(), set(), ["Sydney"], False)[0], "alias_leak")

    def test_name_of_another_place_in_the_label(self):
        index = {"Seoul": {"Q8684"}, "Ansan": {"Q1"}}
        found = va.names_in_labels(["Seoul Semiconductor", "Ansan Greeners"], index)
        f = self.facts(place_names=found["Seoul Semiconductor"])
        self.assertEqual(va.judge_anchor(clue("Seoul Semiconductor"), "Q1", f, set(), [], False)[0],
                         "names_other_place")
        f = self.facts(place_names=found["Ansan Greeners"])        # its own place: alias_leak's business
        self.assertIsNone(va.judge_anchor(clue("Ansan Greeners"), "Q1", f, set(), [], False))

    def test_people_are_not_checked_for_elsewhere_or_aliases(self):
        c = {**clue("Mahathir Mohamad", human=True), "aliases": ["4th Prime Minister of Malaysia"]}
        located = {("Q5", "Putrajaya")}
        self.assertIsNone(va.judge_anchor(c, "Q1", self.facts(located=located), set(), ["Malaysia"], False))


class JudgeVerdicts(unittest.TestCase):
    ANSWER = {"decorative_names": [], "unanchored_names": [], "invented_setting": [], "imagined_reaction": False,
              "unlisted_people": [], "unlisted_places": [], "claim_follows_from_names": True, "claim_tied_to_place": True, "listed_fragments": False,
              "media_untied": False, "misused_names": [], "invented_person_claims": [], "on_theme": True,
              "environmental_claim": True, "coherent": True, "reason": "ok"}

    def verdict(self, it, **kw):
        got = judge.verdict(it, {**self.ANSWER, **kw})
        return got[0] if got else None

    def test_each_new_failure(self):
        it = item(clues=[clue("Sam Elliott", human=True), clue("2000 Summer Olympics", "Q8")], theme="wildfire")
        self.assertIsNone(self.verdict(it))
        self.assertEqual(self.verdict(it, invented_person_claims=["Sam Elliott"]), "judge_invented_person")
        self.assertEqual(self.verdict(it, misused_names=["2000 Summer Olympics"]), "judge_misused")
        self.assertEqual(self.verdict(it, claim_tied_to_place=False), "judge_untied_claim")
        self.assertEqual(self.verdict(it, listed_fragments=True), "judge_fragments")
        self.assertEqual(self.verdict(it, claim_follows_from_names=False), "judge_unconnected")
        self.assertIsNone(self.verdict(item(), claim_follows_from_names=False))   # EXPLICIT: no names to connect
        self.assertEqual(self.verdict(it, media_untied=True), "judge_media")
        self.assertEqual(self.verdict(it, on_theme=False), "judge_off_theme")

    def test_quoted_link_must_be_in_the_post_and_not_just_here(self):
        it = item(clues=[clue("Tomáš Klus", human=True)])
        ok = "Tomáš Klus on stage in his home town, and the crowd left rubbish everywhere after the gig"
        good = [{"name": "Tomáš Klus", "link_words": "in his home town",
                 "connection_words": "the crowd left rubbish everywhere after the gig"}]
        self.assertIsNone(judge.verdict(it, {**self.ANSWER, "evidence": good}, ok))
        bare = [{**good[0], "link_words": "here"}]
        self.assertEqual(judge.verdict(it, {**self.ANSWER, "evidence": bare}, ok)[0], "judge_unconnected")
        made_up = [{**good[0], "link_words": "outside his house"}]
        self.assertEqual(judge.verdict(it, {**self.ANSWER, "evidence": made_up}, ok)[0], "judge_unconnected")
        self.assertEqual(judge.verdict(it, {**self.ANSWER, "evidence": []}, ok)[0], "judge_unconnected")
        short = [{**good[0], "connection_words": "the crowd left rubbish... after the gig"}]
        self.assertIsNone(judge.verdict(it, {**self.ANSWER, "evidence": short}, ok))

    def test_distractor_is_not_failed_as_decorative(self):
        it = item("NOISE_A", noise={"clue_label": "Paul the Apostle", "clue_qid": "Q9"})
        self.assertIsNone(self.verdict(it, decorative_names=["Paul the Apostle"]))

    def test_off_theme_reported_apart_from_no_claim(self):
        it = item(theme="wildfire")
        self.assertEqual(self.verdict(it, environmental_claim=False, on_theme=False), "judge_no_environment")
        self.assertEqual(self.verdict(it, on_theme=False), "judge_off_theme")

    def test_judge_sees_what_a_person_is_known_for(self):
        it = item(clues=[clue("Sam Elliott", human=True, description="actor (born 1944)")])
        self.assertIn("known as: actor (born 1944)", judge.describe(it, "Sam Elliott"))
        self.assertIn("a person from history", judge.describe(item(clues=[{**clue("X Y", human=True), "is_living": False}]), "X Y"))


class RetryFeedback(unittest.TestCase):
    def test_feedback_names_the_problem_and_the_post(self):
        cfg = gen.load_config(SYN / "config.yaml")["prompt"]
        text = gen.retry_feedback({"status": "judge_invented_person", "reason": '"Sam Elliott" - bought water rights',
                                   "text": "Sam Elliott's company bought the water rights"}, cfg)
        self.assertIn("Sam Elliott's company bought the water rights", text)
        self.assertIn("tied a named person to a different place", text)
        self.assertIn("bought water rights", text)


class ExplicitCoverage(unittest.TestCase):
    """2026-11-04: an A or B post is kept only if its place's EXPLICIT post is."""

    def entry(self, condition, place):
        return ({"item_id": f"{condition}-{place}", "condition": condition, "place_qid": place}, {}, "text")

    def test_unpaired_a_and_b_posts_are_dropped(self):
        kept = [self.entry("EXPLICIT", "Q1"), self.entry("A", "Q1"), self.entry("NOISE_B", "Q1"),
                self.entry("B", "Q2"), self.entry("NOISE_A", "Q2"), self.entry("EXPLICIT", "Q3")]
        kept_after, dropped = finalise.split_by_explicit(kept)
        self.assertEqual([e[0]["item_id"] for e in dropped], ["B-Q2", "NOISE_A-Q2"])
        self.assertEqual(len(kept_after), 4)               # an EXPLICIT post alone is kept
        report = [{"item_id": e[0]["item_id"], "place_qid": e[0]["place_qid"], "status": "kept"} for e in kept]
        finalise.apply_no_explicit(report, dropped)
        self.assertEqual([r["status"] for r in report].count("no_explicit"), 2)


if __name__ == "__main__":
    unittest.main()
