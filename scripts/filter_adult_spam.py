"""Strip adult/porn-bot spam out of the Bluesky English dump.

Bluesky's firehose carries a steady stream of solicitation spam ("follow me
for...", OnlyFans promos, findom bait). This removes it from
``generation/synthetic/data/bluesky_en.jsonl`` while leaving ordinary speech alone.

A flat blocklist does not work here, because the vocabulary overlaps with
perfectly normal posts. Measured over the 1,052,788-row dump:

    "bbc"   3,165 posts -- 68.7% the broadcaster, only 3.9% explicit
    "fuck"  6,806 posts -- overwhelmingly ordinary profanity, not spam
    "ass"   4,914 posts -- "badass", "dumbass", "kick ass"

Blocking those outright would delete thousands of legitimate posts, so terms
are split into two tiers:

STRONG      Vocabulary that is unambiguous in practice ("onlyfans", "findom",
            explicit anatomy). A single match drops the post.

AMBIGUOUS   Terms that are only a signal in combination ("bbc", "sex", "daddy").
            These drop the post only alongside a SOLICIT marker -- the "follow
            me / dm me / link in bio" grammar that distinguishes an advert from
            someone just talking.

Ordinary profanity is deliberately NOT filtered: it is a genuine feature of how
people write online, and this dump exists to show that.

Usage:
    python scripts/filter_adult_spam.py --dry-run     # report only
    python scripts/filter_adult_spam.py               # rewrite the dump
"""

import argparse
import json
import pathlib
import os
import random
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Unambiguous in practice: a single hit is enough to drop the post.
STRONG = r"""\b(
    onlyfans|only\s?fans|fansly|findom|femdom|camgirl|camboy|
    cock|cocks|pussy|cunt|cum|cumming|cumshot|creampie|
    anal|blowjob|blowjobs|deepthroat|handjob|rimjob|gangbang|
    dildo|buttplug|hentai|milf|gilf|bbw|
    slut|sluts|slutty|whore|whores|thot|
    tits|titties|boobs|horny|nudes|dickpic|dick\spic|
    nsfw|xxx|bdsm|sexting|sexcam|escorts?\sservice|
    sugar\s?(daddy|baby|mommy)|
    porn(hub|star)?|pornography
)\b"""

# Only meaningful next to a SOLICIT marker.
AMBIGUOUS = r"""\b(
    bbc|sex|sexy|ass|dick|daddy|naked|nude|kink|kinky|fetish|
    sub|dom|domme|breed|breeding|spank|lewd|thirsty|thirst
)\b"""

# The advertising grammar that separates a promo from a person talking.
SOLICIT = r"""(
    follow\s+me|dm\s+me|dms?\s+open|link\s+in\s+bio|check\s+my\s+(bio|page|profile)|
    join\s+my|subscribe|my\s+page|click\s+(here|the\s+link)|free\s+trial|
    tip\s+me|cash\s?app|venmo|paypal|telegram|snap\s?chat|snap\s+me|
    18\+|verify|promo\s?code|buy\s+my|sell(ing)?\s+(content|vids|pics)|
    worship|goon|findom
)"""

STRONG_RE = re.compile(STRONG, re.I | re.X)
AMBIGUOUS_RE = re.compile(AMBIGUOUS, re.I | re.X)
SOLICIT_RE = re.compile(SOLICIT, re.I | re.X)


def classify(text):
    """Return the reason to drop this post, or None to keep it."""
    if STRONG_RE.search(text):
        return "strong"
    if AMBIGUOUS_RE.search(text) and SOLICIT_RE.search(text):
        return "ambiguous+solicit"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=pathlib.Path,
                    default=ROOT / "generation" / "synthetic" / "data" / "bluesky_en.jsonl",
                    help="dump to clean (default: generation/synthetic/data/bluesky_en.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed without rewriting")
    ap.add_argument("--backup", action="store_true",
                    help="keep the pre-filter file as .jsonl.bak (adds ~370 MB)")
    ap.add_argument("--samples", type=int, default=8,
                    help="removed posts to show for eyeballing (default: 8)")
    args = ap.parse_args()

    src = args.path if args.path.is_absolute() else ROOT / args.path
    tmp = src.with_suffix(".cleaned.tmp")

    kept = dropped = 0
    reasons = {"strong": 0, "ambiguous+solicit": 0}
    shown = []
    rng = random.Random(0)

    with src.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            text = json.loads(line)["text"]
            reason = classify(text)
            if reason:
                dropped += 1
                reasons[reason] += 1
                # Reservoir sample so the examples span the whole file.
                if len(shown) < args.samples:
                    shown.append((reason, text))
                elif rng.random() < args.samples / dropped:
                    shown[rng.randrange(args.samples)] = (reason, text)
            else:
                kept += 1
                fout.write(line)

    total = kept + dropped
    print(f"scanned  {total}")
    print(f"removed  {dropped} ({dropped / total * 100:.2f}%)")
    for r, c in reasons.items():
        print(f"           {c:>6}  {r}")
    print(f"kept     {kept}")

    print("\nsample of removed posts:")
    for reason, text in shown:
        flat = " ".join(text.split())[:100]
        # The Windows console is cp1252; these posts are emoji-heavy, so drop
        # anything unencodable rather than dying on the preview.
        safe = flat.encode("ascii", "replace").decode("ascii")
        print(f"  [{reason}] {safe}")

    if args.dry_run:
        tmp.unlink()
        print("\ndry run - nothing written")
        return

    # os.replace is an atomic rename. shutil.move falls back to copy+unlink,
    # and on Windows the unlink loses to whatever is still scanning the file.
    if args.backup:
        backup = src.with_suffix(".jsonl.bak")
        os.replace(src, backup)
        print(f"\noriginal kept at {backup.name}")
    os.replace(tmp, src)
    print(f"rewrote {src.name} ({kept} rows)")


if __name__ == "__main__":
    main()
