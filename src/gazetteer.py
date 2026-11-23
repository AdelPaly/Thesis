"""Gazetteer lookups: candidate places for a surface form, with population.

One backend, `wikidata`: the existing rate-limited wbsearchentities /
wbgetentities client and the shared src/.wikidata_cache. Candidates are
Wikidata QIDs, so a prediction is comparable to a gold QID, and population is
P1082.

There is no GeoNames backend: it would return GeoNames ids, not QIDs, and
exact-QID match could not be computed for its predictions.

Candidates come back highest population first:

    {"qid", "label", "lat", "lon", "population", "country", "continent",
     "description", "sitelinks"}

The Wikidata search matches the start of a label and any alias, so "Seville"
also returns the Province of Seville and "Gao" returns Kaohsiung. With
`same_name=True` the candidates are the results whose English label is the
surface form itself, when there are any; otherwise all results, as before.
"""
from __future__ import annotations

import unicodedata

import wikidata as wd


class GazetteerError(ValueError):
    pass


def _population(entity: dict) -> int | None:
    """P1082 (population) as an int, or None. Prefers the highest-ranked
    non-deprecated claim; several claims means several census years."""
    best = None
    for claim in entity.get("claims", {}).get("P1082", []):
        if claim.get("rank") == "deprecated":
            continue
        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if not isinstance(val, dict):
            continue
        amount = val.get("amount")
        if amount is None:
            continue
        try:
            n = int(float(str(amount).lstrip("+")))
        except ValueError:
            continue
        if claim.get("rank") == "preferred":
            return n
        best = n if best is None else max(best, n)
    return best


def _plain(name: str) -> str:
    """A name without case, accents or punctuation, for comparing labels (the
    same comparison as src/resolver.py)."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in decomposed if c.isalnum()).casefold()


def named_as(name: str, cands: list[dict]) -> list[dict]:
    """The candidates whose English label is `name` itself, if there are any;
    all of them otherwise (a place found only by an alias, "Bombay")."""
    exact = [c for c in cands if c.get("label") and _plain(c["label"]) == _plain(name)]
    return exact or cands


def _wikidata_candidates(name: str, top_n: int) -> list[dict]:
    qids = wd.search_entities(name, limit=top_n)
    if not qids:
        return []
    entities = wd.fetch_entities(qids)
    placed = {q: e for q, e in entities.items() if wd.get_coords_of(e)}
    if not placed:
        return []
    geo = wd.resolve_country_continent(placed)
    out = []
    for qid, ent in placed.items():
        lat, lon = wd.get_coords_of(ent)
        country, continent = geo.get(qid, ("", ""))
        out.append({
            "qid": qid,
            "label": ent.get("labels", {}).get("en", {}).get("value", ""),
            "lat": lat, "lon": lon,
            "population": _population(ent),
            "country": country, "continent": continent,
            "description": ent.get("descriptions", {}).get("en", {}).get("value", name),
            "sitelinks": len(ent.get("sitelinks", {})),
        })
    return out


def lookup(name: str, backend: str = "wikidata", top_n: int = 10,
           min_population: int = 0, same_name: bool = False) -> list[dict]:
    """Candidate places for a surface form, most populous first. With
    `same_name`, only the results named exactly as the surface form when
    there are any (see the module docstring).

    Candidates with no population at all sort last; a gazetteer entry with no
    population figure is not evidence of a large place. Ties (and the
    all-unpopulated case) break on sitelink count, which is the only
    prominence signal available for those rows.
    """
    if backend != "wikidata":
        raise GazetteerError(f"unknown gazetteer backend {backend!r}")
    cands = _wikidata_candidates(name, top_n)
    if same_name:
        cands = named_as(name, cands)

    if min_population:
        cands = [c for c in cands
                 if (c["population"] or 0) >= min_population]
    cands.sort(key=lambda c: (c["population"] or -1, c["sitelinks"]), reverse=True)
    return cands

