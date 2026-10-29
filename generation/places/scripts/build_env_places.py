#!/usr/bin/env python3
"""
Build data/env_places.csv, the place set for the environmental-claims posts.

    py -3 scripts/build_env_places.py

It takes 01_places_final.csv, keeps the towns, islands, lakes, mountains and
national parks, keeps only the heritage sites among the manmade structures
(stadiums, palaces, museums and the like are left out), and adds an env_themes
column: the environmental topics a post about that place may take, separated
by ";". plan.py draws one per item.

The themes start as one list per type and are then cut where they are plainly
wrong for a place (coral bleaching away from the tropics, glacier retreat on a
mountain with no glacier, fracking in a country that has none, drought-style
lake themes for a salt pan). The rules are the dictionaries below; extend them
rather than editing the CSV. Latitude, for the tropics rule, comes from Wikidata
P625 and is cached in data/env_coords.json.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from io_utils import utf8_stdout  # noqa: E402

DATA = HERE.parent / "data"
SRC = DATA / "01_places_final.csv"
DST = DATA / "env_places.csv"
COORDS = DATA / "env_coords.json"

TROPICS = 27   # |latitude| up to this counts as warm enough for coral bleaching

THEMES = {
    "island": [
        "sea-level rise", "coral bleaching", "plastic pollution and recycling",
        "storm intensity", "overfishing", "drinking water supply",
        "energy transition", "offshore drilling", "seabed mining",
        "cruise-ship tourism", "eco-tourism and sustainable local business",
        "imported textile and fast-fashion waste",
        "corporate environmental impact and greenwashing"],
    "lake": [
        "falling water levels", "algal blooms", "pollution runoff", "drought",
        "warming water", "PFAS and chemical contamination", "wastewater discharge",
        "pesticide runoff", "watershed protection",
        "mining or fracking near the shore", "textile dye and factory discharge",
        "sustainable business and supply chains",
        "corporate pollution and company accountability"],
    "mountain": [
        "tourism erosion", "wildfire", "mining", "logging",
        "greenhouse gas emissions", "pesticides",
        "sustainable outdoor-gear and tourism businesses",
        "corporate environmental impact and greenwashing"],
    "national park": [
        "wildfire", "habitat loss", "poaching", "deforestation", "drought",
        "drilling or mining permits", "logging", "climate change", "pesticides",
        "funding cuts", "sustainable tourism businesses",
        "corporate environmental impact and greenwashing"],
    "town": [
        "flooding", "heat waves", "air pollution and particulate matter",
        "drought", "water shortage", "council election and environmental plan",
        "river or sewage pollution", "wastewater treatment",
        "waste collection and recycling", "landfill or incinerator", "fracking",
        "lead or PFAS in drinking water", "mould or damp housing",
        "pesticide spraying", "tree felling or green-space development",
        "traffic fumes", "toxic chemical releases", "contaminated site cleanup",
        "climate change", "sustainable local businesses and zero-waste shops",
        "fast fashion and textile waste", "textile factory pollution",
        "second-hand and repair culture",
        "corporate pollution and company accountability", "greenwashing"],
    # Types added 2026-10-27 (more_nature.py), each feature inside one country.
    "river": [
        "river or sewage pollution", "dams and hydropower", "drought and low water",
        "flooding", "sand and gravel mining", "pesticide and fertiliser runoff",
        "plastic pollution", "fish die-offs", "industrial discharge",
        "wetland and riverbank loss", "water extraction for farming",
        "corporate pollution and company accountability"],
    "bay": [
        "coastal pollution", "oil spills and shipping", "overfishing",
        "plastic pollution and recycling", "sea-level rise", "algal blooms",
        "coastal development", "port expansion and dredging",
        "eco-tourism and sustainable local business",
        "corporate environmental impact and greenwashing"],
    "peninsula": [
        "coastal erosion", "sea-level rise", "plastic pollution and recycling",
        "overfishing", "storm intensity", "drought", "wildfire",
        "coastal development", "tourism pressure", "farming and pesticide runoff",
        "energy transition", "eco-tourism and sustainable local business",
        "corporate environmental impact and greenwashing"],
    "desert": [
        "desertification", "drought", "heat waves", "groundwater depletion",
        "mining", "solar farm development", "dust storms", "off-road tourism damage",
        "habitat loss", "corporate environmental impact and greenwashing"],
    # Types added for the balanced set (more_balance.py), 2026-10-27.
    "beach": [
        "beach erosion", "plastic pollution and recycling", "sewage and water quality",
        "sea-level rise", "overcrowding and tourism pressure", "coastal development",
        "sustainable tourism businesses", "corporate environmental impact and greenwashing"],
    "lagoon": [
        "algal blooms", "pollution runoff", "falling water levels", "overfishing",
        "coastal development", "wetland and riverbank loss", "warming water",
        "corporate pollution and company accountability"],
    "reef": [
        "coral bleaching", "warming water", "overfishing", "plastic pollution and recycling",
        "cruise-ship tourism", "reef restoration", "eco-tourism and sustainable local business",
        "corporate environmental impact and greenwashing"],
    "protected area": [
        "habitat loss", "poaching", "wildfire", "drought", "invasive species",
        "drilling or mining permits", "logging", "climate change", "funding cuts",
        "rewilding and species comebacks", "sustainable tourism businesses",
        "corporate environmental impact and greenwashing"],
    "forest": [
        "logging", "deforestation", "wildfire", "bark beetle and tree disease", "drought",
        "habitat loss", "rewilding and species comebacks", "tree planting schemes",
        "corporate environmental impact and greenwashing"],
    "glacier": [
        "glacier retreat", "meltwater shortage", "flood risk from glacial lakes",
        "climate change", "tourism pressure", "snowline rise",
        "corporate environmental impact and greenwashing"],
    "wetland": [
        "wetland drainage", "drought", "falling water levels", "pollution runoff",
        "bird migration and habitat loss", "invasive species", "water extraction for farming",
        "rewilding and species comebacks", "corporate pollution and company accountability"],
    "canyon": [
        "tourism erosion", "flash floods", "drought", "mining", "dams and hydropower",
        "wildfire", "sustainable outdoor-gear and tourism businesses",
        "corporate environmental impact and greenwashing"],
    "waterfall": [
        "falling water levels", "dams and hydropower", "drought", "tourism pressure",
        "pollution upstream", "habitat loss", "sustainable tourism businesses",
        "corporate environmental impact and greenwashing"],
    "valley": [
        "flooding", "drought", "pesticide and fertiliser runoff", "air pollution and particulate matter",
        "water extraction for farming", "wildfire", "landslides", "tourism pressure",
        "sustainable local businesses and zero-waste shops",
        "corporate pollution and company accountability"],
    "plateau": [
        "drought", "desertification", "mining", "overgrazing", "groundwater depletion",
        "wildfire", "solar farm development", "habitat loss",
        "corporate environmental impact and greenwashing"],
    "heritage": [
        "climate damage", "erosion", "flooding", "heat stress", "tourism wear",
        "air pollution damage", "sustainable tourism businesses",
        "corporate environmental impact and greenwashing"],
}

ARCTIC_QIDS = {"Q26087", "Q25923", "Q79809", "Q226", "Q2030", "Q83232", "Q1763",
               "Q175517", "Q103717", "Q719547", "Q25231", "Q4148644", "Q81178",
               "Q146841", "Q80413", "Q203137"}
ARCTIC_THEMES = ["permafrost thaw", "sea-ice loss", "Arctic oil, gas and mining",
                 "Arctic shipping and cruise traffic"]

# Manmade structures that stay, as heritage sites. A name with "dam" as a word
# is treated as a lake (a reservoir).
HERITAGE = ("leptis", "pyramid", "carthage", "timgad", "angkor", "palmyra", "qumran",
            "great wall", "tikal", "chichen", "machu", "tiwanaku", "saqsaywaman",
            "qubbet", "tendaguru", "flavian", "colosseum", "taj mahal", "kinkaku",
            "wat arun", "hagia")
# A b_only row for this name is never planned (no ambiguous form), so it is left out.
DROP_B_ONLY = ("flavian",)

# Mountains. Each name gets only the snow and ski themes it can plausibly take.
GLACIATED = {"Mount Kilimanjaro", "K2", "Kanchenjunga", "Mount Everest", "Mont Blanc",
             "Denali", "Mount Hood", "Aconcagua", "Cotopaxi volcano", "Huascarán",
             "Mount Kazbek", "Mount Ruapehu"}
SNOWY = {"Mount Etna", "Mount Parnassus", "Mount Olympus", "Popocatépetl",
         "Mount Kumgang", "Mont-Sainte-Anne"}
SKI = {"Mont Blanc", "Mount Hood", "Mont-Sainte-Anne", "Mount Parnassus",
       "Mount Ruapehu", "Mount Etna"}
# Hills, landmarks and volcanoes with no mines, logging or ski slopes to speak of.
LANDMARK_MOUNTAINS = {"Table Mountain", "Mount Royal", "Victoria Peak", "Mount Lofty",
                      "Port Hills", "Parliament Hill", "Uluru", "Piton de la Fournaise",
                      "Mount Vesuvius"}
LANDMARK_THEMES = ["tourism erosion", "wildfire", "air pollution and particulate matter",
                   "sustainable outdoor-gear and tourism businesses",
                   "corporate environmental impact and greenwashing"]

# Places whose type list does not fit, replaced outright.
OVERRIDES = {
    "Etosha Pan": ["falling water levels", "drought", "water shortage",
                   "mining or fracking near the shore",
                   "corporate pollution and company accountability"],
    "Dead Sea": ["falling water levels", "drought", "pollution runoff",
                 "wastewater discharge", "mining or fracking near the shore",
                 "corporate pollution and company accountability",
                 "sustainable business and supply chains"],
    # Islands in lakes and rivers: no sea, reef or cruise-ship themes.
    "Isla del Sol": ["falling water levels", "pollution runoff", "drinking water supply",
                     "plastic pollution and recycling",
                     "eco-tourism and sustainable local business",
                     "corporate environmental impact and greenwashing"],
    "Ada Kaleh": ["falling water levels", "pollution runoff", "drinking water supply",
                  "eco-tourism and sustainable local business",
                  "corporate environmental impact and greenwashing"],
    # A coral cay park is about the reef, not forests or logging.
    "Capricornia Cays National Park": [
        "coral bleaching", "sea-level rise", "storm intensity",
        "plastic pollution and recycling", "climate change", "funding cuts",
        "sustainable tourism businesses",
        "corporate environmental impact and greenwashing"],
}
# Desert parks have no forests to fell.
ARID_PARKS = {"Arches National Park", "Bryce Canyon National Park"}

# Issues that can make any place news, near or far, whatever its fame: a
# little-known island can be in foreign news because of a developer, a
# government or a famous family (2026-10-27). Added to every place; how far a
# given post's claim travels is left to the model (prompt.scale_line).
WIDE_THEMES = [
    "government plans for drilling, fracking or mining",
    "a large construction or infrastructure project",
    "overtourism",
    "national climate policy and broken promises",
    "a company's environmental scandal",
    "protests against an environmental decision",
    "development on protected land",
]


# Fracking is only a live topic where it is done or fought over.
FRACKING_COUNTRIES = {
    "United States", "Canada", "United Kingdom", "Argentina", "Australia", "China",
    "People's Republic of China", "Poland", "Algeria", "Mexico", "Russia",
    "South Africa", "Ukraine", "Romania", "Libya", "New Zealand", "Colombia"}


def latitudes(qids: list[str]) -> dict[str, float]:
    """Latitude per QID from data/env_coords.json, fetching what is missing."""
    cache: dict[str, list[float]] = {}
    if COORDS.exists():
        cache = json.loads(COORDS.read_text(encoding="utf-8"))
    missing = [q for q in qids if q not in cache]
    if missing:
        from wdqs import qid_from_uri, sparql, value_of   # noqa: E402
        for i in range(0, len(missing), 100):
            values = " ".join(f"wd:{q}" for q in missing[i:i + 100])
            for row in sparql(f"SELECT ?p ?c WHERE {{ VALUES ?p {{ {values} }} ?p wdt:P625 ?c }}"):
                match = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", value_of(row, "c") or "")
                if match:
                    cache.setdefault(qid_from_uri(value_of(row, "p")),
                                     [float(match.group(2)), float(match.group(1))])
        COORDS.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {q: v[0] for q, v in cache.items()}


def themes_for(row: dict[str, str], kind: str, lat: float | None) -> list[str]:
    name, qid = row["place_name"], row["wikidata_id"]
    if name in OVERRIDES:
        themes = list(OVERRIDES[name])
    else:
        themes = list(THEMES["island" if kind == "archipelago" else kind])
        if kind in ("island", "archipelago"):
            if lat is not None and abs(lat) <= TROPICS:
                pass
            else:
                themes.remove("coral bleaching")
        if kind == "mountain":
            if name in LANDMARK_MOUNTAINS:
                themes = list(LANDMARK_THEMES)
            else:
                if name in GLACIATED:
                    themes.append("glacier retreat")
                if name in GLACIATED or name in SNOWY:
                    themes += ["snowline rise", "meltwater shortage"]
                if name in SKI:
                    themes.append("ski-resort expansion")
        if kind == "national park" and name in ARID_PARKS:
            themes = [t for t in themes if t not in ("deforestation", "logging")]
        if kind == "town":
            countries = {c.strip() for c in row["country"].split(";")}
            if not countries & FRACKING_COUNTRIES:
                themes.remove("fracking")
    if qid in ARCTIC_QIDS:
        themes += ARCTIC_THEMES
    return themes


def main() -> int:
    utf8_stdout()
    with SRC.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or []) + ["env_themes"]
        rows = list(reader)

    kept = []
    for row in rows:
        name = row["place_name"].lower()
        kind = row["type"]
        if row.get("b_only") == "true" and name.startswith(DROP_B_ONLY):
            continue
        if kind == "manmade structure":
            if "dam" in name.split():
                kind = "lake"
            elif any(h in name for h in HERITAGE):
                kind = "heritage"
            else:
                continue
        kept.append((row, kind))

    lat = latitudes(sorted({row["wikidata_id"] for row, _ in kept}))
    out = []
    for row, kind in kept:
        themes = themes_for(row, kind, lat.get(row["wikidata_id"]))
        themes += [t for t in WIDE_THEMES if t not in themes]
        out.append({**row, "env_themes": ";".join(themes)})

    names = {row["place_name"] for row, _ in kept}
    unmatched = sorted((OVERRIDES.keys() | GLACIATED | SNOWY | SKI | LANDMARK_MOUNTAINS
                        | ARID_PARKS) - names)
    if unmatched:
        print(f"note: rule names with no place: {unmatched}")

    with DST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {len(out)} places to {DST}")
    print(Counter(r["type"] for r in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
