"""Everything the post generator needs about each place in out/places.csv that the file itself does not hold.
Run after reform_places.py:  python place_inputs.py [--out out]
Writes out/place_inputs.jsonl (one record per place) and out/b_need_clues.csv (B places with no usable clue yet).
Queries go through reform_places.sparql and share its cache (out/cache), so a rerun is offline."""
import argparse, csv, json, os, re
from collections import defaultdict

import reform_places as rp

# clue -> place properties and place -> clue properties, as reform_places links them
PROPS = {**rp.FAMILY, **rp.OUT}
PROP_LABEL = {'P19': 'place of birth', 'P20': 'place of death', 'P551': 'residence', 'P937': 'work location',
              'P159': 'headquarters location', 'P740': 'location of formation', 'P276': 'location',
              'P131': 'located in the administrative territorial entity', 'P361': 'part of',
              'P706': 'located in/on physical feature', 'P206': 'located in or next to body of water',
              'P885': 'origin of the watercourse', 'P403': 'mouth of the watercourse', 'P200': 'inflows',
              'P201': 'lake outflow', 'P4552': 'mountain range', 'P6': 'head of government'}
ENTITY_TYPE = {'people': 'person', 'politics': 'person', 'org': 'place_or_org', 'event': 'work_or_event',
               'part': 'place_or_org', 'water': 'place_or_org', 'range': 'place_or_org'}
# Occupation classes -> the kind of person, in order of precedence (politician first). Used only as a hint.
OCCUPATION = {'Q82955': 'politician', 'Q116': 'royal', 'Q2304859': 'royal', 'Q43845': 'business',
              'Q131524': 'business', 'Q15253558': 'activist', 'Q2066131': 'athlete', 'Q901': 'scientist',
              'Q1650915': 'scientist', 'Q36180': 'artist', 'Q483501': 'artist', 'Q639669': 'artist',
              'Q33999': 'artist', 'Q2259532': 'religious', 'Q47064': 'military'}
TOPONYM_FAMILIES = {'part', 'water', 'range'}   # their clues are place names; the prompt allows exactly those names
HUMAN = 'Q5'
# Two-letter country names are kept ("UK"), except those that are everyday English words: the name checks
# ignore case, so "US" would catch every "us".
ENGLISH_2 = {'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no',
             'of', 'ok', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}

# ---------------- environmental themes ----------------
# Theme lists by kind, carried over from places/scripts/build_env_places.py (the v1 set), plus heritage and
# industrial sites. Every place also gets WIDE_THEMES: how far a claim travels is left to the model.
TROPICS = 27   # |latitude| up to this counts as warm enough for coral bleaching
CORP = 'corporate environmental impact and greenwashing'
THEMES = {   # problems: the themes of items planned 'negative'; the good-news themes are in POSITIVE
    'town': ['flooding', 'heat waves', 'air pollution and particulate matter', 'drought', 'water shortage',
             'council election and environmental plan', 'river or sewage pollution', 'wastewater treatment',
             'waste collection and recycling', 'landfill or incinerator', 'fracking', 'lead or PFAS in drinking water',
             'mould or damp housing', 'pesticide spraying', 'tree felling or green-space development', 'traffic fumes',
             'toxic chemical releases', 'contaminated site cleanup', 'climate change', 'fast fashion and textile waste',
             'textile factory pollution', 'corporate pollution and company accountability', 'greenwashing'],
    'island': ['sea-level rise', 'coral bleaching', 'plastic pollution and recycling', 'storm intensity', 'overfishing',
               'drinking water supply', 'energy transition', 'offshore drilling', 'seabed mining', 'cruise-ship tourism',
               'imported textile and fast-fashion waste', CORP],
    'lake': ['falling water levels', 'algal blooms', 'pollution runoff', 'drought', 'warming water',
             'PFAS and chemical contamination', 'wastewater discharge', 'pesticide runoff', 'watershed protection',
             'mining or fracking near the shore', 'textile dye and factory discharge',
             'corporate pollution and company accountability'],
    'mountain': ['tourism erosion', 'wildfire', 'mining', 'logging', 'greenhouse gas emissions', 'pesticides',
                 'snowline rise', CORP],
    'national park': ['wildfire', 'habitat loss', 'poaching', 'deforestation', 'drought', 'drilling or mining permits',
                      'logging', 'climate change', 'pesticides', 'funding cuts', CORP],
    'glacier': ['glacier retreat', 'meltwater shortage', 'flood risk from glacial lakes', 'climate change',
                'tourism pressure', 'snowline rise', CORP],
    'forest': ['logging', 'deforestation', 'wildfire', 'bark beetle and tree disease', 'drought', 'habitat loss', CORP],
    'desert': ['desertification', 'drought', 'heat waves', 'groundwater depletion', 'mining', 'solar farm development',
               'dust storms', 'off-road tourism damage', 'habitat loss', CORP],
    'bay': ['coastal pollution', 'oil spills and shipping', 'overfishing', 'plastic pollution and recycling',
            'sea-level rise', 'algal blooms', 'coastal development', 'port expansion and dredging', CORP],
    'valley': ['flooding', 'drought', 'pesticide and fertiliser runoff', 'air pollution and particulate matter',
               'water extraction for farming', 'wildfire', 'landslides', 'tourism pressure',
               'corporate pollution and company accountability'],
    'heritage': ['climate damage', 'erosion', 'flooding', 'heat stress', 'tourism wear', 'air pollution damage', CORP],
    'energy': ['energy transition', 'power plant emissions',   # coal and gas; wind and solar are 'renewable' 'electricity prices and supply', 'local health worries',
               'river or sea water heating', 'decommissioning and clean-up', 'protests against an energy project',
               'corporate pollution and company accountability', 'greenwashing'],
    'renewable': ['energy transition', 'wind or solar farm expansion', 'birds, bats and habitat', 'noise and the view',
                  'land use and farmland', 'electricity prices and supply', 'protests against an energy project',
                  'subsidies and who profits', 'greenwashing'],
    'nuclear': ['nuclear safety', 'nuclear waste storage', 'river or sea water heating', 'energy transition',
                'decommissioning and clean-up', 'local health worries', 'electricity prices and supply'],
    'dam': ['dams and hydropower', 'falling water levels', 'flooding', 'fish migration and river habitat',
            'drought and low water', 'displaced communities', 'sediment and silt build-up', 'energy transition'],
    'refinery': ['oil refinery emissions', 'toxic chemical releases', 'air pollution and particulate matter',
                 'oil spills', 'flaring', 'local health worries', 'energy transition',
                 'corporate pollution and company accountability'],
    'mine': ['mining', 'tailings and toxic waste', 'water contamination', 'dust and air pollution', 'land subsidence',
             'mine expansion plans', 'local health worries', 'corporate pollution and company accountability'],
    'landfill': ['landfill or incinerator', 'waste collection and recycling', 'smell and pests', 'methane emissions',
                 'groundwater contamination', 'plastic pollution and recycling', 'illegal dumping'],
    'port': ['port expansion and dredging', 'shipping emissions', 'oil spills and shipping', 'coastal pollution',
             'cruise-ship tourism', 'plastic pollution and recycling', 'air pollution and particulate matter',
             'corporate pollution and company accountability'],
}
KIND = {'settlement': 'town', 'island': 'island', 'lake': 'lake', 'reservoir': 'lake', 'mountain': 'mountain',
        'volcano': 'mountain', 'national park': 'national park', 'glacier': 'glacier', 'forest': 'forest',
        'desert': 'desert', 'bay': 'bay', 'gulf': 'bay', 'valley': 'valley', 'archaeological site': 'heritage',
        'ancient city': 'heritage', 'Roman city': 'heritage', 'polis': 'heritage',
        'Ancient Greek archaeological site': 'heritage', 'nuclear plant': 'nuclear', 'hydro plant': 'dam', 'dam': 'dam',
        'oil refinery': 'refinery', 'coal plant': 'energy', 'gas plant': 'energy', 'wind farm': 'renewable',
        'solar plant': 'renewable', 'mine': 'mine', 'landfill': 'landfill', 'port': 'port'}
WIDE_THEMES = ['government plans for drilling, fracking or mining', 'a large construction or infrastructure project',
               'overtourism', 'national climate policy and broken promises', "a company's environmental scandal",
               'protests against an environmental decision', 'development on protected land']
# Good news, for items planned 'positive' (2026-10-30, user: the posts were always negative; people also post about
# clean-ups, new recycling schemes and claimed climate innovations). Claims, like the problems: they need not be true.
POSITIVE = {
    'town': ['a new recycling or deposit-return scheme', 'a repair cafe or tool library', 'a community solar or energy co-op',
             'new bike lanes or electric buses', 'heat pumps, insulation or green roofs', 'cleaner air after a traffic change',
             'sustainable local businesses and zero-waste shops', 'second-hand and repair culture',
             'tree planting or a new green space', "a local company's recycling or take-back scheme",
             'a community garden or composting scheme', 'flood defences or rain gardens that worked'],
    'island': ['reef or seagrass restoration', 'a beach clean-up', 'a plastic ban that is working',
               'solar and batteries replacing diesel', 'eco-tourism and sustainable local business',
               'a marine protected area and fish coming back', 'a rainwater or water-saving scheme'],
    'lake': ['water cleaner than in years', 'fish or birds coming back', 'a wetland or reed-bed filter',
             'a shoreline clean-up', 'less runoff after a ban on fertiliser or chemicals',
             'sustainable business and supply chains'],
    'mountain': ['a restored trail or erosion repair', 'wildlife returning', 'a hut or resort going solar',
                 'sustainable outdoor-gear and tourism businesses', 'a volunteer litter pick on the trails'],
    'national park': ['species comebacks', 'rewilding or a reintroduced species', 'new protection or ranger funding',
                      'sustainable tourism businesses', 'poaching going down', 'a habitat restoration project'],
    'glacier': ['glacier research and monitoring', 'a climate campaign or study starting here',
                'tourism going low-carbon', 'local communities adapting well'],
    'forest': ['rewilding and species comebacks', 'tree planting schemes', 'a forest saved from logging',
               'sustainable forestry or certified timber', 'fire prevention that worked'],
    'desert': ['a solar farm bringing power and jobs', 'tree belts or regreening against desertification',
               'water-saving farming', 'a wildlife conservation success'],
    'bay': ['eco-tourism and sustainable local business', 'seagrass or oyster reef restoration',
            'cleaner water and swimming again', 'ships plugging into shore power', 'dolphins or seals coming back',
            'a beach clean-up'],
    'valley': ['sustainable local businesses and zero-waste shops', 'organic or regenerative farming',
               'river restoration and fish coming back', 'a community energy scheme', 'a flood plain given back to the river'],
    'heritage': ['sustainable tourism businesses', 'restoration work that protects the site', 'low-impact visitor schemes',
                 'research or digital preservation'],
    'energy': ['a switch from coal or gas to renewables', 'the air getting better after a plant closed',
               'carbon capture claims', 'waste heat from the plant warming homes', 'cheaper clean electricity'],
    'renewable': ['new wind or solar capacity', 'community-owned turbines or panels', 'cheaper electricity for locals',
                  'wildlife-friendly design', 'jobs from clean energy'],
    'nuclear': ['low-carbon electricity', 'a safe decommissioning milestone', 'new reactor research or technology',
                'a solution for nuclear waste storage'],
    'dam': ['fish passes or river restoration', 'clean hydropower', 'a dam removal bringing the river back',
            'flood protection that worked'],
    'refinery': ['a switch to biofuels or green hydrogen', 'claims of cut emissions', 'a clean-up of the site',
                 'the plant closing'],
    'mine': ['an old mine turned into a lake or park', 'recycling metals from old electronics', 'claims of cleaner mining',
             'a mine closing and nature coming back'],
    'landfill': ['recycling rates going up', 'landfill gas turned into energy', 'a new composting or sorting plant',
                 'a landfill turned into a park', 'a repair or reuse centre'],
    'port': ['shore power or electric ferries', 'cleaner fuels for ships', 'a harbour clean-up',
             'wildlife back in the harbour', 'a recycling hub at the port'],
}
WIDE_POSITIVE = ['a clean-up day or tree planting', 'an environmental festival or fair', 'a new invention or idea against climate change',
                 'a new law or ban that is working', 'a local group winning a fight for nature',
                 'an organisation offering a new way to recycle products', 'clean energy getting cheaper and better',
                 'a school or university green project']
FRACKING = {'Q30', 'Q16', 'Q145', 'Q414', 'Q408', 'Q148', 'Q36', 'Q262', 'Q96', 'Q159', 'Q258', 'Q212', 'Q218',
            'Q1016', 'Q664', 'Q739'}   # countries where fracking is done or fought over (v1 list)


def strip_country(desc, country_forms):
    """A person's description without the place's country or demonym words: "Jamaican sprinter" -> "sprinter",
    "Prime Minister of Cuba" -> "Prime Minister". The writer must not be handed the country."""
    for form in sorted((f for f in country_forms if len(f) > 2), key=len, reverse=True):
        desc = re.sub(rf'\s*\b(of|in|from)?\s*(the\s+)?{re.escape(form)}\b', '', desc, flags=re.I)
    return re.sub(r'\s+', ' ', desc).strip(' ,-()')


def themes_for(p):
    kind = KIND[p['sub']]
    themes = list(THEMES[kind])
    if kind == 'island' and abs(float(p['lat'])) > TROPICS: themes.remove('coral bleaching')
    if kind == 'town' and p['c'] not in FRACKING: themes.remove('fracking')
    return themes + [t for t in WIDE_THEMES if t not in themes]


def positive_themes_for(p):
    themes = POSITIVE[KIND[p['sub']]]
    return themes + [t for t in WIDE_POSITIVE if t not in themes]


# ---------------- queries ----------------
def by_id(rows, key, value):
    out = defaultdict(list)
    for r in rows:
        if r[value] not in out[r[key]]: out[r[key]].append(r[value])
    return out


def pair_links(pairs, cache):
    """(x, y) -> the PROPS linking them, either way. One fixed predicate per query: QLever times out on a
    variable predicate."""
    out = defaultdict(list)
    for prop in PROPS:
        q = lambda b: ('SELECT DISTINCT ?x ?y WHERE { VALUES (?x ?y) { ' + ' '.join(f'(wd:{x} wd:{y})' for x, y in b) +
                       f' }} {{ ?x wdt:{prop} ?y }} UNION {{ ?y wdt:{prop} ?x }} }}')
        for r in rp.batched(q, pairs, cache, 200):
            out[r['x'], r['y']].append(prop)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out')
    a = ap.parse_args(); cache = os.path.join(a.out, 'cache'); os.makedirs(cache, exist_ok=True)
    places = list(csv.DictReader(open(os.path.join(a.out, 'places.csv'), encoding='utf-8')))
    for p in places: p['clues'] = json.loads(p['clues'])
    ids = [p['i'] for p in places]
    clue_ids = sorted({c[1] for p in places for c in p['clues']})
    countries = sorted({p['c'] for p in places})
    rivals = sorted({p['rival'] for p in places if p['rival']})

    names = by_id(rp.batched(lambda b: f'''SELECT ?i ?n WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i rdfs:label|skos:altLabel ?n
        FILTER(LANG(?n) = "en" || LANG(?n) = "mul") }}''', ids, cache), 'i', 'n')
    regions = by_id(rp.batched(lambda b: f'''SELECT ?i ?n WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i wdt:P131 ?u.
        ?u rdfs:label ?n FILTER(LANG(?n) = "en") }}''', ids, cache), 'i', 'n')
    cnames = by_id(rp.batched(lambda b: f'''SELECT ?i ?n WHERE {{ VALUES ?i {{{rp.V(b)}}}
        {{ ?i rdfs:label|skos:altLabel ?n }} UNION {{ ?i wdt:P1549 ?n }} FILTER(LANG(?n) = "en") }}''', countries, cache), 'i', 'n')
    for forms in cnames.values():                    # "the UK" -> also "UK"; not "the States" -> "States"
        forms += [f[4:] for f in forms if f.lower().startswith('the ') and f[4:].replace('.', '').isupper()
                  and f[4:] not in forms]
    clabel = {r['i']: r['n'] for r in rp.batched(lambda b: f'''SELECT ?i ?n WHERE {{ VALUES ?i {{{rp.V(b)}}}
        ?i rdfs:label ?n FILTER(LANG(?n) = "en") }}''', countries, cache)}
    rlabel = {r['i']: r['n'] for r in rp.batched(lambda b: f'''SELECT ?i ?n WHERE {{ VALUES ?i {{{rp.V(b)}}}
        ?i rdfs:label ?n FILTER(LANG(?n) = "en" || LANG(?n) = "mul") }}''', rivals, cache)}
    caliases = by_id(rp.batched(lambda b: f'''SELECT ?i ?n WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i skos:altLabel ?n
        FILTER(LANG(?n) = "en") }}''', clue_ids, cache), 'i', 'n')
    human = {r['i'] for r in rp.batched(lambda b: f'SELECT ?i WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i wdt:P31 wd:{HUMAN} }}',
                                        clue_ids, cache)}
    # Living people: no date of death and born after 1920. The writer model needs it to bring a person
    # in plausibly (a living footballer gets no statue).
    dead = {r['i'] for r in rp.batched(lambda b: f'SELECT DISTINCT ?i WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i wdt:P570 ?d }}',
                                       sorted(human), cache)}
    born = {r['i']: int(r['y']) for r in rp.batched(lambda b: f'''SELECT ?i (MIN(YEAR(?b)) AS ?y) WHERE {{
        VALUES ?i {{{rp.V(b)}}} ?i wdt:P569 ?b }} GROUP BY ?i''', sorted(human), cache) if r.get('y')}
    living = {q for q in human - dead if born.get(q, 0) > 1920}
    # A person's kind, from their occupations, as the role hint for the writer (reform_places gives people none).
    # The first kind in OCCUPATION order wins: a politician who also wrote books is a politician.
    kinds = defaultdict(set)
    for r in rp.batched(lambda b: f'''SELECT DISTINCT ?i ?c WHERE {{ VALUES ?i {{{rp.V(b)}}}
            VALUES ?c {{{rp.V(list(OCCUPATION))}}} ?i wdt:P106/wdt:P279* ?c }}''', sorted(human), cache):
        kinds[r['i']].add(OCCUPATION[r['c']])
    person_role = {q: next(k for k in dict.fromkeys(OCCUPATION.values()) if k in ks) for q, ks in kinds.items()}
    # What a person is known for (the English description), shown to the writer instead of how they are linked
    # to the place: 85% of person clues are linked by birth or death, and given "born here" the writer kept
    # writing it (2026-10-29, user). The place's own country and demonym words are cut from it per place below.
    pdesc = {r['i']: r['d'] for r in rp.batched(lambda b: f'''SELECT ?i ?d WHERE {{ VALUES ?i {{{rp.V(b)}}}
        ?i schema:description ?d FILTER(LANG(?d) = "en") }}''', sorted(human), cache)}
    # which property links each clue to its place (reform_places keeps only the family)
    linked = pair_links(sorted({(p['i'], c[1]) for p in places for c in p['clues']}), cache)
    # B rule (as reform_places.gate applies it): a clue linked to the top rival by a clue relation can point to both
    to_rival = rp.links([(c[1], p['rival']) for p in places if p['B'] for c in p['clues']], cache)

    out, need = [], []
    for p in places:
        clues = []
        for fam, q, label, sl, role in p['clues']:
            # the family's own property first; P276 clues reform re-filed under 'part' keep P276
            cand = [x for x in linked[p['i'], q] if PROPS[x] == fam] or linked[p['i'], q]
            prop = sorted(cand, key=lambda x: list(PROPS).index(x))[0] if cand else ''
            clues.append(dict(clue_qid=q, clue_label=label, family=fam, property=prop, property_label=PROP_LABEL.get(prop, ''),
                              sitelinks=sl, role=role or person_role.get(q, 'person' if q in human else ''),
                              entity_type=ENTITY_TYPE[fam], is_human=q in human,
                              is_living=q in living if q in human else None,
                              description=strip_country(pdesc.get(q, ''), cnames.get(p['c'], [])) if q in human else '',
                              is_toponym=fam in TOPONYM_FAMILIES, aliases=caliases.get(q, []),
                              linked_to_rival=(q, p['rival']) in to_rival if p['B'] else None))
        b_clues = [c['clue_qid'] for c in clues if p['B'] and not c['linked_to_rival']]
        if p['B'] and not b_clues:
            need.append(dict(i=p['i'], label=p['label'], B=p['B'], rival=p['rival'], clues=len(clues),
                             reason='no clue' if not clues else 'every clue is linked to the rival'))
        out.append(dict(place_qid=p['i'], name_forms=[p['label']] + [n for n in names.get(p['i'], []) if n != p['label']],
                        country_qid=p['c'], country=clabel.get(p['c'], ''),
                        country_forms=[n for n in cnames.get(p['c'], []) if len(n) > 2 or
                                       (len(n) == 2 and n.lower() not in ENGLISH_2)],
                        region_forms=regions.get(p['i'], []), clues=clues, b_clues=b_clues,
                        rival_label=rlabel.get(p['rival'], ''), env_themes=themes_for(p),
                        env_themes_positive=positive_themes_for(p)))

    with open(os.path.join(a.out, 'place_inputs.jsonl'), 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in out)
    with open(os.path.join(a.out, 'b_need_clues.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['i', 'label', 'B', 'rival', 'clues', 'reason']); w.writeheader(); w.writerows(need)
    nb = sum(1 for p in places if p['B'])
    print(f'{len(out)} places; B {nb - len(need)} of {nb} have a clue not linked to the rival; '
          f'{len(need)} listed in b_need_clues.csv; clues with no linking property found: '
          f'{sum(1 for r in out for c in r["clues"] if not c["property"])}')


if __name__ == '__main__': main()
