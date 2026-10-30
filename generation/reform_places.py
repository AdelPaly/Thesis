"""Reform the places set in one run. Stdlib only. Source: QLever Wikidata endpoint (full Wikidata, live updates).
Run:  python reform_places.py --seed 20261028 [--exclude excluded_qids.txt] [--out out]
Every reply is cached in out/cache: a rerun with the same seed and cache gives the same set. The run stops on an
endpoint error instead of skipping data; run it again to resume from the cache."""
import argparse, csv, hashlib, json, math, os, re, time, urllib.parse, urllib.request
from collections import Counter, defaultdict

# ---------------- config (record these values in the thesis) ----------------
FINAL, OVER, RES = 400, 1.0, 0.25            # final per condition; plan = FINAL*OVER; reserve = plan*RES (failed posts are regenerated)
SHARED = .50                                 # share of each A and B plan that must be the same places
SHARE = {'A': {'settlement': .50, 'natural': .30, 'island': .10, 'manmade': .10},   # type mix per condition, same on every continent
         'B': {'settlement': .40, 'natural': .45, 'island': .05, 'manmade': .10}}
MAX_TOWN = .60                               # report warns if settlements exceed this share of all places
CONT_FLOOR = .10                             # continent share ~ sqrt(eligible supply), at least 10 % of the plan
INFRA_PER_CONT = 25                          # EXPLICIT-only infrastructure places per continent (dams, power plants, refineries, mines, ports)
MIN_SL, MIN_POP, MAX_KM2 = 5, 2000, 5000     # size floor; area cap for islands and natural areas
CLUE_MIN_SL, A_CLUES, A_FAMS = 3, 3, 2       # clue needs >=3 Wikipedia articles; A needs 3 clean clue entities from >=2 families
CLUE_BLOCK = ['Q17537576']                   # creative works (albums, songs, films, books) are not clues
EVENT = 'Q1190554'                           # 'event' clues must be occurrences; other things located in the place count as 'part'
RIVAL_MIN_SL, B_MIN_KM = 3, 161              # namesake must be documented and more than 161 km away; B also needs a clue with no FAMILY/OUT link to it
CAP, FLOOR_POP, MAX_WALK = .08, 2e6, 4000    # country cap (share of continent plan); floor for countries >2M; walk limit
FLOOR_TRIES = 20                             # candidates checked per country for its floor place
PBIN = lambda s: 'low' if s < 30 else 'mid' if s < 100 else 'high'
RBIN = lambda r: 'rival_leads' if r <= .5 else 'place_leads' if r >= 2 else 'similar'
CBIN = lambda s: '0' if s == 0 else '1-9' if s < 10 else '>=10'
CONT = {'Q46': 'EU', 'Q48': 'AS', 'Q15': 'AF', 'Q49': 'NA', 'Q18': 'SA', 'Q538': 'OC', 'Q55643': 'OC', 'Q3960': 'OC'}
MAINCONT = {'Q43': 'AS', 'Q232': 'AS', 'Q79': 'AF', 'Q230': 'AS', 'Q227': 'AS', 'Q252': 'AS', 'Q804': 'NA', 'Q399': 'AS', 'Q229': 'EU'}
SETTLE_ROOTS = ['Q486972', 'Q15284']         # human settlement, municipality
NATURAL = {'Q23397': 'lake', 'Q8502': 'mountain', 'Q8072': 'volcano', 'Q46169': 'national park', 'Q35666': 'glacier', 'Q4421': 'forest', 'Q131681': 'reservoir',
           'Q8514': 'desert', 'Q39594': 'bay', 'Q1322134': 'gulf', 'Q39816': 'valley'}
MANMADE = {'Q839954': 'archaeological site', 'Q15661340': 'ancient city', 'Q2202509': 'Roman city', 'Q148837': 'polis',
           'Q93342462': 'Ancient Greek archaeological site'}     # exact classes, no subclasses; no area limit (Wikidata areas are often wrong)
ISLAND = {'Q23442': 'island'}
INFRA = {'Q134447': 'nuclear plant', 'Q15911738': 'hydro plant', 'Q12323': 'dam', 'Q12353044': 'oil refinery', 'Q6558431': 'coal plant',
         'Q2944640': 'gas plant', 'Q194356': 'wind farm', 'Q1003207': 'solar plant', 'Q820477': 'mine', 'Q152810': 'landfill', 'Q44782': 'port'}
BLOCK_CLASSES = ['Q1370598', 'Q692581', 'Q15135589', 'Q105889895', 'Q1530022',   # worship, holy city, pilgrimage, religious site, religious org
                 'Q39614', 'Q5003624', 'Q12735294', 'Q20011797', 'Q152081', 'Q153813', 'Q4895508', 'Q40357']  # cemetery, memorials, camps, battlefield, prison
ATROCITY = ['Q41397', 'Q3199915', 'Q2223653', 'Q152081', 'Q153813']   # genocide, massacre, terror attack, camps
KNOWN_FOR = .5                               # drop a place if such an event in it has >= 50 % of the place's sitelinks
SUBDIV = ['Q123705', 'Q2983893', 'Q188509', 'Q17051044']    # neighbourhood, quarter, suburb, mahalle: below town level
BIG = ['Q1549591', 'Q5119', 'Q1637706']                      # big city, capital, million city: a settlement inside one is a district
BIG_KM = 25                                  # ... if it is also within 25 km of that city (keeps towns in large Chinese prefectures)
DEV = .20                                    # share of places in the dev split, fixed by place before generation
BLOCK_COUNTRIES = ['Q1246', 'Q865', 'Q219060', 'Q40362', 'Q6250', 'Q23334', 'Q23427', 'Q907112', 'Q23681', 'Q34754']  # disputed: country label is contested
FAMILY = {'P19': 'people', 'P20': 'people', 'P551': 'people', 'P937': 'people', 'P159': 'org', 'P740': 'org',
          'P276': 'event', 'P131': 'part', 'P361': 'part', 'P706': 'part', 'P206': 'water', 'P885': 'water', 'P403': 'water'}  # clue -> place
OUT = {'P200': 'water', 'P201': 'water', 'P4552': 'range', 'P6': 'politics'}                                  # place -> clue (P6: mayor/head)
NOT_PART = []                                # classes a 'part' clue may not be (places inside the place); reclue.py fills this
ROLE = {**{c: 'industry' for c in INFRA}, 'Q159719': 'industry', 'Q83405': 'industry', 'Q1248784': 'transport', 'Q12280': 'transport',
        'Q4830453': 'company', 'Q7278': 'party', **{c: 'nature' for c in NATURAL}, 'Q4022': 'nature', 'Q473972': 'nature',
        'Q2385804': 'education', 'Q33506': 'culture'}   # a hint for the writer only; it never changes which clue is picked
NOT_COMPANY = ['Q2385804', 'Q31855', 'Q16917', 'Q33506']    # Wikidata files universities, institutes, hospitals, museums under 'business'
GENERIC = re.compile(r'\b(lake|loch|lac|lago|mount|mont|monte|mountain|mountains|national park|park|river|glacier|volcano|forest|reservoir|island|isle|valley|desert|bay|gulf)\b')
norm = lambda s: ' '.join(s.lower().replace('-', ' ').split())
KIND = {**{c: ('natural', s) for c, s in NATURAL.items()}, **{c: ('island', s) for c, s in ISLAND.items()},
        **{c: ('manmade', s) for c, s in MANMADE.items()}, **{c: ('infra', s) for c, s in INFRA.items()}}
SL_RANGES = [(5, 8), (8, 15), (15, 30), (30, 100), (100, 10 ** 6)]   # big candidate queries run per sitelink range, each checked by COUNT

# ---------------- endpoint ----------------
EP = 'https://qlever.dev/api/wikidata'
PFX = ('PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> '
       'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wikibase: <http://wikiba.se/ontology#> '
       'PREFIX skos: <http://www.w3.org/2004/02/skos/core#> ')
V = lambda ids: ' '.join('wd:' + i for i in ids)
MAX_Q = 6000                                 # longest query sent; qlever.dev refuses URLs over ~8,000 characters (HTTP 414)

def val(v):
    if v.startswith('<'): return v[1:-1].rsplit('/', 1)[-1]
    if v.startswith('"'): return v[1:v.rindex('"')].replace('\\t', ' ').replace('\\"', '"').replace('\\\\', '\\')
    return v

class TooBig(Exception): pass

cpath = lambda q, cache: os.path.join(cache, 'v2-' + hashlib.sha1(q.encode()).hexdigest() + '.json')   # v2: older cache may hold cut-off replies

def full(select, where, cache, lo=MIN_SL, hi=10 ** 6):
    """All rows of SELECT DISTINCT select WHERE where (which binds ?sl), split by sitelink range. Each part is checked
    against a COUNT query; a part whose reply is short or cut off is split again, and the run stops if a single value still fails."""
    out = []
    for a, b in [r for r in SL_RANGES if r[0] < hi and r[1] > lo] if (lo, hi) == (MIN_SL, 10 ** 6) else [(lo, hi)]:
        q = f'SELECT DISTINCT {select} WHERE {{ {where} FILTER(?sl >= {a} && ?sl < {b}) }}'
        try:
            rows = sparql(q, cache)
            n = int(sparql(f'SELECT (COUNT(*) AS ?n) WHERE {{ {{ {q} }} }}', cache)[0]['n'])
            if len(rows) == n: out += rows; continue
            os.remove(cpath(q, cache)); short = f'{len(rows)} of {n} rows'
        except TooBig as e: short = f'cut off: {str(e)[:80]}'
        if b - a <= 1: raise SystemExit(f'reply stays short ({short}) for sitelinks {a}; rerun later')
        m = a * 2 if b > 10 ** 5 else (a + b) // 2
        print(f'  split sitelinks {a}-{b} at {m} ({short})', flush=True)
        out += full(select, where, cache, a, m) + full(select, where, cache, m, b)
    return out

def sparql(q, cache):
    f = cpath(q, cache)
    if os.path.exists(f): return json.load(open(f, encoding='utf-8'))
    req = urllib.request.Request(EP + '?' + urllib.parse.urlencode({'query': PFX + q}),
                                 headers={'Accept': 'application/sparql-results+json', 'User-Agent': 'thesis-places/1.0 (VSE Prague)'})
    cut = 0
    for n in range(30):                              # rate limit: wait and retry. Timeout or cut-off reply: caller splits the batch
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=300).read().decode('utf-8'))   # a cut-off reply fails here
            b = d['results']['bindings']; total = d.get('meta', {}).get('result-size-total')
            if total is not None and int(total) != len(b): raise ConnectionError(f'reply has {len(b)} of {total} rows')
            rows = [{k: v['value'].rsplit('/', 1)[-1] if v['type'] == 'uri' else v['value'] for k, v in r.items()} for r in b]
            with open(f + '.tmp', 'w', encoding='utf-8') as fh: json.dump(rows, fh)
            os.replace(f + '.tmp', f); time.sleep(.5); return rows        # whole file or none, so a stopped run resumes cleanly; pause: shared endpoint
        except Exception as e:
            body = getattr(e, 'read', lambda: b'')()[:300].decode('utf-8', 'replace')
            if 'exception' in body or getattr(e, 'code', None) == 414: raise TooBig(body or str(e))   # QLever's own timeout (429 + JSON body); URL too long
            if getattr(e, 'code', None) is None and (cut := cut + 1) >= 2: raise TooBig(str(e))
            print(f'  wait {min(60 * 2 ** n, 600)}s: {e}', flush=True); time.sleep(min(60 * 2 ** n, 600))
    raise SystemExit('endpoint unreachable for hours; rerun later, the cache keeps finished work')

def batched(make, ids, cache, n=300):
    """Run make(batch) over ids; a batch that is too big is halved. One id that is still too big is logged and dropped."""
    out, todo = [], [ids[i:i + n] for i in range(0, len(ids), n)]
    while todo:
        b = todo.pop()
        if len(b) > 1 and len(PFX + make(b)) > MAX_Q: todo += [b[:len(b) // 2], b[len(b) // 2:]]; continue   # URL length limit
        try: out += sparql(make(b), cache)
        except TooBig as e:
            if len(b) > 1: todo += [b[:len(b) // 2], b[len(b) // 2:]]
            else: print(f'  dropped {b[0]}: reply too big ({str(e)[:80]})', flush=True)
    return out

def pt(w):                                       # WKT point -> (lat, lon); None for unknown value or another globe
    if '(' not in w or 'entity/' in w: return None
    lon, lat = map(float, w[w.index('(') + 1:-1].split()); return lat, lon
def km(a, b):
    (la1, lo1), (la2, lo2) = [tuple(map(math.radians, p)) for p in (a, b)]
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))

# ---------------- candidates ----------------
def pool(cache, countries):
    body = {'settlement': f'VALUES ?root {{{V(SETTLE_ROOTS)}}} ?cls wdt:P279* ?root. ?i wdt:P31 ?cls. ?i wdt:P1082 ?pop. FILTER(?pop >= {MIN_POP})',
            'typed': f'VALUES ?cls {{{V(KIND)}}} ?i wdt:P31 ?cls. OPTIONAL {{ ?i wdt:P2046 ?km2 }}'}
    by = {}
    for kind, b in body.items():
        for r in full('?i ?cls ?sl ?c ?loc ?label ?mul ?pc ?km2', f'''{b} ?i wikibase:sitelinks ?sl.
                ?i wdt:P625 ?loc; wdt:P17 ?c. OPTIONAL {{ ?i rdfs:label ?label FILTER(LANG(?label) = "en") }}
                OPTIONAL {{ ?i rdfs:label ?mul FILTER(LANG(?mul) = "mul") }} OPTIONAL {{ ?i wdt:P30 ?pc }}''', cache):
            t, sub = KIND.get(r['cls'], ('settlement', 'settlement')) if kind == 'typed' else ('settlement', 'settlement')
            if not pt(r.get('loc', '')): continue
            p = by.setdefault(r['i'], dict(i=r['i'], type=t, sub=sub, sl=int(r['sl']), loc=r['loc'], label=r.get('label') or r.get('mul', ''), cs=set(), pcs=set(), km2=0.))
            if kind == 'typed': p.update(type=t, sub=sub)          # an island town counts as island
            p['cs'].add(r['c']); p['pcs'].add(r.get('pc', '')); p['km2'] = max(p['km2'], float(r.get('km2') or 0))
        base = f'{b} ?i wikibase:sitelinks ?sl.'                                    # prescreen: necessary conditions only, the gate decides
        for prop, fam in {**FAMILY, **OUT}.items():
            edge = f'?i wdt:{prop} ?e' if prop in OUT else f'?e wdt:{prop} ?i'
            for r in full('?i', f'{base} {edge}. ?e wikibase:sitelinks ?es. FILTER(?es >= {CLUE_MIN_SL})', cache):
                if r['i'] in by: by[r['i']].setdefault('raw', set()).add(fam)
        for r in full('?i', f'''{base} ?i rdfs:label ?l. FILTER(LANG(?l) = "en")
                ?r rdfs:label ?l. ?r wdt:P625 ?rl. ?r wikibase:sitelinks ?rs. FILTER(?rs >= {RIVAL_MIN_SL} && ?r != ?i)''', cache):
            if r['i'] in by: by[r['i']]['ns'] = True
    out = []
    for p in by.values():
        if len(p['cs']) != 1 or not p['label'] or p['i'] in countries: continue      # multi-country, no English name, or a state itself
        p['c'] = p['cs'].pop(); p['latlon'] = pt(p['loc'])
        pc = [CONT[x] for x in p['pcs'] if x in CONT]
        p['continent'] = pc[0] if len(set(pc)) == 1 else ('AS' if p['latlon'][1] > 60 else 'EU') if p['c'] == 'Q159' \
            else MAINCONT.get(p['c']) or countries.get(p['c'], {}).get('cont')
        if p['continent'] and not (p['type'] in ('island', 'natural') and p['km2'] > MAX_KM2): out.append(p)
    return out

# ---------------- gate: exclusions, clean clue families, namesakes ----------------
def gate(chunk, cache, countries):
    ids, by_sl = [p['i'] for p in chunk], {p['i']: p['sl'] for p in chunk}
    isa = lambda ids, cls: {r['i'] for r in batched(lambda b: f'SELECT DISTINCT ?i WHERE {{ VALUES ?i {{{V(b)}}} VALUES ?x {{{V(cls)}}} ?i wdt:P31/wdt:P279* ?x }}', ids, cache)}
    blocked = {r['i'] for r in batched(lambda b: f'''SELECT DISTINCT ?i WHERE {{ VALUES ?i {{{V(b)}}}
        VALUES ?x {{{V(BLOCK_CLASSES + SUBDIV)}}} ?i wdt:P31/wdt:P279* ?x }}''', ids, cache)}
    by_ll = {p['i']: p['latlon'] for p in chunk}
    for r in batched(lambda b: f'''SELECT DISTINCT ?i ?ul WHERE {{ VALUES ?i {{{V(b)}}} VALUES ?y {{{V(BIG)}}}
            ?i wdt:P131+ ?u. ?u wdt:P31 ?y; wdt:P625 ?ul. ?i wdt:P1082 [] FILTER(?u != ?i) }}''', ids, cache):
        if pt(r['ul']) and km(by_ll[r['i']], pt(r['ul'])) < BIG_KM: blocked.add(r['i'])  # a district, not a town in a large county
    for r in batched(lambda b: f'''SELECT ?i (MAX(?es) AS ?mx) WHERE {{ VALUES ?i {{{V(b)}}} VALUES ?x {{{V(ATROCITY)}}}
            ?e wdt:P31/wdt:P279* ?x; wdt:P276|wdt:P131|wdt:P706 ?i; wikibase:sitelinks ?es }} GROUP BY ?i''', ids, cache):
        if int(r['mx']) >= KNOWN_FOR * by_sl[r['i']]: blocked.add(r['i'])              # place known for a mass killing
    names = defaultdict(set)                         # every name of the place, all languages, for the leak check
    for r in batched(lambda b: f'SELECT ?i ?n WHERE {{ VALUES ?i {{{V(b)}}} ?i rdfs:label|skos:altLabel ?n }}', ids, cache):
        if len(r['n']) > 3: names[r['i']].add(norm(r['n']))
    up = defaultdict(set)
    for r in batched(lambda b: f'SELECT ?i ?ul WHERE {{ VALUES ?i {{{V(b)}}} ?i wdt:P131 ?u. ?u rdfs:label ?ul FILTER(LANG(?ul) = "en") }}', ids, cache):
        up[r['i']].add(r['ul'].lower())
    leak, qual, by_id = {}, set(), {p['i']: p for p in chunk}
    for p in chunk:
        cn = countries.get(p['c'], {})
        if ',' in p['label'] or any(re.search(rf'\b{re.escape(n)}\b', norm(p['label'])) for n in cn.get('names', ())):
            qual.add(p['i'])                          # 'West End, Grand Bahama', 'Grenada': the name already narrows itself, so no B
        core = norm(GENERIC.sub(' ', norm(p['label'])))                                # 'lake ohrid' -> 'ohrid'
        leak[p['i']] = {t for t in {norm(p['label']), core, *names[p['i']], *cn.get('names', ()), *cn.get('dem', ()), *up[p['i']]} if len(t) > 2}
        p['top'] = defaultdict(list)                  # clean clues per family, best first; top 5 kept for the class check

    def fetch(groups, lo, hi):                       # groups: {prop: [ids]}. Fixed predicates in a UNION run fast on QLever
        parts = [(p, i) for p, ids_ in groups.items() for i in ids_]
        def q(b):
            by = defaultdict(list)
            for p, i in b: by[p].append(i)
            arms = ' UNION '.join(f'{{ VALUES ?i {{{V(v)}}} ' + (f'?i wdt:{p} ?e' if p in OUT else f'?e wdt:{p} ?i') + f' BIND("{p}" AS ?p) }}' for p, v in by.items())
            return f'''SELECT ?i ?p ?e ?es ?el WHERE {{ {arms} ?e wikibase:sitelinks ?es; rdfs:label ?el
                FILTER(LANG(?el) = "en") FILTER(?es >= {lo} && ?es <= {hi}) }}'''
        for r in batched(q, parts, cache, 1500):
            yield by_id[r['i']], {**FAMILY, **OUT}[r['p']], r['e'], int(r['es']), r['el']

    def keep(rows):
        for p, f, e, es, el in sorted(rows, key=lambda c: -c[3]):
            if e != p['i'] and not any(t in norm(el) for t in leak[p['i']]) and len(p['top'][f]) < 10: p['top'][f].append((e, el, es))

    allp = {**FAMILY, **OUT}
    keep(fetch({p: ids for p in allp}, 10, 10 ** 6))
    miss = defaultdict(list)                          # families short of clean well-known clues: add the less known ones
    for p in chunk:
        for prop, f in allp.items():
            if f in p.get('raw', {f}) and len(p['top'][f]) < A_CLUES: miss[prop].append(p['i'])
    keep(fetch(miss, CLUE_MIN_SL, 9))
    rivals = defaultdict(list)
    for pred in ('rdfs:label', 'skos:altLabel'):     # namesakes by name or by alias, two light queries
        for r in batched(lambda b: f'''SELECT DISTINCT ?i ?r ?rs ?rl WHERE {{ VALUES ?i {{{V(b)}}} ?i rdfs:label ?l FILTER(LANG(?l) = "en" || LANG(?l) = "mul")
                ?r {pred} ?l. ?r wdt:P625 ?rl; wdt:P17 ?rc; wikibase:sitelinks ?rs. FILTER(?r != ?i && ?rs >= {RIVAL_MIN_SL}) MINUS {{ ?r wdt:P31 wd:Q5 }} }}''', ids, cache):
            if pt(r['rl']) and r['r'] not in {q for q, _, _ in rivals[r['i']]}: rivals[r['i']].append((r['r'], int(r['rs']), pt(r['rl'])))
    isx = defaultdict(set)
    for r in batched(lambda b: f'SELECT DISTINCT ?i ?x WHERE {{ VALUES ?i {{{V(b)}}} VALUES ?x {{{V(BLOCK_CLASSES + CLUE_BLOCK + list(ROLE) + NOT_COMPANY + [EVENT] + NOT_PART)}}} ?i wdt:P31/wdt:P279* ?x }}',
                     sorted({e for p in chunk for v in p.get('top', {}).values() for e, _, _ in v}), cache):
        isx[r['i']].add(r['x'])
    bad = {e for e, xs in isx.items() if xs & set(BLOCK_CLASSES + CLUE_BLOCK)}
    role = {e: r for e, xs in isx.items() for r in [next((ROLE[x] for x in sorted(xs) if x in ROLE and not (ROLE[x] == 'company' and xs & set(NOT_COMPANY))), '')] if r}
    rank = lambda f, c: -c[2]                                                      # most documented clue first (A1 rule)
    rows = []
    for p in chunk:
        if p['i'] in blocked: continue
        top = p.pop('top')
        top['part'] = top.get('part', []) + [c for c in top.get('event', []) if EVENT not in isx[c[0]]]
        top['event'] = [c for c in top.get('event', []) if EVENT in isx[c[0]]]
        top['part'] = [c for c in top['part'] if not isx[c[0]] & set(NOT_PART)]
        top = {f: sorted((c for c in v if c[0] not in bad), key=lambda c: rank(f, c)) for f, v in top.items()}
        R = lambda f, e: 'politics' if f == 'politics' else role.get(e, '')
        best, used = [], set()                        # best clue of each family first, then the best of the rest; no entity twice
        for f, v in sorted(((f, v) for f, v in top.items() if v), key=lambda kv: rank(kv[0], kv[1][0])):
            if v[0][0] not in used and norm(v[0][1]) not in used: best.append([f, *v[0], R(f, v[0][0])]); used |= {v[0][0], norm(v[0][1])}
        for f, c in sorted(((f, c) for f, v in top.items() for c in v if c[0] not in used), key=lambda fc: rank(*fc)):
            if len(best) < A_CLUES and c[0] not in used and norm(c[1]) not in used: best.append([f, *c, R(f, c[0])]); used |= {c[0], norm(c[1])}
        best = sorted(best, key=lambda c: -c[3])[:max(A_CLUES, len({c[0] for c in best}))]
        far = [(q, s, round(km(p['latlon'], ll))) for q, s, ll in rivals[p['i']] if km(p['latlon'], ll) > B_MIN_KM]
        rv = max(far, key=lambda t: t[1], default=None)
        mx = max((c[3] for c in best), default=0); nf = len({c[0] for c in best})
        nrank = 1 + sum(sl > p['sl'] for _, sl, _ in far)                                 # place's rank among its namesakes by sitelinks
        rows.append(dict(p, rank=nrank if rv else '', n_names=len(far) + 1 if rv else '', hooks=';'.join(sorted({c[4] for c in best if c[4]})), clues=best, n_fam=nf, cbin=CBIN(mx), pbin=PBIN(p['sl']), a_ok=len(best) >= A_CLUES and nf >= A_FAMS, b_ok=bool(rv) and p['i'] not in qual,
                         rival=rv[0] if rv else '', rival_sl=rv[1] if rv else '', rival_km=rv[2] if rv else '',
                         rbin=RBIN((p['sl'] + 1) / (rv[1] + 1)) if rv else ''))
    tied = links([(c[1], r['rival']) for r in rows if r['rival'] for c in r['clues']], cache)
    for r in rows:                                    # B needs a clue that cannot point at the rival as well
        r['b_ok'] = r['b_ok'] and any((c[1], r['rival']) not in tied for c in r['clues'])
        yield r

def links(pairs, cache):
    """The (x, y) pairs joined by a clue property (FAMILY or OUT), either way. One fixed predicate per query:
    QLever times out on a variable predicate."""
    out = set()
    for prop in {**FAMILY, **OUT}:
        q = lambda b: ('SELECT DISTINCT ?x ?y WHERE { VALUES (?x ?y) { ' + ' '.join(f'(wd:{x} wd:{y})' for x, y in b) +
                       f' }} {{ ?x wdt:{prop} ?y }} UNION {{ ?y wdt:{prop} ?x }} }}')
        out |= {(r['x'], r['y']) for r in batched(q, sorted(set(pairs)), cache, 200)}
    return out

# ---------------- selection ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True); ap.add_argument('--exclude'); ap.add_argument('--prefer', help='old QIDs to try first, one per line'); ap.add_argument('--out', default='out')
    ap.add_argument('--conts', default='EU,AS,AF,NA,SA,OC'); ap.add_argument('--final', type=int, default=FINAL)
    a = ap.parse_args(); cache = os.path.join(a.out, 'cache'); os.makedirs(cache, exist_ok=True)
    block = set(open(a.exclude).read().split()) if a.exclude else set()
    prefer = set(open(a.prefer).read().split()) if a.prefer else set()
    U = lambda s: (int(hashlib.sha256(f'{a.seed}:{s}'.encode()).hexdigest(), 16) + 1) / (16 ** 64 + 2)   # uniform in (0,1), one per QID
    countries = {}
    for r in sparql('SELECT ?c ?pop ?k ?label ?dem WHERE { ?c wdt:P31 wd:Q6256. OPTIONAL { ?c wdt:P1082 ?pop } OPTIONAL { ?c wdt:P30 ?k } '
                    'OPTIONAL { ?c rdfs:label ?label FILTER(LANG(?label) = "en") } OPTIONAL { ?c wdt:P1549 ?dem FILTER(LANG(?dem) = "en") } }', cache):
        c = countries.setdefault(r['c'], dict(pop=0., ks=set(), dem=set(), label=''))
        c['pop'] = max(c['pop'], float(r.get('pop') or 0)); c['ks'].add(r.get('k')); c['label'] = r.get('label', '').lower()
        if r.get('dem'): c['dem'].add(r['dem'].lower())
    for r in sparql('SELECT ?c ?n WHERE { ?c wdt:P31 wd:Q6256; rdfs:label|skos:altLabel ?n }', cache):      # country names, all languages
        if r['c'] in countries and len(r['n']) > 3: countries[r['c']].setdefault('names', set()).add(norm(r['n']))
    for c in countries.values():
        ks = {CONT[k] for k in c['ks'] if k in CONT}; c['cont'] = ks.pop() if len(ks) == 1 else None
    places = [dict(p, old='yes' if p['i'] in prefer else '') for p in pool(cache, countries) if p['i'] not in block and p['c'] not in BLOCK_COUNTRIES]
    plan = math.ceil(a.final * OVER); conts = a.conts.split(','); sel, log, memo = [], [], {}
    if prefer: log.append(f'old places in the candidate pool: {sum(1 for p in places if p["old"])} of {len(prefer)}')
    BINS = {'A': ('low', 'mid', 'high'), 'B': ('rival_leads', 'similar', 'place_leads')}
    sup = {ct: math.sqrt(sum(1 for p in places if p['continent'] == ct and (len(p.get('raw', ())) >= A_FAMS or p.get('ns')))) for ct in conts}
    w = {ct: max(CONT_FLOOR, sup[ct] / sum(sup.values())) for ct in conts}
    alloc = {ct: round(plan * w[ct] / sum(w.values())) for ct in conts}
    alloc[max(alloc, key=alloc.get)] += plan - sum(alloc.values())                # rounding rest to the largest
    log.append('continent plan: ' + ', '.join(f'{ct}={n}' for ct, n in alloc.items()))
    for ct in conts:
        pc = alloc[ct]; rc = math.ceil(pc * RES); cap = math.ceil(CAP * (pc + rc))
        need = {(cond, t, b): [round(pc * s / 3), round(rc * s / 3)] for cond, bins in BINS.items() for t, s in SHARE[cond].items() for b in bins}
        need['X', 'infra', '-'] = [INFRA_PER_CONT, 0]
        got, per_c, taken = defaultdict(list), Counter(), {}
        open_ = lambda k: len(got[k]) < sum(need[k])

        def useful(p, t, both=False):                 # could this place still fill an open slot? exact after gating, prescreen before
            had = taken.get(p['i'], {}).get('cond', {})                            # a place fills each condition once
            if t == 'infra': return memo.get(p['i'], 1) is not None and 'X' not in had
            if p['i'] in memo:
                g = memo[p['i']]
                if not g: return False
                a = 'A' not in had and g['a_ok'] and open_(('A', t, g['pbin'])) and per_c['A', g['c']] < cap
                b = 'B' not in had and g['b_ok'] and open_(('B', t, g['rbin'])) and per_c['B', g['c']] < cap
            else:
                a = len(p.get('raw', ())) >= A_FAMS and open_(('A', t, PBIN(p['sl']))) and per_c['A', p['c']] < cap
                b = p.get('ns') and any(open_(('B', t, x)) for x in BINS['B']) and per_c['B', p['c']] < cap
            return bool(a and b) if both else bool(a or b)

        def walk(t, order, both=False, one=False):    # both: only places that fill an A and a B slot together; one: stop after one place
            gated = 0
            for n, p in enumerate(order):
                if both and shared[0] >= sh: break
                if not useful(p, t, both): continue
                if p['i'] not in memo:                # gate the next 100 useful places in one go
                    if gated >= MAX_WALK: break
                    batch = [q for q in order[n:] if q['i'] not in memo and useful(q, t, both)][:300]
                    memo.update((q['i'], None) for q in batch)
                    memo.update((g['i'], g) for g in gate(batch, cache, countries)); gated += len(batch)
                    print(f'  {ct} {t}: gated {gated}, filled {sum(len(v) for k, v in got.items() if k[1] == t)}', flush=True)
                g = memo[p['i']]
                if not g or (both and not useful(g, t, True)): continue
                if both: shared[0] += 1
                slots = [('X', 'infra', '-')] if t == 'infra' else [k for k, ok in ((('A', t, g['pbin']), g['a_ok']), (('B', t, g['rbin']), g['b_ok'])) if ok]
                for k in slots:
                    if open_(k) and per_c[k[0], g['c']] < cap and k[0] not in taken.get(g['i'], {}).get('cond', {}):
                        got[k].append(g['i']); per_c[k[0], g['c']] += 1
                        taken.setdefault(g['i'], dict(g, cond={}))['cond'][k[0]] = 'plan' if len(got[k]) <= need[k][0] else 'reserve'
                if one and g['i'] in taken: break
                if not any(open_(k) for k in need if k[1] == t): break

        sh, shared = round(pc * SHARED), [0]
        for t in ('natural', 'island', 'manmade', 'infra', 'settlement'):           # settlement last: it takes up shortfalls
            cand = [p for p in places if p['type'] == t and p['continent'] == ct]
            n_c = Counter(p['c'] for p in cand)                                         # country share ~ sqrt(pop), not Wikidata density
            cand.sort(key=lambda p: -math.log(U(p['i'])) * n_c[p['c']] / math.sqrt(countries.get(p['c'], {}).get('pop') or 1))
            first = {}
            for p in cand: first.setdefault(p['c'], p)
            order, dry = list({p['i']: p for p in [p for p in cand if p['i'] in prefer] + cand}.values()), set()
            big = [c for c in sorted(first, key=U) if t == 'settlement' and countries.get(c, {}).get('pop', 0) >= FLOOR_POP]
            lists = {c: [p for p in order if p['c'] == c][:FLOOR_TRIES] for c in big}
            has = lambda c: any(q['c'] == c for q in taken.values())
            for r0 in range(0, FLOOR_TRIES, 5):                                         # floor: 5 candidates of every country gated together
                batch = [p for c in big if not has(c) for p in lists[c][r0:r0 + 5] if p['i'] not in memo and useful(p, t)]
                for i in range(0, len(batch), 300):
                    memo.update((q['i'], None) for q in batch[i:i + 300]); memo.update((g['i'], g) for g in gate(batch[i:i + 300], cache, countries))
                for c in big:
                    if not has(c): walk(t, lists[c][:r0 + 5], one=True)
            log += [f'{ct}: no qualifying place for {c} in {FLOOR_TRIES} tries' for c in big if not has(c)]
            if t != 'infra': walk(t, order, both=True)                                 # shared places first
            for attempt in range(6):
                walk(t, order)
                short = {k: v for k, v in need.items() if k[1] == t and open_(k)}
                if not short or t != 'settlement': break
                if attempt < 2: cap = math.ceil(cap * 1.5); log.append(f'{ct}: country cap raised to {cap} per condition'); continue
                dry |= set(short)                                                          # a short bin passes its gap to the other bins
                for k, v in short.items():
                    to = [o for o in ((k[0], t, b) for b in BINS[k[0]]) if o not in dry]
                    if not to: continue
                    gap = [max(0, v[0] - len(got[k])), sum(v) - max(len(got[k]), v[0])]
                    v[:] = [min(v[0], len(got[k])), max(0, len(got[k]) - v[0])]
                    for j, o in enumerate(to): need[o][0] += gap[0] // len(to) + (j < gap[0] % len(to)); need[o][1] += gap[1] // len(to) + (j < gap[1] % len(to))
                    log.append(f'{ct} {"/".join(k)}: supply ran out, gap of {sum(gap)} moved to {", ".join(o[2] for o in to)}')
            for k, v in need.items():
                if k[1] == t and open_(k):
                    log.append(f'{ct} {"/".join(k)}: {len(got[k])} of {sum(v)}')
                    if t in ('natural', 'island', 'manmade'):                                          # move the gap to settlements, same condition and bin
                        pm = max(0, v[0] - len(got[k])); s2 = need[k[0], 'settlement', k[2]]
                        s2[0] += pm; s2[1] += sum(v) - len(got[k]) - pm
        sel += [dict(p, continent=ct) for p in taken.values()]
        write(a.out, sel, log, a.seed)                                                  # partial output after every continent
        print(ct, ' '.join(f'{"/".join(k)}={len(v)}/{sum(need[k])}' for k, v in sorted(got.items())), flush=True)
    write(a.out, sel, log, a.seed)

def write(out, sel, log, seed):
    cols = ['i', 'label', 'type', 'sub', 'c', 'continent', 'split', 'lat', 'lon', 'sl', 'pbin', 'n_fam', 'cbin', 'hooks', 'clues', 'old',
            'rival', 'rival_sl', 'rival_km', 'rbin', 'rank', 'n_names', 'A', 'B', 'EXPLICIT']
    with open(os.path.join(out, 'places.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader()
        for p in sel:
            c = p['cond']
            split = 'dev' if int(hashlib.sha256(f'split:{seed}:{p["i"]}'.encode()).hexdigest(), 16) / 16 ** 64 < DEV else 'test'
            w.writerow(dict(p, split=split, lat=p['latlon'][0], lon=p['latlon'][1], A=c.get('A', ''), B=c.get('B', ''),
                            clues=json.dumps(p['clues'], ensure_ascii=False),
                            EXPLICIT='yes'))
    tab = Counter((p['continent'], k, s) for p in sel for k, s in p['cond'].items())
    ov = sum(1 for p in sel if p['cond'].get('A') == p['cond'].get('B') == 'plan')
    with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as f:
        try: idx = json.loads(urllib.request.urlopen(EP + '?cmd=stats', timeout=30).read()).get('name-index', '?')
        except Exception: idx = 'unknown'
        sha = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]
        f.write(f'# Places run\n\nSeed {seed}. Endpoint {EP}, {time.strftime("%Y-%m-%d")}. Index: {idx}. Script sha256: {sha}.\n\n| continent | A plan | A reserve | B plan | B reserve | infra |\n|---|---|---|---|---|---|\n')
        for ct in sorted({p['continent'] for p in sel}):
            f.write(f'| {ct} | {tab[ct, "A", "plan"]} | {tab[ct, "A", "reserve"]} | {tab[ct, "B", "plan"]} | {tab[ct, "B", "reserve"]} | {tab[ct, "X", "plan"]} |\n')
        f.write(f'\nA and B on the same place (plan): {ov}\n\n')
        town = sum(p['type'] == 'settlement' for p in sel) / max(len(sel), 1)
        f.write(f'Settlement share of all places: {town:.0%}' + (f' - ABOVE the {MAX_TOWN:.0%} limit\n\n' if town > MAX_TOWN else '\n\n'))
        for col in ('type', 'pbin', 'rbin', 'cbin'):
            f.write(f'{col}: ' + ', '.join(f'{k or "-"}={n}' for k, n in sorted(Counter(p[col] for p in sel).items())) + '\n\n')
        f.write('Countries: ' + ', '.join(f'{c}={n}' for c, n in Counter(p['c'] for p in sel).most_common()) + '\n\nShortfalls:\n')
        f.write('\n'.join('- ' + l for l in log) or '- none'); f.write('\n')

if __name__ == '__main__': main()
