"""Redo the clues of out/places.csv without birthplace/death (P19, P20), water and range links, without places inside the place as 'part'
clues, and with significant events (P793) added. Places, rivals and
A/B/EXPLICIT slots stay as they are; only clues, n_fam, cbin and hooks change. A place may end with fewer than 3 clues.
Run:  python reclue.py [--out out]   (backs up places.csv to places_born_died.csv first)"""
import argparse, csv, json, os, shutil
import reform_places as rp

DROP = ['P19', 'P20',                           # born here / died here: the person names the place but says nothing about it
        'P206', 'P885', 'P403', 'P200', 'P201', 'P4552']   # water and range: next to the place they all but name it
ADD = {'P793': 'event'}                          # place -> its significant events, beside events located there (P276)
NOT_PART = ['Q56061', 'Q486972', 'Q1048835', 'Q82794', 'Q271669', 'Q15324', 'Q473972', 'Q123705', 'Q35145263']
# a thing inside the place may be a clue (club, company, university, festival), another location may not: admin unit, settlement,
# political territory, region, landform, body of water, protected area, neighbourhood, natural object
VIOLENT = ['Q41397', 'Q3199915', 'Q2223653', 'Q178561', 'Q198']   # genocide, massacre, terror attack, battle, war: no such event as a clue

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out'); a = ap.parse_args()
    cache, path = os.path.join(a.out, 'cache'), os.path.join(a.out, 'places.csv')
    rp.FAMILY = {k: v for k, v in rp.FAMILY.items() if k not in DROP}
    rp.OUT = {**{k: v for k, v in rp.OUT.items() if k not in DROP}, **ADD}
    rp.NOT_PART, rp.CLUE_BLOCK = NOT_PART, rp.CLUE_BLOCK + VIOLENT
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    countries = {}                               # country names and demonyms for the leak check, as in reform_places.main
    for r in rp.sparql('SELECT ?c ?pop ?k ?label ?dem WHERE { ?c wdt:P31 wd:Q6256. OPTIONAL { ?c wdt:P1082 ?pop } OPTIONAL { ?c wdt:P30 ?k } '
                       'OPTIONAL { ?c rdfs:label ?label FILTER(LANG(?label) = "en") } OPTIONAL { ?c wdt:P1549 ?dem FILTER(LANG(?dem) = "en") } }', cache):
        c = countries.setdefault(r['c'], dict(dem=set()))
        if r.get('dem'): c['dem'].add(r['dem'].lower())
    for r in rp.sparql('SELECT ?c ?n WHERE { ?c wdt:P31 wd:Q6256; rdfs:label|skos:altLabel ?n }', cache):
        if r['c'] in countries and len(r['n']) > 3: countries[r['c']].setdefault('names', set()).add(rp.norm(r['n']))
    chunk = [dict(i=r['i'], label=r['label'], c=r['c'], sl=int(r['sl']), latlon=(float(r['lat']), float(r['lon']))) for r in rows]
    new = {}
    for k in range(0, len(chunk), 300):
        new.update((g['i'], g) for g in rp.gate(chunk[k:k + 300], cache, countries)); print(f'  gated {min(k + 300, len(chunk))} of {len(chunk)}', flush=True)
    bak = os.path.join(a.out, 'places_born_died.csv')
    if not os.path.exists(bak): shutil.copyfile(path, bak)       # the first run backs up the original; reruns keep that copy
    rows = list(csv.DictReader(open(bak, encoding='utf-8')))     # always start from the original clues
    lost = []
    for r in rows:
        old = json.loads(r['clues'] or '[]')
        g = new.get(r['i'])
        if g is None:                                # gate dropped it now (Wikidata changed): keep the old clues minus people
            g = dict(clues=[c for c in old if c[0] not in ('people', 'water', 'range', 'part')], b_ok=True)   # stored clues keep no property or class, so drop the whole family
            lost.append(f'{r["i"]} {r["label"]}: not gated again, people/water/range/part clues removed')
        cl = g['clues']
        r.update(clues=json.dumps(cl, ensure_ascii=False), n_fam=len({c[0] for c in cl}), cbin=rp.CBIN(max((c[3] for c in cl), default=0)),
                 hooks=';'.join(sorted({c[4] for c in cl if c[4]})))
        if r['A'] and (len(cl) < rp.A_CLUES or len({c[0] for c in cl}) < rp.A_FAMS): lost.append(f'{r["i"]} {r["label"]}: A {r["A"]} now {len(cl)} clues')
        if r['B'] and not g['b_ok']: lost.append(f'{r["i"]} {r["label"]}: B {r["B"]} has no clue free of the rival')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, 'reclue_report.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Reclue without {", ".join(DROP)}\n\n' + '\n'.join('- ' + l for l in lost) + '\n')
    print(f'{len(rows)} places reclued; {len(lost)} notes in reclue_report.md')

if __name__ == '__main__': main()
