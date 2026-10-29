# Places run

Seed 20261028. Endpoint https://qlever.dev/api/wikidata, 2026-10-29. Index: Complete Wikidata, TTL from https://dumps.wikimedia.org/wikidatawiki/entities (latest-all.ttl.bz2 from 10.08.2026 munged, latest-lexemes.ttl.bz2 from 07.08.2026), updated LIVE. Script sha256: f9872cf06906ce7f.

| continent | A plan | A reserve | B plan | B reserve | infra |
|---|---|---|---|---|---|
| AF | 36 | 9 | 39 | 9 | 25 |
| AS | 66 | 21 | 66 | 18 | 25 |
| EU | 123 | 30 | 126 | 33 | 25 |
| NA | 93 | 24 | 93 | 24 | 25 |
| OC | 36 | 9 | 34 | 4 | 14 |
| SA | 39 | 9 | 39 | 9 | 25 |

A and B on the same place (plan): 164

Settlement share of all places: 47%

type: infra=139, island=46, manmade=50, natural=262, settlement=442

pbin: high=165, low=547, mid=227

rbin: -=364, place_leads=234, rival_leads=161, similar=180

cbin: 0=269, 1-9=86, >=10=584

Countries: Q96=27, Q159=26, Q30=25, Q16=22, Q183=20, Q408=20, Q664=18, Q145=17, Q36=16, Q241=16, Q148=15, Q774=15, Q691=15, Q142=14, Q17=14, Q414=14, Q212=13, Q783=13, Q155=13, Q34=12, Q38=12, Q43=12, Q800=12, Q739=12, Q843=11, Q668=11, Q766=11, Q298=11, Q41=10, Q55=10, Q717=10, Q252=9, Q792=9, Q736=9, Q419=9, Q39=8, Q20=8, Q219=8, Q45=8, Q79=8, Q804=8, Q790=8, Q733=8, Q27=7, Q29=7, Q928=7, Q786=7, Q750=7, Q712=7, Q37=6, Q213=6, Q31=6, Q258=6, Q115=6, Q811=6, Q778=6, Q730=6, Q214=5, Q225=5, Q184=5, Q218=5, Q869=5, Q1033=5, Q242=5, Q77=5, Q686=5, Q221=4, Q403=4, Q40=4, Q215=4, Q33=4, Q229=4, Q230=4, Q836=4, Q854=4, Q227=4, Q794=4, Q232=4, Q884=4, Q889=4, Q1019=4, Q948=4, Q754=4, Q32=3, Q224=3, Q222=3, Q191=3, Q217=3, Q851=3, Q810=3, Q423=3, Q902=3, Q878=3, Q1036=3, Q114=3, Q924=3, Q1011=3, Q262=3, Q1028=3, Q1044=3, Q916=3, Q685=3, Q678=3, Q710=3, Q238=2, Q28=2, Q35=2, Q837=2, Q881=2, Q334=2, Q796=2, Q265=2, Q833=2, Q1037=2, Q1029=2, Q971=2, Q1039=2, Q1000=2, Q958=2, Q1008=2, Q117=2, Q223=2, Q781=2, Q763=2, Q769=2, Q757=2, Q734=2, Q683=2, Q347=1, Q211=1, Q756617=1, Q233=1, Q228=1, Q236=1, Q801=1, Q399=1, Q822=1, Q858=1, Q846=1, Q817=1, Q813=1, Q874=1, Q842=1, Q711=1, Q424=1, Q819=1, Q863=1, Q805=1, Q826=1, Q1032=1, Q1030=1, Q963=1, Q1006=1, Q974=1, Q1041=1, Q977=1, Q1009=1, Q1016=1, Q1005=1, Q967=1, Q953=1, Q1014=1, Q1025=1, Q1049=1, Q965=1, Q986=1, Q962=1, Q929=1, Q912=1, Q1013=1, Q1020=1, Q657=1, Q945=1, Q954=1, Q1045=1, Q970=1, Q760=1, Q21203=1, Q25279=1, Q244=1, Q784=1, Q697=1, Q702=1, Q672=1, Q26988=1, Q709=1

Shortfalls:
- old places in the candidate pool: 528 of 737
- continent plan: EU=125, AS=66, AF=38, NA=93, SA=40, OC=38
- EU A/natural/high: 2 of 15
- EU: no qualifying place for Q43 in 20 tries
- AS A/natural/high: 2 of 9
- AS A/island/high: 0 of 3
- AF A/natural/high: 0 of 5
- NA A/natural/high: 0 of 11
- NA B/natural/rival_leads: 8 of 18
- NA B/natural/place_leads: 15 of 18
- NA A/island/high: 3 of 4
- NA A/manmade/low: 0 of 4
- NA A/manmade/mid: 1 of 4
- NA A/manmade/high: 0 of 4
- NA: country cap raised to 15 per condition
- SA A/natural/high: 1 of 5
- SA A/island/low: 0 of 1
- SA A/manmade/low: 0 of 1
- SA A/manmade/high: 0 of 1
- SA B/manmade/rival_leads: 0 of 1
- SA B/manmade/similar: 0 of 1
- SA: country cap raised to 6 per condition
- OC A/natural/high: 0 of 5
- OC B/natural/rival_leads: 3 of 8
- OC B/natural/similar: 7 of 8
- OC B/natural/place_leads: 5 of 8
- OC A/island/high: 0 of 1
- OC A/manmade/low: 0 of 1
- OC A/manmade/mid: 0 of 1
- OC A/manmade/high: 0 of 1
- OC B/manmade/rival_leads: 0 of 1
- OC B/manmade/similar: 0 of 1
- OC B/manmade/place_leads: 0 of 1
- OC X/infra/-: 14 of 25
- OC: country cap raised to 6 per condition
- OC: country cap raised to 9 per condition
- OC A/settlement/low: supply ran out, gap of 1 moved to mid, high
- OC B/settlement/rival_leads: supply ran out, gap of 7 moved to place_leads
- OC B/settlement/similar: supply ran out, gap of 4 moved to place_leads
- OC B/settlement/place_leads: 11 of 21
