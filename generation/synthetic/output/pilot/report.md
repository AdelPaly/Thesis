# Generation run report

146 items run, 44 of them reserves swapped in. Writer qwen3.5:122b, judge deepseek/deepseek-v4.1-flash.

## Outcome per condition

| condition | items | pass try 1 | pass try 2 | pass try 3 | failed |
|---|---|---|---|---|---|
| A1 | 19 | 15 (79%) | 2 (11%) | 1 (5%) | 1 (5%) |
| A3 | 19 | 12 (63%) | 0 (0%) | 0 (0%) | 7 (37%) |
| B | 19 | 8 (42%) | 5 (26%) | 1 (5%) | 5 (26%) |
| EXPLICIT | 38 | 12 (32%) | 10 (26%) | 6 (16%) | 10 (26%) |
| NOISE_A1 | 24 | 4 (17%) | 3 (12%) | 5 (21%) | 12 (50%) |
| NOISE_B | 27 | 4 (15%) | 3 (11%) | 2 (7%) | 18 (67%) |
| all | 146 | 55 (38%) | 23 (16%) | 15 (10%) | 53 (36%) |

## Why tries failed

Every failed try, also those followed by a passing one.

| condition | reason | tries |
|---|---|---|
| A1 | judge_decorative | 2 |
| A1 | judge_imagined | 2 |
| A1 | length | 1 |
| A1 | judge_incoherent | 1 |
| A1 | judge_no_environment | 1 |
| A3 | length | 8 |
| A3 | judge_decorative | 5 |
| A3 | judge_incoherent | 4 |
| A3 | names_place | 3 |
| A3 | judge_imagined | 1 |
| B | judge_decorative | 6 |
| B | length | 6 |
| B | judge_incoherent | 4 |
| B | judge_imagined | 2 |
| B | judge_no_environment | 2 |
| B | clue_missing | 1 |
| B | wrong_toponym | 1 |
| EXPLICIT | judge_person | 24 |
| EXPLICIT | missing_name | 18 |
| EXPLICIT | judge_place | 5 |
| EXPLICIT | length | 3 |
| EXPLICIT | judge_no_environment | 2 |
| NOISE_A1 | judge_decorative | 22 |
| NOISE_A1 | length | 7 |
| NOISE_A1 | judge_imagined | 6 |
| NOISE_A1 | judge_incoherent | 6 |
| NOISE_A1 | noise_missing | 4 |
| NOISE_A1 | judge_no_environment | 2 |
| NOISE_A1 | judge_place | 1 |
| NOISE_A1 | clue_missing | 1 |
| NOISE_B | judge_decorative | 26 |
| NOISE_B | wrong_toponym | 10 |
| NOISE_B | length | 9 |
| NOISE_B | judge_imagined | 6 |
| NOISE_B | judge_incoherent | 5 |
| NOISE_B | clue_missing | 3 |
| NOISE_B | judge_no_environment | 2 |

## Swaps

- clue swaps (a decorative clue replaced by a spare): 29, in 25 items
  - B Freeport: Petra Martić -> 2022 NACAC Championships
  - NOISE_A1 Kingston: Usain Bolt -> Caribbean Football Union
  - NOISE_A1 Kanpur: Lakshmi Sahgal -> Chhatrapati Shahu Ji Maharaj University
  - NOISE_A1 Glacier National Park: British Columbia Highway 1 -> Bird Woman Falls
  - B Cuba: Faro do Alentejo -> Fialho de Almeida
  - NOISE_A1 Corfu: Prince Philip, Duke of Edinburgh -> Seventh Ottoman–Venetian War
  - NOISE_A1 Corfu: Seventh Ottoman–Venetian War -> Palaiokastritsa
  - A1 Corfu: Prince Philip, Duke of Edinburgh -> Seventh Ottoman–Venetian War
  - NOISE_A1 Abu Qir Bay: Battle of the Nile -> Heracleion
  - NOISE_B Barca: Jawdhar -> Zopyrus
  - NOISE_B Caracol: Yajaw Te' K'inich II -> K’inich Joy K’awiil
  - NOISE_A1 Büyükada: Princess Fahrelnissa Zeid -> Lefter Küçükandonyadis
  - NOISE_A1 Utica: Cato the Younger -> Gnaeus Naevius
  - NOISE_A1 Troy: Achilles -> Hector
  - B Antigua: Lester Bird -> Sir Vivian Richards Stadium
  - A1 Sentosa: Fort Siloso -> Marine Life Park
  - NOISE_B Rioverde: Ana Bárbara -> Ramón Juárez
  - NOISE_A1 Monterrey: Dharius -> Control Machete
  - NOISE_A1 Monterrey: Control Machete -> 1977 CONCACAF Championship
  - NOISE_A1 Florina: Gian Maria Volonte -> Elpida Karamandi
  - NOISE_A1 Florina: Elpida Karamandi -> Gogos House
  - A3 Maastricht: Transavia -> 1981 World Judo Championships
  - B Mykolaiv: Uliana Kravchenko -> Maryan Shved
  - NOISE_B Ede: Cynthia Shalom -> Erinle River
  - NOISE_A1 Sitra: A'ala Hubail -> Applied Science University
  - NOISE_B Wexford: John Banville -> Battle of Vinegar Hill
  - NOISE_B Wexford: Battle of Vinegar Hill -> Environmental Protection Agency
  - NOISE_A1 Ronneburg: Gert Brauer -> Mario Hoyer
  - B Chiva: Rodrigo Tarín -> Enrique Ponce Martínez
- place swaps (all tries failed): 42
  - B set: Mele (NOISE_B, judge_no_environment) -> no matching reserve left
  - B set: Gorakhpur (NOISE_B, wrong_toponym) -> Q2460416
  - B set: Kingston (NOISE_B, judge_decorative) -> Q1925390
  - B set: Suchowola (NOISE_B, judge_decorative) -> Q157034
  - B set: Ithaca (NOISE_B, wrong_toponym) -> no matching reserve left
  - A1 set: Surin (A3, judge_imagined) -> Q535747
  - A1 set: Kingston (NOISE_A1, noise_missing) -> Q81033
  - A1 set: Orsk (NOISE_A1, judge_imagined) -> Q202272
  - A1 set: Syracuse (A3, judge_incoherent) -> Q1309
  - A1 set: Glacier National Park (NOISE_A1, judge_decorative) -> Q645901
  - A1 set: Réunion (NOISE_A1, noise_missing) -> no matching reserve left
  - B set: Namsan (NOISE_B, judge_decorative) -> Q797552
  - B set: Cuba (B, length) -> Q58475
  - B set: Santana (NOISE_B, judge_decorative) -> Q1283170
  - A1 set: Mokhotlong (A3, judge_incoherent) -> no matching reserve left
  - A1 set: K2 (A3, names_place) -> no matching reserve left
  - A1 set: Corfu (NOISE_A1, judge_incoherent) -> Q1027830
  - A1 set: Mount Ararat (NOISE_A1, length) -> no matching reserve left
  - A1 set: Abu Qir Bay (NOISE_A1, judge_decorative) -> Q304933
  - B set: Barca (NOISE_B, judge_decorative) -> no matching reserve left
  - B set: Büyükada (NOISE_B, judge_decorative) -> no matching reserve left
  - B set: Caracol (NOISE_B, clue_missing) -> no matching reserve left
  - A1 set: Büyükada (NOISE_A1, length) -> Q1416753
  - B set: Carthage (B, clue_missing) -> no matching reserve left
  - A1 set: Utica (NOISE_A1, judge_decorative) -> no matching reserve left
  - A1 set: Príncipe (A3, judge_decorative) -> no matching reserve left
  - B set: Cologne (B, judge_decorative) -> no matching reserve left
  - A1 set: Montserrat (NOISE_A1, judge_decorative) -> no matching reserve left
  - B set: Mandu (NOISE_B, wrong_toponym) -> no matching reserve left
  - A1 set: Copán (A3, length) -> no matching reserve left
  - B set: Beru (B, judge_no_environment) -> no matching reserve left
  - B set: Gazipur (NOISE_B, judge_decorative) -> Q241116
  - B set: Rioverde (NOISE_B, judge_decorative) -> Q30616
  - B set: Kruševo (NOISE_B, length) -> Q209126
  - A1 set: Florina (NOISE_A1, judge_incoherent) -> Q554655
  - A1 set: Maastricht (A3, judge_decorative) -> no matching reserve left
  - B set: Babadağ (NOISE_B, judge_decorative) -> no matching reserve left
  - B set: Mykolaiv (B, judge_incoherent) -> Q967818
  - B set: Ede (NOISE_B, length) -> Q1022899
  - A1 set: Gotland (NOISE_A1, noise_missing) -> no matching reserve left
  - B set: Greenville (NOISE_B, wrong_toponym) -> Q2928324
  - B set: Bure (NOISE_B, judge_decorative) -> no matching reserve left

## One post per condition

**A1**, Mandalay (scale local, angle "political accountability on drinking water quality", 19 words, target 15, try 1)

> Neighbors saying the tap water tastes metallic again. Myint Swe promised clean water but the pipes still feel old.

**A3**, Holguin (scale regional, angle "river pollution and political inaction", 24 words, target 21, try 1)

> Saw that report on the river near The Periquera. Bucanero Brewery S.A. keeps dumping, and honestly, Manuel Marrero Cruz's record feels just as murky.

**B**, Freeport (scale global, angle "flooding caused by climate change", 26 words, target 22, try 3)

> Walking past where they held the 2022 NACAC Championships, water is already up to the door in Freeport, proof that climate change is wrecking our coast.

**EXPLICIT**, Ochiai Dam (scale regional, angle "protected land development impacting river flow", 16 words, target 14, try 2)

> View from Ochiai Dam where the council keeps pushing through new builds on protected river land.

**NOISE_A1**, Oron (scale personal, angle "textile factory pollution affecting local water and sports grounds", 26 words, target 11, try 3)

> Saw the dye running out again. Etim Esin used to run these fields clean. Elias Legendre's crew deals with that now too. Avoid the river side.

**NOISE_B**, Spitz (scale regional, angle "corporate greenwashing and pollution", 25 words, target 24, try 1)

> They paint the factory green while the Rätikon air turns sick. Iolu Abil stays quiet as they keep dumping waste into the water at Spitz.

## Length against target

| condition | posts | median target | median length | median off by | within 25% of target |
|---|---|---|---|---|---|
| A1 | 18 | 14.0 | 16.0 | 2.5 | 14 (78%) |
| A3 | 12 | 22.0 | 24.0 | 4.0 | 9 (75%) |
| B | 14 | 19.5 | 20.5 | 3.0 | 11 (79%) |
| EXPLICIT | 28 | 11.0 | 15.5 | 3.0 | 17 (61%) |
| NOISE_A1 | 12 | 12.0 | 23.5 | 11.5 | 1 (8%) |
| NOISE_B | 9 | 15 | 21 | 2 | 5 (56%) |
