# Evaluation methods, configuration design, and bibliography

Thesis: geolocation of text from indirect linguistic mentions.
Compiled 3 October 2026. Written in ASD-STE100 Simplified Technical English.

All references in this file are peer-reviewed publications. No reference is an
arXiv preprint. Each reference shows how it was verified. Section 8 lists the
items that have no published version, and what to do about each one.

---

## 1. Purpose of this file

This file does three things:

1. It gives the rule for how to name an evaluation configuration, and the
   published basis for that rule.
2. It gives the full set of configurations, tier by tier, with a reason and a
   citation for each one.
3. It gives the bibliography, grouped by tier.

---

## 2. The naming rule

**A configuration name denotes the one thing that changes between it and its
neighbours. All other data goes in the fields of the configuration file.**

Apply the rule with this test. Change the property. If the result does not
change, the property is not part of the name.

Example. The configuration `groq_oss` runs the model `openai/gpt-oss` on Groq.
Groq is an inference provider. Groq does not make models. Run the same weights
on a different provider and the accuracy stays the same. Therefore the provider
is not part of the name. Change the model and the accuracy moves 8.2 percentage
points. Therefore the model is part of the name.

Each tier changes along a different axis:

| Tier | What changes | What the name shows |
|---|---|---|
| 0 | the heuristic | the heuristic |
| 1 | the system | the system |
| 2 | the linking mechanism | the mechanism |
| 3 | the retrieval mechanism | the mechanism |
| 4 | the model | the model |
| 5 | the source of the candidates | the source |

### 2.1 Published basis for the rule

Sculley et al. (2015) define *configuration debt* as a specific failure in
machine learning systems. They state that configurations must be versioned and
reviewable. They state that each configuration must differ from its neighbour
in a way that a reader can see. This is the closest published statement of the
rule above.

Mitchell et al. (2019) show that model identity, version and provenance belong
in a structured record with named fields. They do not belong in an informal
label. This supports the separation of the name from the fields.

Wilkinson et al. (2016) give the general form of the same separation. An
artefact gets a persistent identifier. The descriptive data goes in rich
machine-readable metadata. The identifier does not carry the description.

Lipton and Steinhardt (2019) name "failure to identify the sources of empirical
gains" as a specific fault in machine learning work. A name that mixes provider
and model hides the source of a gain.

Narang et al. (2021) show what a controlled comparison looks like. They change
one architectural property at a time against a fixed codebase. Most reported
gains do not survive this control.

### 2.2 Published basis for the fields

Record the provider, the model identifier, the revision and the run date for
every hosted model.

Chen, Zaharia and Zou (2024) measured the behaviour of one commercial model
across four months. Accuracy on one task fell from 84% to 51%. They state that
"the behavior of the 'same' LLM service can change substantially in a relatively
short amount of time". A model name alone therefore does not identify the system
that was evaluated.

Balloccu et al. (2024) surveyed evaluation practice with closed-source API
models. They found that version and access details are frequently not recorded.

Gundersen and Kjensmo (2018) examined 400 AI papers. Between 20% and 30% of the
variables that are needed to reproduce an experiment were documented. No paper
documented all of them.

Pineau et al. (2021) and Dodge et al. (2019) give the two reproducibility
checklists. Both require the exact model, the hyperparameters and the run
configuration.

### 2.3 The corrected schema

Replace this:

```yaml
llm:
  backend: null    # null | gemma | groq_oss | groq_llama
  model: null
```

With this:

```yaml
tier: 4
llm:
  provider: groq              # groq | openrouter | local_hf
  model: openai/gpt-oss-20b   # read the exact variant from the run log
  revision: null              # pin where the provider gives one
  temperature: 0.0
  top_p: 1.0
  prompt_template: v1         # see section 6.4
  run_date: 2026-08-14
```

The `backend` field did two jobs. `gemma` is a model name. `groq_oss` is a
provider plus a model family. The enumeration was not consistent. The split
removes the inconsistency.

One OpenAI-compatible client is sufficient for Groq, OpenRouter and other
providers. The three modules `llm_gemma.py`, `llm_groq.py` and
`llm_openrouter.py` caused the problem. The configuration namespace took its
names from the modules.

Add `tier` as a field. The results table then sorts correctly. The prefix
`baseline_` is then not necessary.

---

## 3. Renames

| Current | New | Reason |
|---|---|---|
| `baseline_majority` | `majority_class` | Standard term for prediction of the most frequent label |
| `baseline_population` | `population_wikidata` | The name must show the gazetteer. The GeoNames backend never ran |
| `two_stage` | `retrieve_rerank` | "Two-stage" describes the whole task. It distinguishes nothing |
| `refined` | `refined` | A named system. No change |
| `mordecai3` | `mordecai3` | A named system. No change |
| `gemma` | `gemma3_12b` | The size is part of what varies |
| `groq_oss` | `gpt_oss` | The model, not the provider |
| `oracle` | `oracle_ceiling` | It is an analysis, not a system |
| `dense_e5` | `retrieval_dense` | The mechanism, not the model |
| `dense_bm25` | `retrieval_lexical` | BM25 is sparse. "dense_bm25" is a contradiction |

The last two are a correction of an earlier error in this project. The mechanism
difference between dense and lexical retrieval is the finding of that tier. The
names must not imply that the two are variants of one mechanism.

---

## 4. The full configuration set

Twelve configurations become twenty. New rows are marked **NEW**.

### Tier 0 - heuristic floors

| Config | Consumes | Status | Basis |
|---|---|---|---|
| `majority_class` | nothing | built | Gritta et al. (2020) |
| `centroid` **NEW** | nothing | to build | Gaman et al. (2020) |
| `population_wikidata` | a toponym | built | Speriosu and Baldridge (2013); Weissenbacher et al. (2019) |
| `char_ngram` **NEW** | whole post | to build | Jauhiainen et al. (2020); Han et al. (2014) |

`centroid` predicts the centre point of the training places. `majority_class`
predicts a discrete place. The two are not the same floor. `majority_class` is
the floor for exact match. `centroid` is the floor for error distance. You
report both metrics, so you need both floors. Gaman et al. (2020) use the
centroid baseline in the VarDial Social Media Variety Geolocation task, and
report that several submitted systems do not beat it.

`char_ngram` is the most valuable new row in this tier. The VarDial SMG task
masks every named entity with a `$NE$` tag. Systems must therefore geolocate
from surface form alone, with no toponym available. That is structurally the
same as your Signal A condition. Jauhiainen et al. (2020) geolocate with HeLI, a
character n-gram model of length 2 to 5, with no toponyms. Gaman et al. (2020)
report that character n-gram Naive Bayes, character n-gram SVM and
character-level CNN systems are all competitive on this task.

This row answers a question that an examiner will ask: does the result come from
the entity clue, or from surface statistics that correlate with region? Without
this row you cannot answer.

Gritta et al. (2020) argue that a population baseline must be mandatory in
geoparsing evaluation. They state that population "has not consistently featured
in geocoding evaluation but is capable of beating many existing systems".
Speriosu and Baldridge (2013) call POPULATION a "challenging" baseline, and
report that only two text-driven resolvers beat it on TR-CoNLL. Their paper is
also about indirect supervision, which makes it doubly relevant.

Armstrong et al. (2009) give the general warning. Reported improvements in
information retrieval across a decade do not accumulate, because baselines are
specified inconsistently.

### Tier 1 - established geoparsers

| Config | Consumes | Status | Basis |
|---|---|---|---|
| `mordecai3` | a toponym | cannot run | see caution below |
| `geoparser_irchel` **NEW** | a toponym | to build | Gomes and Purves (2025), demo |
| `edinburgh` **NEW** | a toponym | to attempt | Grover et al. (2010) |
| `topocluster` **NEW** | a toponym | to attempt | DeLozier et al. (2015) |

This tier currently has no runnable member. That weakens your central claim.
"Geoparsers score zero on Signal A by construction" is at present an argument
from design. With one geoparser that runs, it becomes a measured result.

`geoparser_irchel` is the practical fix. The package supports Python 3.10 to
3.14 and your repository is on 3.14. It uses DuckDB, not Elasticsearch. It does
not pin an old version of torch.

`topocluster` is the most interesting row. DeLozier et al. (2015) resolve
toponyms without a gazetteer. They use geographic word profiles built from
geotagged text. The profiles cover all words, not only toponyms. The system
still needs a toponym as its target, so it belongs in Tier 1. But the mechanism
is closer to your Tier 3 than to the other geoparsers. State this in the
discussion.

An alternative to installing each system: EUPEG (Wang and Hu, 2019) is a
published benchmarking platform that already covers the Edinburgh Geoparser,
CLAVIN, GeoTxt, TopoCluster and Mordecai over six corpora. Consider whether to
run your data through EUPEG instead of installing four systems.

**CAUTION on Mordecai 3.** Mordecai 3 exists only as an arXiv preprint. The
author's own CV lists it under "working papers" with no venue. Under your own
no-arXiv rule you cannot cite it. Mordecai 1 (Halterman, 2017) is published in
the Journal of Open Source Software, which is peer-reviewed. Cite Mordecai 1, or
remove the row.

**CAUTION on CLAVIN.** CLAVIN has no peer-reviewed paper. It is software only.
The CLIFF-CLAVIN item is a non-archival workshop abstract with no DOI. Cite
CLAVIN through third-party evaluations such as Wang and Hu (2019) or Lee et al.
(2019), or do not cite it.

**CAUTION on the Irchel Geoparser.** There is no peer-reviewed publication for
it today. A JOSS submission is in review (issue #10365, opened 2 April 2026).
A GIScience 2025 demo exists but demos are non-archival. Cite it as software
with a URL and a version number. State in the thesis that no peer-reviewed
description was available. Check the JOSS issue again before you submit.

### Tier 2 - entity linking

| Config | Consumes | Status | Basis |
|---|---|---|---|
| `retrieve_rerank` | any entity | built | Wu et al. (2020) |
| `refined` | any entity | built | Ayoola et al. (2022) |
| `blink` **NEW** | any entity | to build | Wu et al. (2020) |
| `rel` or `mgenre` | any entity | to build | van Hulst et al. (2020); De Cao et al. (2022) |

`blink` is a strong addition. Wu et al. (2020) define the bi-encoder retrieval
plus cross-encoder rerank architecture. Your `retrieve_rerank` is a hand-built
version of the same pattern. Adding BLINK converts your own component into a
comparison against the published reference. That is a better position than
presenting a hand-built pipeline with no standard to compare against.

Choose `mgenre` over `rel` if you want the larger architectural distance.
mGENRE is autoregressive and uses no candidate index. ReFinED and BLINK both
use one.

Kolitsas et al. (2018) give the pipeline against end-to-end framing. One of
their findings matters for you: under domain shift, from news to tweets, the
pipeline configuration was best or second best. Your data is social media text,
so it is out of domain for most linkers. Report this.

Candidate generation is the measured bottleneck, in your work and in the
literature. Your failure attribution found that 38.5% of `retrieve_rerank`
errors are a gold place that was retrieved and then not ranked first. Zhou et al.
(2020) report a 16.9% gain in top-30 gold candidate recall and show that recall
at the candidate stage caps everything downstream. Hachey et al. (2013) state
that entity linking comparison "has focused on disambiguation accuracy, making
it difficult to determine how search impacts performance". Lai et al. (2022)
address candidate retrieval over Wikidata specifically, which is your knowledge
base.

**No published work does exactly what your Tier 2 does.** No paper was found
that links a non-place entity and then follows a Wikidata property such as P19
to get a location. Krause and Cohen (2020) derive a geolocation for a Wikipedia
page from textual and categorical knowledge-base data, which is the same move.
Middleton et al. (2018) benchmark a DBpedia entity-recognition-and-
disambiguation approach against gazetteer methods for location extraction. State
the gap. Do not claim a precedent that does not exist.

### Tier 3 - retrieval without mention detection

| Config | Consumes | Status | Basis |
|---|---|---|---|
| `retrieval_dense` | whole post | to build | Karpukhin et al. (2020) |
| `retrieval_lexical` | whole post | to build | Robertson and Zaragoza (2009) |
| `retrieval_hybrid` **NEW** | whole post | to build | Bruch et al. (2023); Chen et al. (2022) |
| `grid_lm` **NEW** | whole post | to build | Wing and Baldridge (2011) |

`grid_lm` is the most valuable new row in the whole set. Wing and Baldridge
(2011) compare a document's language model against a language model for each
cell of a geodesic grid, and take the best match. No gazetteer. No mention
detection. That is structurally identical to your dense retrieval tier, with
grid cells in place of place descriptions. It is the canonical prior method for
your exact task, and it is from 2011. Roller et al. (2012) refine it with an
adaptive k-d grid.

Add it, and your Tier 3 stops being an untested idea and becomes a modern
version of an established method. That is a much easier position to defend.

Eisenstein et al. (2010) are the foundational citation for the premise of your
whole thesis: non-toponym vocabulary alone carries geographic signal.

`retrieval_hybrid` fuses the dense and lexical scores. Chen et al. (2022) show
that hybrid retrieval generalises better zero-shot than either mechanism alone.
Bruch et al. (2023) give the formal analysis of fusion functions. Add this row
only if you fuse. Do not add it if you only compare.

Justification for running both dense and lexical: Thakur et al. (2021) show that
BM25 is a robust baseline and that dense retrievers often underperform out of
domain. Luan et al. (2021) give the theoretical capacity limit of fixed-length
dense encodings. Sciavolino et al. (2021) is the most relevant: dense retrievers
underperform BM25 on entity-centric questions and generalise poorly to unseen
entities. Your indirect geographic cues are often rare named entities, so this
result applies directly to you.

**CAUTION on the embedding model.** Your reranker uses
`intfloat/multilingual-e5-base`. The multilingual E5 paper is an arXiv technical
report. The original E5 paper is also arXiv-only. Neither meets your citation
rule. Three options:

1. Cite Wang et al. (2024), "Improving Text Embeddings with Large Language
   Models", ACL 2024. This is the only peer-reviewed E5-family paper. It covers
   E5-mistral, not your checkpoint. State this.
2. Change the embedding model to BGE. The C-Pack paper (Xiao et al., 2024) is
   published at SIGIR 2024. You then have a peer-reviewed citation for the model
   you actually ran.
3. Change to mGTE (Zhang et al., 2024), published at EMNLP 2024 Industry Track.

Option 2 is the cleanest if multilingual coverage is equal. Decide before you
run the final experiments, not after.

**The similarity-function experiment.** No published paper was found whose main
research question is cosine against dot product against Euclidean in dense
retrieval. The closest is ColBERT (Khattab and Zaharia, 2020), which reports
both cosine and squared L2 variants as a secondary result, and states that inner
product outperformed cosine for single-vector reranking. Bachrach et al. (2014)
show that maximum inner product search and Euclidean nearest-neighbour search
are interconvertible, so the three conditions differ only when vector norms
vary. Steck et al. (2024) warn that cosine similarity on learned embeddings can
give arbitrary results depending on regularisation. Zhou et al. (2022) show that
cosine underestimates similarity for high-frequency words.

The absence of a dedicated study strengthens this experiment. Keep it as an
ablation inside `retrieval_dense`, not as a separate row.

### Tier 4 - parametric

| Config | Consumes | Status | Basis |
|---|---|---|---|
| `gemma3_12b` | whole post | built | Manvi et al. (2024a) |
| `gpt_oss` | whole post | built | Alzahrani et al. (2024) |
| third model **NEW** | whole post | to build | Mizrahi et al. (2024) |

The argument for a third model is the same as for Tier 2. Two models give a
line. Three give a trend.

Alzahrani et al. (2024) is the strongest citation for multi-model evaluation.
Minor and meaningless perturbations, such as the order of answer options, move
model rankings by up to 8 positions. A single model's score is therefore not a
measurement of "can language models do this".

Manvi et al. (2024a) show that language models hold substantial parametric
geographic knowledge, and that this knowledge is systematically biased against
lower-socioeconomic regions. This is the main caveat for the tier. Manvi et al.
(2024b) show that the knowledge can be elicited by prompting alone, which is the
justification that the tier measures something real. Gurnee and Tegmark (2024)
show that language models learn linear internal representations of spatial
coordinates, which explains the mechanism.

**A protocol change, not a new row.** Mizrahi et al. (2024) evaluated 20 models
across 39 tasks with 6.5 million instances. They show that single-template
evaluation is unreliable and that conclusions change with a paraphrase. Sclar et
al. (2024) measured up to 76 accuracy points of spread from formatting changes
alone on one model. Run every Tier 4 configuration under at least three prompt
templates. Report the spread. Otherwise a between-model difference of 8.2
percentage points cannot be distinguished from within-model prompt variance.

This is the single most important finding in this file for your Tier 4 results.

**Keep one model local.** Chen, Zaharia and Zou (2024) show silent API drift.
A locally-hosted model gives you one row that stays reproducible.

### Tier 5 - combination

| Config | Consumes | Status | Basis |
|---|---|---|---|
| `grounded_llm_retrieval` | post plus candidates from Tier 3 | to build | Hu et al. (2024) |
| `grounded_llm_entity` **NEW** | post plus candidates from Tier 2 | to build | Hu et al. (2024) |

The split is a real contrast and costs little. Both rows reuse components you
already have. The question they answer is whether the source of the candidates
matters, or only the fact of grounding.

Hu et al. (2024) is the direct precedent. Candidate toponym entries are
retrieved from GeoNames and given to a lightweight open-source language model,
which selects the referent. That is your Tier 5 design. Hu et al. (2025) add
batching and speed optimisation, with about a 7-fold speedup at higher accuracy.

Zhou et al. (2024), Img2Loc, is the same retrieve-then-select pattern in the
image modality, published at SIGIR 2024.

Lewis et al. (2020) is the canonical formulation of retrieve-then-generate.

**CAUTION on Spatial-RAG.** Spatial-RAG is arXiv-only. It has no DOI and no
proceedings record. Do not cite it. Use Hu et al. (2024) and Campo et al. (2026)
instead.

---

## 5. What is still absent

There is no fine-tuned model in this design. This is defensible at your dataset
size. It also changes the research question from "can existing systems do this"
to "can a system be trained to do this", which is a different thesis. Prepare
the answer. An examiner will ask.

---

## 6. Statistical testing

### 6.1 McNemar's test for paired binary outcomes

Dietterich (1998) shows that McNemar's test has low Type I error when two
classifiers are compared on a single test set. This is the canonical
justification. Japkowicz and Shah (2011) give the textbook treatment with the
assumptions stated.

### 6.2 Wilcoxon signed-rank for paired distances

Demšar (2006) recommends the Wilcoxon signed-rank test as the robust
non-parametric test for a comparison of two classifiers across paired
observations.

**State this caveat in the thesis.** Demšar's setting is a comparison across
datasets. Your paired distance errors are items inside one test set. Cite Demšar
for the properties of the test. Cite Dror et al. (2018) for its application to
your paired-measurement design. Do not import Demšar's design as a whole.

### 6.3 Bootstrap resampling by place

Field and Welsh (2007) is the direct citation. They analyse bootstrap schemes
for clustered data and show that resampling whole clusters, not individual
observations, gives consistent variance estimates under within-cluster
correlation. Your items share places, so items are not independent. Resample by
place.

Davison and Hinkley (1997) give the textbook treatment. Koehn (2004) gives the
standard bootstrap procedure in natural language processing.

**CAUTION on cluster count.** Cameron, Gelbach and Miller (2008) show that the
naive cluster bootstrap under-covers when the number of clusters is small. You
have 102 places. That is adequate but not large. Report the number of clusters
with every interval.

### 6.4 Statistical power

Card et al. (2020) show that most comparisons in natural language processing are
underpowered. Your table has about 20 configurations. The number of pairwise
comparisons grows with the square of that. Many comparisons on 102 place
clusters will be underpowered.

Do two things. Decide in advance which pairwise comparisons matter, and test
only those. Report the minimum detectable effect beside each p-value.

Søgaard et al. (2014) show that p-values are inflated when sample dependence and
multiple comparisons are ignored. Both apply to you.

---

## 7. Bibliography

Verification key: **[P]** the publisher or proceedings page was fetched.
**[C]** the record was verified through the Crossref metadata API, because the
publisher blocks automated access.

### 7.1 Configuration, reproducibility, experimental method

Balloccu, S., Schmidtová, P., Lango, M., Dušek, O. (2024). Leak, Cheat, Repeat:
Data Contamination and Evaluation Malpractices in Closed-Source LLMs.
*Proceedings of EACL 2024, Volume 1: Long Papers*, 67-93.
DOI 10.18653/v1/2024.eacl-long.5 **[C]**

Chen, L., Zaharia, M., Zou, J. (2024). How Is ChatGPT's Behavior Changing Over
Time? *Harvard Data Science Review* 6(2). DOI 10.1162/99608f92.5317da47 **[P]**

Dodge, J., Gururangan, S., Card, D., Schwartz, R., Smith, N. A. (2019). Show
Your Work: Improved Reporting of Experimental Results. *Proceedings of
EMNLP-IJCNLP 2019*, 2185-2194. DOI 10.18653/v1/D19-1224 **[P]**

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé
III, H., Crawford, K. (2021). Datasheets for Datasets. *Communications of the
ACM* 64(12), 86-92. DOI 10.1145/3458723 **[C]**

Gundersen, O. E., Kjensmo, S. (2018). State of the Art: Reproducibility in
Artificial Intelligence. *Proceedings of the AAAI Conference on Artificial
Intelligence* 32(1). DOI 10.1609/aaai.v32i1.11503 **[P]**

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018).
Deep Reinforcement Learning That Matters. *Proceedings of the AAAI Conference on
Artificial Intelligence* 32(1). DOI 10.1609/aaai.v32i1.11694 **[C]**

Idowu, S., Strüber, D., Berger, T. (2023). Asset Management in Machine Learning:
State-of-research and State-of-practice. *ACM Computing Surveys* 55(7),
Article 144. DOI 10.1145/3543847 **[C]**

Lipton, Z. C., Steinhardt, J. (2019). Troubling Trends in Machine Learning
Scholarship. *ACM Queue* 17(1), 45-77. DOI 10.1145/3317287.3328534 **[C]**

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B.,
Spitzer, E., Raji, I. D., Gebru, T. (2019). Model Cards for Model Reporting.
*Proceedings of FAT\* 2019*, 220-229. DOI 10.1145/3287560.3287596 **[C]**

Narang, S., Chung, H. W., Tay, Y., et al. (2021). Do Transformer Modifications
Transfer Across Implementations and Applications? *Proceedings of EMNLP 2021*,
5758-5773. DOI 10.18653/v1/2021.emnlp-main.465 **[C]**

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A.,
d'Alché-Buc, F., Fox, E., Larochelle, H. (2021). Improving Reproducibility in
Machine Learning Research. *Journal of Machine Learning Research* 22(164), 1-20.
https://www.jmlr.org/papers/v22/20-303.html **[P]**

Sculley, D., Holt, G., Golovin, D., et al. (2015). Hidden Technical Debt in
Machine Learning Systems. *Advances in Neural Information Processing Systems
28*, 2503-2511. **[P]** Confirm the page range against the printed volume.

Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data
management and stewardship. *Scientific Data* 3, 160018.
DOI 10.1038/sdata.2016.18 **[C]**

### 7.2 Baselines and evaluation practice

Armstrong, T. G., Moffat, A., Webber, W., Zobel, J. (2009). Improvements that
don't add up: ad-hoc retrieval results since 1998. *Proceedings of CIKM 2009*,
601-610. DOI 10.1145/1645953.1646031 **[C]**

Fuhr, N. (2018). Some Common Mistakes In IR Evaluation, And How They Can Be
Avoided. *ACM SIGIR Forum* 51(3), 32-41. DOI 10.1145/3190580.3190586 **[C]**

Gritta, M., Pilehvar, M. T., Limsopatham, N., Collier, N. (2018). What's missing
in geographical parsing? *Language Resources and Evaluation* 52(2), 603-623.
DOI 10.1007/s10579-017-9385-8 **[P]** Introduces WikToR, built so that
prominence heuristics mislead.

Gritta, M., Pilehvar, M. T., Collier, N. (2020). A pragmatic guide to geoparsing
evaluation. *Language Resources and Evaluation* 54(3), 683-712.
DOI 10.1007/s10579-019-09475-3 **[P]** Introduces GeoWebNews and the Pragmatic
Taxonomy of Toponyms, which your dataset design uses.

Speriosu, M., Baldridge, J. (2013). Text-Driven Toponym Resolution using
Indirect Supervision. *Proceedings of ACL 2013, Volume 1*, 1466-1476.
https://aclanthology.org/P13-1144/ **[P]**

Weissenbacher, D., Magge, A., O'Connor, K., Scotch, M., Gonzalez-Hernandez, G.
(2019). SemEval-2019 Task 12: Toponym Resolution in Scientific Papers.
*Proceedings of SemEval-2019*, 907-916. DOI 10.18653/v1/S19-2155 **[P]**

Zhang, Z., Bethard, S. (2025). A survey on geocoding: algorithms and datasets
for toponym resolution. *Language Resources and Evaluation* 59(2), 1775-1796.
DOI 10.1007/s10579-024-09730-2 **[C]**

### 7.3 Surface-form and gazetteer-free geolocation

Eisenstein, J., O'Connor, B., Smith, N. A., Xing, E. P. (2010). A Latent
Variable Model for Geographic Lexical Variation. *Proceedings of EMNLP 2010*,
1277-1287. https://aclanthology.org/D10-1124/ **[P]**

Gaman, M., Hovy, D., Ionescu, R. T., et al. (2020). A Report on the VarDial
Evaluation Campaign 2020. *Proceedings of the 7th VarDial Workshop*, 1-14.
https://aclanthology.org/2020.vardial-1.1/ **[P]** Documents the centroid
baseline and the `$NE$` entity-masking protocol.

Han, B., Cook, P., Baldwin, T. (2014). Text-Based Twitter User Geolocation
Prediction. *Journal of Artificial Intelligence Research* 49, 451-500.
DOI 10.1613/jair.4200 **[C]**

Jauhiainen, T., Jauhiainen, H., Lindén, K. (2020). Experiments in Language
Variety Geolocation and Dialect Identification. *Proceedings of the 7th VarDial
Workshop*, 220-231. https://aclanthology.org/2020.vardial-1.21/ **[P]**

Roller, S., Speriosu, M., Rallapalli, S., Wing, B., Baldridge, J. (2012).
Supervised Text-based Geolocation Using Language Models on an Adaptive Grid.
*Proceedings of EMNLP-CoNLL 2012*, 1500-1510.
https://aclanthology.org/D12-1137/ **[P]**

Wing, B., Baldridge, J. (2011). Simple supervised document geolocation with
geodesic grids. *Proceedings of ACL-HLT 2011*, 955-964.
https://aclanthology.org/P11-1096/ **[P]**

Wing, B., Baldridge, J. (2014). Hierarchical Discriminative Classification for
Text-Based Geolocation. *Proceedings of EMNLP 2014*, 336-348.
DOI 10.3115/v1/D14-1039 **[P]**

### 7.4 Geoparsers and their evaluation

DeLozier, G., Baldridge, J., London, L. (2015). Gazetteer-Independent Toponym
Resolution Using Geographic Word Profiles. *Proceedings of the AAAI Conference
on Artificial Intelligence* 29(1), 2382-2388. DOI 10.1609/aaai.v29i1.9531 **[P]**

Grover, C., Tobin, R., Byrne, K., Woollard, M., Reid, J., Dunn, S., Ball, J.
(2010). Use of the Edinburgh geoparser for georeferencing digitized historical
collections. *Philosophical Transactions of the Royal Society A* 368(1925),
3875-3889. DOI 10.1098/rsta.2010.0149 **[C]**

Halterman, A. (2017). Mordecai: Full Text Geoparsing and Event Geocoding.
*Journal of Open Source Software* 2(9), 91. DOI 10.21105/joss.00091 **[P]**

Hu, X., Zhou, Z., Li, H., Hu, Y., Gu, F., Kersten, J., Fan, H., Klan, F. (2023).
Location Reference Recognition from Texts: A Survey and Comparison. *ACM
Computing Surveys* 56(5), 1-37. DOI 10.1145/3625819 **[C]**

Karimzadeh, M., Pezanowski, S., MacEachren, A. M., Wallgrün, J. O. (2019).
GeoTxt: A scalable geoparsing system for unstructured text geolocation.
*Transactions in GIS* 23(1), 118-136. DOI 10.1111/tgis.12510 **[C]**

Lee, S. J., Liu, H., Ward, M. D. (2019). Lost in Space: Geolocation in Event
Data. *Political Science Research and Methods* 7(4), 871-888.
DOI 10.1017/psrm.2018.23 **[P]**

Leppämäki, T., Toivonen, T., Hiippala, T. (2024). Geographical and linguistic
perspectives on developing geoparsers with generic resources. *International
Journal of Geographical Information Science* 38(10), 2039-2060.
DOI 10.1080/13658816.2024.2369539 **[C]**

Melo, F., Martins, B. (2017). Automated Geocoding of Textual Documents: A Survey
of Current Approaches. *Transactions in GIS* 21(1), 3-38.
DOI 10.1111/tgis.12212 **[C]**

Wang, J., Hu, Y. (2019). Enhancing spatial and textual analysis with EUPEG: An
extensible and unified platform for evaluating geoparsers. *Transactions in GIS*
23(6), 1393-1419. DOI 10.1111/tgis.12579 **[C]**

Wang, J., Hu, Y. (2019). Are we there yet? Evaluating state-of-the-art neural
network based geoparsers using EUPEG as a benchmark platform. *Proceedings of
the 3rd ACM SIGSPATIAL International Workshop on Geospatial Humanities*, 1-6.
DOI 10.1145/3356991.3365470 **[C]**

### 7.5 Entity linking

Ayoola, T., Tyagi, S., Fisher, J., Christodoulopoulos, C., Pierleoni, A. (2022).
ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking.
*Proceedings of NAACL 2022, Industry Track*, 209-220.
DOI 10.18653/v1/2022.naacl-industry.24 **[P]**

De Cao, N., Wu, L., Popat, K., et al. (2022). Multilingual Autoregressive Entity
Linking. *Transactions of the ACL* 10, 274-290. DOI 10.1162/tacl_a_00460 **[P]**

Hachey, B., Radford, W., Nothman, J., Honnibal, M., Curran, J. R. (2013).
Evaluating Entity Linking with Wikipedia. *Artificial Intelligence* 194,
130-150. DOI 10.1016/j.artint.2012.04.005 **[C]**

Kolitsas, N., Ganea, O.-E., Hofmann, T. (2018). End-to-End Neural Entity
Linking. *Proceedings of CoNLL 2018*, 519-529. DOI 10.18653/v1/K18-1050 **[P]**

Krause, A., Cohen, W. W. (2020). Deriving Geolocations in Wikipedia.
*Proceedings of CIKM 2020*, 3293-3296. DOI 10.1145/3340531.3417459 **[C]**

Lai, T., Ji, H., Zhai, C. (2022). Improving Candidate Retrieval with Entity
Profile Generation for Wikidata Entity Linking. *Findings of ACL 2022*,
3696-3711. DOI 10.18653/v1/2022.findings-acl.292 **[P]**

Middleton, S. E., Kordopatis-Zilos, G., Papadopoulos, S., Kompatsiaris, Y.
(2018). Location Extraction from Social Media: Geoparsing, Location
Disambiguation, and Geotagging. *ACM Transactions on Information Systems* 36(4).
DOI 10.1145/3202662 **[C]**

Möller, C., Lehmann, J., Usbeck, R. (2022). Survey on English Entity Linking on
Wikidata: Datasets and approaches. *Semantic Web* 13(6), 925-966.
DOI 10.3233/SW-212865 **[C]**

Röder, M., Usbeck, R., Ngonga Ngomo, A.-C. (2018). GERBIL - Benchmarking Named
Entity Recognition and Linking consistently. *Semantic Web* 9(5), 605-625.
DOI 10.3233/SW-170286 **[C]**

van Hulst, J. M., Hasibi, F., Dercksen, K., Balog, K., de Vries, A. P. (2020).
REL: An Entity Linker Standing on the Shoulders of Giants. *Proceedings of SIGIR
2020*, 2197-2200. DOI 10.1145/3397271.3401416 **[C]**

Wu, L., Petroni, F., Josifoski, M., Riedel, S., Zettlemoyer, L. (2020). Scalable
Zero-shot Entity Linking with Dense Entity Retrieval. *Proceedings of EMNLP
2020*, 6397-6407. DOI 10.18653/v1/2020.emnlp-main.519 **[P]**

Zhang, Z., Bethard, S. (2023). Improving Toponym Resolution with Better
Candidate Generation, Transformer-based Reranking, and Two-Stage Resolution.
*Proceedings of \*SEM 2023*, 48-60. DOI 10.18653/v1/2023.starsem-1.6 **[P]**

Zhou, S., Rijhwani, S., Wieting, J., Carbonell, J., Neubig, G. (2020). Improving
Candidate Generation for Low-resource Cross-lingual Entity Linking.
*Transactions of the ACL* 8, 109-124. DOI 10.1162/tacl_a_00303 **[P]**

### 7.6 Retrieval

Bachrach, Y., Finkelstein, Y., Gilad-Bachrach, R., Katzir, L., Koenigstein, N.,
Nice, N., Paquet, U. (2014). Speeding up the Xbox recommender system using a
Euclidean transformation for inner-product spaces. *Proceedings of RecSys 2014*,
257-264. DOI 10.1145/2645710.2645741 **[C]**

Bruch, S., Gai, S., Ingber, A. (2023). An Analysis of Fusion Functions for
Hybrid Retrieval. *ACM Transactions on Information Systems* 42(1), 1-35.
DOI 10.1145/3596512 **[C]**

Chen, T., Zhang, M., Lu, J., Bendersky, M., Najork, M. (2022). Out-of-Domain
Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models. *Advances in
Information Retrieval (ECIR 2022)*, 95-110. DOI 10.1007/978-3-030-99736-6_7
**[C]**

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih,
W. (2020). Dense Passage Retrieval for Open-Domain Question Answering.
*Proceedings of EMNLP 2020*, 6769-6781. DOI 10.18653/v1/2020.emnlp-main.550
**[P]**

Khattab, O., Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search
via Contextualized Late Interaction over BERT. *Proceedings of SIGIR 2020*,
39-48. DOI 10.1145/3397271.3401075 **[C]**

Luan, Y., Eisenstein, J., Toutanova, K., Collins, M. (2021). Sparse, Dense, and
Attentional Representations for Text Retrieval. *Transactions of the ACL* 9,
329-345. DOI 10.1162/tacl_a_00369 **[P]**

Muennighoff, N., Tazi, N., Magne, L., Reimers, N. (2023). MTEB: Massive Text
Embedding Benchmark. *Proceedings of EACL 2023*, 2014-2037.
DOI 10.18653/v1/2023.eacl-main.148 **[P]**

Purves, R. S., Clough, P., Jones, C. B., Hall, M. H., Murdock, V. (2018).
Geographic Information Retrieval: Progress and Challenges in Spatial Search of
Text. *Foundations and Trends in Information Retrieval* 12(2-3), 164-318.
DOI 10.1561/1500000034 **[C]**

Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*, 3980-3990.
DOI 10.18653/v1/D19-1410 **[C]**

Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25
and Beyond. *Foundations and Trends in Information Retrieval* 3(4), 333-389.
DOI 10.1561/1500000019 **[C]** The deposited Crossref record for this DOI is
corrupted. Use the citation as given here.

Sciavolino, C., Zhong, Z., Lee, J., Chen, D. (2021). Simple Entity-Centric
Questions Challenge Dense Retrievers. *Proceedings of EMNLP 2021*, 6138-6148.
DOI 10.18653/v1/2021.emnlp-main.496 **[P]**

Steck, H., Ekanadham, C., Kallus, N. (2024). Is Cosine-Similarity of Embeddings
Really About Similarity? *Companion Proceedings of the ACM Web Conference 2024*,
887-890. DOI 10.1145/3589335.3651526 **[C]**

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. (2021). BEIR:
A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
Models. *Proceedings of the NeurIPS Track on Datasets and Benchmarks, Round 2*.
**[P]** Cite with the full track name. There is no DOI.

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F. (2024). Improving
Text Embeddings with Large Language Models. *Proceedings of ACL 2024, Long
Papers*, 11897-11916. DOI 10.18653/v1/2024.acl-long.642 **[P]**

Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., Nie, J.-Y. (2024).
C-Pack: Packed Resources For General Chinese Embeddings. *Proceedings of SIGIR
2024*, 641-649. DOI 10.1145/3626772.3657878 **[C]** The published venue for the
BGE family.

Zhang, X., Zhang, Y., Long, D., et al. (2024). mGTE: Generalized Long-Context
Text Representation and Reranking Models for Multilingual Text Retrieval.
*Proceedings of EMNLP 2024, Industry Track*, 1393-1412.
DOI 10.18653/v1/2024.emnlp-industry.103 **[P]**

Zhou, K., Ethayarajh, K., Card, D., Jurafsky, D. (2022). Problems with Cosine as
a Measure of Embedding Similarity for High Frequency Words. *Proceedings of ACL
2022, Short Papers*, 401-423. DOI 10.18653/v1/2022.acl-short.45 **[P]**

### 7.7 Language models, geographic knowledge, grounding

Alzahrani, N., Alyahya, H. A., Alnumay, Y., et al. (2024). When Benchmarks are
Targets: Revealing the Sensitivity of Large Language Model Leaderboards.
*Proceedings of ACL 2024, Volume 1: Long Papers*, 13787-13805.
DOI 10.18653/v1/2024.acl-long.744 **[P]**

Bhandari, P., Anastasopoulos, A., Pfoser, D. (2023). Are Large Language Models
Geospatially Knowledgeable? *Proceedings of SIGSPATIAL 2023*, 1-4.
DOI 10.1145/3589132.3625625 **[C]**

Campo, M., Conde, J., Alonso, Á., Huecas, G., Salvachúa, J., Reviriego, P.
(2026). Real-time Spatial Retrieval Augmented Generation for Urban Environments.
*ACM Transactions on Intelligent Systems and Technology*. DOI 10.1145/3831675
**[C]** Online first. Confirm the volume and pages before you submit.

Gurnee, W., Tegmark, M. (2024). Language Models Represent Space and Time.
*Proceedings of ICLR 2024*. **[search only]** Confirm the proceedings page
before you submit.

Hu, X., Kersten, J., Klan, F., Farzana, S. M. (2024). Toponym resolution
leveraging lightweight and open-source large language models and geo-knowledge.
*International Journal of Geographical Information Science* 40(3), 670-697.
DOI 10.1080/13658816.2024.2405182 **[C]** Published online 24 September 2024.
Assigned to a 2026 print issue. Choose one convention and keep it.

Hu, X., Kersten, J., Klan, F. (2025). Scalable Toponym Resolution with LLMs:
Accuracy and Speed Optimizations. *GeoExT 2025, CEUR Workshop Proceedings* 3969,
46-51. https://ceur-ws.org/Vol-3969/paper6.pdf **[P]**

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing
Systems 33*, 9459-9474. **[P]**

Manvi, R., Khanna, S., Burke, M., Lobell, D., Ermon, S. (2024a). Large Language
Models are Geographically Biased. *Proceedings of ICML 2024, PMLR* 235,
34654-34669. https://proceedings.mlr.press/v235/manvi24a.html **[P]**
This replaces the non-existent "Moayeri et al." citation used earlier in this
project.

Manvi, R., Khanna, S., Mai, G., Burke, M., Lobell, D., Ermon, S. (2024b).
GeoLLM: Extracting Geospatial Knowledge from Large Language Models.
*Proceedings of ICLR 2024*. **[search only]** Confirm before you submit.

Mizrahi, M., Kaplan, G., Malkin, D., Dror, R., Shahaf, D., Stanovsky, G. (2024).
State of What Art? A Call for Multi-Prompt LLM Evaluation. *Transactions of the
ACL* 12, 933-949. DOI 10.1162/tacl_a_00681 **[P]**

Sclar, M., Choi, Y., Tsvetkov, Y., Suhr, A. (2024). Quantifying Language Models'
Sensitivity to Spurious Features in Prompt Design. *Proceedings of ICLR 2024*.
**[P]**

Staab, R., Vero, M., Balunović, M., Vechev, M. (2024). Beyond Memorization:
Violating Privacy via Inference with Large Language Models. *Proceedings of ICLR
2024*. **[P]** Infers location and other attributes from real Reddit profiles at
up to 85% top-1 accuracy. Relevant to your ethics section.

Stillman, N., Kruspe, A. (2025). Biased Geolocation in LLMs: Experiments on
Probing LLMs for Geographic Knowledge and Reasoning. *GeoExT 2025, CEUR Workshop
Proceedings* 3969, 52-64. https://ceur-ws.org/Vol-3969/paper7.pdf **[P]**

Zhou, Z., Zhang, J., Guan, Z., Hu, M., Lao, N., Mu, L., Li, S., Mai, G. (2024).
Img2Loc: Revisiting Image Geolocalization using Multi-modality Foundation Models
and Image-based Retrieval-Augmented Generation. *Proceedings of SIGIR 2024*,
2749-2754. DOI 10.1145/3626772.3657673 **[C]**

### 7.8 Statistics

Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-Based
Improvements for Inference with Clustered Errors. *The Review of Economics and
Statistics* 90(3), 414-427. DOI 10.1162/rest.90.3.414 **[C]**

Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K., Jurafsky, D.
(2020). With Little Power Comes Great Responsibility. *Proceedings of EMNLP
2020*, 9263-9274. DOI 10.18653/v1/2020.emnlp-main.745 **[P]**

Davison, A. C., Hinkley, D. V. (1997). *Bootstrap Methods and their
Application*. Cambridge University Press. DOI 10.1017/CBO9780511802843 **[C]**

Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data
Sets. *Journal of Machine Learning Research* 7(1), 1-30.
https://www.jmlr.org/papers/v7/demsar06a.html **[P]**

Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised
Classification Learning Algorithms. *Neural Computation* 10(7), 1895-1923.
DOI 10.1162/089976698300017197 **[C]** Some secondary sources give the pages as
1885-1924. That is incorrect.

Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). The Hitchhiker's Guide
to Testing Statistical Significance in Natural Language Processing. *Proceedings
of ACL 2018, Volume 1*, 1383-1392. DOI 10.18653/v1/P18-1128 **[P]**

Field, C. A., Welsh, A. H. (2007). Bootstrapping Clustered Data. *Journal of the
Royal Statistical Society Series B* 69(3), 369-390.
DOI 10.1111/j.1467-9868.2007.00593.x **[C]** The direct citation for resampling
by place.

Japkowicz, N., Shah, M. (2011). *Evaluating Learning Algorithms: A
Classification Perspective*. Cambridge University Press.
DOI 10.1017/CBO9780511921803 **[C]**

Koehn, P. (2004). Statistical Significance Tests for Machine Translation
Evaluation. *Proceedings of EMNLP 2004*, 388-395.
https://aclanthology.org/W04-3250/ **[P]**

Søgaard, A., Johannsen, A., Plank, B., Hovy, D., Martínez Alonso, H. (2014).
What's in a p-value in NLP? *Proceedings of CoNLL 2014*, 1-10.
DOI 10.3115/v1/W14-1601 **[P]**

---

## 8. Items with no citable publication

Act on each of these before you submit.

| Item | Status | Action |
|---|---|---|
| Mordecai 3 | arXiv preprint only. Listed as a working paper on the author's CV | Cite Halterman (2017) for Mordecai 1, or remove the row |
| CLAVIN | No paper. CLIFF-CLAVIN is a non-archival abstract with no DOI | Cite through Wang and Hu (2019) or Lee et al. (2019) |
| Irchel Geoparser | JOSS submission in review since 2 April 2026. GIScience 2025 demo is non-archival | Cite as software with a version. Recheck the JOSS issue |
| E5 original and multilingual E5 | arXiv only. Multilingual E5 is a technical report | See section 4, Tier 3. Prefer a change to BGE |
| GTE original | arXiv only | Cite mGTE (Zhang et al., 2024) |
| Spatial-RAG | arXiv only. No DOI, no proceedings record | Do not cite. Use Hu et al. (2024) |
| Mai et al., Foundation Models for GeoAI | No published version found | Treat as a preprint. Do not cite |
| GENRE (monolingual) | ICLR 2021, OpenReview only, no DOI | Prefer mGENRE (De Cao et al., 2022) |
| "GeoCoDe" | Not found in Crossref or web search | The name appears to be incorrect. Check your source |
| "Moayeri et al." | Does not exist | Replaced by Manvi et al. (2024a) |

### Items to confirm manually

- Sculley et al. (2015): the page range 2503-2511.
- Gurnee and Tegmark (2024), Manvi et al. (2024b): fetch the ICLR proceedings
  pages.
- Campo et al. (2026): volume and pages, when assigned.
- Efron and Tibshirani: cite as 1993 or 1994, then stay consistent.
- Hu et al. IJGIS: cite as 2024 or 2026, then stay consistent.

---

## 9. Summary of actions

1. Rename the ten configurations in section 3.
2. Split `llm.backend` into `llm.provider` and `llm.model`. Add `tier`.
3. Add `char_ngram` and `centroid` to Tier 0.
4. Install `geoparser` to make Tier 1 runnable.
5. Add `grid_lm` to Tier 3. This is the strongest single addition.
6. Add `blink` to Tier 2.
7. Run every Tier 4 configuration under three or more prompt templates.
8. Decide the embedding model. Prefer BGE for a citable reference.
9. Resample by place, not by item. Report the cluster count.
10. Select the pairwise comparisons in advance. Report the minimum detectable
    effect.
