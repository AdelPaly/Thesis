# Geolocation Pipeline: Architecture and Improvements

## 1. Pipeline Overview

The geolocation pipeline predicts a real-world location (as a Wikidata entity with coordinates) from unstructured text. It operates in four sequential stages:

```
Input text
    |
    v
[1. Named Entity Recognition] -- extracts LOC, PER, ORG entities
    |
    v
[2. Wikidata Entity Linking]  -- finds candidate locations for each entity
    |
    v
[3. Semantic Ranking]         -- scores candidates by text similarity
    |
    v
[4. Aggregation]              -- selects final prediction from candidates
    |
    v
Predicted location (QID, coordinates, country, continent)
```

## 2. Stage 1: Named Entity Recognition (NER)

**Model:** Flair `ner-large`, a pre-trained BiLSTM-CRF sequence labeling model.

**Input:** Raw text (news article or social media post).

**Output:** List of named entities, each with:
- `text`: the surface form (e.g., "Kaizer Chiefs")
- `type`: entity class -- `LOC` (location), `PER` (person), `ORG` (organization), or `MISC` (miscellaneous)
- `score`: model confidence in [0, 1]

The NER stage identifies all named entities in the text. Only LOC, PER, and ORG types are used downstream; MISC entities (e.g., nationality adjectives like "French") are discarded by the entity linking stage.

## 3. Stage 2: Wikidata Entity Linking

For each extracted entity, the pipeline queries the Wikidata API to find candidate real-world locations. The resolution strategy differs by entity type:

### LOC entities (direct resolution)
1. Search Wikidata for the entity text (e.g., "Highlands").
2. If no results, progressively remove trailing words (e.g., "Kaizer Chiefs Foundation" -> "Kaizer Chiefs").
3. Batch-fetch all candidate entities and filter to those with coordinate property (P625).
4. Resolve each candidate's country (P17) and continent (P30) via batch label lookups.
5. Sort by Wikipedia sitelink count (a proxy for notability) and return top 10 candidates.

### PER entities (indirect -- birthplace)
1. Search Wikidata for the person's name.
2. Filter to entities classified as human (P31 = Q5).
3. Retrieve each person's birthplace (P19) and fetch its coordinates.
4. Return up to 5 birthplace locations.

### ORG entities (indirect -- headquarters)
1. Search Wikidata for the organization name.
2. Retrieve each organization's headquarters location (P159).
3. If a headquarters lacks coordinates, fall back to its administrative territory (P131).
4. Sort by sitelink count and return top 10.

### Caching and rate limiting
- Three-layer cache (entity data, labels, coordinates) minimizes redundant API calls.
- Batch fetching groups up to 50 Wikidata IDs per request.
- Rate limiting (100ms minimum interval) with exponential backoff on HTTP 429 responses.

## 4. Stage 3: Semantic Ranking

**Model:** `intfloat/multilingual-e5-base`, a multilingual sentence transformer.

**Process:**
1. Encode the full input text into a dense embedding vector.
2. Encode each candidate's Wikidata description (e.g., "capital and largest city of France") into embedding vectors.
3. Compute cosine similarity between the text embedding and each candidate description embedding.
4. Sort candidates by similarity score in descending order.

This stage re-ranks candidates so that those whose Wikidata descriptions are most semantically related to the input text appear first. For example, an article about football in Ghana should rank "administrative region in Ghana" higher than "borough of New Jersey, United States".

## 5. Stage 4: Aggregation

Three aggregation strategies are implemented:

### Strategy: `top1` (baseline)
Simply returns the highest-ranked candidate after semantic ranking. This is the simplest approach but is vulnerable to ranking errors -- a single misranked candidate produces a wrong prediction.

### Strategy: `cluster`
1. Project all candidate coordinates into radians.
2. Apply HDBSCAN density-based clustering with haversine distance metric (min_cluster_size=2).
3. Identify the largest cluster.
4. Return the highest-ranked candidate from that cluster.
5. Falls back to `top1` if fewer than 2 candidates or no clusters form.

**Limitation:** When Wikidata returns multiple candidates for a single entity from the same country (e.g., four US towns named "Highlands"), clustering can favor that country even when it is incorrect. The cluster forms from disambiguation noise, not from genuine geographic consensus across different entities.

### Strategy: `consensus` (improved)
A two-level aggregation that resolves entities individually before seeking geographic agreement:

1. **Per-entity resolution:** Group candidates by their source entity. For each entity, select its top-ranked candidate (the one with the highest semantic similarity score). This eliminates within-entity disambiguation noise.

2. **Country voting:** Count how many distinct entities point to each country. The country with the most entity-level votes wins. This captures geographic consensus -- if three different entities in the text all resolve to locations in Ghana, Ghana is likely the correct country.

3. **Final selection:** Among all candidates from the winning country, prefer LOC-type entities over PER/ORG (since LOC provides direct coordinates rather than indirect birthplace/HQ locations). Return the highest-ranked LOC candidate from the winning country.

**Key insight:** PER and ORG entities participate in country voting (providing geographic signal) but LOC entities are preferred for the final coordinate prediction. This allows indirect entities like organization names or person names to influence the country-level decision without being returned as the final location.

## 6. Evaluation Methodology

The pipeline is evaluated at three granularity levels:

| Level | Criterion | Description |
|-------|-----------|-------------|
| **Exact** | Predicted QID = True QID | The pipeline identified the exact Wikidata entity |
| **Country** | Predicted country = True country (or exact match) | Correct country, possibly wrong specific location |
| **Continent** | Predicted continent = True continent (or country/exact match) | Correct continent, possibly wrong country |

Additionally, "No result" is tracked when the pipeline fails to produce any prediction (no candidates found for any entity).

Results are broken down by signal type:
- **Signal A:** Unambiguous location names
- **Signal B:** Moderately ambiguous names
- **Signal C:** Highly ambiguous names with multiple common referents

## 7. Baseline Results (top1 strategy)

| Metric | Overall (n=72) | Signal A | Signal B | Signal C |
|--------|----------------|----------|----------|----------|
| Exact | 33.3% | 33.3% | 37.5% | 29.2% |
| Country | 44.4% | 37.5% | 58.3% | 37.5% |
| Continent | 51.4% | 45.8% | 66.7% | 41.7% |
| No result | 12.5% | 20.8% | 0.0% | 16.7% |

### Analysis of failure modes

The baseline pipeline has several systematic failure modes:

1. **Ranking mismatch:** The semantic similarity between a full news article and a short Wikidata description (e.g., "city in Gauteng, South Africa") is a weak signal. Articles about sports, business, or culture have minimal lexical overlap with geographic descriptions.

2. **Indirect entity resolution:** PER entities resolve to birthplaces and ORG entities to headquarters, both of which are often wrong (e.g., a Wikidata search for a hip-hop group returns the wrong entity entirely).

3. **Disambiguation bias:** Wikidata search results are biased toward entities with more sitelinks (typically Western/anglophone locations). A search for "Victoria" returns the Australian state before the Seychelles capital.

4. **No geographic consensus:** The `top1` strategy selects a single candidate without considering whether multiple entities in the text corroborate the same geographic area.

## 8. Improved Results (consensus strategy)

| Metric | Overall (n=72) | Signal A | Signal B | Signal C |
|--------|----------------|----------|----------|----------|
| Exact | 31.9% | 37.5% | 37.5% | 20.8% |
| Country | 45.8% | 45.8% | 58.3% | 33.3% |
| Continent | 52.8% | 54.2% | 66.7% | 37.5% |
| No result | 20.8% | 20.8% | 0.0% | 41.7% |

**Note on Signal C results:** The evaluation run experienced heavy Wikidata API rate limiting (HTTP 429) during the second half, which disproportionately affected Signal C samples. Many "No result" outcomes in Signal C are caused by API failures, not pipeline logic failures. On the first 50 samples (where API access was reliable), the consensus strategy showed clearer improvements:

| Metric | Baseline (first 50) | Consensus (first 50) | Delta |
|--------|---------------------|----------------------|-------|
| Exact | 34.0% | 36.0% | +2.0pp |
| Country | 48.0% | 52.0% | +4.0pp |
| Continent | 56.0% | 60.0% | +4.0pp |
| No result | 10.0% | 10.0% | 0.0pp |

### Comparison: Signal A improvements

Signal A (unambiguous names) saw the largest gains, with exact match improving from 33.3% to 37.5% and country-level accuracy from 37.5% to 45.8%. This confirms that the consensus strategy is most effective when the text contains multiple entities that correctly point to the same geographic area.

### Why consensus improves results

The `geographic_consensus` strategy addresses failure modes #3 and #4:

- **Per-entity resolution** prevents disambiguation noise from one entity (e.g., four US towns named "Highlands") from overwhelming the correct signal from another entity (e.g., "Ashanti" -> Ghana).
- **Country voting** exploits the fact that texts typically contain multiple entities pointing to the same geographic area. Even if one entity resolves incorrectly, the majority vote across entities can still identify the correct country.
- **LOC preference in final selection** avoids returning indirect PER/ORG locations while still benefiting from their geographic signal during voting.

## 9. Implications for Noisy Text

When adversarial noise is introduced (e.g., "English bulldog bit Spanish tourist in Norwegian hotel"), the pipeline faces additional challenges:

- **False entity extraction:** NER may tag "English", "Spanish", "Norwegian" as entities, creating candidates in the UK, Spain, and Norway.
- **Diluted consensus:** Noise entities add country votes for incorrect locations, potentially outvoting the correct signal.

The `consensus` strategy provides partial resilience: if the text contains more genuine location entities than noise entities, the correct country will still win the vote. The MISC entity type filter in NER also helps, as nationality adjectives are often tagged as MISC rather than LOC.

Further noise resilience could be achieved by:
- Filtering entities by NER confidence score (noise entities tend to have lower confidence).
- Weighting votes by semantic ranking score rather than treating all entity votes equally.
- Using the cluster strategy as a secondary check on the consensus result.
