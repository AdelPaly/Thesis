# Retired prediction records

Per-item records for configurations removed from the run grid. Kept because
they are the evidence behind findings that are still cited, and because
re-running them would cost API calls and GPU time for numbers that already
exist.

The live scorer does not see them: `scripts/score.py` globs
`results/predictions/*.jsonl`, which does not match a subdirectory. To report
one, pass it explicitly:

```bash
python scripts/score.py --predictions results/predictions/retired/refined_search__*.jsonl
```

| configuration | removed because | finding it still supports |
|---|---|---|
| `refined_search` | ReFinED plus a search fallback — a patch on a method, not a method | the fallback trades abstention for confident error: `span_not_linked` 23.8% → 12.7%, `wrong_entity` 33.7% → 41.8%, +1.2pp exact |
| `refined_groq_oss` | routed skeleton with a swapped fallback model | model swap inside an existing tier |
| `refined_groq_llama` | routed skeleton with a swapped fallback model | model swap inside an existing tier |
| `groq_llama` | second Groq model; `groq_oss` kept as the LLM contrast | would have given a 0.2pp contrast against gemma (42.3 vs 42.5) — no sensitivity signal |

`refined_gemma` is deliberately **not** here. Its configuration was removed but
its result is still reported, as the routing comparison that currently beats
the rebuilt gate, so its records stay in `results/predictions/`.

To restore any of these configurations, recover its YAML from git history —
see the commit that removed it.
