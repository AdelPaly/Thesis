"""Google Gemma handler.

The model id and the cache name come from the configuration (`llm.model`,
`llm.cache_name`); nothing here names a model. `build()` is the factory the
pipeline calls, and instances are memoised per (model, cache) pair so a run
loads one client.
"""
import os
import time

import google.generativeai as genai
from google.api_core import exceptions as gax

from llm_geo import LLMGeoPipeline

_TRANSIENT = (gax.InternalServerError, gax.ServiceUnavailable,
              gax.ResourceExhausted, gax.DeadlineExceeded)

_instances = {}


class GemmaPipeline(LLMGeoPipeline):
    def __init__(self, model_id, cache_name):
        super().__init__(cache_name)
        self.model_id = model_id
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_id)

    def _call_llm(self, prompt):
        cfg = genai.types.GenerationConfig(temperature=0)
        delay = 2
        for attempt in range(5):
            try:
                return self.model.generate_content(prompt, generation_config=cfg).text
            except _TRANSIENT as e:
                if attempt == 4:
                    raise
                print(f" [retry {attempt+1}/4 after {type(e).__name__}]", end="", flush=True)
                time.sleep(delay)
                delay *= 2


def build(model_id, cache_name):
    key = (model_id, cache_name)
    if key not in _instances:
        _instances[key] = GemmaPipeline(model_id, cache_name)
    return _instances[key]
