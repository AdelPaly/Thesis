import os
import time
import google.generativeai as genai
from google.api_core import exceptions as gax
from llm_geo import LLMGeoPipeline

_TRANSIENT = (gax.InternalServerError, gax.ServiceUnavailable, gax.ResourceExhausted, gax.DeadlineExceeded)


class GemmaPipeline(LLMGeoPipeline):
    def __init__(self):
        super().__init__("gemma")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel("gemma-3-12b-it")

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


_pipeline = GemmaPipeline()
geolocate_gemma = _pipeline.geolocate
save_cache = _pipeline.save_cache
