"""Configuration as data.

A configuration is a YAML file in configs/. Nothing about a configuration --
which NER model, which linker, which LLM, which thresholds -- lives in code.
`load_config` reads one file, fills defaults, validates the combination, and
returns a frozen `PipelineConfig` whose `config_hash` is a sha256 over the
canonicalised resolved dict, computed at load time.

Adding a configuration means adding one file here. It does not mean editing
the pipeline, the runner, or a registry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATASET_REGISTRY = CONFIG_DIR / "datasets.yaml"

PARADIGMS = {"ner_dependent", "ner_free", "geoparser", "routed", "baseline"}
LINKING_BACKENDS = {"refined", "wikidata_search", "none"}
LLM_BACKENDS = {None, "gemma", "groq_oss", "groq_llama"}
ROUTING_GATES = {"legacy_is_geographic_qid", "input_properties"}
ROUTING_BRANCHES = {"llm", "linking", "geoparser"}
BASELINE_STRATEGIES = {"most_populous", "majority_dev_place"}


class ConfigError(ValueError):
    """Raised when a configuration file is missing a key or contradicts itself."""


@dataclass(frozen=True)
class NERConfig:
    model: str | None = None
    entity_types: list[str] | None = None   # None means keep every tag


@dataclass(frozen=True)
class LinkingConfig:
    backend: str = "none"
    refined_model: str | None = None
    refined_entity_set: str | None = None
    search_fallback: bool = False
    top_n: int = 10


@dataclass(frozen=True)
class RerankConfig:
    model: str | None = None
    enabled: bool = False


@dataclass(frozen=True)
class LLMConfig:
    backend: str | None = None
    model: str | None = None
    # cache_name addresses the on-disk response cache (src/.llm_cache/<name>.pkl).
    # Kept separate from `model` so existing caches stay addressable after a
    # model slug is rewritten; defaults to a slugified model id.
    cache_name: str | None = None


@dataclass(frozen=True)
class RoutingConfig:
    enabled: bool = False
    gate: str | None = None
    fallback: str | None = None
    # Gate thresholds. Fitted on dev only.
    ambiguity_threshold: int = 2       # >= this many gazetteer hits == ambiguous
    gazetteer_top_n: int = 10          # how many hits to ask for when counting


@dataclass(frozen=True)
class BaselineConfig:
    strategy: str | None = None        # most_populous | majority_dev_place
    gazetteer: str | None = None       # geonames
    min_population: int = 0


@dataclass(frozen=True)
class GeoparserConfig:
    backend: str | None = None         # mordecai3
    spacy_model: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    paradigm: str
    ner: NERConfig = field(default_factory=NERConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    geoparser: GeoparserConfig = field(default_factory=GeoparserConfig)
    description: str = ""
    config_hash: str = ""
    source_path: str = ""

    @property
    def uses_ner(self) -> bool:
        return self.ner.model is not None

    @property
    def uses_llm(self) -> bool:
        return self.llm.backend is not None

    def behavioural_dict(self) -> dict:
        """The part of the config that changes what the pipeline does.
        `description` and `source_path` are excluded: documentation and file
        location must not move the hash."""
        d = asdict(self)
        d.pop("description", None)
        d.pop("config_hash", None)
        d.pop("source_path", None)
        return d


def _canonical_hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _section(raw: dict, key: str) -> dict:
    """A section that is absent, or present but null, means 'all defaults'."""
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"section '{key}' must be a mapping, got {type(value).__name__}")
    return value


def _check_keys(section_name: str, raw: dict, cls) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in section '{section_name}'; "
            f"allowed: {sorted(allowed)}"
        )


def _slug(model_id: str | None) -> str | None:
    if not model_id:
        return None
    return model_id.replace("/", "_").replace("-", "_").replace(".", "_")


def from_dict(raw: dict, source_path: str = "") -> PipelineConfig:
    """Build and validate a PipelineConfig from an already-parsed mapping."""
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a YAML mapping")

    known_sections = {"name", "paradigm", "description", "ner", "linking", "rerank",
                      "llm", "routing", "baseline", "geoparser"}
    unknown = set(raw) - known_sections
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {sorted(unknown)}; "
                          f"allowed: {sorted(known_sections)}")

    name = raw.get("name")
    if not name:
        raise ConfigError("'name' is required")
    paradigm = raw.get("paradigm")
    if paradigm not in PARADIGMS:
        raise ConfigError(f"paradigm must be one of {sorted(PARADIGMS)}, got {paradigm!r}")

    sections = {
        "ner": (NERConfig, _section(raw, "ner")),
        "linking": (LinkingConfig, _section(raw, "linking")),
        "rerank": (RerankConfig, _section(raw, "rerank")),
        "llm": (LLMConfig, _section(raw, "llm")),
        "routing": (RoutingConfig, _section(raw, "routing")),
        "baseline": (BaselineConfig, _section(raw, "baseline")),
        "geoparser": (GeoparserConfig, _section(raw, "geoparser")),
    }
    built = {}
    for key, (cls, sub) in sections.items():
        _check_keys(key, sub, cls)
        built[key] = cls(**sub)

    llm = built["llm"]
    if llm.backend is not None and llm.cache_name is None:
        built["llm"] = LLMConfig(llm.backend, llm.model, _slug(llm.model))

    cfg = PipelineConfig(
        name=name,
        paradigm=paradigm,
        description=raw.get("description", "") or "",
        source_path=source_path,
        **built,
    )
    _validate(cfg)
    return PipelineConfig(**{**cfg.__dict__,
                             "config_hash": _canonical_hash(cfg.behavioural_dict())})


def _validate(cfg: PipelineConfig) -> None:
    n, li, rr, llm = cfg.ner, cfg.linking, cfg.rerank, cfg.llm
    ro, ba, gp = cfg.routing, cfg.baseline, cfg.geoparser

    if li.backend not in LINKING_BACKENDS:
        raise ConfigError(f"linking.backend must be one of {sorted(LINKING_BACKENDS)}")
    if llm.backend not in LLM_BACKENDS:
        raise ConfigError("llm.backend must be one of "
                          f"{sorted(str(b) for b in LLM_BACKENDS)}")
    if li.top_n < 1:
        raise ConfigError("linking.top_n must be >= 1")
    if rr.enabled and not rr.model:
        raise ConfigError("rerank.enabled is true but rerank.model is not set")
    if llm.backend and not llm.model:
        raise ConfigError(f"llm.backend is {llm.backend!r} but llm.model is not set")
    if n.entity_types is not None and not n.entity_types:
        raise ConfigError("ner.entity_types is an empty list; use null to keep every tag")
    if li.backend == "refined" and not (li.refined_model and li.refined_entity_set):
        raise ConfigError("linking.backend is 'refined' but refined_model / "
                          "refined_entity_set are not both set")

    if cfg.paradigm == "ner_dependent":
        if not n.model:
            raise ConfigError("paradigm 'ner_dependent' requires ner.model")
        if li.backend == "none":
            raise ConfigError("paradigm 'ner_dependent' requires a linking backend")
        if ro.enabled:
            raise ConfigError("paradigm 'ner_dependent' cannot have routing.enabled")

    elif cfg.paradigm == "ner_free":
        if not llm.backend:
            raise ConfigError("paradigm 'ner_free' requires an llm.backend")
        if n.model:
            raise ConfigError("paradigm 'ner_free' must not set ner.model -- that is "
                              "what makes it NER-free")
        if ro.enabled:
            raise ConfigError("paradigm 'ner_free' cannot have routing.enabled")

    elif cfg.paradigm == "routed":
        if not ro.enabled:
            raise ConfigError("paradigm 'routed' requires routing.enabled: true")
        if ro.gate not in ROUTING_GATES:
            raise ConfigError(f"routing.gate must be one of {sorted(ROUTING_GATES)}")
        if ro.fallback not in ROUTING_BRANCHES:
            raise ConfigError(f"routing.fallback must be one of {sorted(ROUTING_BRANCHES)}")
        if not n.model:
            raise ConfigError("paradigm 'routed' requires ner.model -- the gate reads "
                              "NER spans off the input")
        if ro.fallback == "llm" and not llm.backend:
            raise ConfigError("routing.fallback is 'llm' but llm.backend is not set")
        if ro.fallback == "geoparser" and not gp.backend:
            raise ConfigError("routing.fallback is 'geoparser' but geoparser.backend is not set")

    elif cfg.paradigm == "baseline":
        if ba.strategy not in BASELINE_STRATEGIES:
            raise ConfigError(f"baseline.strategy must be one of {sorted(BASELINE_STRATEGIES)}")
        if ba.strategy == "most_populous":
            if not n.model:
                raise ConfigError("baseline 'most_populous' requires ner.model")
            if not ba.gazetteer:
                raise ConfigError("baseline 'most_populous' requires baseline.gazetteer")

    elif cfg.paradigm == "geoparser":
        if not gp.backend:
            raise ConfigError("paradigm 'geoparser' requires geoparser.backend")


def load_config(path: str | Path) -> PipelineConfig:
    """Load one configuration file, by path or by bare name."""
    path = Path(path)
    if not path.exists():
        for candidate in (CONFIG_DIR / path.name,
                          CONFIG_DIR / f"{path.name}.yaml",
                          CONFIG_DIR / f"{path.stem}.yaml"):
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise ConfigError(f"no such configuration file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return from_dict(raw, source_path=str(path.relative_to(REPO_ROOT)))
    except ConfigError as e:
        raise ConfigError(f"{path.name}: {e}") from None


def load_all_configs(config_dir: str | Path = CONFIG_DIR) -> dict[str, PipelineConfig]:
    """Every configuration in the directory, keyed by name. The directory *is*
    the registry -- there is no list of modes anywhere in code."""
    out = {}
    for p in sorted(Path(config_dir).glob("*.yaml")):
        if p.name == DATASET_REGISTRY.name:
            continue
        cfg = load_config(p)
        if cfg.name in out:
            raise ConfigError(f"duplicate configuration name {cfg.name!r} "
                              f"({out[cfg.name].source_path} and {cfg.source_path})")
        out[cfg.name] = cfg
    return out


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    description: str = ""
    noisy: bool = False

    def resolved_path(self) -> Path:
        return REPO_ROOT / self.path


def load_datasets(registry: str | Path = DATASET_REGISTRY) -> dict[str, DatasetSpec]:
    raw = yaml.safe_load(Path(registry).read_text(encoding="utf-8")) or {}
    return {k: DatasetSpec(name=k, **v) for k, v in raw.items()}
