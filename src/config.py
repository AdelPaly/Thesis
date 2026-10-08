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
from dataclasses import dataclass, field, asdict, fields, replace
from datetime import date
from pathlib import Path

import yaml

from keys import legacy_config_name

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATASET_REGISTRY = CONFIG_DIR / "datasets.yaml"

PARADIGMS = {"ner_dependent", "ner_free", "baseline", "dense_retrieval",
             "grounded_llm", "geoparser", "grid_lm"}
LINKING_BACKENDS = {"refined", "wikidata_search", "none"}
LLM_PROVIDERS = {None, "google", "openrouter"}
PROMPT_TEMPLATES = {"v1"}
TIERS = range(0, 6)
BASELINE_STRATEGIES = {"most_populous", "majority_dev_place", "char_ngram"}
GAZETTEERS = {"wikidata"}
DENSE_POOLS = {"gold_only", "gold_plus_predicted"}
CANDIDATE_SOURCES = {"dense", "linking"}


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
    top_n: int = 10


@dataclass(frozen=True)
class RerankConfig:
    model: str | None = None
    enabled: bool = False


@dataclass(frozen=True)
class LLMConfig:
    # Who serves the model, kept apart from the model itself: the same weights
    # on another provider are the same model.
    provider: str | None = None        # google | openrouter
    model: str | None = None
    revision: str | None = None        # pinned where the provider offers one
    # None means "not sent": the provider's own default applies.
    temperature: float | None = None
    top_p: float | None = None
    prompt_template: str = "v1"
    # Date of the run that produced the predictions. Provenance only: a hosted
    # model can change behind an unchanged id, so the id alone does not say
    # what was evaluated. Excluded from config_hash.
    run_date: str | None = None
    # cache_name addresses the on-disk response cache (src/.llm_cache/<name>.pkl).
    # Kept separate from `model` so existing caches stay addressable after a
    # model slug is rewritten; defaults to a slugified model id.
    cache_name: str | None = None


@dataclass(frozen=True)
class BaselineConfig:
    strategy: str | None = None        # most_populous | majority_dev_place | char_ngram
    gazetteer: str | None = None       # wikidata
    min_population: int = 0
    # char_ngram: n-gram lengths, inclusive.
    ngram_min: int = 2
    ngram_max: int = 5


@dataclass(frozen=True)
class DenseConfig:
    """Tier 3: dense retrieval over candidate place descriptions."""
    encoder: str | None = None
    pool: str = "gold_plus_predicted"
    top_k: int = 10
    # What text represents a place in the index. This defines what the tier can
    # possibly retrieve, so it is a parameter and it is recorded in the index
    # metadata rather than being implicit in the code.
    index_fields: list[str] | None = None
    # E5 is trained with asymmetric prefixes for retrieval. They are parameters
    # so the choice is visible in the config rather than buried in the encoder
    # call, and so the index cache key can include them.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass(frozen=True)
class GroundedConfig:
    """Tier 5: retrieve candidates, then have the LLM select from them."""
    candidate_source: str | None = None      # dense | linking
    top_k: int = 10


@dataclass(frozen=True)
class GeoparserConfig:
    """Tier 1: an established geoparser, run as published."""
    system: str | None = None          # irchel
    version: str | None = None         # the installed version must match
    recognizer: str | None = None      # spaCy model
    entity_types: list[str] | None = None
    resolver: str | None = None        # sentence-transformers model
    gazetteer: str | None = None


@dataclass(frozen=True)
class GridConfig:
    """Tier 3: a language model per cell of a latitude/longitude grid."""
    cell_degrees: float | None = None


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    paradigm: str
    ner: NERConfig = field(default_factory=NERConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    grounded: GroundedConfig = field(default_factory=GroundedConfig)
    geoparser: GeoparserConfig = field(default_factory=GeoparserConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    # Presentation metadata: which tier of the evaluation this configuration
    # belongs to (0-5, by what kind of signal it can consume), and a variant
    # letter for configurations that share a tier. Excluded from config_hash
    # with `description`, because they order report rows and do not change a
    # single prediction -- forcing a re-run over row ordering would be absurd.
    tier: int | None = None
    variant: str | None = None
    description: str = ""
    config_hash: str = ""
    source_path: str = ""

    def behavioural_dict(self) -> dict:
        """The part of the config that changes what the pipeline does.
        `description` and `source_path` are excluded: documentation and file
        location must not move the hash."""
        d = asdict(self)
        for presentation_only in ("tier", "variant", "description", "config_hash",
                                  "source_path"):
            d.pop(presentation_only, None)
        d["llm"].pop("run_date")
        # Hash-stability shim: GroundedConfig.allow_abstention was removed (it
        # was never read) but every existing prediction file was hashed with it
        # present and True. Re-adding it keeps those hashes, and so resume,
        # valid. Delete this line only if you accept re-running everything.
        d["grounded"]["allow_abstention"] = True
        # Hash-stability shim: a renamed configuration hashes under its old
        # name, so prediction files written before the rename stay valid.
        d["name"] = legacy_config_name(self.name) or self.name
        # Hash-stability shim: a configuration with no LLM hashes its llm
        # section in the shape it had before `backend` was split into
        # `provider` and the sampling fields, so its existing prediction files
        # stay valid.
        if d["llm"]["provider"] is None:
            d["llm"] = {"backend": None, "model": None, "cache_name": None}
        return _drop_late_defaults(d)


# Fields added after prediction files existed. While a field holds its default
# it is left out of the hash, so adding it changed no existing configuration's
# hash; a configuration that sets it to anything else hashes it.
_LATE_FIELDS = {
    "baseline": {"ngram_min": 2, "ngram_max": 5},
}


# Sections added after prediction files existed, left out of the hash while
# every field holds its default, for the same reason.
_LATE_SECTIONS = {
    "geoparser": GeoparserConfig,
    "grid": GridConfig,
}


def _drop_late_defaults(d: dict) -> dict:
    for section, cls in _LATE_SECTIONS.items():
        if d.get(section) == asdict(cls()):
            del d[section]
    for section, defaults in _LATE_FIELDS.items():
        for key, default in defaults.items():
            if d[section].get(key) == default:
                del d[section][key]
    return d


def _canonical_hash(d: dict) -> str:
    # HAZARD: the hash covers every section, including ones irrelevant to the
    # paradigm. Adding any defaulted field to any config dataclass changes every
    # config's hash and invalidates all existing prediction files on resume
    # (they are reported as "different config_hash and will be replaced").
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

    known_sections = ({f.name for f in fields(PipelineConfig)}
                      - {"config_hash", "source_path"})
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
        "baseline": (BaselineConfig, _section(raw, "baseline")),
        "dense": (DenseConfig, _section(raw, "dense")),
        "grounded": (GroundedConfig, _section(raw, "grounded")),
        "geoparser": (GeoparserConfig, _section(raw, "geoparser")),
        "grid": (GridConfig, _section(raw, "grid")),
    }
    built = {}
    for key, (cls, sub) in sections.items():
        _check_keys(key, sub, cls)
        if isinstance(sub.get("run_date"), date):
            # YAML reads an unquoted 2026-08-14 as a date.
            sub = {**sub, "run_date": sub["run_date"].isoformat()}
        built[key] = cls(**sub)

    llm = built["llm"]
    if llm.provider is not None and llm.cache_name is None:
        built["llm"] = replace(llm, cache_name=_slug(llm.model))

    cfg = PipelineConfig(
        name=name,
        paradigm=paradigm,
        tier=raw.get("tier"),
        variant=raw.get("variant"),
        description=raw.get("description", "") or "",
        source_path=source_path,
        **built,
    )
    _validate(cfg)
    return replace(cfg, config_hash=_canonical_hash(cfg.behavioural_dict()))


def _require_type(where: str, value, expected: type, *, allow_none: bool = False) -> None:
    """Reject wrong-typed values (bool is not accepted as an int)."""
    if value is None and allow_none:
        return
    if isinstance(value, bool) and expected is not bool:
        ok = False
    else:
        ok = isinstance(value, expected)
    if not ok:
        raise ConfigError(f"{where} must be {expected.__name__}, "
                          f"got {type(value).__name__} ({value!r})")


def _require_str_list(where: str, value, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be a list of strings, got {value!r}")


def _validate_types(cfg: PipelineConfig) -> None:
    _require_type("tier", cfg.tier, int)
    _require_type("variant", cfg.variant, str, allow_none=True)
    for where, value in (("llm.temperature", cfg.llm.temperature),
                         ("llm.top_p", cfg.llm.top_p)):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, float))):
            raise ConfigError(f"{where} must be a number, got {value!r}")
    _require_str_list("ner.entity_types", cfg.ner.entity_types, allow_none=True)
    _require_str_list("dense.index_fields", cfg.dense.index_fields, allow_none=True)
    _require_type("linking.top_n", cfg.linking.top_n, int)
    _require_type("dense.top_k", cfg.dense.top_k, int)
    _require_type("grounded.top_k", cfg.grounded.top_k, int)
    _require_type("baseline.min_population", cfg.baseline.min_population, int)
    _require_type("baseline.ngram_min", cfg.baseline.ngram_min, int)
    _require_type("baseline.ngram_max", cfg.baseline.ngram_max, int)
    _require_type("rerank.enabled", cfg.rerank.enabled, bool)
    _require_type("dense.query_prefix", cfg.dense.query_prefix, str)
    _require_type("dense.passage_prefix", cfg.dense.passage_prefix, str)
    for where, value in (("ner.model", cfg.ner.model),
                         ("linking.refined_model", cfg.linking.refined_model),
                         ("linking.refined_entity_set", cfg.linking.refined_entity_set),
                         ("rerank.model", cfg.rerank.model),
                         ("llm.model", cfg.llm.model),
                         ("llm.cache_name", cfg.llm.cache_name),
                         ("llm.revision", cfg.llm.revision),
                         ("llm.run_date", cfg.llm.run_date),
                         ("llm.prompt_template", cfg.llm.prompt_template),
                         ("dense.encoder", cfg.dense.encoder),
                         ("baseline.strategy", cfg.baseline.strategy)):
        _require_type(where, value, str, allow_none=True)


def _validate_common(cfg: PipelineConfig) -> None:
    linking, rerank, llm = cfg.linking, cfg.rerank, cfg.llm
    if linking.backend not in LINKING_BACKENDS:
        raise ConfigError(f"linking.backend must be one of {sorted(LINKING_BACKENDS)}")
    if cfg.tier not in TIERS:
        raise ConfigError(f"tier must be an integer from 0 to 5, got {cfg.tier!r}")
    if llm.provider not in LLM_PROVIDERS:
        raise ConfigError("llm.provider must be one of "
                          f"{sorted(str(b) for b in LLM_PROVIDERS)}")
    if llm.prompt_template not in PROMPT_TEMPLATES:
        raise ConfigError(f"llm.prompt_template must be one of {sorted(PROMPT_TEMPLATES)}")
    if linking.top_n < 1:
        raise ConfigError("linking.top_n must be >= 1")
    if cfg.dense.pool not in DENSE_POOLS:
        raise ConfigError(f"dense.pool must be one of {sorted(DENSE_POOLS)}")
    if cfg.dense.top_k < 1:
        raise ConfigError("dense.top_k must be >= 1")
    if cfg.grounded.top_k < 1:
        raise ConfigError("grounded.top_k must be >= 1")
    if rerank.enabled and not rerank.model:
        raise ConfigError("rerank.enabled is true but rerank.model is not set")
    if llm.provider and not llm.model:
        raise ConfigError(f"llm.provider is {llm.provider!r} but llm.model is not set")
    if cfg.ner.entity_types is not None and not cfg.ner.entity_types:
        raise ConfigError("ner.entity_types is an empty list; use null to keep every tag")
    if cfg.baseline.gazetteer is not None and cfg.baseline.gazetteer not in GAZETTEERS:
        raise ConfigError(f"baseline.gazetteer must be one of {sorted(GAZETTEERS)}")
    if linking.backend == "refined" and not (linking.refined_model
                                             and linking.refined_entity_set):
        raise ConfigError("linking.backend is 'refined' but refined_model / "
                          "refined_entity_set are not both set")


def _validate_ner_dependent(cfg: PipelineConfig) -> None:
    if not cfg.ner.model:
        raise ConfigError("paradigm 'ner_dependent' requires ner.model")
    if cfg.linking.backend == "none":
        raise ConfigError("paradigm 'ner_dependent' requires a linking backend")


def _validate_ner_free(cfg: PipelineConfig) -> None:
    if not cfg.llm.provider:
        raise ConfigError("paradigm 'ner_free' requires an llm.provider")
    if cfg.ner.model:
        raise ConfigError("paradigm 'ner_free' must not set ner.model -- that is "
                          "what makes it NER-free")


def _validate_baseline(cfg: PipelineConfig) -> None:
    baseline = cfg.baseline
    if baseline.strategy not in BASELINE_STRATEGIES:
        raise ConfigError(f"baseline.strategy must be one of {sorted(BASELINE_STRATEGIES)}")
    if baseline.strategy == "most_populous":
        if not cfg.ner.model:
            raise ConfigError("baseline 'most_populous' requires ner.model")
        if not baseline.gazetteer:
            raise ConfigError("baseline 'most_populous' requires baseline.gazetteer")
    if baseline.strategy == "char_ngram":
        if cfg.ner.model or baseline.gazetteer:
            raise ConfigError("baseline 'char_ngram' must not set ner.model or "
                              "baseline.gazetteer -- surface form only is the point")
        if not 1 <= baseline.ngram_min <= baseline.ngram_max:
            raise ConfigError("baseline.ngram_min / ngram_max must satisfy "
                              "1 <= ngram_min <= ngram_max")


def _validate_dense_retrieval(cfg: PipelineConfig) -> None:
    if not cfg.dense.encoder:
        raise ConfigError("paradigm 'dense_retrieval' requires dense.encoder")
    if not cfg.dense.index_fields:
        raise ConfigError("paradigm 'dense_retrieval' requires dense.index_fields "
                          "-- what goes into the index defines what the tier can "
                          "retrieve, so it must be stated")
    if cfg.ner.model:
        raise ConfigError("paradigm 'dense_retrieval' must not set ner.model -- "
                          "having no mention-detection stage is the point of the "
                          "tier")
    if cfg.llm.provider:
        raise ConfigError("paradigm 'dense_retrieval' must not set llm.provider")


def _validate_grounded_llm(cfg: PipelineConfig) -> None:
    grounded = cfg.grounded
    if grounded.candidate_source not in CANDIDATE_SOURCES:
        raise ConfigError("grounded.candidate_source must be one of "
                          f"{sorted(CANDIDATE_SOURCES)}")
    if not cfg.llm.provider:
        raise ConfigError("paradigm 'grounded_llm' requires an llm.provider")
    if grounded.candidate_source == "dense" and not cfg.dense.encoder:
        raise ConfigError("grounded.candidate_source is 'dense' but dense.encoder "
                          "is not set")
    if grounded.candidate_source == "linking":
        if not cfg.ner.model:
            raise ConfigError("grounded.candidate_source is 'linking' but "
                              "ner.model is not set")
        if cfg.linking.backend == "none":
            raise ConfigError("grounded.candidate_source is 'linking' but "
                              "linking.backend is 'none'")


GEOPARSER_SYSTEMS = {"irchel"}


def _validate_geoparser(cfg: PipelineConfig) -> None:
    gp = cfg.geoparser
    if gp.system not in GEOPARSER_SYSTEMS:
        raise ConfigError(f"geoparser.system must be one of {sorted(GEOPARSER_SYSTEMS)}")
    for key in ("version", "recognizer", "resolver", "gazetteer"):
        if not isinstance(getattr(gp, key), str) or not getattr(gp, key):
            raise ConfigError(f"paradigm 'geoparser' requires geoparser.{key}")
    _require_str_list("geoparser.entity_types", gp.entity_types)
    if cfg.ner.model or cfg.llm.provider or cfg.linking.backend != "none":
        raise ConfigError("paradigm 'geoparser' runs the system as published: it must "
                          "not set ner.model, llm.provider or linking.backend")


def _validate_grid_lm(cfg: PipelineConfig) -> None:
    degrees = cfg.grid.cell_degrees
    if isinstance(degrees, bool) or not isinstance(degrees, (int, float)) or degrees <= 0:
        raise ConfigError("paradigm 'grid_lm' requires grid.cell_degrees > 0")
    if cfg.ner.model or cfg.llm.provider or cfg.linking.backend != "none":
        raise ConfigError("paradigm 'grid_lm' must not set ner.model, llm.provider "
                          "or linking.backend -- words in, cell out is the point")


_PARADIGM_VALIDATORS = {
    "ner_dependent": _validate_ner_dependent,
    "ner_free": _validate_ner_free,
    "baseline": _validate_baseline,
    "dense_retrieval": _validate_dense_retrieval,
    "grounded_llm": _validate_grounded_llm,
    "geoparser": _validate_geoparser,
    "grid_lm": _validate_grid_lm,
}


def _validate(cfg: PipelineConfig) -> None:
    _validate_types(cfg)
    _validate_common(cfg)
    _PARADIGM_VALIDATORS[cfg.paradigm](cfg)


def load_config(path: str | Path) -> PipelineConfig:
    """Load one configuration file, by path or by bare name ("refined")."""
    path = Path(path)
    if not path.exists():
        for candidate in (CONFIG_DIR / path.name, CONFIG_DIR / f"{path.name}.yaml"):
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise ConfigError(f"no such configuration file: {path}")
    path = path.resolve()
    try:
        source_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raise ConfigError("file is empty")
        return from_dict(raw, source_path=source_path)
    except (ConfigError, yaml.YAMLError) as e:
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

    def resolved_path(self) -> Path:
        return REPO_ROOT / self.path


def load_datasets(registry: str | Path = DATASET_REGISTRY) -> dict[str, DatasetSpec]:
    raw = yaml.safe_load(Path(registry).read_text(encoding="utf-8")) or {}
    return {k: DatasetSpec(name=k, **v) for k, v in raw.items()}
