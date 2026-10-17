"""BLINK entity linking (tier 2).

BLINK as published: a bi-encoder retrieves the top-k of its 5,903,527
Wikipedia entities for each mention, and a cross-encoder rescores those k with
the mention context and picks one. The models, the entity catalogue and the
precomputed entity encodings are BLINK's own downloads; the bi-encoder,
cross-encoder and their input processing are BLINK's own code (imported from
.venv/src/blink).

One thing differs from BLINK's `main_dense`, and only in how the arithmetic is
laid out: `main_dense` holds the 5.9M x 1024 float32 entity matrix (24 GB) in
memory and scores each mention against all of it. That does not fit in this
machine's memory, so every mention of a run is scored in one pass over the
file, reading one chunk at a time (a plain read, not a memory map, so pages
already scored do not stay resident) and keeping a running top-k. It is the
same exact inner-product search, not an approximate index.

The cross-encoder's chosen Wikipedia title is mapped to a Wikidata QID by the
enwiki sitelink.

Linking results are cached on disk keyed by the mention and its context, so a
re-run or a resumed run does not repeat the pass.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np

from config import REPO_ROOT, ConfigError

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("BLINK_MODELS")
                  or REPO_ROOT / ".venv" / "src" / "blink" / "models")
CACHE_PATH = Path(__file__).resolve().parent / ".blink_cache" / "links.pkl"

N_ENTITIES = 5903527
DIM = 1024
CHUNK_ROWS = 200_000
ENCODE_BATCH = 16


def _mention_key(sample: dict, top_k: int) -> tuple:
    return (sample["context_left"], sample["mention"], sample["context_right"], top_k)


def make_sample(text: str, start: int, end: int) -> dict:
    """BLINK's mention record, lower-cased as BLINK's own annotator does."""
    return {"label": "unknown", "label_id": -1,
            "context_left": text[:start].lower(),
            "mention": text[start:end].lower(),
            "context_right": text[end:].lower()}


class BlinkLinker:
    def __init__(self, top_k: int):
        import torch  # noqa: F401  (fail here, not mid-run, if torch is absent)
        from blink.biencoder.biencoder import load_biencoder
        from blink.crossencoder.crossencoder import load_crossencoder

        self.top_k = top_k
        paths = {name: MODELS_DIR / name for name in (
            "biencoder_wiki_large.json", "biencoder_wiki_large.bin",
            "crossencoder_wiki_large.json", "crossencoder_wiki_large.bin",
            "entity.jsonl", "all_entities_large.t7")}
        missing = [p.name for p in paths.values() if not p.exists()]
        if missing:
            raise ConfigError(f"BLINK model files missing from {MODELS_DIR}: {missing}")

        with paths["biencoder_wiki_large.json"].open() as f:
            self.bi_params = json.load(f)
        self.bi_params["path_to_model"] = str(paths["biencoder_wiki_large.bin"])
        with paths["crossencoder_wiki_large.json"].open() as f:
            self.cross_params = json.load(f)
        self.cross_params["path_to_model"] = str(paths["crossencoder_wiki_large.bin"])
        self.biencoder = load_biencoder(self.bi_params)
        self.crossencoder = load_crossencoder(self.cross_params)
        self.biencoder.model.eval()
        self.crossencoder.model.eval()

        self.encodings = paths["all_entities_large.t7"]
        self._offset = self._matrix_offset(self.encodings)
        self.catalogue = paths["entity.jsonl"]
        self._offsets = self._line_offsets(self.catalogue)
        self._cache = self._load_cache()

    # ------------------------------------------------------------ resources

    @staticmethod
    def _matrix_offset(path: Path) -> int:
        """Byte offset of the entity matrix in all_entities_large.t7.

        The file is a single float32 tensor in torch's legacy serialisation:
        a small pickled header followed by the raw storage, so the storage is
        the last N_ENTITIES * DIM * 4 bytes of the file."""
        size = path.stat().st_size
        data_bytes = N_ENTITIES * DIM * 4
        if size < data_bytes:
            raise ConfigError(f"{path.name} is {size} bytes, smaller than its "
                              f"{data_bytes}-byte matrix; the download is incomplete")
        return size - data_bytes

    def _rows(self, start: int, stop: int) -> np.ndarray:
        """Entity rows [start, stop), read from disk."""
        with self.encodings.open("rb") as f:
            f.seek(self._offset + start * DIM * 4)
            data = np.fromfile(f, dtype="<f4", count=(stop - start) * DIM)
        return data.reshape(stop - start, DIM)

    @staticmethod
    def _line_offsets(path: Path) -> np.ndarray:
        """Byte offset of every line of entity.jsonl, cached beside it."""
        index = path.with_suffix(".offsets.npy")
        if index.exists():
            return np.load(index)
        offsets = np.empty(N_ENTITIES, dtype=np.int64)
        n = 0
        with path.open("rb") as f:
            pos = 0
            for line in f:
                offsets[n] = pos
                pos += len(line)
                n += 1
        if n != N_ENTITIES:
            raise ConfigError(f"{path.name} has {n} lines, expected {N_ENTITIES}")
        np.save(index, offsets)
        return offsets

    def entity(self, local_id: int) -> dict:
        with self.catalogue.open("rb") as f:
            f.seek(int(self._offsets[local_id]))
            return json.loads(f.readline())

    @staticmethod
    def _load_cache() -> dict:
        if CACHE_PATH.exists():
            with CACHE_PATH.open("rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self._cache, f)
        os.replace(tmp, CACHE_PATH)

    # --------------------------------------------------------------- linking

    def _encode_contexts(self, samples: list[dict]):
        import torch
        from blink.biencoder.data_process import process_mention_data

        _, tensor_data = process_mention_data(
            samples, self.biencoder.tokenizer, self.bi_params["max_context_length"],
            self.bi_params["max_cand_length"], silent=True, logger=None, debug=False)
        context_input = tensor_data.tensors[0]
        out = []
        with torch.no_grad():
            for i in range(0, len(context_input), ENCODE_BATCH):
                out.append(self.biencoder.encode_context(context_input[i:i + ENCODE_BATCH]))
        return torch.cat(out).numpy().astype("float32")

    def _search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k inner product of every query against all entities."""
        k = self.top_k
        best_scores = np.full((len(queries), k), -np.inf, dtype="float32")
        best_ids = np.zeros((len(queries), k), dtype=np.int64)
        for start in range(0, N_ENTITIES, CHUNK_ROWS):
            chunk = self._rows(start, min(start + CHUNK_ROWS, N_ENTITIES))
            scores = queries @ chunk.T
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores, part, axis=1)
            merged_scores = np.concatenate([best_scores, part_scores], axis=1)
            merged_ids = np.concatenate([best_ids, part + start], axis=1)
            keep = np.argsort(-merged_scores, axis=1, kind="stable")[:, :k]
            best_scores = np.take_along_axis(merged_scores, keep, axis=1)
            best_ids = np.take_along_axis(merged_ids, keep, axis=1)
            log.info("BLINK bi-encoder pass: %d / %d entities",
                     min(start + CHUNK_ROWS, N_ENTITIES), N_ENTITIES)
        return best_ids, best_scores

    def _rerank(self, samples: list[dict], nns: np.ndarray):
        from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

        from blink.crossencoder.data_process import prepare_crossencoder_data
        from blink.crossencoder.train_cross import evaluate, modify

        ids = {int(i) for row in nns for i in row}
        entities = {i: self.entity(i) for i in ids}
        id2title = {i: e["title"] for i, e in entities.items()}
        id2text = {i: e["text"] for i, e in entities.items()}
        labels = [-1] * len(samples)
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.crossencoder.tokenizer, samples, labels, [list(r) for r in nns],
            id2title, id2text, keep_all=True)
        context_input = modify(context_input, candidate_input,
                               self.cross_params["max_seq_length"])
        data = TensorDataset(context_input, label_input)
        loader = DataLoader(data, sampler=SequentialSampler(data),
                            batch_size=self.cross_params["eval_batch_size"])
        res = evaluate(self.crossencoder, loader, "cpu", None,
                       self.bi_params["max_context_length"], zeshel=False, silent=True)
        return np.asarray(res["logits"]), id2title

    def link_all(self, samples: list[dict]) -> None:
        """Link every sample not already cached, in one pass."""
        todo, seen = [], set()
        for s in samples:
            key = _mention_key(s, self.top_k)
            if key not in self._cache and key not in seen:
                seen.add(key)
                todo.append(s)
        if not todo:
            return
        log.info("BLINK: linking %d mention(s)", len(todo))
        queries = self._encode_contexts(todo)
        nns, bi_scores = self._search(queries)
        logits, id2title = self._rerank(todo, nns)
        for s, ids, bis, cross in zip(todo, nns, bi_scores, logits):
            order = np.argsort(-cross, kind="stable")
            self._cache[_mention_key(s, self.top_k)] = [
                {"local_id": int(ids[j]), "title": id2title[int(ids[j])],
                 "biencoder_score": float(bis[j]), "crossencoder_score": float(cross[j])}
                for j in order]
        self.save_cache()

    def link(self, sample: dict) -> list[dict]:
        """The cross-encoder's ranking of the top-k entities for one mention,
        best first."""
        key = _mention_key(sample, self.top_k)
        if key not in self._cache:
            self.link_all([sample])
        return self._cache[key]


_linkers: dict[int, BlinkLinker] = {}


def get_linker(top_k: int) -> BlinkLinker:
    if top_k not in _linkers:
        _linkers[top_k] = BlinkLinker(top_k)
    return _linkers[top_k]
