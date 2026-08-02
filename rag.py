"""SQLite deployment memory used by the conference RAG planner."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_MEMORY_INDEX: dict[str, dict[str, Any]] = {}


def _index_key(path: str) -> str:
    return str(Path(path).resolve())


def invalidate_memory(path: str) -> None:
    _MEMORY_INDEX.pop(_index_key(path), None)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _finite_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{label} must contain finite numbers")
    return vector


def _require_trace(trace: dict[str, Any]) -> None:
    required = {
        "timestamp",
        "embedding",
        "telemetry",
        "intent",
        "dag",
        "placement",
        "outcome",
    }
    missing = required.difference(trace)
    if missing:
        raise ValueError(f"trace is missing {sorted(missing)}")
    timestamp = float(trace["timestamp"])
    if not math.isfinite(timestamp):
        raise ValueError("trace timestamp must be finite")
    _finite_vector(trace["embedding"], "trace embedding")
    outcome = trace["outcome"]
    for key in ("latency_ms", "resource_use", "sla_violations", "utility", "reward"):
        if key not in outcome or not math.isfinite(float(outcome[key])):
            raise ValueError(f"trace outcome requires finite {key}")


def initialize_memory(path: str) -> None:
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(memory_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                embedding TEXT NOT NULL,
                telemetry TEXT NOT NULL,
                intent TEXT NOT NULL,
                dag TEXT NOT NULL,
                placement TEXT NOT NULL,
                outcome TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS traces_time ON traces(timestamp DESC)"
        )


def record_trace(path: str, trace: dict[str, Any], max_traces: int) -> str:
    _require_trace(trace)
    if max_traces < 1:
        raise ValueError("max_traces must be positive")
    initialize_memory(path)
    trace_id = str(trace.get("id") or uuid.uuid4())
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO traces
            (id, timestamp, embedding, telemetry, intent, dag, placement, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                float(trace["timestamp"]),
                _json(trace["embedding"]),
                _json(trace["telemetry"]),
                _json(trace["intent"]),
                _json(trace["dag"]),
                _json(trace["placement"]),
                _json(trace["outcome"]),
            ),
        )
        excess = connection.execute(
            "SELECT MAX(COUNT(*) - ?, 0) FROM traces", (max_traces,)
        ).fetchone()[0]
        if excess:
            connection.execute(
                "DELETE FROM traces WHERE id IN (SELECT id FROM traces ORDER BY timestamp ASC LIMIT ?)",
                (int(excess),),
            )
    index = _MEMORY_INDEX.get(_index_key(path))
    if index is not None:
        _index_trace(index, {**trace, "id": trace_id}, max_traces)
    return trace_id


def _tokens(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, item in value.items():
            result.update(_tokens(item, f"{prefix}/{key}"))
        return result
    if isinstance(value, list):
        result = set()
        for index, item in enumerate(value):
            result.update(_tokens(item, f"{prefix}/{index}"))
        return result
    return {f"{prefix}={value}"}


def _token_jaccard(left_tokens: set[str], right_tokens: set[str]) -> float:
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _row_trace(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "embedding": json.loads(row["embedding"]),
        "telemetry": json.loads(row["telemetry"]),
        "intent": json.loads(row["intent"]),
        "dag": json.loads(row["dag"]),
        "placement": json.loads(row["placement"]),
        "outcome": json.loads(row["outcome"]),
    }


def _empty_index(dimension: int, capacity: int) -> dict[str, Any]:
    return {
        "embeddings": np.empty((capacity, dimension), dtype=np.float32),
        "norms": np.empty(capacity, dtype=np.float32),
        "timestamps": np.empty(capacity, dtype=np.float64),
        "traces": [],
        "dag_tokens": [],
        "placement_tokens": [],
        "structure_cache": {},
        "id_to_index": {},
        "size": 0,
    }


def _grow_index(index: dict[str, Any], maximum: int) -> None:
    current = len(index["embeddings"])
    target = min(maximum, max(16, current * 2, int(index["size"]) + 1))
    if target <= current:
        return
    dimension = index["embeddings"].shape[1]
    embeddings = np.empty((target, dimension), dtype=np.float32)
    norms = np.empty(target, dtype=np.float32)
    timestamps = np.empty(target, dtype=np.float64)
    size = int(index["size"])
    embeddings[:size] = index["embeddings"][:size]
    norms[:size] = index["norms"][:size]
    timestamps[:size] = index["timestamps"][:size]
    index["embeddings"] = embeddings
    index["norms"] = norms
    index["timestamps"] = timestamps


def _index_trace(index: dict[str, Any], trace: dict[str, Any], max_traces: int) -> None:
    trace_id = str(trace["id"])
    vector = np.asarray(trace["embedding"], dtype=np.float32)
    if vector.shape != (index["embeddings"].shape[1],):
        raise ValueError("trace embedding dimension changed")
    if trace_id in index["id_to_index"]:
        position = int(index["id_to_index"][trace_id])
    elif int(index["size"]) < max_traces:
        if int(index["size"]) == len(index["embeddings"]):
            _grow_index(index, max_traces)
        position = int(index["size"])
        index["size"] = position + 1
        index["traces"].append({})
        index["dag_tokens"].append(set())
        index["placement_tokens"].append(set())
    else:
        position = int(np.argmin(index["timestamps"][: int(index["size"])]))
        index["id_to_index"].pop(index["traces"][position]["id"], None)
    stored = {key: value for key, value in trace.items() if key != "embedding"}
    index["embeddings"][position] = vector
    index["norms"][position] = float(np.linalg.norm(vector))
    index["timestamps"][position] = float(trace["timestamp"])
    index["traces"][position] = stored
    index["dag_tokens"][position] = _tokens(trace["dag"])
    index["placement_tokens"][position] = _tokens(trace["placement"])
    index["id_to_index"][trace_id] = position
    index["structure_cache"].clear()


def _load_index(path: str, dimension: int) -> dict[str, Any]:
    key = _index_key(path)
    if key in _MEMORY_INDEX:
        index = _MEMORY_INDEX[key]
        if index["embeddings"].shape[1] != dimension:
            raise ValueError("query and trace embeddings have different dimensions")
        return index
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        traces = [_row_trace(row) for row in connection.execute("SELECT * FROM traces")]
    index = _empty_index(dimension, max(1, len(traces)))
    for trace in traces:
        _index_trace(index, trace, max(1, len(traces)))
    _MEMORY_INDEX[key] = index
    return index


def retrieve_traces(
    path: str,
    query: dict[str, Any],
    top_k: int,
    decay_seconds: float,
    now: float,
) -> list[dict[str, Any]]:
    required = {"embedding", "dag", "placement"}
    missing = required.difference(query)
    if missing:
        raise ValueError(f"query is missing {sorted(missing)}")
    query_embedding = _finite_vector(query["embedding"], "query embedding")
    if top_k < 1 or decay_seconds <= 0.0:
        raise ValueError("top_k and decay_seconds must be positive")
    initialize_memory(path)
    index = _load_index(path, len(query_embedding))
    size = int(index["size"])
    if not size:
        return []
    vector = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(vector))
    cosine = np.zeros(size, dtype=np.float32)
    denominators = index["norms"][:size] * query_norm
    np.divide(
        index["embeddings"][:size] @ vector,
        denominators,
        out=cosine,
        where=denominators > 0.0,
    )
    semantic = (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0
    dag_tokens = _tokens(query["dag"])
    placement_tokens = _tokens(query["placement"])
    structure_key = (frozenset(dag_tokens), frozenset(placement_tokens))
    structure = index["structure_cache"].get(structure_key)
    if structure is None:
        structure = np.fromiter(
            (
                0.5 * _token_jaccard(dag_tokens, index["dag_tokens"][position])
                + 0.5
                * _token_jaccard(placement_tokens, index["placement_tokens"][position])
                for position in range(size)
            ),
            dtype=np.float32,
            count=size,
        )
        if len(index["structure_cache"]) == 16:
            index["structure_cache"].pop(next(iter(index["structure_cache"])))
        index["structure_cache"][structure_key] = structure
    recency = np.exp(-np.maximum(0.0, now - index["timestamps"][:size]) / decay_seconds)
    scores = (semantic + structure + recency) / 3.0
    count = min(top_k, size)
    if count == size:
        selected = np.arange(size)
    else:
        selected = np.argpartition(scores, size - count)[size - count :]
    ordered = sorted(
        selected.tolist(),
        key=lambda position: (scores[position], index["timestamps"][position]),
        reverse=True,
    )
    result = []
    for position in ordered:
        trace = {
            **index["traces"][position],
            "embedding": index["embeddings"][position].tolist(),
            "retrieval": {
                "cosine": float(semantic[position]),
                "jaccard": float(structure[position]),
                "recency": float(recency[position]),
                "score": float(scores[position]),
            },
        }
        result.append(trace)
    return result


def memory_size(path: str) -> int:
    initialize_memory(path)
    with sqlite3.connect(path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0])
