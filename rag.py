"""Self-contained topology-aware deployment-trace memory.

The memory is intentionally implemented with SQLite and plain Python.  A
region is a logical shard, indexed independently from the other regions, and
each stored row represents the graph bundle associated with one deployment
trace: service DAG, infrastructure topology, placement/routing graph, rollout
metadata, and QoS outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

EMBEDDING_DIM = 128
LSH_BITS = 12
TRACE_FIELDS = {
    "trace_id",
    "region_id",
    "controller_id",
    "created_at",
    "priority",
    "success",
    "service_dag",
    "topology",
    "query_context",
    "placement_routing",
    "rollout_trace",
    "qos_summary",
    "embedding",
    "metadata",
}
QUERY_FIELDS = {
    "window_id",
    "controller_id",
    "region_id",
    "sim_time_s",
    "service_dags",
    "topology",
    "telemetry",
    "qos_intent",
    "slice_tag",
    "policy_id",
    "current_placement",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_exact_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - value.keys()
    unexpected = value.keys() - required
    if missing or unexpected:
        raise ValueError(
            f"{label} keys are invalid; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def validate_query(query: dict[str, Any]) -> None:
    if not isinstance(query, dict):
        raise TypeError("planner query must be an object")
    _require_exact_keys(query, QUERY_FIELDS, "planner query")
    for key in ("window_id", "controller_id", "region_id", "slice_tag", "policy_id"):
        if not isinstance(query[key], (str, int)) or isinstance(query[key], bool):
            raise TypeError(f"planner query {key} must be a string or integer")
    _finite_number(query["sim_time_s"], "planner query sim_time_s")
    if not isinstance(query["service_dags"], list):
        raise TypeError("planner query service_dags must be a list")
    for key in ("topology", "telemetry", "qos_intent"):
        if not isinstance(query[key], dict):
            raise TypeError(f"planner query {key} must be an object")
    if not isinstance(query["current_placement"], (dict, list)):
        raise TypeError("planner query current_placement must be an object or list")


def validate_trace(trace: dict[str, Any]) -> None:
    if not isinstance(trace, dict):
        raise TypeError("deployment trace must be an object")
    _require_exact_keys(trace, TRACE_FIELDS, "deployment trace")
    for key in ("trace_id", "region_id", "controller_id"):
        if not isinstance(trace[key], (str, int)) or isinstance(trace[key], bool):
            raise TypeError(f"deployment trace {key} must be a string or integer")
    _finite_number(trace["created_at"], "deployment trace created_at")
    priority = _finite_number(trace["priority"], "deployment trace priority")
    if not 0.0 <= priority <= 1.0:
        raise ValueError("deployment trace priority must be in [0, 1]")
    if not isinstance(trace["success"], bool):
        raise TypeError("deployment trace success must be boolean")
    if not isinstance(trace["service_dag"], (dict, list)):
        raise TypeError("deployment trace service_dag must be an object or list")
    for key in ("topology", "query_context", "placement_routing", "rollout_trace", "qos_summary", "metadata"):
        if not isinstance(trace[key], dict):
            raise TypeError(f"deployment trace {key} must be an object")
    embedding = trace["embedding"]
    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
        raise ValueError(f"deployment trace embedding must contain {EMBEDDING_DIM} values")
    for index, component in enumerate(embedding):
        _finite_number(component, f"deployment trace embedding[{index}]")


def initialize_memory(memory_path: str) -> None:
    path = Path(memory_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS deployment_traces (
                trace_id TEXT PRIMARY KEY,
                region_id TEXT NOT NULL,
                controller_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                priority REAL NOT NULL CHECK(priority >= 0.0 AND priority <= 1.0),
                success INTEGER NOT NULL CHECK(success IN (0, 1)),
                service_dag_json TEXT NOT NULL,
                topology_json TEXT NOT NULL,
                query_context_json TEXT NOT NULL,
                placement_routing_json TEXT NOT NULL,
                rollout_trace_json TEXT NOT NULL,
                qos_summary_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_bucket INTEGER NOT NULL,
                topology_signature_json TEXT NOT NULL,
                placement_signature_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(deployment_traces)")
        }
        if "embedding_bucket" not in columns:
            connection.execute(
                "ALTER TABLE deployment_traces ADD COLUMN embedding_bucket INTEGER"
            )
            rows = connection.execute(
                "SELECT trace_id, embedding_json FROM deployment_traces"
            ).fetchall()
            connection.executemany(
                "UPDATE deployment_traces SET embedding_bucket = ? WHERE trace_id = ?",
                [
                    (_embedding_bucket(json.loads(embedding_json)), trace_id)
                    for trace_id, embedding_json in rows
                ],
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trace_region_priority
            ON deployment_traces(region_id, priority DESC, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trace_controller_time
            ON deployment_traces(controller_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trace_expiry
            ON deployment_traces(expires_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trace_region_embedding
            ON deployment_traces(region_id, embedding_bucket, expires_at)
            """
        )


def _flatten_tokens(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        tokens: list[str] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            tokens.extend(_flatten_tokens(value[key], child_prefix))
        return tokens
    if isinstance(value, list):
        tokens = []
        for item in value:
            tokens.extend(_flatten_tokens(item, prefix))
        return tokens
    if isinstance(value, float):
        value = round(value, 3)
    return [f"{prefix}={value}"]


def hashed_embedding(value: Any, dimensions: int = EMBEDDING_DIM) -> list[float]:
    """Create a deterministic local embedding without an external model."""

    if dimensions <= 0:
        raise ValueError("embedding dimensions must be positive")
    vector = [0.0] * dimensions
    for token in _flatten_tokens(value):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


def build_query_embedding(query: dict[str, Any]) -> list[float]:
    validate_query(query)
    semantic_view = {
        "service_dags": query["service_dags"],
        "topology": query["topology"],
        "telemetry": query["telemetry"],
        "qos_intent": query["qos_intent"],
        "slice_tag": query["slice_tag"],
        "policy_id": query["policy_id"],
        "current_placement": query["current_placement"],
    }
    return hashed_embedding(semantic_view)


def _embedding_bucket(embedding: list[float]) -> int:
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(f"embedding bucket requires {EMBEDDING_DIM} values")
    bucket = 0
    for bit in range(LSH_BITS):
        projection = 0.0
        for index, component in enumerate(embedding):
            digest = hashlib.blake2b(
                f"{bit}:{index}".encode("ascii"), digest_size=1
            ).digest()[0]
            projection += float(component) * (1.0 if digest & 1 else -1.0)
        if projection >= 0.0:
            bucket |= 1 << bit
    return bucket


def _neighbor_buckets(bucket: int) -> list[int]:
    neighbors = [bucket]
    neighbors.extend(bucket ^ (1 << first) for first in range(LSH_BITS))
    neighbors.extend(
        bucket ^ (1 << first) ^ (1 << second)
        for first in range(LSH_BITS)
        for second in range(first + 1, LSH_BITS)
    )
    return neighbors


def _node_identifier(node: Any) -> str:
    if isinstance(node, (str, int)) and not isinstance(node, bool):
        return str(node)
    if not isinstance(node, dict):
        raise TypeError("graph nodes must be identifiers or objects")
    for key in ("id", "node_id", "microservice_id", "name"):
        if key in node:
            return str(node[key])
    raise ValueError("graph node object requires id, node_id, microservice_id, or name")


def _graph_nodes(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "nodes" in graph:
        raw_nodes = graph["nodes"]
    elif "microservices" in graph:
        raw_nodes = graph["microservices"]
    elif "node_ids" in graph:
        raw_nodes = []
        node_ids = graph["node_ids"]
        if not isinstance(node_ids, list):
            raise TypeError("graph node_ids must be a list")
        for index, node_id in enumerate(node_ids):
            attributes = {"id": node_id}
            for key, values in graph.items():
                if key in {"node_ids", "links", "edges"}:
                    continue
                if isinstance(values, list) and len(values) == len(node_ids):
                    attributes[key] = values[index]
            raw_nodes.append(attributes)
    else:
        raw_nodes = []
    if not isinstance(raw_nodes, list):
        raise TypeError("graph nodes must be a list")
    nodes = []
    for raw_node in raw_nodes:
        identifier = _node_identifier(raw_node)
        attributes = raw_node if isinstance(raw_node, dict) else {"id": raw_node}
        nodes.append((identifier, attributes))
    return nodes


def _edge_endpoints(edge: Any) -> tuple[str, str]:
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return str(edge[0]), str(edge[1])
    if not isinstance(edge, dict):
        raise TypeError("graph edges must be endpoint pairs or objects")
    source_key = next((key for key in ("source", "src", "from") if key in edge), None)
    target_key = next((key for key in ("target", "destination", "dst", "to") if key in edge), None)
    if source_key is None or target_key is None:
        raise ValueError("graph edge object requires source/target endpoints")
    return str(edge[source_key]), str(edge[target_key])


def _graph_edges(graph: dict[str, Any]) -> list[tuple[str, str]]:
    if "edges" in graph:
        raw_edges = graph["edges"]
    elif "links" in graph:
        raw_edges = graph["links"]
    else:
        raw_edges = []
    if not isinstance(raw_edges, list):
        raise TypeError("graph edges must be a list")
    return [_edge_endpoints(edge) for edge in raw_edges]


def _graphs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        if not all(isinstance(graph, dict) for graph in value):
            raise TypeError("service DAG collection must contain objects")
        return value
    raise TypeError("graph value must be an object or list of objects")


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _single_graph_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_nodes = {identifier for identifier, _ in _graph_nodes(left)}
    right_nodes = {identifier for identifier, _ in _graph_nodes(right)}
    left_edges = set(_graph_edges(left))
    right_edges = set(_graph_edges(right))
    node_score = _jaccard(left_nodes, right_nodes)
    edge_score = _jaccard(left_edges, right_edges)
    left_size = len(left_nodes) + len(left_edges)
    right_size = len(right_nodes) + len(right_edges)
    size_score = min(left_size, right_size) / max(left_size, right_size) if max(left_size, right_size) else 0.0
    return 0.25 * node_score + 0.55 * edge_score + 0.20 * size_score


def _dag_collection_similarity(left: Any, right: Any) -> float:
    left_graphs = _graphs(left)
    right_graphs = _graphs(right)
    if not left_graphs or not right_graphs:
        return 0.0
    return sum(max(_single_graph_similarity(graph, candidate) for candidate in right_graphs) for graph in left_graphs) / len(
        left_graphs
    )


def topology_similarity(query: dict[str, Any], trace: dict[str, Any]) -> float:
    infrastructure_score = _single_graph_similarity(query["topology"], trace["topology"])
    service_score = _dag_collection_similarity(query["service_dags"], trace["service_dag"])
    return 0.65 * infrastructure_score + 0.35 * service_score


def _placement_pairs(value: Any) -> set[tuple[str, str]]:
    if isinstance(value, dict):
        for key in ("assignments", "placements", "placement"):
            if key in value:
                return _placement_pairs(value[key])
        pairs: set[tuple[str, str]] = set()
        for key, target in value.items():
            if isinstance(target, (str, int)) and not isinstance(target, bool):
                pairs.add((str(key), str(target)))
            elif isinstance(target, dict):
                for node_id, probability in target.items():
                    if isinstance(probability, (int, float)) and not isinstance(probability, bool) and probability > 0.0:
                        pairs.add((str(key), str(node_id)))
        return pairs
    if isinstance(value, list):
        pairs = set()
        for placement in value:
            if placement == "<eop>":
                continue
            if isinstance(placement, (list, tuple)) and len(placement) == 2:
                pairs.add((str(placement[0]), str(placement[1])))
                continue
            if isinstance(placement, dict):
                microservice_key = next((key for key in ("placement_id", "microservice_id", "microservice", "service") if key in placement), None)
                node_key = next((key for key in ("node_id", "node", "target") if key in placement), None)
                if microservice_key is not None and node_key is not None:
                    if microservice_key == "placement_id":
                        placement_id = str(placement[microservice_key])
                    elif "request_id" in placement and "replica_id" in placement:
                        placement_id = (
                            f"{placement['request_id']}/{placement[microservice_key]}/{placement['replica_id']}"
                        )
                    else:
                        placement_id = str(placement[microservice_key])
                    pairs.add((placement_id, str(placement[node_key])))
        return pairs
    raise TypeError("placement must be an object or list")


def placement_overlap(query: dict[str, Any], trace: dict[str, Any]) -> float:
    current = _placement_pairs(query["current_placement"])
    historical = _placement_pairs(trace["placement_routing"])
    return _jaccard(current, historical)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(0.0, min(1.0, cosine))


def _node_embeddings(graph: dict[str, Any], dimensions: int = 24) -> tuple[dict[str, list[float]], dict[str, set[str]]]:
    nodes = _graph_nodes(graph)
    edges = _graph_edges(graph)
    adjacency = {identifier: set() for identifier, _ in nodes}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
    embeddings = {}
    for identifier, attributes in nodes:
        node_view = {"attributes": attributes, "degree": len(adjacency[identifier])}
        embeddings[identifier] = hashed_embedding(node_view, dimensions)
    return embeddings, adjacency


def _attention_layer(embeddings: dict[str, list[float]], adjacency: dict[str, set[str]]) -> dict[str, list[float]]:
    updated = {}
    for node_id, own in embeddings.items():
        neighbors = [neighbor for neighbor in adjacency[node_id] if neighbor in embeddings]
        if not neighbors:
            updated[node_id] = own
            continue
        logits = [math.exp(cosine_similarity(own, embeddings[neighbor])) for neighbor in neighbors]
        normalizer = sum(logits)
        message = [0.0] * len(own)
        for neighbor, logit in zip(neighbors, logits):
            attention = logit / normalizer
            for index, component in enumerate(embeddings[neighbor]):
                message[index] += attention * component
        combined = [0.65 * own[index] + 0.35 * message[index] for index in range(len(own))]
        norm = math.sqrt(sum(component * component for component in combined))
        updated[node_id] = [component / norm for component in combined] if norm else combined
    return updated


def _graph_attention_match(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_embeddings, left_adjacency = _node_embeddings(left)
    right_embeddings, right_adjacency = _node_embeddings(right)
    if not left_embeddings or not right_embeddings:
        return 0.0
    for _ in range(2):
        left_embeddings = _attention_layer(left_embeddings, left_adjacency)
        right_embeddings = _attention_layer(right_embeddings, right_adjacency)
    matches = [
        max(cosine_similarity(embedding, candidate) for candidate in right_embeddings.values())
        for embedding in left_embeddings.values()
    ]
    return sum(matches) / len(matches)


def graph_attention_relevance(query: dict[str, Any], trace: dict[str, Any]) -> float:
    infrastructure = _graph_attention_match(query["topology"], trace["topology"])
    query_dags = _graphs(query["service_dags"])
    trace_dags = _graphs(trace["service_dag"])
    if query_dags and trace_dags:
        service = sum(max(_graph_attention_match(dag, candidate) for candidate in trace_dags) for dag in query_dags) / len(
            query_dags
        )
    else:
        service = 0.0
    return 0.65 * infrastructure + 0.35 * service


def _trace_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "trace_id": row["trace_id"],
        "region_id": row["region_id"],
        "controller_id": row["controller_id"],
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
        "priority": row["priority"],
        "success": bool(row["success"]),
        "service_dag": json.loads(row["service_dag_json"]),
        "topology": json.loads(row["topology_json"]),
        "query_context": json.loads(row["query_context_json"]),
        "placement_routing": json.loads(row["placement_routing_json"]),
        "rollout_trace": json.loads(row["rollout_trace_json"]),
        "qos_summary": json.loads(row["qos_summary_json"]),
        "embedding": json.loads(row["embedding_json"]),
        "metadata": json.loads(row["metadata_json"]),
    }


def prune_memory(memory_path: str, now: float, max_traces: int) -> dict[str, int]:
    if max_traces <= 0:
        raise ValueError("max_traces must be positive")
    initialize_memory(memory_path)
    with sqlite3.connect(memory_path) as connection:
        expired = connection.execute("DELETE FROM deployment_traces WHERE expires_at <= ?", (now,)).rowcount
        count = connection.execute("SELECT COUNT(*) FROM deployment_traces").fetchone()[0]
        overflow = max(0, count - max_traces)
        if overflow:
            connection.execute(
                """
                DELETE FROM deployment_traces
                WHERE trace_id IN (
                    SELECT trace_id FROM deployment_traces
                    ORDER BY priority ASC, success ASC, created_at ASC
                    LIMIT ?
                )
                """,
                (overflow,),
            )
    return {"expired": expired, "capacity_pruned": overflow}


def record_trace(memory_path: str, trace: dict[str, Any], ttl_seconds: float, max_traces: int) -> str:
    validate_trace(trace)
    ttl = _finite_number(ttl_seconds, "trace TTL")
    if ttl <= 0.0:
        raise ValueError("trace TTL must be positive")
    initialize_memory(memory_path)
    topology_signature = {
        "nodes": sorted(identifier for identifier, _ in _graph_nodes(trace["topology"])),
        "edges": sorted(_graph_edges(trace["topology"])),
    }
    placement_signature = sorted(_placement_pairs(trace["placement_routing"]))
    with sqlite3.connect(memory_path) as connection:
        connection.execute(
            """
            INSERT INTO deployment_traces (
                trace_id, region_id, controller_id, created_at, expires_at,
                priority, success, service_dag_json, topology_json,
                query_context_json, placement_routing_json, rollout_trace_json,
                qos_summary_json, embedding_json, embedding_bucket,
                topology_signature_json, placement_signature_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(trace["trace_id"]),
                str(trace["region_id"]),
                str(trace["controller_id"]),
                float(trace["created_at"]),
                float(trace["created_at"]) + ttl,
                float(trace["priority"]),
                int(trace["success"]),
                _canonical_json(trace["service_dag"]),
                _canonical_json(trace["topology"]),
                _canonical_json(trace["query_context"]),
                _canonical_json(trace["placement_routing"]),
                _canonical_json(trace["rollout_trace"]),
                _canonical_json(trace["qos_summary"]),
                _canonical_json(trace["embedding"]),
                _embedding_bucket(trace["embedding"]),
                _canonical_json(topology_signature),
                _canonical_json(placement_signature),
                _canonical_json(trace["metadata"]),
            ),
        )
    prune_memory(memory_path, time.time(), max_traces)
    return str(trace["trace_id"])


def _validate_similarity_weights(weights: dict[str, float]) -> None:
    required = {"latent", "topology", "placement"}
    if not isinstance(weights, dict):
        raise TypeError("similarity_weights must be an object")
    _require_exact_keys(weights, required, "similarity_weights")
    values = [_finite_number(weights[key], f"similarity_weights.{key}") for key in sorted(required)]
    if any(value < 0.0 for value in values):
        raise ValueError("similarity weights must be non-negative")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("similarity weights must sum to 1")


def retrieve_traces(
    memory_path: str,
    query: dict[str, Any],
    retrieval_pool: int,
    top_k: int,
    similarity_weights: dict[str, float],
    soft_prune_threshold: float,
    now: float,
) -> list[dict[str, Any]]:
    """Retrieve indexed coarse matches, softly prune, and graph-rerank to top five.

    The sizes are passed explicitly so the strict configuration validator can
    enforce the paper-aligned values without hidden runtime defaults.
    """

    validate_query(query)
    _validate_similarity_weights(similarity_weights)
    if retrieval_pool <= 0 or top_k <= 0 or top_k > retrieval_pool:
        raise ValueError("retrieval sizes must satisfy 0 < top_k <= retrieval_pool")
    threshold = _finite_number(soft_prune_threshold, "soft_prune_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("soft_prune_threshold must be in [0, 1]")
    initialize_memory(memory_path)
    query_embedding = build_query_embedding(query)
    buckets = _neighbor_buckets(_embedding_bucket(query_embedding))
    placeholders = ",".join("?" for _ in buckets)
    with sqlite3.connect(memory_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT * FROM deployment_traces
            WHERE region_id = ? AND expires_at > ?
              AND embedding_bucket IN ({placeholders})
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            (
                str(query["region_id"]),
                now,
                *buckets,
                retrieval_pool * 8,
            ),
        ).fetchall()
        if len(rows) < top_k:
            rows = connection.execute(
                """
                SELECT * FROM deployment_traces
                WHERE region_id = ? AND expires_at > ?
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
                """,
                (str(query["region_id"]), now, retrieval_pool),
            ).fetchall()

    coarse = []
    for row in rows:
        trace = _trace_from_row(row)
        latent = cosine_similarity(query_embedding, trace["embedding"])
        structural = topology_similarity(query, trace)
        overlap = placement_overlap(query, trace)
        composite = (
            similarity_weights["latent"] * latent
            + similarity_weights["topology"] * structural
            + similarity_weights["placement"] * overlap
        )
        trace["scores"] = {
            "latent": latent,
            "topology": structural,
            "placement": overlap,
            "composite": composite,
        }
        coarse.append(trace)
    coarse.sort(key=lambda trace: (trace["scores"]["composite"], trace["priority"]), reverse=True)
    coarse = coarse[:retrieval_pool]

    for trace in coarse:
        lifetime = trace["expires_at"] - trace["created_at"]
        freshness = max(0.0, min(1.0, (trace["expires_at"] - now) / lifetime)) if lifetime > 0.0 else 0.0
        quality = 0.7 * trace["priority"] + 0.3 * float(trace["success"])
        signal = 0.55 * trace["scores"]["composite"] + 0.25 * quality + 0.20 * freshness
        attention = 1.0 / (1.0 + math.exp(-8.0 * (signal - threshold)))
        graph_score = graph_attention_relevance(query, trace)
        rerank_score = 0.65 * trace["scores"]["composite"] * attention + 0.35 * graph_score
        trace["scores"].update(
            {
                "freshness": freshness,
                "soft_attention": attention,
                "graph_rerank": graph_score,
                "rerank": rerank_score,
            }
        )
    coarse.sort(key=lambda trace: (trace["scores"]["rerank"], trace["priority"]), reverse=True)
    return coarse[:top_k]


def memory_stats(memory_path: str, now: float) -> dict[str, Any]:
    initialize_memory(memory_path)
    with sqlite3.connect(memory_path) as connection:
        rows = connection.execute(
            """
            SELECT region_id, COUNT(*), SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END)
            FROM deployment_traces GROUP BY region_id ORDER BY region_id
            """,
            (now,),
        ).fetchall()
    shards = {region_id: {"stored": stored, "active": active} for region_id, stored, active in rows}
    return {
        "stored": sum(shard["stored"] for shard in shards.values()),
        "active": sum(shard["active"] for shard in shards.values()),
        "regional_shards": shards,
    }


def successful_traces(memory_path: str, region_id: str | int, limit: int, now: float) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("successful trace limit must be positive")
    initialize_memory(memory_path)
    with sqlite3.connect(memory_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM deployment_traces
            WHERE region_id = ? AND success = 1 AND expires_at > ?
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            (str(region_id), now, limit),
        ).fetchall()
    return [_trace_from_row(row) for row in rows]


def new_trace_id(region_id: str | int, controller_id: str | int) -> str:
    return f"{region_id}:{controller_id}:{uuid.uuid4().hex}"


def _benchmark_query(seed: int, variant: int, now: float) -> dict[str, Any]:
    """Build a compact, valid planner query for a measured TopoRAG run."""

    region_id = f"benchmark-region-{seed % 7}"
    edge_suffix = variant % 4
    topology = {
        "nodes": [
            {"id": f"edge-{edge_suffix}-0", "tier": "edge", "cpu": 32},
            {"id": f"edge-{edge_suffix}-1", "tier": "edge", "cpu": 24},
            {"id": f"cloud-{edge_suffix}", "tier": "cloud", "cpu": 96},
        ],
        "edges": [
            [f"edge-{edge_suffix}-0", f"edge-{edge_suffix}-1"],
            [f"edge-{edge_suffix}-0", f"cloud-{edge_suffix}"],
            [f"edge-{edge_suffix}-1", f"cloud-{edge_suffix}"],
        ],
    }
    service_dag = {
        "nodes": [
            {"id": "ingest", "replicas": 2},
            {"id": "render", "replicas": 4 + variant % 5},
            {"id": "encode", "replicas": 2},
        ],
        "edges": [["ingest", "render"], ["render", "encode"]],
    }
    current_placement = {
        "assignments": {
            "ingest": f"edge-{edge_suffix}-0",
            "render": f"edge-{edge_suffix}-1",
            "encode": f"cloud-{edge_suffix}",
        }
    }
    query = {
        "window_id": f"benchmark-{seed}-{variant}",
        "controller_id": f"controller-{variant % 8}",
        "region_id": region_id,
        "sim_time_s": now,
        "service_dags": [service_dag],
        "topology": topology,
        "telemetry": {
            "arrival_rate": 30 + variant,
            "mean_cpu_utilization": round(0.45 + 0.01 * (variant % 10), 3),
        },
        "qos_intent": {"latency_ms": 20 + variant % 4, "availability": 0.999},
        "slice_tag": "immersive-xr",
        "policy_id": "llm-mappo",
        "current_placement": current_placement,
    }
    validate_query(query)
    return query


def _benchmark_payload(seed: int, variant: int, now: float) -> dict[str, Any]:
    """Serialize one validated synthetic trace shape used by batch insertion."""

    query = _benchmark_query(seed, variant, now)
    service_dag = query["service_dags"][0]
    placement_routing = {
        "assignments": query["current_placement"]["assignments"],
        "routing": {"ingest->render": [0.6, 0.4], "render->encode": [1.0]},
    }
    embedding = [round(component, 6) if component else 0 for component in build_query_embedding(query)]
    trace = {
        "trace_id": f"benchmark-template-{seed}-{variant}",
        "region_id": query["region_id"],
        "controller_id": query["controller_id"],
        "created_at": now - 1.0,
        "priority": 0.5,
        "success": True,
        "service_dag": service_dag,
        "topology": query["topology"],
        "query_context": {
            "slice_tag": query["slice_tag"],
            "policy_id": query["policy_id"],
            "telemetry": query["telemetry"],
        },
        "placement_routing": placement_routing,
        "rollout_trace": {"normalized_reward": 0.7 + 0.01 * (variant % 10)},
        "qos_summary": {
            "latency_ms": 16 + variant % 8,
            "sla_met": variant % 5 != 0,
        },
        "embedding": embedding,
        "metadata": {"source": "synthetic-memory-scaling", "seed": seed, "variant": variant},
    }
    validate_trace(trace)
    topology_signature = {
        "nodes": sorted(identifier for identifier, _ in _graph_nodes(trace["topology"])),
        "edges": sorted(_graph_edges(trace["topology"])),
    }
    placement_signature = sorted(_placement_pairs(trace["placement_routing"]))
    return {
        "region_id": str(trace["region_id"]),
        "service_dag_json": _canonical_json(trace["service_dag"]),
        "topology_json": _canonical_json(trace["topology"]),
        "query_context_json": _canonical_json(trace["query_context"]),
        "placement_routing_json": _canonical_json(trace["placement_routing"]),
        "rollout_trace_json": _canonical_json(trace["rollout_trace"]),
        "qos_summary_json": _canonical_json(trace["qos_summary"]),
        "embedding_json": _canonical_json(trace["embedding"]),
        "embedding_bucket": _embedding_bucket(trace["embedding"]),
        "topology_signature_json": _canonical_json(topology_signature),
        "placement_signature_json": _canonical_json(placement_signature),
        "metadata_json": _canonical_json(trace["metadata"]),
    }


def _insert_synthetic_traces(
    memory_path: str,
    trace_count: int,
    seed: int,
    now: float,
    ttl_seconds: float,
) -> None:
    """Populate an isolated memory in bounded, transactional batches."""

    if trace_count <= 0:
        raise ValueError("trace_count must be positive")
    ttl = _finite_number(ttl_seconds, "benchmark trace TTL")
    if ttl <= 100.0:
        raise ValueError("benchmark trace TTL must exceed the generated trace age")
    initialize_memory(memory_path)
    payloads = [_benchmark_payload(seed, variant, now) for variant in range(16)]
    statement = """
        INSERT INTO deployment_traces (
            trace_id, region_id, controller_id, created_at, expires_at,
            priority, success, service_dag_json, topology_json,
            query_context_json, placement_routing_json, rollout_trace_json,
            qos_summary_json, embedding_json, embedding_bucket,
            topology_signature_json, placement_signature_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch_size = 5000
    with sqlite3.connect(memory_path) as connection:
        for start in range(0, trace_count, batch_size):
            stop = min(start + batch_size, trace_count)
            rows = []
            for index in range(start, stop):
                payload = payloads[index % len(payloads)]
                created_at = now - ((index * 37 + seed) % 1000) / 10.0
                priority = ((index * 7919 + seed) % 1000) / 1000.0
                rows.append(
                    (
                        f"benchmark:{seed}:{trace_count}:{index}",
                        payload["region_id"],
                        f"controller-{index % 8}",
                        created_at,
                        created_at + ttl,
                        priority,
                        int((index + seed) % 5 != 0),
                        payload["service_dag_json"],
                        payload["topology_json"],
                        payload["query_context_json"],
                        payload["placement_routing_json"],
                        payload["rollout_trace_json"],
                        payload["qos_summary_json"],
                        payload["embedding_json"],
                        payload["embedding_bucket"],
                        payload["topology_signature_json"],
                        payload["placement_signature_json"],
                        payload["metadata_json"],
                    )
                )
            connection.executemany(statement, rows)
            connection.commit()
        stored = connection.execute("SELECT COUNT(*) FROM deployment_traces").fetchone()[0]
        if stored != trace_count:
            raise RuntimeError(f"inserted {stored} traces, expected {trace_count}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()


def _sqlite_storage_mib(memory_path: str) -> float:
    """Measure allocated SQLite database bytes after checkpointing its WAL."""

    with sqlite3.connect(memory_path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    allocated_bytes = page_count * page_size
    file_bytes = Path(memory_path).stat().st_size
    if allocated_bytes != file_bytes:
        raise RuntimeError(
            f"SQLite allocation mismatch: page allocation={allocated_bytes}, file size={file_bytes}"
        )
    return allocated_bytes / (1024.0 * 1024.0)


def _percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        raise ValueError("percentile samples cannot be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure_memory_scaling(
    config: dict[str, Any],
    seed: int,
    trace_counts: tuple[int, ...],
    repetitions: int,
) -> list[dict[str, Any]]:
    """Measure real retrieval latency and SQLite size at explicit trace counts."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("benchmark seed must be an integer")
    if not trace_counts or any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in trace_counts):
        raise ValueError("trace_counts must contain positive integers")
    if len(set(trace_counts)) != len(trace_counts):
        raise ValueError("trace_counts must be unique")
    if repetitions < 3:
        raise ValueError("repetitions must be at least three")

    planner = config["planner"]
    retrieval_pool = int(planner["retrieval_pool"])
    top_k = int(planner["retrieval_top_k"])
    if min(trace_counts) < top_k:
        raise ValueError("every trace count must be at least retrieval_top_k")
    ttl_seconds = float(planner["planning_interval_s"]) * int(planner["memory_ttl_windows"])
    now = 1_800_000_000.0 + seed
    query = _benchmark_query(seed, seed % 16, now)
    measurements = []

    for trace_count in trace_counts:
        with tempfile.TemporaryDirectory(prefix=f"toporag-{seed}-{trace_count}-") as directory:
            memory_path = str(Path(directory) / "memory.sqlite")
            _insert_synthetic_traces(memory_path, trace_count, seed, now, ttl_seconds)
            storage_mib = _sqlite_storage_mib(memory_path)

            retrieval_arguments = (
                memory_path,
                query,
                retrieval_pool,
                top_k,
                planner["similarity_weights"],
                float(planner["soft_prune_threshold"]),
                now,
            )
            for _ in range(2):
                warm_result = retrieve_traces(*retrieval_arguments)
                if len(warm_result) != top_k:
                    raise RuntimeError(f"warm retrieval returned {len(warm_result)} traces, expected {top_k}")

            latencies_ms = []
            for _ in range(repetitions):
                started = time.perf_counter()
                retrieved = retrieve_traces(*retrieval_arguments)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if len(retrieved) != top_k:
                    raise RuntimeError(f"retrieval returned {len(retrieved)} traces, expected {top_k}")
                latencies_ms.append(elapsed_ms)

            measurements.append(
                {
                    "trace_count": trace_count,
                    "retrieval_latency_p25_ms": _percentile(latencies_ms, 0.25),
                    "retrieval_latency_p50_ms": _percentile(latencies_ms, 0.50),
                    "retrieval_latency_p75_ms": _percentile(latencies_ms, 0.75),
                    "memory_usage_mib": storage_mib,
                    "latencies_ms": latencies_ms,
                }
            )
    return measurements


def smoke_memory_scaling(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Run the same benchmark method at two small sizes for fast verification."""

    return measure_memory_scaling(config, seed, (128, 512), 3)


def benchmark_memory_scaling(
    config: dict[str, Any],
    seed: int,
    run_id: str,
    configuration_hash: str,
    result_row: Any,
) -> list[dict[str, Any]]:
    """Return measured Figure 11b rows at the paper's memory-size points."""

    measurements = measure_memory_scaling(
        config,
        seed,
        (10_000, 50_000, 100_000, 200_000, 500_000),
        9,
    )
    rows = []
    for measurement in measurements:
        trace_count = int(measurement["trace_count"])
        common = {
            "run_id": run_id,
            "configuration_hash": configuration_hash,
            "seed": seed,
            "figure": 11,
            "panel": "b",
            "method": "toporag",
            "scenario": "toporag_scaling",
            "workload": "trace_memory",
            "x": trace_count / 1000.0,
            "x_label": f"{trace_count // 1000}k",
        }
        for metric, key, unit in (
            ("retrieval_latency_p25", "retrieval_latency_p25_ms", "ms"),
            ("retrieval_latency_p50", "retrieval_latency_p50_ms", "ms"),
            ("retrieval_latency_p75", "retrieval_latency_p75_ms", "ms"),
            ("memory_usage", "memory_usage_mib", "MiB"),
        ):
            rows.append(
                result_row(
                    **common,
                    metric=metric,
                    value=float(measurement[key]),
                    unit=unit,
                )
            )
    return rows


# Small compatibility alias for old callers that only initialized retrieval.
def initialize_rag(memory_path: str) -> None:
    initialize_memory(memory_path)
