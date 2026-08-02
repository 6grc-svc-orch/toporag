"""Measured evaluation suite for the paper's Figures 4–11.

This file intentionally contains the experiment record and aggregation logic
beside the evaluation driver.  Running it writes only measured records using
the fixed schema consumed by ``figures/generate_figures.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from baselines import METHOD_IDS, method_id

FIGURE_ONLY_METHODS = {
    "full",
    "no_toporag",
    "no_reward_shaping",
    "no_llm_guidance",
    "toporag",
    "workers_1",
    "workers_2",
    "workers_3",
    "workers_4",
    "workers_5",
    "lora_rank",
    "frozen_layer_ratio",
    "planning_frequency",
    "learning_rate",
    "clip_epsilon",
    "entropy_coefficient",
    "trust_threshold",
    "interpolation_decay",
}

FIGURE4_METHODS = ("LLM-MAPPO", "RSPPO", "GNN-DRL", "mu-DDRL", "CROMADQN")


RESULT_COLUMNS = (
    "run_id",
    "config_hash",
    "seed",
    "figure",
    "panel",
    "method",
    "scenario",
    "workload",
    "x",
    "x_label",
    "metric",
    "value",
    "unit",
)


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def make_run_id(configuration_hash: str) -> str:
    return f"paper-{configuration_hash}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"


def result_row(
    *,
    run_id: str,
    configuration_hash: str,
    seed: int,
    figure: int,
    panel: str,
    method: str,
    scenario: str,
    workload: str,
    x: float,
    x_label: str,
    metric: str,
    value: float,
    unit: str,
) -> dict[str, Any]:
    if method not in METHOD_IDS.values() and method not in FIGURE_ONLY_METHODS:
        raise ValueError(f"Unsupported result method identifier {method!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"Result {metric!r} must be finite, received {value!r}")
    return {
        "run_id": run_id,
        "config_hash": configuration_hash,
        "seed": int(seed),
        "figure": int(figure),
        "panel": str(panel),
        "method": method,
        "scenario": str(scenario),
        "workload": str(workload),
        "x": float(x),
        "x_label": str(x_label),
        "metric": str(metric),
        "value": float(value),
        "unit": str(unit),
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty measured result set")
    for row in rows:
        missing = set(RESULT_COLUMNS).difference(row)
        extra = set(row).difference(RESULT_COLUMNS)
        if missing or extra:
            raise ValueError(f"Invalid result row; missing={sorted(missing)}, extra={sorted(extra)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_training_record(path: Path, record: dict[str, Any]) -> None:
    required = {
        "run_id",
        "config_hash",
        "seed",
        "method",
        "workload",
        "episode",
        "normalized_cumulative_reward",
        "end_to_end_latency_ms",
        "load_balance_index",
        "sla_violation_rate_percent",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise KeyError(f"Training record is missing fields: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_training_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Measured training log does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "episode" not in record or "method" not in record:
                raise ValueError(f"Malformed training record at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Measured training log is empty: {path}")
    return records


def convergence_episode(
    rewards: list[float],
    window: int,
    minimum_episode: int,
    patience: int,
    relative_delta: float,
) -> int:
    if window < 2 or patience < 1 or minimum_episode < window:
        raise ValueError("Invalid convergence settings")
    values = np.asarray(rewards, dtype=np.float64)
    if len(values) < minimum_episode + patience * window:
        return len(values)
    moving = np.convolve(values, np.ones(window) / window, mode="valid")
    stable = 0
    for moving_index in range(window, len(moving), window):
        episode = moving_index + window
        if episode < minimum_episode:
            continue
        previous = moving[moving_index - window]
        current = moving[moving_index]
        relative = abs(current - previous) / max(abs(previous), 1e-8)
        stable = stable + 1 if relative < relative_delta else 0
        if stable >= patience:
            return episode
    return len(values)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value set")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def planner_worker_rows(
    run_id: str,
    configuration_hash: str,
    seed: int,
    controller_counts: list[int],
    workers: list[int],
    planning_interval_s: float,
    measured_service_time_s: float,
) -> list[dict[str, Any]]:
    rows = []
    for worker_count in workers:
        for controllers in controller_counts:
            plans_per_minute = (controllers / planning_interval_s) * 60.0
            utilization = (controllers / planning_interval_s) * measured_service_time_s / worker_count
            rows.append(
                result_row(
                    run_id=run_id,
                    configuration_hash=configuration_hash,
                    seed=seed,
                    figure=11,
                    panel="a",
                    method=f"workers_{worker_count}",
                    scenario="planner_scaling",
                    workload="planning_cadence_600s",
                    x=plans_per_minute,
                    x_label="plans/min",
                    metric="worker_utilization",
                    value=utilization,
                    unit="ratio",
                )
            )
    return rows


def records_to_figure4_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    metric_map = {
        "normalized_cumulative_reward": ("a", "normalized_cumulative_reward", "ratio"),
        "end_to_end_latency_ms": ("b", "end_to_end_latency", "ms"),
        "load_balance_index": ("c", "load_balance_index", "index"),
        "sla_violation_rate_percent": ("d", "sla_violation_rate", "percent"),
    }
    for record in records:
        for source, (panel, metric, unit) in metric_map.items():
            rows.append(
                result_row(
                    run_id=str(record["run_id"]),
                    configuration_hash=str(record["config_hash"]),
                    seed=int(record["seed"]),
                    figure=4,
                    panel=panel,
                    method=str(record["method"]),
                    scenario="training_convergence",
                    workload=str(record["workload"]),
                    x=float(record["episode"]),
                    x_label="episode",
                    metric=metric,
                    value=float(record[source]),
                    unit=unit,
                )
            )
    return rows


def run_evaluation(config_path: str = "configs/paper.json") -> None:
    """Run configured measured scenarios through the trained policy driver."""
    from env.constants import load_config
    from train_agentMulti import evaluate_experiments

    config = load_config(config_path)
    rows, spatial = evaluate_experiments(config)
    training_directory = Path(config["paths"]["training_logs"])
    for method in FIGURE4_METHODS:
        for seed in config["evaluation"]["seeds"]:
            path = training_directory / f"{method_id(method)}-seed-{seed}.jsonl"
            rows.extend(records_to_figure4_rows(read_training_records(path)))
    figure_data = Path(config["paths"]["figure_data"])
    write_results(figure_data / "results.csv", rows)
    required_spatial = {
        "x",
        "y",
        "utilization_before",
        "utilization_after",
        "node_x",
        "node_y",
    }
    missing_spatial = sorted(required_spatial.difference(spatial))
    if missing_spatial:
        raise KeyError(f"Spatial results are missing arrays: {missing_spatial}")
    np.savez_compressed(figure_data / "spatial_grids.npz", **spatial)
    print(f"Measured evaluation records: {figure_data / 'results.csv'}")


if __name__ == "__main__":
    run_evaluation()
