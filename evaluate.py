"""Run the conference paper's measured evaluation for Figures 3--6."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/paper.json")
METHODS = ("RAG-DRL", "DQN Scheduler", "DRP")
TRAINING_COLUMNS = (
    "method",
    "seed",
    "episode",
    "reward",
    "score",
    "latency_ms",
    "resource_use",
    "sla_violation",
)
LATENCY_SLA_SCALES = ((26, 3), (30, 5), (33, 6), (37, 8))
UTILIZATION_SCALES = ((26, 3), (28, 4), (30, 5), (32, 5), (34, 6), (37, 8))
EXPECTED_EPISODES = 4000
EXPECTED_TRIALS = 1000

RESULT_COLUMNS = (
    "run_id",
    "config_hash",
    "figure",
    "method",
    "trial",
    "seed",
    "episode",
    "microservices",
    "replicas",
    "metric",
    "value",
    "unit",
)

METRIC_FIELDS = {
    "end_to_end_latency": "end_to_end_latency_ms",
    "sla_violation_rate": "sla_violation_rate_percent",
    "resource_utilization": "resource_utilization_percent",
}


def configuration_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def make_run_id(config_hash: str) -> str:
    return f"conference-{config_hash}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def result_row(
    *,
    run_id: str,
    config_hash: str,
    figure: int,
    method: str,
    trial: int,
    seed: int,
    episode: int,
    microservices: int,
    replicas: int,
    metric: str,
    value: float,
    unit: str,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, received {method!r}")
    if figure not in {3, 4, 5, 6}:
        raise ValueError("figure must be in [3, 6]")
    return {
        "run_id": str(run_id),
        "config_hash": str(config_hash),
        "figure": int(figure),
        "method": method,
        "trial": int(trial),
        "seed": int(seed),
        "episode": int(episode),
        "microservices": int(microservices),
        "replicas": int(replicas),
        "metric": str(metric),
        "value": _finite_number(value, metric),
        "unit": str(unit),
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty measured result set")
    for index, row in enumerate(rows):
        if tuple(row) != RESULT_COLUMNS:
            raise ValueError(
                f"result row {index} columns must be exactly {list(RESULT_COLUMNS)}; observed={list(row)}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_training_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Measured training log does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        observed = tuple(reader.fieldnames or ())
        if observed != TRAINING_COLUMNS:
            raise ValueError(
                f"{path} columns must be exactly {list(TRAINING_COLUMNS)}; observed={list(observed)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                seed = int(row["seed"])
                episode = int(row["episode"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: seed and episode must be integers"
                ) from error
            method = row["method"].strip()
            if method not in METHODS:
                raise ValueError(f"{path}:{line_number}: unknown method {method!r}")
            record: dict[str, Any] = {
                "method": method,
                "seed": seed,
                "episode": episode,
            }
            for field in (
                "reward",
                "score",
                "latency_ms",
                "resource_use",
                "sla_violation",
            ):
                try:
                    raw_value = float(row[field])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: {field} must be numeric"
                    ) from error
                record[field] = _finite_number(
                    raw_value, f"{field} at {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Measured training log is empty: {path}")
    return records


def load_training_rows(
    config: dict[str, Any], run_id: str, config_hash: str
) -> list[dict[str, Any]]:
    path = Path(config["paths"]["training_log"])
    records = read_training_records(path)
    configured_seeds = tuple(int(seed) for seed in config["experiment"]["seeds"])
    if not configured_seeds or len(set(configured_seeds)) != len(configured_seeds):
        raise ValueError("experiment.seeds must contain distinct training seeds")
    observed_seeds = {int(record["seed"]) for record in records}
    if observed_seeds != set(configured_seeds):
        raise ValueError(
            f"{path} seeds must match experiment.seeds; observed={sorted(observed_seeds)}"
        )
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for trial, seed in enumerate(configured_seeds):
            selected = [
                record
                for record in records
                if record["method"] == method and int(record["seed"]) == seed
            ]
            episodes = [int(record["episode"]) for record in selected]
            if len(episodes) != EXPECTED_EPISODES or set(episodes) != set(
                range(1, EXPECTED_EPISODES + 1)
            ):
                raise ValueError(
                    f"{path} must contain each episode 1--4000 exactly once for {method}/seed-{seed}"
                )
            for record in sorted(selected, key=lambda item: int(item["episode"])):
                rows.append(
                    result_row(
                        run_id=run_id,
                        config_hash=config_hash,
                        figure=3,
                        method=method,
                        trial=trial,
                        seed=seed,
                        episode=int(record["episode"]),
                        microservices=0,
                        replicas=0,
                        metric="reward",
                        value=float(record["score"]),
                        unit="score",
                    )
                )
    return rows


def _require_export(module: Any, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise TypeError(
            f"{module.__name__} must export callable {name}() for measured evaluation"
        )
    return value


# Simulator integration is intentionally isolated in the following adapters.
def load_evaluation_context(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    simulator = importlib.import_module("simulator")
    load_config = _require_export(simulator, "load_config")
    load_scenario = _require_export(simulator, "load_scenario")
    config = load_config(config_path)
    scenario_path = Path(config["data"]["scenario"])
    if not scenario_path.is_file():
        raise FileNotFoundError(
            f"Prepared scenario is missing: {scenario_path}. Run python prepare_data.py first."
        )
    scenario = load_scenario(scenario_path)
    if not isinstance(config, dict) or not isinstance(scenario, dict):
        raise TypeError("simulator.load_config/load_scenario must return dictionaries")
    return config, scenario, simulator


def reset_trial(
    simulator: Any,
    config: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    microservices: int,
    replicas: int,
) -> Any:
    reset = _require_export(simulator, "reset")
    return reset(config, scenario, (microservices, replicas), seed)


def trial_observation(simulator: Any, state: Any) -> dict[str, Any]:
    observation = _require_export(simulator, "observe")(state)
    if not isinstance(observation, dict):
        raise TypeError("simulator.observe must return a dictionary")
    return observation


def advance_trial(simulator: Any, state: Any, action: Any) -> tuple[Any, bool]:
    transition = _require_export(simulator, "step")(state, action)
    if not isinstance(transition, tuple) or len(transition) != 4:
        raise TypeError("simulator.step must return (state, reward, done, info)")
    _, _, done, _ = transition
    if not isinstance(done, bool):
        raise TypeError("simulator.step done flag must be boolean")
    return state, done


def measured_metrics(simulator: Any, state: Any) -> dict[str, float]:
    metrics = _require_export(simulator, "episode_metrics")(state)
    if not isinstance(metrics, dict):
        raise TypeError("simulator.episode_metrics must return a dictionary")
    missing = sorted(set(METRIC_FIELDS.values()).difference(metrics))
    if missing:
        raise KeyError(f"simulator.episode_metrics is missing fields: {missing}")
    return {
        key: _finite_number(metrics[source], f"episode metric {source}")
        for key, source in METRIC_FIELDS.items()
    }


def load_policy(config: dict[str, Any], method: str) -> Any:
    """Load the method artifact selected by the project configuration."""

    training = importlib.import_module("train_agent")
    return _require_export(training, "load_policy")(config, method)


def select_action(
    method: str, policy: Any, observation: dict[str, Any], state: Any
) -> Any:
    training = importlib.import_module("train_agent")
    return _require_export(training, "policy_action")(
        policy, observation, method, state
    )


def run_trial(
    simulator: Any,
    config: dict[str, Any],
    scenario: dict[str, Any],
    method: str,
    policy: Any,
    seed: int,
    microservices: int,
    replicas: int,
) -> dict[str, float]:
    training = importlib.import_module("train_agent")
    _require_export(training, "seed_everything")(seed)
    _require_export(training, "reset_evaluation_state")(policy)
    state = reset_trial(simulator, config, scenario, seed, microservices, replicas)
    done = bool(state["done"])
    while not done:
        observation = trial_observation(simulator, state)
        action = select_action(method, policy, observation, state)
        state, done = advance_trial(simulator, state, action)
    return measured_metrics(simulator, state)


def validate_evaluation_config(config: dict[str, Any]) -> None:
    if int(config["experiment"]["episodes"]) != EXPECTED_EPISODES:
        raise ValueError("experiment.episodes must be 4000 for Figure 3")
    if int(config["experiment"]["monte_carlo_runs"]) != EXPECTED_TRIALS:
        raise ValueError("experiment.monte_carlo_runs must be 1000")
    if tuple(config["experiment"]["methods"]) != METHODS:
        raise ValueError(f"experiment.methods must be exactly {list(METHODS)}")
    latency_scales = tuple(
        (int(scale["microservices"]), int(scale["replicas"]))
        for scale in config["experiment"]["latency_scales"]
    )
    utilization_scales = tuple(
        (int(scale["microservices"]), int(scale["replicas"]))
        for scale in config["experiment"]["utilization_scales"]
    )
    if len(latency_scales) != len(LATENCY_SLA_SCALES) or set(latency_scales) != set(
        LATENCY_SLA_SCALES
    ):
        raise ValueError(
            f"experiment.latency_scales must be exactly {list(LATENCY_SLA_SCALES)}"
        )
    if len(utilization_scales) != len(UTILIZATION_SCALES) or set(
        utilization_scales
    ) != set(UTILIZATION_SCALES):
        raise ValueError(
            f"experiment.utilization_scales must be exactly {list(UTILIZATION_SCALES)}"
        )
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def run_monte_carlo_rows(
    config: dict[str, Any],
    scenario: dict[str, Any],
    simulator: Any,
    run_id: str,
    config_hash: str,
) -> list[dict[str, Any]]:
    policies = {method: load_policy(config, method) for method in METHODS}
    base_seed = int(config["seed"])
    rows: list[dict[str, Any]] = []
    all_scales = tuple(sorted(set(LATENCY_SLA_SCALES) | set(UTILIZATION_SCALES)))
    for microservices, replicas in all_scales:
        for trial in range(EXPECTED_TRIALS):
            seed = base_seed + trial
            for method in METHODS:
                metrics = run_trial(
                    simulator,
                    config,
                    scenario,
                    method,
                    policies[method],
                    seed,
                    microservices,
                    replicas,
                )
                common = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "method": method,
                    "trial": trial,
                    "seed": seed,
                    "episode": 0,
                    "microservices": microservices,
                    "replicas": replicas,
                }
                if (microservices, replicas) in LATENCY_SLA_SCALES:
                    rows.append(
                        result_row(
                            **common,
                            figure=4,
                            metric="end_to_end_latency",
                            value=metrics["end_to_end_latency"],
                            unit="ms",
                        )
                    )
                    rows.append(
                        result_row(
                            **common,
                            figure=5,
                            metric="sla_violation_rate",
                            value=metrics["sla_violation_rate"],
                            unit="percent",
                        )
                    )
                if (microservices, replicas) in UTILIZATION_SCALES:
                    rows.append(
                        result_row(
                            **common,
                            figure=6,
                            metric="resource_utilization",
                            value=metrics["resource_utilization"],
                            unit="percent",
                        )
                    )
        print(f"Measured 1000 shared trials for {microservices}µS/{replicas}R")
    return rows


def run_evaluation(config_path: Path = CONFIG_PATH) -> Path:
    config, scenario, simulator = load_evaluation_context(config_path)
    validate_evaluation_config(config)
    config_hash = configuration_hash(config)
    run_id = make_run_id(config_hash)
    rows = load_training_rows(config, run_id, config_hash)
    rows.extend(run_monte_carlo_rows(config, scenario, simulator, run_id, config_hash))
    result_path = Path(config["paths"]["figure_data"]) / "results.csv"
    write_results(result_path, rows)
    print(f"Measured results: {result_path}")
    print("Render them with: python figures/generate_figures.py")
    return result_path


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
