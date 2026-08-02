"""Render the conference paper's measured Figures 3--6."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


FIGURE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = FIGURE_DIR / "data" / "results.csv"
OUTPUT_DIR = FIGURE_DIR / "output"

METHODS = ("RAG-DRL", "DQN Scheduler", "DRP")
METHOD_COLORS = {
    "RAG-DRL": "#1f77b4",
    "DQN Scheduler": "#ff7f0e",
    "DRP": "#d62728",
}
METHOD_MARKERS = {
    "RAG-DRL": "o",
    "DQN Scheduler": "v",
    "DRP": "D",
}
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

FIGURE_SPECS = {
    3: ("reward", "score", None),
    4: ("end_to_end_latency", "ms", LATENCY_SLA_SCALES),
    5: ("sla_violation_rate", "percent", LATENCY_SLA_SCALES),
    6: ("resource_utilization", "percent", UTILIZATION_SCALES),
}


def _integer(value: str, column: str, line_number: int) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(
            f"{RESULTS_PATH}:{line_number}: {column} must be an integer"
        ) from error
    return number


def _number(value: str, column: str, line_number: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"{RESULTS_PATH}:{line_number}: {column} must be numeric"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{RESULTS_PATH}:{line_number}: {column} must be finite")
    return number


def load_results(path: Path = RESULTS_PATH) -> list[dict[str, object]]:
    """Load and fully validate raw experiment rows."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Measured figure input is missing: {path}. Run evaluate.py first."
        )
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        observed_columns = tuple(reader.fieldnames or ())
        if observed_columns != RESULT_COLUMNS:
            raise ValueError(
                f"{path} columns must be exactly {list(RESULT_COLUMNS)}; observed={list(observed_columns)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            for column in ("run_id", "config_hash", "method", "metric", "unit"):
                if not raw[column].strip():
                    raise ValueError(f"{path}:{line_number}: {column} cannot be empty")
            row: dict[str, object] = {
                "run_id": raw["run_id"].strip(),
                "config_hash": raw["config_hash"].strip(),
                "figure": _integer(raw["figure"], "figure", line_number),
                "method": raw["method"].strip(),
                "trial": _integer(raw["trial"], "trial", line_number),
                "seed": _integer(raw["seed"], "seed", line_number),
                "episode": _integer(raw["episode"], "episode", line_number),
                "microservices": _integer(
                    raw["microservices"], "microservices", line_number
                ),
                "replicas": _integer(raw["replicas"], "replicas", line_number),
                "metric": raw["metric"].strip(),
                "value": _number(raw["value"], "value", line_number),
                "unit": raw["unit"].strip(),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"Measured figure input is empty: {path}")
    validate_results(rows)
    return rows


def _rows_for(
    rows: Iterable[dict[str, object]], figure: int, method: str
) -> list[dict[str, object]]:
    return [row for row in rows if row["figure"] == figure and row["method"] == method]


def validate_results(rows: list[dict[str, object]]) -> None:
    figures = {int(row["figure"]) for row in rows}
    if figures != set(FIGURE_SPECS):
        raise ValueError(
            f"Result figures must be exactly 3--6; observed={sorted(figures)}"
        )
    methods = {str(row["method"]) for row in rows}
    if methods != set(METHODS):
        raise ValueError(
            f"Result methods must be exactly {list(METHODS)}; observed={sorted(methods)}"
        )
    run_ids = {str(row["run_id"]) for row in rows}
    hashes = {str(row["config_hash"]) for row in rows}
    if len(run_ids) != 1 or len(hashes) != 1:
        raise ValueError(
            "All measured rows must belong to one run and one configuration"
        )

    for figure, (metric, unit, scales) in FIGURE_SPECS.items():
        figure_rows = [row for row in rows if row["figure"] == figure]
        if {str(row["metric"]) for row in figure_rows} != {metric}:
            raise ValueError(f"Figure {figure} must contain only metric {metric!r}")
        if {str(row["unit"]) for row in figure_rows} != {unit}:
            raise ValueError(f"Figure {figure} must use unit {unit!r}")
        if {str(row["method"]) for row in figure_rows} != set(METHODS):
            raise ValueError(f"Figure {figure} is missing a conference method")
        if scales is None:
            _validate_training_rows(figure_rows)
        else:
            _validate_trial_rows(figure, figure_rows, scales)


def _validate_training_rows(rows: list[dict[str, object]]) -> None:
    by_run: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["episode"]) not in range(1, EXPECTED_EPISODES + 1):
            raise ValueError("Figure 3 episodes must be in [1, 4000]")
        if int(row["microservices"]) != 0 or int(row["replicas"]) != 0:
            raise ValueError("Figure 3 rows must use zero for scale fields")
        by_run[(str(row["method"]), int(row["trial"]))].append(row)

    expected_episodes = set(range(1, EXPECTED_EPISODES + 1))
    seed_sets: dict[str, set[int]] = {}
    for method in METHODS:
        method_runs = [key for key in by_run if key[0] == method]
        if not method_runs:
            raise ValueError(f"Figure 3 contains no training run for {method}")
        for key in method_runs:
            run_rows = by_run[key]
            episodes = [int(row["episode"]) for row in run_rows]
            if len(episodes) != EXPECTED_EPISODES or set(episodes) != expected_episodes:
                raise ValueError(
                    f"Figure 3 training run {key} must contain each episode 1--4000 exactly once"
                )
        seed_sets[method] = {
            int(row["seed"]) for row in rows if row["method"] == method
        }
    if len({frozenset(values) for values in seed_sets.values()}) != 1:
        raise ValueError("Figure 3 methods must use the same training seeds")


def _validate_trial_rows(
    figure: int,
    rows: list[dict[str, object]],
    scales: tuple[tuple[int, int], ...],
) -> None:
    expected_trials = set(range(EXPECTED_TRIALS))
    if any(int(row["episode"]) != 0 for row in rows):
        raise ValueError(f"Figure {figure} trial rows must use zero for episode")
    observed_scales = {
        (int(row["microservices"]), int(row["replicas"])) for row in rows
    }
    if observed_scales != set(scales):
        raise ValueError(
            f"Figure {figure} scales must be exactly {list(scales)}; observed={sorted(observed_scales)}"
        )
    for method in METHODS:
        for scale in scales:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (int(row["microservices"]), int(row["replicas"])) == scale
            ]
            trials = [int(row["trial"]) for row in selected]
            seeds = [int(row["seed"]) for row in selected]
            if len(trials) != EXPECTED_TRIALS or set(trials) != expected_trials:
                raise ValueError(
                    f"Figure {figure} {method} {scale} must contain trials 0--999 exactly once"
                )
            if len(set(seeds)) != EXPECTED_TRIALS:
                raise ValueError(
                    f"Figure {figure} {method} {scale} must contain 1000 distinct seeds"
                )
    for scale in scales:
        trial_seed_maps = {
            method: {
                int(row["trial"]): int(row["seed"])
                for row in rows
                if row["method"] == method
                and (int(row["microservices"]), int(row["replicas"])) == scale
            }
            for method in METHODS
        }
        if (
            len({tuple(sorted(values.items())) for values in trial_seed_maps.values()})
            != 1
        ):
            raise ValueError(f"Figure {figure} scale {scale} does not use shared seeds")

    values = np.asarray([float(row["value"]) for row in rows], dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError(f"Figure {figure} measurements cannot be negative")
    if figure in {5, 6} and np.any(values > 100.0):
        raise ValueError(f"Figure {figure} percentages must be in [0, 100]")


def mean_and_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty measured series")
    mean = float(array.mean())
    ci = (
        0.0
        if array.size == 1
        else 1.959963984540054 * float(array.std(ddof=1)) / math.sqrt(array.size)
    )
    return mean, ci


def figure_3(rows: list[dict[str, object]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for method in METHODS:
        selected = _rows_for(rows, 3, method)
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            grouped[int(row["episode"])].append(float(row["value"]))
        episodes = np.arange(1, EXPECTED_EPISODES + 1)
        summaries = np.asarray(
            [mean_and_ci(grouped[int(episode)]) for episode in episodes]
        )
        mean = summaries[:, 0]
        ci = summaries[:, 1]
        color = METHOD_COLORS[method]
        axis.plot(episodes, mean, color=color, linewidth=1.5, label=method)
        axis.fill_between(
            episodes, mean - ci, mean + ci, color=color, alpha=0.16, linewidth=0
        )
    axis.set_xlabel("Training Episodes")
    axis.set_ylabel("Reward")
    axis.set_xlim(0, EXPECTED_EPISODES)
    axis.legend(frameon=False)
    axis.grid(True, linewidth=0.5, alpha=0.45)
    figure.tight_layout()
    return figure


def _scale_figure(
    rows: list[dict[str, object]],
    figure_number: int,
    scales: tuple[tuple[int, int], ...],
    ylabel: str,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    positions = np.arange(len(scales), dtype=np.float64)
    for method in METHODS:
        means = []
        intervals = []
        for microservices, replicas in scales:
            values = [
                float(row["value"])
                for row in rows
                if row["figure"] == figure_number
                and row["method"] == method
                and row["microservices"] == microservices
                and row["replicas"] == replicas
            ]
            mean, ci = mean_and_ci(values)
            means.append(mean)
            intervals.append(ci)
        axis.errorbar(
            positions,
            means,
            yerr=intervals,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.6,
            markersize=5,
            capsize=3,
            label=method,
        )
    axis.set_xticks(
        positions,
        [f"{microservices}µS/{replicas}R" for microservices, replicas in scales],
    )
    axis.set_xlabel("Network Scale")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)
    axis.grid(True, linewidth=0.5, alpha=0.45)
    figure.tight_layout()
    return figure


def render_all(rows: list[dict[str, object]], output_dir: Path = OUTPUT_DIR) -> None:
    builders = {
        3: lambda: figure_3(rows),
        4: lambda: _scale_figure(rows, 4, LATENCY_SLA_SCALES, "Latency (ms)"),
        5: lambda: _scale_figure(rows, 5, LATENCY_SLA_SCALES, "SLA Violation Rate (%)"),
        6: lambda: _scale_figure(
            rows, 6, UTILIZATION_SCALES, "Resource Utilization (%)"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="conference_figures_", dir=output_dir
    ) as directory:
        temporary_dir = Path(directory)
        rendered: list[tuple[Path, Path]] = []
        for number, builder in builders.items():
            figure = builder()
            for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
                temporary_path = temporary_dir / f"Figure{number}.{suffix}"
                figure.savefig(temporary_path, bbox_inches="tight", **options)
                rendered.append((temporary_path, output_dir / temporary_path.name))
            plt.close(figure)
        for source, destination in rendered:
            os.replace(source, destination)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )
    rows = load_results()
    render_all(rows)
    print(f"Generated Figures 3--6 in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
