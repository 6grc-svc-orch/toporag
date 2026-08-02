"""Generate the paper's Figures 4--11 from measured experiment records."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


FIGURE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = FIGURE_DIR / "data" / "results.csv"
SPATIAL_PATH = FIGURE_DIR / "data" / "spatial_grids.npz"
OUTPUT_DIR = FIGURE_DIR / "output"

REQUIRED_COLUMNS = (
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
EXPECTED_SEEDS = frozenset({646, 647, 648, 649, 650})
EXPECTED_FEASIBILITY_TRIALS = 500

CORE_METHODS = ("llm_mappo", "rsppo", "gnn_drl", "mu_ddrl", "cromadqn")
LOAD_METHODS = (
    "llm_mappo",
    "llm_mappo_no_toporag",
    "rsppo",
    "gnn_drl",
    "mu_ddrl",
    "cromadqn",
)
ABLATIONS = ("full", "no_toporag", "no_reward_shaping", "no_llm_guidance")
SENSITIVITY_PARAMETERS = (
    "lora_rank",
    "frozen_layer_ratio",
    "planning_frequency",
    "learning_rate",
    "clip_epsilon",
    "entropy_coefficient",
    "trust_threshold",
    "interpolation_decay",
)
SENSITIVITY_RESPONSES = (
    "end_to_end_latency",
    "sla_violation_rate",
    "convergence_time",
    "resource_efficiency",
    "planning_overhead",
    "adaptation_quality",
)
WORKER_METHODS = ("workers_1", "workers_2", "workers_3", "workers_4", "workers_5")

METHOD_LABELS = {
    "llm_mappo": "LLM-MAPPO",
    "llm_mappo_no_toporag": "LLM-MAPPO w/o TopoRAG",
    "rsppo": "RSPPO",
    "gnn_drl": "GNN-DRL",
    "mu_ddrl": "μ-DDRL",
    "cromadqn": "CROMADQN",
    "full": "Full LLM-MAPPO",
    "no_toporag": "Without TopoRAG",
    "no_reward_shaping": "Without Reward Shaping",
    "no_llm_guidance": "Without LLM Guidance",
    "workers_1": "C=1",
    "workers_2": "C=2",
    "workers_3": "C=3",
    "workers_4": "C=4",
    "workers_5": "C=5",
}
PARAMETER_LABELS = {
    "lora_rank": "LoRA rank",
    "frozen_layer_ratio": "Frozen layer ratio",
    "planning_frequency": "Planning frequency",
    "learning_rate": "Learning rate",
    "clip_epsilon": "Clip epsilon",
    "entropy_coefficient": "Entropy coefficient",
    "trust_threshold": "Trust threshold",
    "interpolation_decay": "Interpolation decay",
}
RESPONSE_LABELS = {
    "end_to_end_latency": "End-to-end\nlatency",
    "sla_violation_rate": "SLA\nviolations",
    "convergence_time": "Convergence\nspeed",
    "resource_efficiency": "Resource\nefficiency",
    "planning_overhead": "Planning\noverhead",
    "adaptation_quality": "Adaptation\nquality",
}
METHOD_COLORS = {
    "llm_mappo": "#168aad",
    "llm_mappo_no_toporag": "#8f3b76",
    "rsppo": "#6c5ce7",
    "gnn_drl": "#b23a6f",
    "mu_ddrl": "#f39c12",
    "cromadqn": "#e4572e",
    "full": "#2e8b57",
    "no_toporag": "#4c86b7",
    "no_reward_shaping": "#e0a722",
    "no_llm_guidance": "#dc143c",
    "workers_1": "#277da1",
    "workers_2": "#f8961e",
    "workers_3": "#43aa8b",
    "workers_4": "#e63946",
    "workers_5": "#8e5cc2",
}
METHOD_MARKERS = ("o", "v", "s", "^", "D", "P")


PANEL_SPECS = {
    (4, "a"): {
        "scenario": "training_convergence",
        "workload": "diurnal_24h",
        "metrics": {"normalized_cumulative_reward": "ratio"},
        "methods": CORE_METHODS,
        "minimum_points": 2,
    },
    (4, "b"): {
        "scenario": "training_convergence",
        "workload": "diurnal_24h",
        "metrics": {"end_to_end_latency": "ms"},
        "methods": CORE_METHODS,
        "minimum_points": 2,
    },
    (4, "c"): {
        "scenario": "training_convergence",
        "workload": "diurnal_24h",
        "metrics": {"load_balance_index": "index"},
        "methods": CORE_METHODS,
        "minimum_points": 2,
    },
    (4, "d"): {
        "scenario": "training_convergence",
        "workload": "diurnal_24h",
        "metrics": {"sla_violation_rate": "percent"},
        "methods": CORE_METHODS,
        "minimum_points": 2,
    },
    (5, "a"): {
        "scenario": "arrival_rate_scaling",
        "workload": "metaverse_mixed",
        "metrics": {"end_to_end_latency": "ms"},
        "methods": LOAD_METHODS,
        "minimum_points": 5,
        "expected_x": (50.0, 100.0, 175.0, 225.0, 300.0),
    },
    (5, "b"): {
        "scenario": "arrival_rate_scaling",
        "workload": "metaverse_mixed",
        "metrics": {"sla_violation_rate": "percent"},
        "methods": LOAD_METHODS,
        "minimum_points": 5,
        "expected_x": (50.0, 100.0, 175.0, 225.0, 300.0),
    },
    (5, "c"): {
        "scenario": "arrival_rate_scaling",
        "workload": "metaverse_mixed",
        "metrics": {"load_balance_index": "index"},
        "methods": LOAD_METHODS,
        "minimum_points": 5,
        "expected_x": (50.0, 100.0, 175.0, 225.0, 300.0),
    },
    (5, "d"): {
        "scenario": "arrival_rate_scaling",
        "workload": "metaverse_mixed",
        "metrics": {"convergence_time": "episodes"},
        "methods": LOAD_METHODS,
        "minimum_points": 5,
        "expected_x": (50.0, 100.0, 175.0, 225.0, 300.0),
    },
    (6, "a"): {
        "scenario": "service_complexity",
        "workload": "metaverse_service_dag",
        "metrics": {"end_to_end_latency": "ms"},
        "methods": CORE_METHODS,
        "minimum_points": 4,
        "expected_labels": ("26MS/3R", "30MS/5R", "33MS/6R", "37MS/8R"),
    },
    (6, "b"): {
        "scenario": "service_complexity",
        "workload": "metaverse_service_dag",
        "metrics": {"sla_violation_rate": "percent"},
        "methods": CORE_METHODS,
        "minimum_points": 4,
        "expected_labels": ("26MS/3R", "30MS/5R", "33MS/6R", "37MS/8R"),
    },
    (7, "a"): {
        "scenario": "node_failures",
        "workload": "metaverse_mixed",
        "metrics": {"service_recovery_time": "s"},
        "methods": CORE_METHODS,
        "minimum_points": 6,
        "expected_x": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    },
    (7, "b"): {
        "scenario": "traffic_burst",
        "workload": "metaverse_mixed",
        "metrics": {"latency_to_baseline": "ratio"},
        "methods": CORE_METHODS,
        "minimum_points": 5,
    },
    (8, "a"): {
        "scenario": "resource_utilization",
        "workload": "diurnal_24h",
        "metrics": {"compute_utilization": "percent"},
        "methods": CORE_METHODS,
        "minimum_points": 2,
    },
    (8, "b"): {
        "scenario": "resource_utilization",
        "workload": "diurnal_24h",
        "metrics": {"bandwidth_utilization": "percent"},
        "methods": CORE_METHODS,
        "minimum_points": 2,
    },
    (9, "a"): {
        "scenario": "component_ablation",
        "workload": "metaverse_mixed",
        "metrics": {"end_to_end_latency": "ms"},
        "methods": ABLATIONS,
        "minimum_points": 1,
    },
    (9, "b"): {
        "scenario": "component_ablation",
        "workload": "metaverse_mixed",
        "metrics": {"sla_violation_rate": "percent"},
        "methods": ABLATIONS,
        "minimum_points": 1,
    },
    (9, "c"): {
        "scenario": "component_ablation",
        "workload": "metaverse_mixed",
        "metrics": {"convergence_time": "episodes"},
        "methods": ABLATIONS,
        "minimum_points": 1,
    },
    (9, "d"): {
        "scenario": "hyperparameter_sensitivity",
        "workload": "metaverse_mixed",
        "metrics": {"sensitivity": "percent"},
        "methods": SENSITIVITY_PARAMETERS,
        "minimum_points": 6,
        "expected_labels": SENSITIVITY_RESPONSES,
    },
    (10, "a"): {
        "scenario": "network_density",
        "workload": "metaverse_mixed",
        "metrics": {"feasible_plan_rate": "percent"},
        "methods": ABLATIONS,
        "minimum_points": 3,
        "expected_labels": ("urban", "medium", "suburban"),
    },
    (10, "b"): {
        "scenario": "semantic_drift",
        "workload": "metaverse_mixed",
        "metrics": {"semantic_drift": "score"},
        "methods": ABLATIONS,
        "minimum_points": 10,
    },
    (11, "a"): {
        "scenario": "planner_scaling",
        "workload": "planning_cadence_600s",
        "metrics": {"worker_utilization": "ratio"},
        "methods": WORKER_METHODS,
        "minimum_points": 2,
    },
    (11, "b"): {
        "scenario": "toporag_scaling",
        "workload": "trace_memory",
        "metrics": {
            "retrieval_latency_p25": "ms",
            "retrieval_latency_p50": "ms",
            "retrieval_latency_p75": "ms",
            "memory_usage": "MiB",
        },
        "methods": ("toporag",),
        "minimum_points": 2,
    },
}


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Measured figure input is missing: {RESULTS_PATH}. Run evaluate.py before generating figures."
        )

    frame = pd.read_csv(RESULTS_PATH)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{RESULTS_PATH} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{RESULTS_PATH} contains no measured records")

    text_columns = (
        "run_id",
        "config_hash",
        "panel",
        "method",
        "scenario",
        "workload",
        "x_label",
        "metric",
        "unit",
    )
    for column in text_columns:
        if frame[column].isna().any():
            raise ValueError(f"Column {column!r} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Column {column!r} contains empty values")

    for column in ("seed", "figure", "x", "value"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column!r} contains non-finite values")

    if not np.equal(frame["seed"], np.floor(frame["seed"])).all():
        raise ValueError("Seed values must be integers")
    if not np.equal(frame["figure"], np.floor(frame["figure"])).all():
        raise ValueError("Figure values must be integers")
    frame["seed"] = frame["seed"].astype(int)
    frame["figure"] = frame["figure"].astype(int)

    validate_results(frame)
    return frame


def validate_results(frame: pd.DataFrame) -> None:
    expected_panels = set(PANEL_SPECS)
    observed_panels = set(zip(frame["figure"], frame["panel"], strict=False))
    if observed_panels != expected_panels:
        missing = sorted(expected_panels - observed_panels)
        unexpected = sorted(observed_panels - expected_panels)
        raise ValueError(f"Figure panel coverage mismatch; missing={missing}, unexpected={unexpected}")

    for key, spec in PANEL_SPECS.items():
        figure, panel_name = key
        panel = frame[(frame["figure"] == figure) & (frame["panel"] == panel_name)]
        _expect_values(panel, "scenario", {spec["scenario"]}, key)
        _expect_values(panel, "workload", {spec["workload"]}, key)
        _expect_values(panel, "method", set(spec["methods"]), key)
        _expect_values(panel, "metric", set(spec["metrics"]), key)

        for metric, unit in spec["metrics"].items():
            metric_rows = panel[panel["metric"] == metric]
            _expect_values(metric_rows, "unit", {unit}, key)
            _validate_series_coverage(metric_rows, spec, key, metric)

    _validate_metric_ranges(frame)
    _validate_feasibility_trials(frame)
    _validate_retrieval_quantiles(frame)


def _validate_feasibility_trials(frame: pd.DataFrame) -> None:
    trials = frame[(frame["figure"] == 10) & (frame["panel"] == "a")]
    counts = trials.groupby(["seed", "method", "x_label"], sort=True).size()
    if counts.empty or not (counts == EXPECTED_FEASIBILITY_TRIALS).all():
        invalid = counts[counts != EXPECTED_FEASIBILITY_TRIALS].to_dict()
        raise ValueError(
            "Figure 10a requires exactly "
            f"{EXPECTED_FEASIBILITY_TRIALS} binary trials per seed/method/density; "
            f"invalid={invalid}"
        )


def _expect_values(frame: pd.DataFrame, column: str, expected: set[str], key: tuple[int, str]) -> None:
    observed = set(frame[column].unique())
    if observed != expected:
        raise ValueError(
            f"Figure {key[0]}{key[1]} has invalid {column} values; "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )


def _validate_series_coverage(
    frame: pd.DataFrame,
    spec: dict[str, object],
    key: tuple[int, str],
    metric: str,
) -> None:
    reference_points: set[tuple[float, str]] | None = None
    for method in spec["methods"]:
        method_rows = frame[frame["method"] == method]
        observed_seeds = set(method_rows["seed"].unique())
        if observed_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"Figure {key[0]}{key[1]} {metric}/{method} has incomplete seeds; "
                f"expected={sorted(EXPECTED_SEEDS)}, observed={sorted(observed_seeds)}"
            )

        for seed in sorted(EXPECTED_SEEDS):
            seed_rows = method_rows[method_rows["seed"] == seed]
            points = set(zip(seed_rows["x"].astype(float), seed_rows["x_label"], strict=False))
            if len(points) < int(spec["minimum_points"]):
                raise ValueError(
                    f"Figure {key[0]}{key[1]} {metric}/{method}/seed {seed} has "
                    f"{len(points)} x points; at least {spec['minimum_points']} are required"
                )
            if reference_points is None:
                reference_points = points
            elif points != reference_points:
                raise ValueError(
                    f"Figure {key[0]}{key[1]} {metric} must use identical x points for every method and seed"
                )

    if reference_points is None:
        raise ValueError(f"Figure {key[0]}{key[1]} {metric} contains no measurements")

    observed_x = {point[0] for point in reference_points}
    if "expected_x" in spec and observed_x != set(spec["expected_x"]):
        raise ValueError(
            f"Figure {key[0]}{key[1]} {metric} has invalid x values; "
            f"expected={list(spec['expected_x'])}, observed={sorted(observed_x)}"
        )

    observed_labels = {point[1] for point in reference_points}
    if "expected_labels" in spec and observed_labels != set(spec["expected_labels"]):
        raise ValueError(
            f"Figure {key[0]}{key[1]} {metric} has invalid x labels; "
            f"expected={list(spec['expected_labels'])}, observed={sorted(observed_labels)}"
        )


def _validate_metric_ranges(frame: pd.DataFrame) -> None:
    nonnegative_metrics = {
        "end_to_end_latency",
        "load_balance_index",
        "sla_violation_rate",
        "convergence_time",
        "service_recovery_time",
        "latency_to_baseline",
        "compute_utilization",
        "bandwidth_utilization",
        "sensitivity",
        "feasible_plan_rate",
        "semantic_drift",
        "worker_utilization",
        "retrieval_latency_p25",
        "retrieval_latency_p50",
        "retrieval_latency_p75",
        "memory_usage",
    }
    bounded_percent_metrics = {
        "compute_utilization",
        "bandwidth_utilization",
        "feasible_plan_rate",
        "sla_violation_rate",
        "sensitivity",
    }

    for metric in nonnegative_metrics:
        values = frame.loc[frame["metric"] == metric, "value"]
        if (values < 0).any():
            raise ValueError(f"Metric {metric!r} contains negative measurements")
    for metric in bounded_percent_metrics:
        values = frame.loc[frame["metric"] == metric, "value"]
        if (values > 100).any():
            raise ValueError(f"Metric {metric!r} must be measured in the range 0--100 percent")
    feasibility = frame.loc[frame["metric"] == "feasible_plan_rate", "value"]
    if not feasibility.isin((0.0, 100.0)).all():
        raise ValueError("Feasible-plan trials must be preserved as binary 0/100 measurements")


def _validate_retrieval_quantiles(frame: pd.DataFrame) -> None:
    panel = frame[(frame["figure"] == 11) & (frame["panel"] == "b")]
    metric_points = {
        metric: set(zip(rows["x"].astype(float), rows["x_label"], strict=False))
        for metric, rows in panel.groupby("metric")
    }
    reference_points = metric_points["retrieval_latency_p50"]
    if any(points != reference_points for points in metric_points.values()):
        raise ValueError("Figure 11b metrics must use identical memory-graph sizes")

    latency = panel[panel["metric"].str.startswith("retrieval_latency")]
    per_seed = (
        latency.groupby(["seed", "x", "metric"], as_index=False, sort=True)["value"]
        .mean()
        .pivot(index=["seed", "x"], columns="metric", values="value")
    )
    if (per_seed["retrieval_latency_p25"] > per_seed["retrieval_latency_p50"]).any() or (
        per_seed["retrieval_latency_p50"] > per_seed["retrieval_latency_p75"]
    ).any():
        raise ValueError("Figure 11b retrieval quantiles must satisfy p25 <= p50 <= p75")


def load_spatial_grids() -> dict[str, np.ndarray]:
    if not SPATIAL_PATH.is_file():
        raise FileNotFoundError(
            f"Measured spatial input is missing: {SPATIAL_PATH}. Run evaluate.py before generating figures."
        )

    expected_keys = {"x", "y", "utilization_before", "utilization_after", "node_x", "node_y"}
    with np.load(SPATIAL_PATH, allow_pickle=False) as archive:
        observed_keys = set(archive.files)
        if observed_keys != expected_keys:
            raise ValueError(
                f"{SPATIAL_PATH} keys must be exactly {sorted(expected_keys)}; observed={sorted(observed_keys)}"
            )
        arrays = {key: np.asarray(archive[key], dtype=float).copy() for key in expected_keys}

    for name, array in arrays.items():
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"Spatial array {name!r} must be non-empty and finite")

    before = arrays["utilization_before"]
    after = arrays["utilization_after"]
    if before.ndim != 2 or after.shape != before.shape:
        raise ValueError("Spatial utilization grids must be two-dimensional and have identical shapes")
    if (before < 0).any() or (before > 100).any() or (after < 0).any() or (after > 100).any():
        raise ValueError("Spatial utilization grids must contain percentages in the range 0--100")

    x_values = arrays["x"]
    y_values = arrays["y"]
    if x_values.ndim == 1 and y_values.ndim == 1:
        if x_values.size != before.shape[1] or y_values.size != before.shape[0]:
            raise ValueError("One-dimensional x/y coordinates do not match the utilization grid")
        if not (np.diff(x_values) > 0).all() or not (np.diff(y_values) > 0).all():
            raise ValueError("One-dimensional spatial coordinates must be strictly increasing")
    elif x_values.ndim == 2 and y_values.ndim == 2:
        if x_values.shape != before.shape or y_values.shape != before.shape:
            raise ValueError("Two-dimensional x/y coordinates must match the utilization grid")
    else:
        raise ValueError("Spatial x/y coordinates must both be one-dimensional or both be two-dimensional")

    if arrays["node_x"].ndim != 1 or arrays["node_y"].ndim != 1:
        raise ValueError("Node coordinates must be one-dimensional")
    if arrays["node_x"].shape != arrays["node_y"].shape:
        raise ValueError("Node x/y coordinate arrays must have identical shapes")
    return arrays


def panel_rows(frame: pd.DataFrame, figure: int, panel: str) -> pd.DataFrame:
    return frame[(frame["figure"] == figure) & (frame["panel"] == panel)].copy()


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    per_seed = frame.groupby(groups + ["seed"], as_index=False, sort=True)["value"].mean()
    summary = per_seed.groupby(groups, as_index=False, sort=True)["value"].agg(["mean", "std", "count"])
    summary = summary.reset_index()
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    return summary


def apply_axes_style(axis: plt.Axes) -> None:
    axis.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    ylabel: str,
    title: str,
    xlabel: str,
) -> None:
    stats = summarize(frame, ["method", "x", "x_label"])
    for index, method in enumerate(methods):
        rows = stats[stats["method"] == method].sort_values("x")
        x_values = rows["x"].to_numpy(dtype=float)
        means = rows["mean"].to_numpy(dtype=float)
        ci95 = rows["ci95"].to_numpy(dtype=float)
        axis.plot(
            x_values,
            means,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[index % len(METHOD_MARKERS)],
            markevery=max(1, len(rows) // 10),
            linewidth=1.8,
            markersize=4,
            label=METHOD_LABELS[method],
        )
        axis.fill_between(x_values, means - ci95, means + ci95, color=METHOD_COLORS[method], alpha=0.12)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    apply_axes_style(axis)


def plot_grouped_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    ylabel: str,
    title: str,
    xlabel: str,
) -> None:
    stats = summarize(frame, ["method", "x", "x_label"])
    point_rows = stats[["x", "x_label"]].drop_duplicates().sort_values("x")
    x_values = point_rows["x"].to_numpy(dtype=float)
    x_labels = point_rows["x_label"].tolist()
    positions = np.arange(len(x_values), dtype=float)
    width = 0.82 / len(methods)

    for index, method in enumerate(methods):
        rows = stats[stats["method"] == method].set_index("x").loc[x_values]
        offset = (index - (len(methods) - 1) / 2) * width
        axis.bar(
            positions + offset,
            rows["mean"],
            width,
            yerr=rows["ci95"],
            capsize=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(positions, x_labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    apply_axes_style(axis)


def plot_single_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    ylabel: str,
    title: str,
) -> None:
    stats = summarize(frame, ["method"]).set_index("method").loc[list(methods)]
    positions = np.arange(len(methods))
    labels = [METHOD_LABELS[method].replace(" ", "\n", 1) for method in methods]
    axis.bar(
        positions,
        stats["mean"],
        yerr=stats["ci95"],
        capsize=3,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="#202020",
        linewidth=0.6,
    )
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    apply_axes_style(axis)


def shared_legend(figure: plt.Figure, axes: np.ndarray, columns: int) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=columns, frameon=False, bbox_to_anchor=(0.5, 1.01))


def figure_4(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    settings = (
        ("a", "Normalized cumulative reward", "(a) Cumulative reward"),
        ("b", "End-to-end latency (ms)", "(b) Mean end-to-end latency"),
        ("c", "Load balance index", "(c) QoS load balancing index"),
        ("d", "SLA violation rate (%)", "(d) SLA violation rate"),
    )
    for axis, (panel, ylabel, title) in zip(axes.flat, settings, strict=True):
        plot_lines(axis, panel_rows(frame, 4, panel), CORE_METHODS, ylabel, title, "Training episode")
    shared_legend(figure, axes, 5)
    figure.suptitle("Figure 4. Training convergence comparison", y=1.06, fontsize=14)
    figure.tight_layout()
    return figure


def figure_5(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    settings = (
        ("a", "End-to-end latency (ms)", "(a) End-to-end latency"),
        ("b", "SLA violation rate (%)", "(b) SLA violation rate"),
        ("c", "Load imbalance index", "(c) QoS load balancing index"),
        ("d", "Convergence time (episodes)", "(d) Convergence time"),
    )
    for axis, (panel, ylabel, title) in zip(axes.flat, settings, strict=True):
        plot_grouped_bars(
            axis,
            panel_rows(frame, 5, panel),
            LOAD_METHODS,
            ylabel,
            title,
            "Arrival rate (requests/s)",
        )
    shared_legend(figure, axes, 3)
    figure.suptitle("Figure 5. Performance under varying arrival rates", y=1.06, fontsize=14)
    figure.tight_layout()
    return figure


def figure_6(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    settings = (
        ("a", "Mean end-to-end latency (ms)", "(a) End-to-end latency"),
        ("b", "SLA violation rate (%)", "(b) SLA violation rate"),
    )
    for axis, (panel, ylabel, title) in zip(axes, settings, strict=True):
        rows = panel_rows(frame, 6, panel)
        plot_lines(axis, rows, CORE_METHODS, ylabel, title, "Configuration (microservices/replicas)")
        labels = rows[["x", "x_label"]].drop_duplicates().sort_values("x")
        axis.set_xticks(labels["x"], labels["x_label"])
    shared_legend(figure, axes, 5)
    figure.suptitle("Figure 6. Performance under varying system complexity", y=1.08, fontsize=14)
    figure.tight_layout()
    return figure


def figure_7(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_lines(
        axes[0],
        panel_rows(frame, 7, "a"),
        CORE_METHODS,
        "Service recovery time (s)",
        "(a) Recovery after edge-node failures",
        "Node failure rate (%)",
    )
    axes[0].axhline(120.0, color="#d62728", linestyle="--", linewidth=1.4, label="2-minute SLA")
    plot_lines(
        axes[1],
        panel_rows(frame, 7, "b"),
        CORE_METHODS,
        "Latency relative to baseline",
        "(b) Adaptation to sudden traffic bursts",
        "Time after burst start (s)",
    )
    axes[1].axvspan(0.0, 20.0, color="#f4a261", alpha=0.17, label="Burst interval")
    handles, labels = axes[0].get_legend_handles_labels()
    burst_handles, burst_labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles + burst_handles[-1:],
        labels + burst_labels[-1:],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle("Figure 7. Reliability and adaptation under dynamic conditions", y=1.09, fontsize=14)
    figure.tight_layout()
    return figure


def _spatial_mesh(spatial: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if spatial["x"].ndim == 1:
        return np.meshgrid(spatial["x"], spatial["y"])
    return spatial["x"], spatial["y"]


def figure_8(frame: pd.DataFrame, spatial: dict[str, np.ndarray]) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_lines(
        axes[0, 0],
        panel_rows(frame, 8, "a"),
        CORE_METHODS,
        "Compute utilization (%)",
        "(a) Compute utilization over time",
        "Simulation time (s)",
    )
    plot_lines(
        axes[0, 1],
        panel_rows(frame, 8, "b"),
        CORE_METHODS,
        "Bandwidth utilization (%)",
        "(b) Bandwidth utilization over time",
        "Simulation time (s)",
    )

    mesh_x, mesh_y = _spatial_mesh(spatial)
    for axis, key, title in (
        (axes[1, 0], "utilization_before", "(c) Load balance before orchestration"),
        (axes[1, 1], "utilization_after", "(d) Load balance after orchestration"),
    ):
        image = axis.pcolormesh(mesh_x, mesh_y, spatial[key], shading="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
        axis.scatter(
            spatial["node_x"],
            spatial["node_y"],
            s=13,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )
        axis.set_xlabel("X (km)")
        axis.set_ylabel("Y (km)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Resource utilization (%)", fraction=0.046, pad=0.04)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle("Figure 8. Resource utilization and spatial load balancing", y=1.055, fontsize=14)
    figure.tight_layout()
    return figure


def _sensitivity_heatmap(axis: plt.Axes, frame: pd.DataFrame) -> None:
    per_seed = frame.groupby(["method", "x_label", "seed"], as_index=False)["value"].mean()
    means = per_seed.groupby(["method", "x_label"], as_index=False)["value"].mean()
    matrix = (
        means.pivot(index="method", columns="x_label", values="value")
        .loc[list(SENSITIVITY_PARAMETERS), list(SENSITIVITY_RESPONSES)]
        .to_numpy(dtype=float)
    )
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=float(matrix.max()))
    axis.set_xticks(np.arange(len(SENSITIVITY_RESPONSES)), [RESPONSE_LABELS[key] for key in SENSITIVITY_RESPONSES])
    axis.set_yticks(np.arange(len(SENSITIVITY_PARAMETERS)), [PARAMETER_LABELS[key] for key in SENSITIVITY_PARAMETERS])
    axis.tick_params(axis="x", labelrotation=42, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    threshold = matrix.max() * 0.55
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", fontsize=7, color=color)
    axis.set_title("(d) Hyperparameter sensitivity")
    axis.figure.colorbar(image, ax=axis, label="Sensitivity (%)", fraction=0.046, pad=0.04)


def figure_9(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    settings = (
        ("a", "Mean end-to-end latency (ms)", "(a) Latency impact"),
        ("b", "SLA violation rate (%)", "(b) Service-quality impact"),
        ("c", "Episodes to convergence", "(c) Training-efficiency impact"),
    )
    for axis, (panel, ylabel, title) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), settings, strict=True):
        plot_single_bars(axis, panel_rows(frame, 9, panel), ABLATIONS, ylabel, title)
    _sensitivity_heatmap(axes[1, 1], panel_rows(frame, 9, "d"))
    figure.suptitle("Figure 9. Ablation and hyperparameter sensitivity", y=1.02, fontsize=14)
    figure.tight_layout()
    return figure


def _feasibility_bars(axis: plt.Axes, frame: pd.DataFrame) -> None:
    labels = ("urban", "medium", "suburban")
    label_to_x = frame[["x_label", "x"]].drop_duplicates().set_index("x_label")["x"].to_dict()
    x_values = np.array([label_to_x[label] for label in labels], dtype=float)
    positions = np.arange(len(labels), dtype=float)
    width = 0.82 / len(ABLATIONS)
    for index, method in enumerate(ABLATIONS):
        means = []
        lower_errors = []
        upper_errors = []
        for x_value in x_values:
            trials = frame[(frame["method"] == method) & (frame["x"] == x_value)]["value"]
            successes = int((trials == 100.0).sum())
            total = len(trials)
            if total == 0:
                raise ValueError(f"Figure 10a has no feasibility trials for {method}/{x_value}")
            proportion = successes / total
            z = 1.959963984540054
            denominator = 1.0 + z * z / total
            center = (proportion + z * z / (2.0 * total)) / denominator
            radius = (
                z
                * np.sqrt(
                    proportion * (1.0 - proportion) / total
                    + z * z / (4.0 * total * total)
                )
                / denominator
            )
            low = max(0.0, center - radius) * 100.0
            high = min(1.0, center + radius) * 100.0
            mean = proportion * 100.0
            means.append(mean)
            lower_errors.append(mean - low)
            upper_errors.append(high - mean)
        offset = (index - (len(ABLATIONS) - 1) / 2) * width
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=np.asarray((lower_errors, upper_errors)),
            capsize=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(positions, ("Urban\n(high density)", "Medium\ndensity", "Suburban\n(low density)"))
    axis.set_ylabel("Feasible-plan rate (%)")
    axis.set_xlabel("Network environment")
    axis.set_title("(a) Pre-agent plan feasibility")
    apply_axes_style(axis)


def _drift_cdf(axis: plt.Axes, frame: pd.DataFrame) -> None:
    inset = axis.inset_axes([0.56, 0.52, 0.41, 0.39])
    for method in ABLATIONS:
        values = np.sort(frame.loc[frame["method"] == method, "value"].to_numpy(dtype=float))
        cumulative = np.arange(1, values.size + 1, dtype=float) / values.size
        survival = (values.size - np.arange(values.size, dtype=float)) / values.size
        axis.plot(values, cumulative, color=METHOD_COLORS[method], linewidth=1.9, label=METHOD_LABELS[method])
        inset.plot(values, survival, color=METHOD_COLORS[method], linewidth=1.2)
    axis.axvline(0.1, color="#d62728", linestyle=":", linewidth=1.3, label="Trust threshold")
    axis.set_xlabel("Semantic drift score")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_ylim(0, 1.02)
    axis.set_title("(b) Semantic drift distribution")
    apply_axes_style(axis)
    inset.axvline(0.1, color="#d62728", linestyle=":", linewidth=1.0)
    inset.set_yscale("log")
    inset.set_xlabel("Drift", fontsize=7)
    inset.set_ylabel("Tail probability", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(True, linewidth=0.5, alpha=0.55)


def figure_10(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    _feasibility_bars(axes[0], panel_rows(frame, 10, "a"))
    _drift_cdf(axes[1], panel_rows(frame, 10, "b"))
    handles, labels = axes[0].get_legend_handles_labels()
    drift_handles, drift_labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles + drift_handles[-1:],
        labels + drift_labels[-1:],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle("Figure 10. Plan feasibility and semantic drift", y=1.1, fontsize=14)
    figure.tight_layout()
    return figure


def _planner_utilization(axis: plt.Axes, frame: pd.DataFrame) -> None:
    plot_lines(
        axis,
        frame,
        WORKER_METHODS,
        "Worker utilization",
        "(a) Planner worker utilization vs. arrival rate",
        "Arrival rate (plans/min)",
    )
    axis.axhline(0.6, color="#d62728", linestyle="--", linewidth=1.3, label="Stability threshold")
    secondary = axis.secondary_xaxis("top", functions=(lambda rate: rate * 10.0, lambda controllers: controllers / 10.0))
    secondary.set_xlabel("Controllers at a 600-second cadence")


def _toporag_scaling(axis: plt.Axes, frame: pd.DataFrame) -> None:
    stats = summarize(frame, ["metric", "x", "x_label"])
    metric_rows = {}
    for metric in PANEL_SPECS[(11, "b")]["metrics"]:
        metric_rows[metric] = stats[stats["metric"] == metric].sort_values("x")

    x_values = metric_rows["retrieval_latency_p50"]["x"].to_numpy(dtype=float)
    p25 = metric_rows["retrieval_latency_p25"]["mean"].to_numpy(dtype=float)
    p50 = metric_rows["retrieval_latency_p50"]["mean"].to_numpy(dtype=float)
    p75 = metric_rows["retrieval_latency_p75"]["mean"].to_numpy(dtype=float)
    memory = metric_rows["memory_usage"]["mean"].to_numpy(dtype=float)

    axis.plot(x_values, p50, color="#5b8e15", marker="o", linewidth=1.9, label="p50 retrieval latency")
    axis.fill_between(x_values, p25, p75, color="#8aac47", alpha=0.23, label="Retrieval IQR (p25--p75)")
    axis.set_xlabel("Memory graph size (thousand traces)")
    axis.set_ylabel("Retrieval latency (ms)")
    axis.set_title("(b) TopoRAG scaling with memory size")
    apply_axes_style(axis)

    memory_axis = axis.twinx()
    memory_axis.plot(x_values, memory, color="#8c8c8c", marker="s", linewidth=1.5, label="Memory usage")
    memory_axis.set_ylabel("Memory usage (MiB)")
    handles, labels = axis.get_legend_handles_labels()
    memory_handles, memory_labels = memory_axis.get_legend_handles_labels()
    axis.legend(handles + memory_handles, labels + memory_labels, loc="upper left", frameon=False)


def figure_11(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    _planner_utilization(axes[0], panel_rows(frame, 11, "a"))
    _toporag_scaling(axes[1], panel_rows(frame, 11, "b"))
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", frameon=False, ncol=2)
    figure.suptitle("Figure 11. Commercial deployment feasibility", y=1.08, fontsize=14)
    figure.tight_layout()
    return figure


def render_all(frame: pd.DataFrame, spatial: dict[str, np.ndarray]) -> None:
    builders = {
        4: lambda: figure_4(frame),
        5: lambda: figure_5(frame),
        6: lambda: figure_6(frame),
        7: lambda: figure_7(frame),
        8: lambda: figure_8(frame, spatial),
        9: lambda: figure_9(frame),
        10: lambda: figure_10(frame),
        11: lambda: figure_11(frame),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="paper_figures_", dir=OUTPUT_DIR
    ) as temporary:
        temporary_dir = Path(temporary)
        generated: list[tuple[Path, Path]] = []
        for number, builder in builders.items():
            figure = builder()
            for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
                temporary_path = temporary_dir / f"fig{number:02d}.{suffix}"
                figure.savefig(temporary_path, bbox_inches="tight", **options)
                generated.append((temporary_path, OUTPUT_DIR / temporary_path.name))
            plt.close(figure)

        for source, destination in generated:
            os.replace(source, destination)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    frame = load_results()
    spatial = load_spatial_grids()
    render_all(frame, spatial)
    print(f"Generated Figures 4--11 in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
