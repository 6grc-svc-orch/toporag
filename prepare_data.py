"""Generate the configured paper-scale scenario artifacts."""

from pathlib import Path

import numpy as np

from env.constants import load_config
from env.envMulti import prepare_scenario
from template import save_scenario


def main(config_path: str = "configs/paper.json") -> None:
    config = load_config(config_path)
    scenario = prepare_scenario(config, seed=config["simulation"]["seed"])
    scenario_path = save_scenario(scenario, config["paths"]["generated_scenario"])
    topology = scenario["topology"]
    topology_path = Path(config["paths"]["generated_scenario"]).with_name("paper_topology.npz")
    np.savez_compressed(
        topology_path,
        node_ids=topology["node_ids"],
        node_tiers=topology["node_tiers"],
        node_controller_ids=topology["node_controller_ids"],
        coordinates_km=topology["coordinates_km"],
        capacity=topology["capacity"],
        radio_profiles=topology["radio_profiles"],
        latency_ms=topology["latency_ms"],
        node_active=topology["node_active"],
        base_station_coordinates_km=topology["base_station_coordinates_km"],
        base_station_edge_ids=topology["base_station_edge_ids"],
        link_sources=topology["link_sources"],
        link_destinations=topology["link_destinations"],
        link_capacity_mbps=topology["link_capacity_mbps"],
        link_delay_ms=topology["link_delay_ms"],
        link_reliability=topology["link_reliability"],
    )
    print(
        f"Prepared {len(topology['node_ids'])} infrastructure nodes, "
        f"{len(scenario['services'])} DAG templates, and {len(scenario['requests'])} requests."
    )
    print(f"Scenario: {scenario_path}")
    print(f"Topology arrays: {topology_path}")


if __name__ == "__main__":
    main()
