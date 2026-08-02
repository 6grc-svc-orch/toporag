"""Prepare and validate the conference-paper scenario.

Run directly with ``python prepare_data.py``.  Select ``data.source`` in the
single paper config; external mode deliberately fails unless all three strict
CSV adapters receive their required columns and paper-scale row counts.
"""

from simulator import (
    feasible_action,
    load_config,
    load_scenario,
    observation,
    prepare_scenario,
    reset_environment,
    save_scenario,
    step_environment,
)


def main() -> None:
    config = load_config("configs/paper.json")
    output = save_scenario(prepare_scenario(config), config["data"]["scenario"])
    scenario = load_scenario(output)

    assert len(scenario["esp_coordinates"]) == 3233
    assert len(scenario["controller_coordinates"]) == 16
    assert len(scenario["taxi_coordinates"]) == 4328
    assert int(scenario["service_feature_dim"]) == 5
    assert scenario["capacity"].shape == (3233, 2)

    scale = config["experiment"]["latency_scales"][0]
    state = reset_environment(config, scenario, scale, int(config["seed"]))
    first = observation(state)
    assert first["service_nodes"].shape[0] == int(scale["microservices"])
    assert first["path_features"].shape[2] == 3
    assert first["placement_mask"].any()

    completed = None
    decisions = 0
    while completed is None:
        _, _, _, info = step_environment(state, feasible_action(state))
        completed = info["completed_request"]
        decisions += 1
    expected = int(scale["microservices"]) * int(scale["replicas"])
    assert decisions == expected
    assert len(state["active"][-1]["nodes"]) == expected
    assert (state["residual_nodes"] >= -1e-9).all()
    assert (state["residual_links"] >= -1e-9).all()

    print(f"scenario: {output}")
    print("infrastructure: 3,233 ESPs, 16 edge controllers, 1 cloud")
    print("mobility: 4,328 taxi traces at 0x-2x speed")
    print(
        f"validated: {decisions} exact-replica placement decisions with three-path routing"
    )


if __name__ == "__main__":
    main()
