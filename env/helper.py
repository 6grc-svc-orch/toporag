import numpy as np
from env.constants import N_MS
from typing import TypedDict
from numpy.typing import NDArray


class FogEnvState(TypedDict):
    node_feats: NDArray[np.float64]
    link_feats: NDArray[np.float64]
    service_feats: NDArray[np.float64]
    ms_feats: NDArray[np.float64]
    link_ms_feats: NDArray[np.float64]
    placement_status: NDArray[np.float64]
    next_ms_idx: int
    next_service_idx: int


class FogEnvAssertionsMixin:
    """All assertion checks for internal consistency"""

    # def _assert_p_nodes(self, p_nodes, ms_idx):
    #     assert len(p_nodes) > 0, f"No previous node found for microservice {ms_idx - 1}"

    def _assert_next_service_idx(self):
        expected = 0
        actual = self.state["next_service_idx"]
        assert actual == expected, f"`next_service_idx` != {expected}, got {actual}"

    def _assert_next_ms_idx(self):
        expected = 0
        actual = self.state["next_ms_idx"]
        assert actual == expected, f"`next_ms_idx` != {expected}, got {actual}"

    def _assert_next_ms_idx_nms(self):
        actual = self.state["next_ms_idx"]
        assert actual <= N_MS, f"`next_ms_idx` > `N_MS`: {actual} > {N_MS}"

    def _assert_next_ms_idx_all(self):
        expected = 0
        actual = self.state["next_ms_idx"]
        assert actual == expected, f"All microservices placed, but `next_ms_idx` != {expected}, got {actual}"

    # def _assert_node_idx(self, node_idxs, svc_idx):
    #     assert len(node_idxs) > 0, f"Service {svc_idx} not placed on any node"

    def _assert_node_claim(self, node):
        avail = self.state["node_feats"][node, :2]
        total = self.state["node_feats"][node, 2:4]
        assert np.allclose(
            avail, np.minimum(avail, total), atol=1e-6
        ), f"Node {node} available resources exceed total: {avail} > {total}"

    def _assert_link_claim(self, l_idx):
        avail = self.state["link_feats"][l_idx, 0]
        total = self.state["link_feats"][l_idx, 1]
        check = np.allclose(avail, np.minimum(avail, total), atol=1e-6)
        if not check:
            # f"Link {l_idx} available bandwidth exceeds total: {avail} > {total}"
            # TODO: a bad workaround for now
            self.state["link_feats"][l_idx, 0] = np.minimum(avail, total)

    def _assert_placement_all(self):
        assert np.all(self.state["placement_status"] >= 0), "Negative values found in placement status"

    def _assert_placed_nodes(self, placed_nodes, svc_duplicate_count):
        count = len(placed_nodes)
        assert count <= svc_duplicate_count, f"Service placed on too many nodes: {count} > {svc_duplicate_count}"

    def _assert_agent_idx(self, agent_idx: int):
        assert 0 <= agent_idx < self.num_agents, "Agent index out of bounds"
