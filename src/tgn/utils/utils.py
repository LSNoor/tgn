from typing import Optional

from numpy import int64

from .NeighborFinder import NeighborFinder
from .data_processing import Data


def get_neighbor_finder(
    data: Data, uniform: bool, max_node_idx: Optional[int64] = None
):
    max_node_idx = (
        max(data.sources.max(), data.destinations.max())
        if max_node_idx is None
        else max_node_idx
    )
    # The graph is encoded in the adjacency list this way
    # adj_list[source_list_idx] = [(destination_node_idx, edge_idx, timestamp) ... ]
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)
