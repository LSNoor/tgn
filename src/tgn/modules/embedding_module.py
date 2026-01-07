from tgn.model import TemporalAttentionLayer

import torch
from torch import nn
import numpy as np
import math


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        memory,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_features,
        n_edge_features,
        n_time_features,
        embedding_dimension,
        device,
        dropout,
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features: int = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(
        self,
        memory,
        source_nodes,
        timestamps,
        n_layers,
        n_neighbors=20,
        time_diffs=None,
        use_time_proj=True,
    ):
        return NotImplemented


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(
        self,
        memory,
        source_nodes,
        timestamps,
        n_layers,
        n_neighbors=20,
        time_diffs=None,
        use_time_proj=True,
    ):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(
        self,
        node_features,
        edge_features,
        memory,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_features,
        n_edge_features,
        n_time_features,
        embedding_dimension,
        device,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        n_neighbors=1,
    ):
        super().__init__(
            node_features,
            edge_features,
            memory,
            neighbor_finder,
            time_encoder,
            n_layers,
            n_node_features,
            n_edge_features,
            n_time_features,
            embedding_dimension,
            device,
            dropout,
        )

        import torch.nn.init as init

        # Modernized the code here
        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.in_features)

                init.normal_(self.weight, mean=0.0, std=stdv)

                if self.bias is not None:
                    init.normal_(self.bias, mean=0.0, std=stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(
        self,
        memory,
        source_nodes,
        timestamps,
        n_layers,
        n_neighbors=20,
        time_diffs=None,
        use_time_proj=True,
    ):
        source_embeddings = memory[source_nodes, :] * (
            1 + self.embedding_layer(time_diffs.unsqueeze(1))
        )

        return source_embeddings


class GraphEmbedding(EmbeddingModule):
    def __init__(
        self,
        node_features,
        edge_features,
        memory,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_features,
        n_edge_features,
        n_time_features,
        embedding_dimension,
        device,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
    ):
        super().__init__(
            node_features,
            edge_features,
            memory,
            neighbor_finder,
            time_encoder,
            n_layers,
            n_node_features,
            n_edge_features,
            n_time_features,
            embedding_dimension,
            device,
            dropout,
        )

        self.use_memory = use_memory
        self.device = device

    def compute_embedding(
        self,
        memory,
        source_nodes,
        timestamps,
        n_layers,
        n_neighbors=20,
        time_diffs=None,
        use_time_proj=True,
    ):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert n_layers >= 0

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(
            torch.from_numpy(timestamps).float().to(self.device), dim=1
        )

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(
            torch.zeros_like(timestamps_torch)
        )

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            source_node_conv_embeddings = self.compute_embedding(
                memory,
                source_nodes,
                timestamps,
                n_layers=n_layers - 1,
                n_neighbors=n_neighbors,
            )

            neighbors, edge_idxs, edge_times = (
                self.neighbor_finder.get_temporal_neighbor(
                    source_nodes, timestamps, n_neighbors=n_neighbors
                )
            )

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(
                memory,
                neighbors,
                np.repeat(timestamps, n_neighbors),
                n_layers=n_layers - 1,
                n_neighbors=n_neighbors,
            )

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(
                len(source_nodes), effective_n_neighbors, -1
            )
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(
                n_layers,
                source_node_conv_embeddings,
                source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings,
                edge_features,
                mask,
            )

            return source_embedding

    def aggregate(
        self,
        n_layers,
        source_node_features,
        source_nodes_time_embedding,
        neighbor_embeddings,
        edge_time_embeddings,
        edge_features,
        mask,
    ):
        return NotImplemented
