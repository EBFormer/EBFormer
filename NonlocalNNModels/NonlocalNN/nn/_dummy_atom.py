import numpy as np
import torch.nn.functional
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, Irreps, FullyConnectedTensorProduct
from e3nn.o3 import spherical_harmonics

from torch import nn

from NonlocalNN._keys import DUMMY_NODE_FEATURES_KEY, DUMMY_EDGE_FEATURES_KEY
from NonlocalNN.nn._layer_norm import EquivariantLayerNorm
from nequip.data import AtomicDataDict
from nequip.nn._graph_mixin import GraphModuleMixin


class BesselBasisNoCutoff(nn.Module):
    r_max: float

    def __init__(self, num_basis=8, trainable=True, eps=1e-6):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123

        Parameters
        ----------
        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasisNoCutoff, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * torch.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1))

        return numerator / (x.unsqueeze(-1) + self.eps)


class DummyNodes(GraphModuleMixin, torch.nn.Module):
    """
    Includes virtual nodes in the graph that keep track of global features.
    NOTE! THIS CLASS IS WRITTEN ONLY FOR NANOSLABS WITH NORMAL DIRECTION Z!!
    """
    def __init__(
        self,
        irreps_in,
        irreps_dummy_nodes=None,
        irreps_query=None,
        irreps_key=None,
        bessel_basis_size=None,
        bessel_trainable=True,
        conditional_weight_generator_hidden=8,
        distance_from_surface=5.0, # Angstroms
        normalization=None,
    ):
        # Assume that the bottom and top global features are already included in the input irreps
        super().__init__()

        if irreps_query is None:
            irreps_query = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        if irreps_key is None:
            irreps_key = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]

        if irreps_dummy_nodes is None:
            irreps_dummy_nodes = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]

        if bessel_basis_size is None:
            bessel_basis_size = 8
        self.bessel_basis = BesselBasisNoCutoff(num_basis=bessel_basis_size, trainable=bessel_trainable)

        # TODO: Make the number of dummies dynamic
        irreps_out = irreps_in.copy()
        irreps_out[DUMMY_NODE_FEATURES_KEY] = irreps_dummy_nodes
        irreps_out[DUMMY_EDGE_FEATURES_KEY] = Irreps(f'{bessel_basis_size}x0e')

        # Query posed by the dummy node
        self.query_dummy_linear = Linear(irreps_dummy_nodes, irreps_query)
        # Key weight-generating matrix
        self.key_dummy_linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], irreps_key, internal_weights=False, shared_weights=False)
        self.key_dummy_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.key_dummy_linear.weight_numel], act=torch.nn.functional.silu)

        # Value weight-generating matrix
        self.value_dummy_linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], irreps_dummy_nodes, internal_weights=False, shared_weights=False)
        self.value_dummy_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.value_dummy_linear.weight_numel], act=torch.nn.functional.silu)

        # Query posed by the graph nodes
        self.query_graph_linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], irreps_query)
        # Key weight-generating matrix
        self.key_graph_linear = Linear(irreps_dummy_nodes, irreps_key, internal_weights=False, shared_weights=False)
        self.key_graph_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.key_graph_linear.weight_numel], act=torch.nn.functional.silu)
        # Value weight-generating matrix
        self.value_graph_linear = Linear(irreps_dummy_nodes, irreps_in[AtomicDataDict.NODE_FEATURES_KEY], internal_weights=False, shared_weights=False)
        self.value_graph_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.value_graph_linear.weight_numel], act=torch.nn.functional.silu)

        # Dot product between the query and key
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key

        self.dot = FullyConnectedTensorProduct(irreps_query, irreps_key, Irreps('0e'))

        # Distance from the surface of each dummy node
        self.distance_from_surface = distance_from_surface
        # Initial dummy node embedding
        self._init_dummy_node_embedding = nn.Parameter(torch.randn(irreps_dummy_nodes.dim))

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.POSITIONS_KEY
            ],
            irreps_out=irreps_out,
        )

        if normalization is None:
            self.node_normalization = lambda x: x
            self.dummy_normalization = lambda x: x
        else:
            self.node_normalization = EquivariantLayerNorm(irreps_in=irreps_in[AtomicDataDict.NODE_FEATURES_KEY], mode=normalization)
            self.dummy_normalization = EquivariantLayerNorm(irreps_in=irreps_dummy_nodes, mode=normalization)

    def _update_dummy_edge_features(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Update the dummy edge features in the graph. Use a Bessel basis for the dummy edge features.

        data[DUMMY_EDGE_FEATURES_KEY] should be a tensor of shape (n_edges, n_features, n_dummies (i.e. n_cleave_planes))
        """

        # Check if the dummy edge features are already included in the data
        if DUMMY_EDGE_FEATURES_KEY in data:
            # If they are, return the data as is
            return data

        z_positions = data[AtomicDataDict.POSITIONS_KEY][:, 2]
        device = z_positions.device

        # Otherwise, calculate the dummy edge features
        if AtomicDataDict.BATCH_KEY in data:
            batch_indices = data[AtomicDataDict.BATCH_KEY]
            n_batches = torch.max(batch_indices).detach().cpu().numpy() + 1

            z_min = torch.scatter_reduce(input=torch.full(size=(n_batches,),
                                                          fill_value=float('inf'),
                                                          dtype=z_positions.dtype,
                                                          device=device),
                                         dim=0, index=batch_indices, src=z_positions, reduce='amin')
            z_max = torch.scatter_reduce(input=torch.full(size=(n_batches,),
                                                          fill_value=-float('inf'),
                                                          dtype=z_positions.dtype,
                                                          device=device),
                                         dim=0, index=batch_indices, src=z_positions, reduce='amax')

            # Calculate dummy edge lengths (should be only the distance in the normal direction to the confinement)
            bottom_dummy_edge_lengths = z_positions - z_min[batch_indices] + self.distance_from_surface
            top_dummy_edge_lengths = z_max[batch_indices] + self.distance_from_surface - z_positions
        else:
            z_min = torch.min(z_positions)
            z_max = torch.max(z_positions)

            # Calculate dummy edge lengths (should be only the distance in the normal direction to the confinement)
            bottom_dummy_edge_lengths = z_positions - z_min + self.distance_from_surface
            top_dummy_edge_lengths = z_max + self.distance_from_surface - z_positions

        # Embed the dummy edge lengths using a Bessel basis (technically sinc)
        bottom_dummy_edge_features = self.bessel_basis(bottom_dummy_edge_lengths)
        top_dummy_edge_features = self.bessel_basis(top_dummy_edge_lengths)
        # Include the dummy edge lengths in the data (N_atoms, n_dummies, n_features)
        data[DUMMY_EDGE_FEATURES_KEY] = torch.cat([bottom_dummy_edge_features.unsqueeze(1), top_dummy_edge_features.unsqueeze(1)],
                                                    dim=1)

        return data

    def _update_dummy_nodes(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Update the dummy nodes in the graph. Uses an attention mechanism per dummy node to collect features from
        every node in the graph.
        """

        if DUMMY_NODE_FEATURES_KEY not in data:
            # Initialize the dummy node embeddings with constant learned embeddings
            # TODO: Make this general to different number of dummy nodes
            if AtomicDataDict.BATCH_KEY in data:
                n_batches = torch.max(data[AtomicDataDict.BATCH_KEY]).detach().cpu().numpy() + 1
                # (n_batches, n_dummies, n_features)
                data[DUMMY_NODE_FEATURES_KEY] = self._init_dummy_node_embedding.unsqueeze(0).repeat(n_batches, 2, 1)
            else:
                # (n_dummies, n_features)
                data[DUMMY_NODE_FEATURES_KEY] = self._init_dummy_node_embedding.unsqueeze(0).repeat(2, 1)

        device = data[DUMMY_NODE_FEATURES_KEY].device
        if AtomicDataDict.BATCH_KEY in data:
            batch_indices = data[AtomicDataDict.BATCH_KEY]
            n_batches = torch.max(batch_indices).detach().cpu().numpy() + 1

            dummy_features = data[DUMMY_NODE_FEATURES_KEY]
            node_features = data[AtomicDataDict.NODE_FEATURES_KEY]
            dummy_edge_embeddings = data[DUMMY_EDGE_FEATURES_KEY]

            # Normalize the dummy features and node features (pre-LN GPT-style normalization)
            norm_dummy_features = self.dummy_normalization(dummy_features)
            norm_node_features = self.node_normalization(node_features)

            dummy_q = self.query_dummy_linear(norm_dummy_features)
            bottom_dummy_q, top_dummy_q = dummy_q[:, 0, :], dummy_q[:, 1, :]
            bottom_dummy_queries = bottom_dummy_q[batch_indices]
            top_dummy_queries = top_dummy_q[batch_indices]

            bottom_dummy_k = self.key_dummy_linear(norm_node_features, self.key_dummy_weight_generator(dummy_edge_embeddings)[:, 0, :])
            top_dummy_k = self.key_dummy_linear(norm_node_features, self.key_dummy_weight_generator(dummy_edge_embeddings)[:, 1, :])

            bottom_dummy_v = self.value_dummy_linear(norm_node_features, self.value_dummy_weight_generator(dummy_edge_embeddings)[:, 0, :])
            top_dummy_v = self.value_dummy_linear(norm_node_features, self.value_dummy_weight_generator(dummy_edge_embeddings)[:, 1, :])

            # Calculate the attention weights
            # Scaling by the dimension of the query as Vaswani et al. (2017)
            # Choose the larger dimensions to scale by
            scale_dim = np.max((self.irreps_query.dim, self.irreps_key.dim))
            bottom_attention_scores = self.dot(bottom_dummy_queries, bottom_dummy_k) / np.sqrt(scale_dim)
            top_attention_scores = self.dot(top_dummy_queries, top_dummy_k) / np.sqrt(scale_dim)

            # Implement a stable softmax by subtracting the maximum value
            # bottom_max_val = torch.max(bottom_attention_scores)
            # top_max_val = torch.max(top_attention_scores)

            bottom_max_vals = torch.scatter_reduce(torch.zeros((n_batches, 1), device=device, dtype=bottom_attention_scores.dtype),
                                                   index=batch_indices.unsqueeze(1), src=bottom_attention_scores, dim=0, reduce='amax')
            top_max_vals = torch.scatter_reduce(torch.zeros((n_batches, 1), device=device, dtype=top_attention_scores.dtype),
                                                index=batch_indices.unsqueeze(1), src=top_attention_scores, dim=0, reduce='amax')

            bottom_attention_scores_scaled = bottom_attention_scores - bottom_max_vals[batch_indices]
            top_attention_scores_scaled = top_attention_scores - top_max_vals[batch_indices]

            bottom_exp = torch.exp(bottom_attention_scores_scaled)
            top_exp = torch.exp(top_attention_scores_scaled)

            bottom_z = torch.scatter_add(torch.zeros((n_batches, 1), device=device, dtype=bottom_exp.dtype),
                                         index=batch_indices.unsqueeze(1), src=bottom_exp, dim=0)
            top_z = torch.scatter_add(torch.zeros((n_batches, 1), device=device, dtype=top_exp.dtype),
                                            index=batch_indices.unsqueeze(1), src=top_exp, dim=0)

            bottom_attention_weights = bottom_exp / bottom_z[batch_indices]
            top_attention_weights = top_exp / top_z[batch_indices]

            # Calculate the weighted sum of the values
            bottom_weighted_values = bottom_attention_weights * bottom_dummy_v
            top_weighted_values = top_attention_weights * top_dummy_v

            # Linear layer to update the dummy node features


            # Sum back into the dummy node features (with skip connection)
            bottom_updated_features = torch.scatter_add(input=dummy_features[:, 0, :], index=batch_indices.unsqueeze(1).repeat(1, dummy_features.shape[2]),
                                                            src=bottom_weighted_values, dim=0)
            top_updated_features = torch.scatter_add(input=dummy_features[:, 1, :], index=batch_indices.unsqueeze(1).repeat(1, dummy_features.shape[2]),
                                                            src=top_weighted_values, dim=0)

            # Update the dummy node features
            data[DUMMY_NODE_FEATURES_KEY] = torch.cat([bottom_updated_features.unsqueeze(1), top_updated_features.unsqueeze(1)], dim=1)
        else:
            raise NotImplementedError

        return data

    def _update_graph_nodes(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Update the graph nodes in the graph.
        """

        if AtomicDataDict.BATCH_KEY in data:
            batch_indices = data[AtomicDataDict.BATCH_KEY]

            node_features = data[AtomicDataDict.NODE_FEATURES_KEY]
            dummy_features = data[DUMMY_NODE_FEATURES_KEY]
            dummy_edge_embeddings = data[DUMMY_EDGE_FEATURES_KEY]

            # Normalize the dummy features and node features (pre-LN GPT-style normalization)
            norm_dummy_features = self.dummy_normalization(dummy_features)
            norm_node_features = self.node_normalization(node_features)

            graph_q = self.query_graph_linear(norm_node_features)

            bottom_graph_k = self.key_graph_linear(norm_dummy_features[batch_indices, 0, :], self.key_graph_weight_generator(dummy_edge_embeddings[:, 0, :]))
            top_graph_k = self.key_graph_linear(norm_dummy_features[batch_indices, 1, :], self.key_graph_weight_generator(dummy_edge_embeddings[:, 1, :]))

            bottom_graph_v = self.value_graph_linear(norm_dummy_features[batch_indices, 0, :], self.value_graph_weight_generator(dummy_edge_embeddings[:, 0, :]))
            top_graph_v = self.value_graph_linear(norm_dummy_features[batch_indices, 1, :], self.value_graph_weight_generator(dummy_edge_embeddings[:, 1, :]))

            # Calculate the attention weights
            # Scaling by the dimension of the query as Vaswani et al. (2017)
            # Choose the larger dimensions to scale by
            scale_dim = np.max((self.irreps_query.dim, self.irreps_key.dim))
            bottom_attention_scores = self.dot(graph_q, bottom_graph_k) / np.sqrt(scale_dim)
            top_attention_scores = self.dot(graph_q, top_graph_k) / np.sqrt(scale_dim)

            # Implement a stable softmax by subtracting the maximum value
            max_val = torch.max(bottom_attention_scores, top_attention_scores)
            bottom_attention_scores_scaled = bottom_attention_scores - max_val
            top_attention_scores_scaled = top_attention_scores - max_val

            bottom_exp, top_exp = torch.exp(bottom_attention_scores_scaled), torch.exp(top_attention_scores_scaled)
            z = bottom_exp + top_exp
            bottom_norm_score = bottom_exp / z
            top_norm_score = top_exp / z

            # Update the node features (with a residual connection)
            data[AtomicDataDict.NODE_FEATURES_KEY] += (bottom_norm_score * bottom_graph_v) + (top_norm_score * top_graph_v)
        else:
            raise NotImplementedError

        return data

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Forward pass through the dummy nodes.
        """
        data = self._update_dummy_edge_features(data)
        data = self._update_dummy_nodes(data)
        data = self._update_graph_nodes(data)

        return data


class DirectionalDummyNodes(GraphModuleMixin, torch.nn.Module):
    """
    Includes virtual nodes in the graph that keep track of global features with higher-order spherical harmonics.
    NOTE! THIS CLASS IS WRITTEN ONLY FOR NANOSLABS WITH NORMAL DIRECTION Z!!
    """
    def __init__(
        self,
        irreps_in,
        irreps_dummy_nodes=None,
        irreps_query=None,
        irreps_key=None,
        bessel_basis_size=None,
        bessel_trainable=True,
        spherical_harmonics_lmax=0,
        conditional_weight_generator_hidden=8,
        distance_from_surface=5.0, # Angstroms
        normalization=None,
    ):
        # Assume that the bottom and top global features are already included in the input irreps
        super().__init__()

        if irreps_query is None:
            irreps_query = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        if irreps_key is None:
            irreps_key = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]

        if irreps_dummy_nodes is None:
            irreps_dummy_nodes = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]

        if bessel_basis_size is None:
            bessel_basis_size = 8
        self.bessel_basis = BesselBasisNoCutoff(num_basis=bessel_basis_size, trainable=bessel_trainable)

        # TODO: Make the number of dummies dynamic
        irreps_out = irreps_in.copy()
        irreps_out[DUMMY_NODE_FEATURES_KEY] = irreps_dummy_nodes
        irreps_out[DUMMY_EDGE_FEATURES_KEY] = Irreps(f'{bessel_basis_size}x0e')
        irreps_out['dummy_edge_sh'] = Irreps.spherical_harmonics(spherical_harmonics_lmax, p=1)

        self.irreps_sh = irreps_out['dummy_edge_sh']

        # Query posed by the dummy node
        self.query_dummy_linear = Linear(irreps_dummy_nodes, irreps_query)
        # Key weight-generating matrix
        self.key_dummy_linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], irreps_key, internal_weights=False, shared_weights=False)
        self.key_dummy_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.key_dummy_linear.weight_numel], act=torch.nn.functional.silu)

        # Value weight-generating matrix
        self.value_dummy_linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], irreps_dummy_nodes, internal_weights=False, shared_weights=False)
        self.value_dummy_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.value_dummy_linear.weight_numel], act=torch.nn.functional.silu)

        # Query posed by the graph nodes
        self.query_graph_linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], irreps_query)
        # Key weight-generating matrix
        self.key_graph_tp = FullyConnectedTensorProduct(irreps_dummy_nodes, self.irreps_sh, irreps_key, shared_weights=False)
        self.key_graph_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.key_graph_tp.weight_numel], act=torch.nn.functional.silu)
        # Value weight-generating matrix
        self.value_graph_tp = FullyConnectedTensorProduct(irreps_dummy_nodes, self.irreps_sh, irreps_in[AtomicDataDict.NODE_FEATURES_KEY], shared_weights=False)
        self.value_graph_weight_generator = FullyConnectedNet([bessel_basis_size, conditional_weight_generator_hidden,
                                                             self.value_graph_tp.weight_numel], act=torch.nn.functional.silu)

        # Dot product between the query and key
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key

        self.dot = FullyConnectedTensorProduct(irreps_query, irreps_key, Irreps('0e'))

        # Distance from the surface of each dummy node
        self.distance_from_surface = distance_from_surface
        # Initial dummy node embedding
        self._init_dummy_node_embedding = nn.Parameter(torch.randn(irreps_dummy_nodes.dim))

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.POSITIONS_KEY
            ],
            irreps_out=irreps_out,
        )

        if normalization is None:
            self.node_normalization = lambda x: x
            self.dummy_normalization = lambda x: x
        else:
            self.node_normalization = EquivariantLayerNorm(irreps_in=irreps_in[AtomicDataDict.NODE_FEATURES_KEY], mode=normalization)
            self.dummy_normalization = EquivariantLayerNorm(irreps_in=irreps_dummy_nodes, mode=normalization)

    def _update_dummy_edge_features(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Update the dummy edge features in the graph. Use a Bessel basis for the dummy edge features.

        data[DUMMY_EDGE_FEATURES_KEY] should be a tensor of shape (n_edges, n_features, n_dummies (i.e. n_cleave_planes))
        """

        # Check if the dummy edge features are already included in the data
        if DUMMY_EDGE_FEATURES_KEY in data:
            # If they are, return the data as is
            return data

        atomic_positions = data[AtomicDataDict.POSITIONS_KEY]

        z_positions = atomic_positions[:, 2]
        device = z_positions.device

        # Otherwise, calculate the dummy edge features
        if AtomicDataDict.BATCH_KEY in data:
            batch_indices = data[AtomicDataDict.BATCH_KEY]
            n_batches = torch.max(batch_indices).detach().cpu().numpy() + 1

            z_min = torch.scatter_reduce(input=torch.full(size=(n_batches,),
                                                          fill_value=float('inf'),
                                                          dtype=z_positions.dtype,
                                                          device=device),
                                         dim=0, index=batch_indices, src=z_positions, reduce='amin')
            z_max = torch.scatter_reduce(input=torch.full(size=(n_batches,),
                                                          fill_value=-float('inf'),
                                                          dtype=z_positions.dtype,
                                                          device=device),
                                         dim=0, index=batch_indices, src=z_positions, reduce='amax')

            # Calculate dummy edge lengths (should be only the distance in the normal direction to the confinement)
            bottom_dummy_edge_lengths = z_positions - z_min[batch_indices] + self.distance_from_surface
            top_dummy_edge_lengths = z_max[batch_indices] + self.distance_from_surface - z_positions
        else:
            z_min = torch.min(z_positions)
            z_max = torch.max(z_positions)

            # Calculate dummy edge lengths (should be only the distance in the normal direction to the confinement)
            bottom_dummy_edge_lengths = z_positions - z_min + self.distance_from_surface
            top_dummy_edge_lengths = z_max + self.distance_from_surface - z_positions

        # Create vectors from atoms pointing to the dummy nodes
        bottom_dummy_edge_vectors = torch.zeros_like(atomic_positions)
        bottom_dummy_edge_vectors[:, 2] = -bottom_dummy_edge_lengths # negative because the bottom dummy node is below
        bottom_dummy_edge_sh = spherical_harmonics(self.irreps_sh, bottom_dummy_edge_vectors, True, normalization='component')
        top_dummy_edge_vectors = torch.zeros_like(atomic_positions)
        top_dummy_edge_vectors[:, 2] = top_dummy_edge_lengths
        top_dummy_edge_sh = spherical_harmonics(self.irreps_sh, top_dummy_edge_vectors, True, normalization='component')

        # Embed the dummy edge lengths using a Bessel basis (technically sinc)
        bottom_dummy_edge_features = self.bessel_basis(bottom_dummy_edge_lengths)
        top_dummy_edge_features = self.bessel_basis(top_dummy_edge_lengths)
        # Include the dummy edge lengths in the data (N_atoms, n_dummies, n_features)
        data[DUMMY_EDGE_FEATURES_KEY] = torch.cat([bottom_dummy_edge_features.unsqueeze(1), top_dummy_edge_features.unsqueeze(1)],
                                                    dim=1)
        data['dummy_edge_sh'] = torch.cat([bottom_dummy_edge_sh.unsqueeze(2), top_dummy_edge_sh.unsqueeze(2)],
                                                    dim=2)

        return data

    def _update_dummy_nodes(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Update the dummy nodes in the graph. Uses an attention mechanism per dummy node to collect features from
        every node in the graph.
        """

        if DUMMY_NODE_FEATURES_KEY not in data:
            # Initialize the dummy node embeddings with constant learned embeddings
            # TODO: Make this general to different number of dummy nodes
            if AtomicDataDict.BATCH_KEY in data:
                n_batches = torch.max(data[AtomicDataDict.BATCH_KEY]).detach().cpu().numpy() + 1
                # (n_batches, n_dummies, n_features)
                data[DUMMY_NODE_FEATURES_KEY] = self._init_dummy_node_embedding.unsqueeze(0).repeat(n_batches, 2, 1)
            else:
                # (n_dummies, n_features)
                data[DUMMY_NODE_FEATURES_KEY] = self._init_dummy_node_embedding.unsqueeze(0).repeat(2, 1)

        device = data[DUMMY_NODE_FEATURES_KEY].device
        if AtomicDataDict.BATCH_KEY in data:
            batch_indices = data[AtomicDataDict.BATCH_KEY]
            n_batches = torch.max(batch_indices).detach().cpu().numpy() + 1

            dummy_features = data[DUMMY_NODE_FEATURES_KEY]
            node_features = data[AtomicDataDict.NODE_FEATURES_KEY]
            dummy_edge_embeddings = data[DUMMY_EDGE_FEATURES_KEY]

            # Normalize the dummy features and node features (pre-LN GPT-style normalization)
            norm_dummy_features = self.dummy_normalization(dummy_features)
            norm_node_features = self.node_normalization(node_features)

            dummy_q = self.query_dummy_linear(norm_dummy_features)
            bottom_dummy_q, top_dummy_q = dummy_q[:, 0, :], dummy_q[:, 1, :]
            bottom_dummy_queries = bottom_dummy_q[batch_indices]
            top_dummy_queries = top_dummy_q[batch_indices]

            bottom_dummy_k = self.key_dummy_linear(norm_node_features, self.key_dummy_weight_generator(dummy_edge_embeddings)[:, 0, :])
            top_dummy_k = self.key_dummy_linear(norm_node_features, self.key_dummy_weight_generator(dummy_edge_embeddings)[:, 1, :])

            bottom_dummy_v = self.value_dummy_linear(norm_node_features, self.value_dummy_weight_generator(dummy_edge_embeddings)[:, 0, :])
            top_dummy_v = self.value_dummy_linear(norm_node_features, self.value_dummy_weight_generator(dummy_edge_embeddings)[:, 1, :])

            # Calculate the attention weights
            # Scaling by the dimension of the query as Vaswani et al. (2017)
            # Choose the larger dimensions to scale by
            scale_dim = np.max((self.irreps_query.dim, self.irreps_key.dim))
            bottom_attention_scores = self.dot(bottom_dummy_queries, bottom_dummy_k) / np.sqrt(scale_dim)
            top_attention_scores = self.dot(top_dummy_queries, top_dummy_k) / np.sqrt(scale_dim)

            # Implement a stable softmax by subtracting the maximum value
            bottom_max_vals = torch.scatter_reduce(torch.zeros((n_batches, 1), device=device, dtype=bottom_attention_scores.dtype),
                                                   index=batch_indices.unsqueeze(1), src=bottom_attention_scores, dim=0, reduce='amax')
            top_max_vals = torch.scatter_reduce(torch.zeros((n_batches, 1), device=device, dtype=top_attention_scores.dtype),
                                                index=batch_indices.unsqueeze(1), src=top_attention_scores, dim=0, reduce='amax')

            bottom_attention_scores_scaled = bottom_attention_scores - bottom_max_vals[batch_indices]
            top_attention_scores_scaled = top_attention_scores - top_max_vals[batch_indices]

            bottom_exp = torch.exp(bottom_attention_scores_scaled)
            top_exp = torch.exp(top_attention_scores_scaled)

            bottom_z = torch.scatter_add(torch.zeros((n_batches, 1), device=device, dtype=bottom_exp.dtype),
                                         index=batch_indices.unsqueeze(1), src=bottom_exp, dim=0)
            top_z = torch.scatter_add(torch.zeros((n_batches, 1), device=device, dtype=top_exp.dtype),
                                            index=batch_indices.unsqueeze(1), src=top_exp, dim=0)

            bottom_attention_weights = bottom_exp / bottom_z[batch_indices]
            top_attention_weights = top_exp / top_z[batch_indices]

            # Calculate the weighted sum of the values
            bottom_weighted_values = bottom_attention_weights * bottom_dummy_v
            top_weighted_values = top_attention_weights * top_dummy_v

            # Sum back into the dummy node features (with skip connection)
            bottom_updated_features = torch.scatter_add(input=dummy_features[:, 0, :], index=batch_indices.unsqueeze(1).repeat(1, dummy_features.shape[2]),
                                                            src=bottom_weighted_values, dim=0)
            top_updated_features = torch.scatter_add(input=dummy_features[:, 1, :], index=batch_indices.unsqueeze(1).repeat(1, dummy_features.shape[2]),
                                                            src=top_weighted_values, dim=0)

            # Update the dummy node features
            data[DUMMY_NODE_FEATURES_KEY] = torch.cat([bottom_updated_features.unsqueeze(1), top_updated_features.unsqueeze(1)], dim=1)
        else:
            raise NotImplementedError

        return data

    def _update_graph_nodes(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Update the graph nodes in the graph.
        """

        if AtomicDataDict.BATCH_KEY in data:
            batch_indices = data[AtomicDataDict.BATCH_KEY]

            node_features = data[AtomicDataDict.NODE_FEATURES_KEY]
            dummy_features = data[DUMMY_NODE_FEATURES_KEY]
            dummy_edge_embeddings = data[DUMMY_EDGE_FEATURES_KEY]
            dummy_edge_sh = data['dummy_edge_sh']

            # Normalize the dummy features and node features (pre-LN GPT-style normalization)
            norm_dummy_features = self.dummy_normalization(dummy_features)
            norm_node_features = self.node_normalization(node_features)

            graph_q = self.query_graph_linear(norm_node_features)

            bottom_graph_k = self.key_graph_tp(norm_dummy_features[batch_indices, 0, :], dummy_edge_sh[:, :, 0],
                                               self.key_graph_weight_generator(dummy_edge_embeddings[:, 0, :]))
            top_graph_k = self.key_graph_tp(norm_dummy_features[batch_indices, 1, :], dummy_edge_sh[:, :, 1],
                                            self.key_graph_weight_generator(dummy_edge_embeddings[:, 1, :]))

            bottom_graph_v = self.value_graph_tp(norm_dummy_features[batch_indices, 0, :], dummy_edge_sh[:, :, 0],
                                                 self.value_graph_weight_generator(dummy_edge_embeddings[:, 0, :]))
            top_graph_v = self.value_graph_tp(norm_dummy_features[batch_indices, 1, :], dummy_edge_sh[:, :, 1],
                                              self.value_graph_weight_generator(dummy_edge_embeddings[:, 1, :]))

            # Calculate the attention weights
            # Scaling by the dimension of the query as Vaswani et al. (2017)
            # Choose the larger dimensions to scale by
            scale_dim = np.max((self.irreps_query.dim, self.irreps_key.dim))
            bottom_attention_scores = self.dot(graph_q, bottom_graph_k) / np.sqrt(scale_dim)
            top_attention_scores = self.dot(graph_q, top_graph_k) / np.sqrt(scale_dim)

            # Implement a stable softmax by subtracting the maximum value
            max_val = torch.max(bottom_attention_scores, top_attention_scores)
            bottom_attention_scores_scaled = bottom_attention_scores - max_val
            top_attention_scores_scaled = top_attention_scores - max_val

            bottom_exp, top_exp = torch.exp(bottom_attention_scores_scaled), torch.exp(top_attention_scores_scaled)
            z = bottom_exp + top_exp
            bottom_norm_score = bottom_exp / z
            top_norm_score = top_exp / z

            # Update the node features (with a residual connection)
            data[AtomicDataDict.NODE_FEATURES_KEY] += (bottom_norm_score * bottom_graph_v) + (top_norm_score * top_graph_v)
        else:
            raise NotImplementedError

        return data

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Forward pass through the dummy nodes.
        """
        data = self._update_dummy_edge_features(data)
        data = self._update_dummy_nodes(data)
        data = self._update_graph_nodes(data)

        return data
