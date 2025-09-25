from typing import Dict, Callable

import torch

from NonlocalNN.nn._dummy_atom import DummyNodes, DirectionalDummyNodes, NanowireDirectionalDummyNodes, NanowireDummyNodes
from nequip.data import AtomicDataDict
from nequip.nn import (
    InteractionBlock, ConvNetLayer, GraphModuleMixin,
)

from e3nn.o3 import Linear

class NxConvNetLayersUnsharedWithDummyAtomsNanowire(GraphModuleMixin, torch.nn.Module):
    """
        NxConvNetLayers is a neural network module that applies a series of identical convolutional layers to the input data.
        This version includes dummy atoms to keep track of global features.

        Args:
            irreps_in: Dictionary of input irreducible representations.
            feature_irreps_hidden: Irreducible representations for the hidden features.
            n_convolutions: Number of convolutional layers to apply.
            convolution: Convolutional layer class to use.
            convolution_kwargs: Additional arguments for the convolutional layer.
            num_layers: Number of layers in each convolutional block.
            resnet: Whether to use residual connections.
            nonlinearity_type: Type of nonlinearity to use.
            nonlinearity_scalars: Dictionary of scalar nonlinearities.
            nonlinearity_gates: Dictionary of gate nonlinearities.
        """

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        n_convolutions: int,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        irreps_key=None,
        irreps_query=None,
        irreps_dummy_nodes=None,
        normalization=None,
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        # Translates input feature size to hidden feature size so we can use a single convnet of the same dimension
        # N times (translates input to output feature size)
        self.linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], feature_irreps_hidden)

        # Convolutional layer input feature size is the hidden feature size
        conv_irreps_in = irreps_in.copy()
        conv_irreps_in[AtomicDataDict.NODE_FEATURES_KEY] = feature_irreps_hidden

        # Create a separate convnet for each convolution
        # This is done by creating a list of convnets
        self.n_convolutions = n_convolutions
        self.convnets = torch.nn.ModuleList()
        for i in range(n_convolutions):
            self.convnets.append(ConvNetLayer(irreps_in=conv_irreps_in,
                         feature_irreps_hidden=feature_irreps_hidden,
                         convolution=convolution,
                         convolution_kwargs=convolution_kwargs,
                         num_layers=num_layers,
                         resnet=resnet,
                         nonlinearity_type=nonlinearity_type,
                         nonlinearity_scalars=nonlinearity_scalars,
                         nonlinearity_gates=nonlinearity_gates))

        # Add dummy nodes to keep track of global features
        self.dummy_layer = NanowireDummyNodes(
            irreps_in=self.convnets[-1].irreps_out,
            irreps_dummy_nodes=irreps_dummy_nodes,
            irreps_query=irreps_query,
            irreps_key=irreps_key,
            normalization=normalization
        )

        # Update the output irreps based on the convnet output
        self.irreps_out.update(self.dummy_layer.irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Translates input feature size to hidden feature size
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.linear(data[AtomicDataDict.NODE_FEATURES_KEY])

        # Apply the convnet n times
        for module in self.convnets:
            data = module(data)
            data = self.dummy_layer(data)
        return data

lass NxConvNetLayersUnsharedWithDirectionalDummyAtomsNanowire(GraphModuleMixin, torch.nn.Module):
    """
        NxConvNetLayers is a neural network module that applies a series of distinct convolutional layers to the input data.
        This version includes dummy atoms to keep track of global features.

        Args:
            irreps_in: Dictionary of input irreducible representations.
            feature_irreps_hidden: Irreducible representations for the hidden features.
            n_convolutions: Number of convolutional layers to apply.
            convolution: Convolutional layer class to use.
            convolution_kwargs: Additional arguments for the convolutional layer.
            num_layers: Number of layers in each convolutional block.
            resnet: Whether to use residual connections.
            nonlinearity_type: Type of nonlinearity to use.
            nonlinearity_scalars: Dictionary of scalar nonlinearities.
            nonlinearity_gates: Dictionary of gate nonlinearities.
        """

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        n_convolutions: int,
        convolution=InteractionBlock,
        l_max: int = 0,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        irreps_key=None,
        irreps_query=None,
        irreps_dummy_nodes=None,
        normalization=None,
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        # Translates input feature size to hidden feature size
        self.linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], feature_irreps_hidden)

        # Convolutional layer input feature size is the hidden feature size
        conv_irreps_in = irreps_in.copy()
        conv_irreps_in[AtomicDataDict.NODE_FEATURES_KEY] = feature_irreps_hidden

        # Create a separate convnet for each convolution
        # This is done by creating a list of convnets
        self.n_convolutions = n_convolutions
        self.convnets = torch.nn.ModuleList()
        for i in range(n_convolutions):
            self.convnets.append(ConvNetLayer(irreps_in=conv_irreps_in,
                         feature_irreps_hidden=feature_irreps_hidden,
                         convolution=convolution,
                         convolution_kwargs=convolution_kwargs,
                         num_layers=num_layers,
                         resnet=resnet,
                         nonlinearity_type=nonlinearity_type,
                         nonlinearity_scalars=nonlinearity_scalars,
                         nonlinearity_gates=nonlinearity_gates))

        # Add dummy nodes to keep track of global features
        self.dummy_layer = NanowireDirectionalDummyNodes(
            irreps_in=self.convnets[-1].irreps_out,
            irreps_dummy_nodes=irreps_dummy_nodes,
            irreps_query=irreps_query,
            irreps_key=irreps_key,
            spherical_harmonics_lmax=l_max,
            normalization=normalization,
        )

        # Update the output irreps based on the convnet output
        self.irreps_out.update(self.dummy_layer.irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Translates input feature size to hidden feature size
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.linear(data[AtomicDataDict.NODE_FEATURES_KEY])

        # Apply the convnet n times
        for module in self.convnets:
            data = module(data)
            data = self.dummy_layer(data)
        return data

class NxConvNetLayers(GraphModuleMixin, torch.nn.Module):
    """
        NxConvNetLayers is a neural network module that applies a series of identical convolutional layers to the input data.

        Args:
            irreps_in: Dictionary of input irreducible representations.
            feature_irreps_hidden: Irreducible representations for the hidden features.
            n_convolutions: Number of convolutional layers to apply.
            convolution: Convolutional layer class to use.
            convolution_kwargs: Additional arguments for the convolutional layer.
            num_layers: Number of layers in each convolutional block.
            resnet: Whether to use residual connections.
            nonlinearity_type: Type of nonlinearity to use.
            nonlinearity_scalars: Dictionary of scalar nonlinearities.
            nonlinearity_gates: Dictionary of gate nonlinearities.
        """

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        n_convolutions: int,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        # Translates input feature size to hidden feature size so we can use a single convnet of the same dimension
        # N times (translates input to output feature size)
        self.linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], feature_irreps_hidden)

        # Convolutional layer input feature size is the hidden feature size
        conv_irreps_in = irreps_in.copy()
        conv_irreps_in[AtomicDataDict.NODE_FEATURES_KEY] = feature_irreps_hidden

        self.convnet = ConvNetLayer(irreps_in=conv_irreps_in,
                         feature_irreps_hidden=feature_irreps_hidden,
                         convolution=convolution,
                         convolution_kwargs=convolution_kwargs,
                         num_layers=num_layers,
                         resnet=resnet,
                         nonlinearity_type=nonlinearity_type,
                         nonlinearity_scalars=nonlinearity_scalars,
                         nonlinearity_gates=nonlinearity_gates)
        self.n_convolutions = n_convolutions

        # Update the output irreps based on the convnet output
        self.irreps_out.update(self.convnet.irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Translates input feature size to hidden feature size
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.linear(data[AtomicDataDict.NODE_FEATURES_KEY])

        # Apply the convnet n times
        for i in range(self.n_convolutions):
            data = self.convnet(data)
        return data

class NxConvNetLayersUnsharedWithDummyAtoms(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        n_convolutions: int,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        irreps_key=None,
        irreps_query=None,
        irreps_dummy_nodes=None,
        normalization=None,
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        # Translates input feature size to hidden feature size so we can use a single convnet of the same dimension
        # N times (translates input to output feature size)
        self.linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], feature_irreps_hidden)

        # Convolutional layer input feature size is the hidden feature size
        conv_irreps_in = irreps_in.copy()
        conv_irreps_in[AtomicDataDict.NODE_FEATURES_KEY] = feature_irreps_hidden

        # Create a separate convnet for each convolution
        # This is done by creating a list of convnets
        self.n_convolutions = n_convolutions
        self.convnets = torch.nn.ModuleList()
        for i in range(n_convolutions):
            self.convnets.append(ConvNetLayer(irreps_in=conv_irreps_in,
                         feature_irreps_hidden=feature_irreps_hidden,
                         convolution=convolution,
                         convolution_kwargs=convolution_kwargs,
                         num_layers=num_layers,
                         resnet=resnet,
                         nonlinearity_type=nonlinearity_type,
                         nonlinearity_scalars=nonlinearity_scalars,
                         nonlinearity_gates=nonlinearity_gates))

        # Add dummy nodes to keep track of global features
        self.dummy_layer = DummyNodes(
            irreps_in=self.convnets[-1].irreps_out,
            irreps_dummy_nodes=irreps_dummy_nodes,
            irreps_query=irreps_query,
            irreps_key=irreps_key,
            normalization=normalization
        )

        # Update the output irreps based on the convnet output
        self.irreps_out.update(self.dummy_layer.irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Translates input feature size to hidden feature size
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.linear(data[AtomicDataDict.NODE_FEATURES_KEY])

        # Apply the convnet n times
        for module in self.convnets:
            data = module(data)
            data = self.dummy_layer(data)
        return data


class NxConvNetLayersUnsharedWithDirectionalDummyAtoms(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        n_convolutions: int,
        convolution=InteractionBlock,
        l_max: int = 0,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
        irreps_key=None,
        irreps_query=None,
        irreps_dummy_nodes=None,
        normalization=None,
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        # Translates input feature size to hidden feature size
        self.linear = Linear(irreps_in[AtomicDataDict.NODE_FEATURES_KEY], feature_irreps_hidden)

        # Convolutional layer input feature size is the hidden feature size
        conv_irreps_in = irreps_in.copy()
        conv_irreps_in[AtomicDataDict.NODE_FEATURES_KEY] = feature_irreps_hidden

        # Create a separate convnet for each convolution
        # This is done by creating a list of convnets
        self.n_convolutions = n_convolutions
        self.convnets = torch.nn.ModuleList()
        for i in range(n_convolutions):
            self.convnets.append(ConvNetLayer(irreps_in=conv_irreps_in,
                         feature_irreps_hidden=feature_irreps_hidden,
                         convolution=convolution,
                         convolution_kwargs=convolution_kwargs,
                         num_layers=num_layers,
                         resnet=resnet,
                         nonlinearity_type=nonlinearity_type,
                         nonlinearity_scalars=nonlinearity_scalars,
                         nonlinearity_gates=nonlinearity_gates))

        # Add dummy nodes to keep track of global features
        self.dummy_layer = DirectionalDummyNodes(
            irreps_in=self.convnets[-1].irreps_out,
            irreps_dummy_nodes=irreps_dummy_nodes,
            irreps_query=irreps_query,
            irreps_key=irreps_key,
            spherical_harmonics_lmax=l_max,
            normalization=normalization,
        )

        # Update the output irreps based on the convnet output
        self.irreps_out.update(self.dummy_layer.irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Translates input feature size to hidden feature size
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.linear(data[AtomicDataDict.NODE_FEATURES_KEY])

        # Apply the convnet n times
        for module in self.convnets:
            data = module(data)
            data = self.dummy_layer(data)
        return data