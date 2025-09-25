import logging
from typing import Optional

from NonlocalNN._keys import LDOS_KEY, MEAN_DOS_KEY, LJx_KEY, MEAN_Jx_KEY
from NonlocalNN.nn import AtomwiseReduceBasic, NxConvNetLayers
from NonlocalNN.nn._Nxconvnetlayer import NxConvNetLayersUnsharedWithDummyAtoms, \
    NxConvNetLayersUnsharedWithDirectionalDummyAtoms, NxConvNetLayersUnsharedWithDummyAtomsNanowire
from nequip.data import AtomicDataset, AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.model import builder_utils
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)
from nequip.utils import instantiate

def JxDOSModelNoSharingDummyNodeNanowire(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model architecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)
    output_size = config.get("target_size", 1)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    dummy_normalization = config.get('dummy_normalization', 'none')

    # add convnet layers
    layers[f"layer_NxConvnet"] = (NxConvNetLayersUnsharedWithDummyAtomsNanowire, dict(n_convolutions=num_layers,
                                                                              normalization=dummy_normalization))

    # DOS Output Layers
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # K: Why does it say that it discards? Isn't there a path from L > 0 to L = 0?
            # -- output block --
            "conv_to_output_hidden_dos": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='dos_hidden'),
            ),
            "output_hidden_to_scalar_dos": (
                AtomwiseLinear,
                dict(field='dos_hidden', irreps_out=f"{output_size}x0e", out_field=LDOS_KEY),
            ),
        }
    )

    # Jx Output Layers
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # K: Why does it say that it discards? Isn't there a path from L > 0 to L = 0?
            # -- output block --
            "conv_to_output_hidden_Jx": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='Jx_hidden'),
            ),
            "output_hidden_to_scalar_Jx": (
                AtomwiseLinear,
                dict(field='Jx_hidden', irreps_out=f"{output_size}x0e", out_field=LJx_KEY),
            ),
        }
    )

    # Reduce to per structure
    type_mapper, _ = instantiate(TypeMapper, optional_args=config)
    mask_atoms = config.get('mask_atoms', [])
    layers["per_atom_dos"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LDOS_KEY,
            out_field=MEAN_DOS_KEY,
        ),
    )
    layers["per_atom_Jx"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LJx_KEY,
            out_field=MEAN_Jx_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
    
def JxDOSModelWeightShared(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:

    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)
    output_size = config.get("target_size", 1)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    # add convnet layers
    layers[f"layer_NxConvnet"] = (NxConvNetLayers, dict(n_convolutions=num_layers))

    # DOS Output Layers
    layers.update(
        {
            # -- output block --
            "conv_to_output_hidden_dos": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='dos_hidden'),
            ),
            "output_hidden_to_scalar_dos": (
                AtomwiseLinear,
                dict(field='dos_hidden', irreps_out=f"{output_size}x0e", out_field=LDOS_KEY),
            ),
        }
    )

    # Jx Output Layers
    layers.update(
        {
            # -- output block --
            "conv_to_output_hidden_Jx": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='Jx_hidden'),
            ),
            "output_hidden_to_scalar_Jx": (
                AtomwiseLinear,
                dict(field='Jx_hidden', irreps_out=f"{output_size}x0e", out_field=LJx_KEY),
            ),
        }
    )

    # Reduce to per structure
    type_mapper, _ = instantiate(TypeMapper, optional_args=config)
    mask_atoms = config.get('mask_atoms', [])
    layers["per_atom_dos"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LDOS_KEY,
            out_field=MEAN_DOS_KEY,
        ),
    )
    layers["per_atom_Jx"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LJx_KEY,
            out_field=MEAN_Jx_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )


def JxDOSModelNoSharingDummyNode(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:

    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)
    output_size = config.get("target_size", 1)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    dummy_normalization = config.get('dummy_normalization', 'none')

    # add convnet layers
    layers[f"layer_NxConvnet"] = (NxConvNetLayersUnsharedWithDummyAtoms, dict(n_convolutions=num_layers,
                                                                              normalization=dummy_normalization))

    # DOS Output Layers
    layers.update(
        {
            # -- output block --
            "conv_to_output_hidden_dos": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='dos_hidden'),
            ),
            "output_hidden_to_scalar_dos": (
                AtomwiseLinear,
                dict(field='dos_hidden', irreps_out=f"{output_size}x0e", out_field=LDOS_KEY),
            ),
        }
    )

    # Jx Output Layers
    layers.update(
        {
            # -- output block --
            "conv_to_output_hidden_Jx": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='Jx_hidden'),
            ),
            "output_hidden_to_scalar_Jx": (
                AtomwiseLinear,
                dict(field='Jx_hidden', irreps_out=f"{output_size}x0e", out_field=LJx_KEY),
            ),
        }
    )

    # Reduce to per structure
    type_mapper, _ = instantiate(TypeMapper, optional_args=config)
    mask_atoms = config.get('mask_atoms', [])
    layers["per_atom_dos"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LDOS_KEY,
            out_field=MEAN_DOS_KEY,
        ),
    )
    layers["per_atom_Jx"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LJx_KEY,
            out_field=MEAN_Jx_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )


def JxDOSModelNoSharedDirectionalDummyNode(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:

    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)
    output_size = config.get("target_size", 1)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    dummy_normalization = config.get('dummy_normalization', None)

    layers[f"layer_NxDirectionalConvnet"] = (NxConvNetLayersUnsharedWithDirectionalDummyAtoms, dict(n_convolutions=num_layers,
                                                                                                    l_max=config['l_max'],
                                                                                                    normalization=dummy_normalization))

    # DOS Output Layers
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # K: Why does it say that it discards? Isn't there a path from L > 0 to L = 0?
            # -- output block --
            "conv_to_output_hidden_dos": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='dos_hidden'),
            ),
            "output_hidden_to_scalar_dos": (
                AtomwiseLinear,
                dict(field='dos_hidden', irreps_out=f"{output_size}x0e", out_field=LDOS_KEY),
            ),
        }
    )

    # Jx Output Layers
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # K: Why does it say that it discards? Isn't there a path from L > 0 to L = 0?
            # -- output block --
            "conv_to_output_hidden_Jx": (
                AtomwiseLinear,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field='Jx_hidden'),
            ),
            "output_hidden_to_scalar_Jx": (
                AtomwiseLinear,
                dict(field='Jx_hidden', irreps_out=f"{output_size}x0e", out_field=LJx_KEY),
            ),
        }
    )

    # Reduce to per structure
    type_mapper, _ = instantiate(TypeMapper, optional_args=config)
    mask_atoms = config.get('mask_atoms', [])
    layers["per_atom_dos"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LDOS_KEY,
            out_field=MEAN_DOS_KEY,
        ),
    )
    layers["per_atom_Jx"] = (
        AtomwiseReduceBasic,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LJx_KEY,
            out_field=MEAN_Jx_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )