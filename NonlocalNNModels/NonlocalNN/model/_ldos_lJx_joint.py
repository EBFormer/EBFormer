import logging
from typing import Optional

from NonlocalNN._keys import LDOS_KEY, MEAN_DOS_KEY, LJx_KEY, MEAN_Jx_KEY
from NonlocalNN.nn._Nxconvnetlayer import NxConvNetLayersUnsharedWithDummyAtoms, \
    NxConvNetLayersUnsharedWithDirectionalDummyAtoms
from NonlocalNN.nn._atomwise import AtomwiseReduceBasicNonnegative
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


def lJxlDOSModelNoSharingDummyNodeNonnegative(
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
    layers[f"layer_NxConvnet"] = (NxConvNetLayersUnsharedWithDummyAtoms, dict(n_convolutions=num_layers))

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
        AtomwiseReduceBasicNonnegative,
        dict(
            type_mapper=type_mapper,
            mask_atoms=mask_atoms,
            reduce="mean",
            field=LDOS_KEY,
            out_field=MEAN_DOS_KEY,
        ),
    )
    layers["per_atom_Jx"] = (
        AtomwiseReduceBasicNonnegative,
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

def lJxlDOSModelUnsharedDirectionalDummy(
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

    # add convnet layers
    layers[f"layer_NxDirectionalConvnet"] = (NxConvNetLayersUnsharedWithDirectionalDummyAtoms, dict(n_convolutions=num_layers,
                                                                                                    l_max=config['l_max'],
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

    # No pooling is done for local training -- This is handled in the rescale layer

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )