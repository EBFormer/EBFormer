import logging
import torch
import warnings

from typing import Optional, Union, Sequence, List

from e3nn.util.jit import compile_mode

from NonlocalNN._keys import MEAN_DOS_KEY, MEAN_Jx_KEY
from NonlocalNN.nn._rescale import RescaleOutputNonnegative
from nequip.model._scaling import _compute_stats, RESCALE_THRESHOLD, RescaleOutput
from nequip.data import AtomicDataset, AtomicDataDict

from nequip.nn import GraphModuleMixin
from nequip.utils import dtype_from_name

def RescaleMultiplePredictionsNonnegative(
        model: GraphModuleMixin,
        config,
        initialize: bool,
        dataset: Optional[AtomicDataset] = None,
):
    """
    K: Added to do the global rescale of the predictions that are not only energy or forces.

    Rescale the predictions of a model based on a global scale and shift. The global scale and shift are derived from the
    dataset, and must be defined through the config file through the 'string' type definition.
    """
    module_prefix = "global_rescale"

    return GlobalRescaleVectorialMultipleValuesNonnegative(
        model=model,
        config=config,
        dataset=dataset,
        initialize=initialize,
        module_prefix=module_prefix,
        default_scale=None,
        default_shift=None,
        default_scale_keys=[MEAN_DOS_KEY, MEAN_Jx_KEY],
        default_shift_keys=[MEAN_DOS_KEY, MEAN_Jx_KEY],
    )

def GlobalRescaleVectorialMultipleValuesNonnegative(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    module_prefix: str,
    default_scale: Union[str, float, list],
    default_shift: Union[str, float, list],
    default_scale_keys: list,
    default_shift_keys: list,
    dataset: Optional[AtomicDataset] = None,
):
    """
    Allows shifts and scales to be vectors.
    Required change in default value on loading model.

    If ``initialize`` is false, doesn't compute statistics.
    """

    global_scale: List = config.get(f"{module_prefix}_scale", default_scale)
    global_shift: List = config.get(f"{module_prefix}_shift", default_shift)

    target_size = config.get("target_size", 1)

    if global_shift is not None:
        logging.warning(
            f"!!!! Careful global_shift is set to {global_shift}."
            f"The model for {default_shift_keys} will no longer be size extensive"
        )

    if global_shift is None:
        global_shift = []
    if global_scale is None:
        global_scale = []

    assert len(global_scale) == len(global_shift), "Global scale and shift must have the same length"

    # = Get statistics of training dataset =
    if initialize:
        str_names = global_scale + global_shift

        # = Compute shifts and scales =
        if len(str_names) > 0:
            computed_stats = _compute_stats(
                str_names=str_names,
                dataset=dataset,
                stride=config.dataset_statistics_stride,
            )

        global_scale_values = global_scale.copy()
        for i, value in enumerate(global_scale):
            s = value
            global_scale_values[i] = computed_stats[str_names.index(value)]
            logging.info(f"Replace string {s} to {global_scale_values[i]}")

            # We often have small rescaling values for DOS at energies below band gap. Currently, we simply allow to NN
            # to pass scaled when the scale is very small.
            # TODO: Ideal solution ignores any constant rescaling values below a threshold
            # (either set to mean when shift is present or nn output when shift is not present).
            if global_scale_values is not None and torch.any(global_scale_values[i] < RESCALE_THRESHOLD):
                warnings.warn(
                    f"Global scaling {value} contains small values! Small scales are unused, scaling factor set to 1.0!"
                )
                ith_scale = global_scale_values[i]
                ith_scale[ith_scale < RESCALE_THRESHOLD] = 1.0
                global_scale_values[i] = ith_scale

        global_shift_values = global_shift.copy()
        for i, value in enumerate(global_shift):
            s = value
            global_shift_values[i] = computed_stats[str_names.index(value)]
            logging.info(f"Replace string {s} to {global_shift_values[i]}")

    else:
        # Put dummy values
        global_shift_values = global_shift.copy()
        for i in range(len(global_scale)):
            if global_scale is not None:
                global_shift_values[i] = torch.tensor([1.0]*target_size) # it has some kind of value

        global_scale_values = global_scale.copy()
        for i in range(len(global_shift)):
            if global_shift is not None:
                global_scale_values[i] = torch.tensor([0.0]*target_size) # same,

    error_string = "keys need to be a list"
    assert isinstance(default_scale_keys, list), error_string
    assert isinstance(default_shift_keys, list), error_string

    # == Build the model ==
    for i in range(len(global_scale)):
        # Gets the key from the quantity string
        scale_key = [global_scale[i][len('dataset_'):global_scale[i].rfind("_")]]
        shift_key = [global_shift[i][len('dataset_'):global_shift[i].rfind("_")]]
        scale_by = global_scale_values[i]
        shift_by = global_shift_values[i]

        assert scale_key == shift_key, "Global scale and shift must work on the same keys"

        model = RescaleOutputNonnegative(
                    model=model,
                    scale_keys=scale_key,
                    scale_by=scale_by,
                    shift_keys=shift_key,
                    shift_by=shift_by,
                    shift_trainable=False,
                    scale_trainable=False,
                    default_dtype=config.get("default_dtype", None),
                    config=config,
                )

    return model
