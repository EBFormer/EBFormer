import warnings
from typing import Sequence, List, Union, Optional

import torch
from e3nn.util.jit import compile_mode
from torch_runstats.scatter import scatter_mean

from NonlocalNN._keys import LDOS_KEY, MEAN_DOS_KEY, LJx_KEY, MEAN_Jx_KEY
from NonlocalNN.nn import AtomwiseReduceBasic
from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.nn import GraphModuleMixin, RescaleOutput
from nequip.utils import instantiate

@compile_mode("script")
class RescaleOutputNonnegative(RescaleOutput):
    """Wrap a model and rescale its outputs when in ``eval()`` mode.

    Note that scaling/shifting is always done (casting into) ``default_dtype``, even if ``model_dtype`` is lower precision.

    Args:
        model : GraphModuleMixin
            The model whose outputs are to be rescaled.
        scale_keys : list of keys, default []
            Which fields to rescale.
        shift_keys : list of keys, default []
            Which fields to shift after rescaling.
        scale_by : floating or Tensor, default 1.
            The scaling factor by which to multiply fields in ``scale``.
        shift_by : floating or Tensor, default 0.
            The shift to add to fields in ``shift``.
        irreps_in : dict, optional
            Extra inputs expected by this beyond those of `model`; this is only present for compatibility.
    """

    scale_keys: List[str]
    shift_keys: List[str]
    scale_trainble: bool
    rescale_trainable: bool
    _all_keys: List[str]

    has_scale: bool
    has_shift: bool

    default_dtype: torch.dtype

    def __init__(
            self,
            model: GraphModuleMixin,
            scale_keys: Union[Sequence[str], str] = [],
            shift_keys: Union[Sequence[str], str] = [],
            scale_by=None,
            shift_by=None,
            shift_trainable: bool = False,
            scale_trainable: bool = False,
            default_dtype: Optional[str] = None,
            irreps_in: dict = {},
            config: dict = {},
    ):
        super().__init__(model=model, scale_keys=scale_keys,
                         shift_keys=shift_keys, scale_by=scale_by,
                         shift_by=shift_by, shift_trainable=shift_trainable,
                         scale_trainable=scale_trainable,
                         default_dtype=default_dtype, irreps_in=irreps_in)

        type_mapper, _ = instantiate(TypeMapper, optional_args=config)
        mask_atoms = config.get('mask_atoms', [])

        if LDOS_KEY in shift_keys or LDOS_KEY in scale_keys:
            warnings.warn("LDOS key is being shifted and/or scaled! Rescaling module is conducting a mean reduction on the LDOS key!")
            self.irreps_out.update({MEAN_DOS_KEY: self.irreps_out[LDOS_KEY]})
            self.LDOS_REDUCE = AtomwiseReduceBasic(
                type_mapper=type_mapper,
                mask_atoms=mask_atoms,
                reduce="mean",
                field=LDOS_KEY,
                out_field=MEAN_DOS_KEY)

        if LJx_KEY in shift_keys or LJx_KEY in scale_keys:
            warnings.warn("LJx key is being shifted and/or scaled! Rescaling module is conducting a mean reduction on the LJx key!")
            self.irreps_out.update({MEAN_Jx_KEY: self.irreps_out[LJx_KEY]})
            self.LJx_REDUCE = AtomwiseReduceBasic(
                type_mapper=type_mapper,
                mask_atoms=mask_atoms,
                reduce="mean",
                field=LJx_KEY,
                out_field=MEAN_Jx_KEY)

    def make_zscore_nonnegative(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Data is unscaled, so we need to ensure non-negativity in the scaled output
        # prediction * scale + shift >= 0
        # prediction >= -shift / scale
        # Note that scale_by elements are guaranteed to have magnitude > 1e-6
        for field in self._all_keys:
            v = data[field].to(dtype=self.default_dtype)

            scale = self.scale_by if self.has_scale and field in self.scale_keys else torch.tensor(1.0,
                                                                                                   dtype=self.default_dtype)
            shift = self.shift_by if self.has_shift and field in self.shift_keys else torch.tensor(0.0,
                                                                                                   dtype=self.default_dtype)

            lower_z_score = -shift.expand(v.shape) / scale.expand(v.shape)
            data[field] = torch.max(v, lower_z_score)
        return data

    def make_real_nonnegative(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        for field in self._all_keys:
            v = data[field].to(dtype=self.default_dtype)
            data[field] = torch.relu(v)
        return data

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.model(data)
        data = self.make_zscore_nonnegative(data)

        if self.training:
            # no scaling, but still need to promote for consistent dtype behavior
            # this is hopefully a no-op in most circumstances due to a
            # preceeding PerSpecies rescale promoting to default_dtype anyway:
            for field in self._all_keys:
                data[field] = data[field].to(dtype=self.default_dtype)
        else:
            # Scale then shift
            # * and + promote dtypes by default, but not when the other
            # operand is a scalar, which `scale/shift_by` are.
            # We solve this by expanding `scale/shift_by` to tensors
            # This is free and doesn't allocate new memory on CUDA:
            # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
            # confirmed in PyTorch slack
            # https://pytorch.slack.com/archives/C3PDTEV8E/p1671652283801129
            if self.has_scale:
                for field in self.scale_keys:
                    v = data[field]
                    data[field] = v * self.scale_by.expand(v.shape)
            if self.has_shift:
                for field in self.shift_keys:
                    v = data[field]
                    data[field] = v + self.shift_by.expand(v.shape)

            # Update downstream graph-pooled fields
            if LDOS_KEY in self.scale_keys or LDOS_KEY in self.shift_keys:
                data = self.LDOS_REDUCE(data)
            if LJx_KEY in self.scale_keys or LJx_KEY in self.shift_keys:
                data = self.LJx_REDUCE(data)

        return data

    @torch.jit.export
    def scale(
            self,
            data: AtomicDataDict.Type,
            force_process: bool = False,
    ) -> AtomicDataDict.Type:
        """Apply rescaling to ``data``, in place.

        Only processes the data if the module is in ``eval()`` mode, unless ``force_process`` is ``True``.

        Args:
            data (map-like): a dict, ``AtomicDataDict``, ``AtomicData``, ``torch_geometric.data.Batch``, or anything else dictionary-like
            force_process (bool): if ``True``, scaling will be done regardless of whether the model is in train or evaluation mode.
        Returns:
            ``data``, modified in place
        """
        data = data.copy()
        data = self.make_zscore_nonnegative(data)

        if self.training and not force_process:
            return data
        else:
            if self.has_scale:
                for field in self.scale_keys:
                    if field in data:
                        data[field] = data[field] * self.scale_by
            if self.has_shift:
                for field in self.shift_keys:
                    if field in data:
                        data[field] = data[field] + self.shift_by

            # Update downstream graph-pooled fields
            if LDOS_KEY in self.scale_keys or LDOS_KEY in self.shift_keys:
                data = self.LDOS_REDUCE(data)
            if LJx_KEY in self.scale_keys or LJx_KEY in self.shift_keys:
                data = self.LJx_REDUCE(data)

            return data

    @torch.jit.export
    def unscale(
            self,
            data: AtomicDataDict.Type,
            force_process: bool = False,
    ) -> AtomicDataDict.Type:
        """Apply the inverse of the rescaling operation to ``data``, in place.

        Only processes the data if the module is in ``train()`` mode, unless ``force_process`` is ``True``.

        Args:
            data (map-like): a dict, ``AtomicDataDict``, ``AtomicData``, ``torch_geometric.data.Batch``, or anything else dictionary-like
            force_process (bool): if ``True``, unscaling will be done regardless of whether the model is in train or evaluation mode.
        Returns:
            ``data``
        """
        data = data.copy()
        data = self.make_real_nonnegative(data)
        if self.training or force_process:
            # To invert, -shift then divide by scale
            if self.has_shift:
                for field in self.shift_keys:
                    if field in data:
                        data[field] = data[field] - self.shift_by
            if self.has_scale:
                for field in self.scale_keys:
                    if field in data:
                        data[field] = data[field] / self.scale_by
            return data
        else:
            return data
