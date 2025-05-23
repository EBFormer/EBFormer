import warnings
from typing import Optional, List

import torch.nn.functional

from torch_runstats.scatter import scatter, scatter_mean

from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.nn._graph_mixin import GraphModuleMixin

class AtomwiseSiLU(GraphModuleMixin, torch.nn.Module):
    def __init__(
            self,
            field: str,
            out_field: Optional[str] = None,
            irreps_in={},
    ):
        super().__init__()
        self.field = field
        self.out_field = field if out_field is None else out_field

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        field = data[self.field]
        data[self.out_field] = torch.nn.functional.silu(field)
        return data


class AtomwiseReduceBasic(GraphModuleMixin, torch.nn.Module):
    """
    Basic atomwise reduction module
    """
    constant: float

    def __init__(
            self,
            field: str,
            mask_atoms: Optional[List[int]] = None,
            type_mapper: Optional[TypeMapper] = None,
            out_field: Optional[str] = None,
            reduce="sum",
            avg_num_atoms=None,
            irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self.mask_atoms = mask_atoms
        self.type_mapper = type_mapper

        if self.mask_atoms:
            assert type_mapper is not None, "TypeMapper must be provided if mask_atoms is specified!"
            warnings.warn("Masking atoms is enabled, ensure that no example "
                          "in the dataset has all atoms masked (undefined behavior)!")

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        field = data[self.field]

        mask = None
        # JIT can't cast NoneType to bool, so use short-circuting logic to check None status and emptiness
        if self.mask_atoms is not None and len(self.mask_atoms) > 0:
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
            mask_atoms = self.type_mapper.transform(torch.tensor(self.mask_atoms, device=atom_types.device))
            mask = torch.logical_not(sum(atom_types == mask_atom for mask_atom in mask_atoms).bool()).flatten()

            # Remove masked atoms from field (required to have accurate mean counts)
            # NOTE, this requires that not all atom types are masked!
            field = field[mask.bool()]

        if AtomicDataDict.BATCH_KEY in data:
            # Remove masked atoms from batch indices
            scatter_idx = data[AtomicDataDict.BATCH_KEY] if mask is None else data[AtomicDataDict.BATCH_KEY][
                mask.bool()]
            if self.reduce == 'sum':
                result = scatter(
                    field,
                    scatter_idx,
                    dim=0,
                    dim_size=len(data[AtomicDataDict.BATCH_PTR_KEY]) - 1,
                    reduce=self.reduce,
                )
            elif self.reduce == 'mean':
                result = scatter_mean(
                    src=field,
                    index=scatter_idx,
                    dim=0,
                    dim_size=len(data[AtomicDataDict.BATCH_PTR_KEY]) - 1,
                )
            else:
                raise NotImplementedError(f'Reduce method {self.reduce} not implemented!')
        else:
            # We can significantly simplify and avoid scatters
            if self.reduce == "sum":
                result = field.sum(dim=0, keepdim=True)
            elif self.reduce == "mean":
                result = field.mean(dim=0, keepdim=True)
            else:
                assert False
        if self.constant != 1.0:
            result = result * self.constant
        data[self.out_field] = result
        return data


class AtomwiseReduceBasicNonnegative(GraphModuleMixin, torch.nn.Module):
    """
    Basic atomwise reduction module. Implements a relu on the output to ensure non-negativity.
    """
    constant: float

    def __init__(
            self,
            field: str,
            mask_atoms: Optional[List[int]] = None,
            type_mapper: Optional[TypeMapper] = None,
            out_field: Optional[str] = None,
            reduce="sum",
            avg_num_atoms=None,
            irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self.mask_atoms = mask_atoms
        self.type_mapper = type_mapper

        if self.mask_atoms:
            assert type_mapper is not None, "TypeMapper must be provided if mask_atoms is specified!"
            warnings.warn("Masking atoms is enabled, ensure that no example "
                          "in the dataset has all atoms masked (undefined behavior)!")

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        field = data[self.field]

        mask = None
        # JIT can't cast NoneType to bool, so use short-circuting logic to check None status and emptiness
        if self.mask_atoms is not None and len(self.mask_atoms) > 0:
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
            mask_atoms = self.type_mapper.transform(torch.tensor(self.mask_atoms, device=atom_types.device))
            mask = torch.logical_not(sum(atom_types == mask_atom for mask_atom in mask_atoms).bool()).flatten()

            # Remove masked atoms from field (required to have accurate mean counts)
            # NOTE, this requires that not all atom types are masked!
            field = field[mask.bool()]

        if AtomicDataDict.BATCH_KEY in data:
            # Remove masked atoms from batch indices
            scatter_idx = data[AtomicDataDict.BATCH_KEY] if mask is None else data[AtomicDataDict.BATCH_KEY][
                mask.bool()]
            if self.reduce == 'sum':
                result = scatter(
                    field,
                    scatter_idx,
                    dim=0,
                    dim_size=len(data[AtomicDataDict.BATCH_PTR_KEY]) - 1,
                    reduce=self.reduce,
                )
            elif self.reduce == 'mean':
                result = scatter_mean(
                    src=field,
                    index=scatter_idx,
                    dim=0,
                    dim_size=len(data[AtomicDataDict.BATCH_PTR_KEY]) - 1,
                )
            else:
                raise NotImplementedError(f'Reduce method {self.reduce} not implemented!')
        else:
            # We can significantly simplify and avoid scatters
            if self.reduce == "sum":
                result = field.sum(dim=0, keepdim=True)
            elif self.reduce == "mean":
                result = field.mean(dim=0, keepdim=True)
            else:
                assert False
        if self.constant != 1.0:
            result = result * self.constant
        # Apply relu to ensure non-negativity
        result = torch.nn.functional.relu(result)
        data[self.out_field] = result
        return data

