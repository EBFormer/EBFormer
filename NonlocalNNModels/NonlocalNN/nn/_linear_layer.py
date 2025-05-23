from typing import Optional, List

import torch
from e3nn.o3 import Linear, Irreps

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class LinearLayer(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str,
        out_field: str,
        irreps_in={},
        irreps_out={},
        activation=True,
        biases=True,
        residual=False,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

        self.residual = residual
        if self.residual:
            assert irreps_out[out_field] == irreps_in[field], "For residual connections, input and output irreps must match."

        self.linear = Linear(irreps_in=irreps_in[self.field],
                             irreps_out=irreps_out[self.out_field],
                             biases=biases)
        self.has_activation = activation

        # Silu activation function is used by default for invariant features
        self.activation = torch.nn.functional.silu

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        field = data[self.field]
        data[self.out_field] = self.linear(field)

        if self.has_activation:
            data[self.out_field] = self.activation(data[self.out_field])
        if self.residual:
            data[self.out_field] += field

        return data