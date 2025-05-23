import warnings

from e3nn.nn import NormActivation
from torch import nn
from e3nn.o3 import Irreps
import torch

class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, mode='norm'):
        super().__init__()

        irreps = irreps_in
        slices = irreps.slices()      # list of slice objects
        units  = list(irreps)         # list of (mul, Irrep) pairs

        self.scalar_slices     = []
        self.non_scalar_slices = []

        for slc, (mul, ir) in zip(slices, units):
            s, e = slc.start, slc.stop
            l, p = ir.l, ir.p

            if l > 1:
                raise ValueError(f"EquivariantLayerNorm only supports up to l=1 irreps, got {l}.")

            if l == 0:
                self.scalar_slices.append((s, e))
            else:
                self.non_scalar_slices.append((s, e, mul, l, p))

        # one LayerNorm for *all* scalars concatenated
        total_s = sum(e - s for s, e in self.scalar_slices)
        self.scalar_norm = nn.LayerNorm(total_s, eps=eps) if total_s > 0 else None

        # NormActivation on the block-norm of each vector/tensor irrep
        # identity on scalars, normalize=True → divide by norm

        blocks = []
        for s, e, mul, l, p in self.non_scalar_slices:
            parity_char = "e" if p == 1 else "o"
            blocks.append(f"{mul}x{l}{parity_char}")
        non_scalar_irreps = Irreps(" + ".join(blocks))

        # Raise a warning of the normalization mode
        warnings.warn(f'Using {mode} mode for EquivariantLayerNorm')

        if mode == "norm":
            self.norm_act = NormActivation(
                non_scalar_irreps,
                scalar_nonlinearity=lambda x: 1.0,
                normalize=True
            ) if len(non_scalar_irreps) > 0 else None
        elif mode == "component":
            self.norm_act = lambda x: x / (torch.mean(x * x, dim=-1, keepdim=True) + eps) ** 0.5
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def forward(self, x):
        y = x.clone()

        # ——— normalize scalars ———
        if self.scalar_norm is not None:
            # gather and concat
            parts = [x[..., s:e] for s, e in self.scalar_slices]
            scalars = torch.cat(parts, dim=-1)
            scalars_norm = self.scalar_norm(scalars)
            # split back and assign
            sizes = [e - s for s, e in self.scalar_slices]
            for (s, e), chunk in zip(self.scalar_slices, scalars_norm.split(sizes, dim=-1)):
                y[..., s:e] = chunk

        # ——— normalize vectors/tensors ———
        if self.non_scalar_slices and self.norm_act is not None:
            parts = [x[..., s:e] for s, e, *_ in self.non_scalar_slices]
            non_scalars = torch.cat(parts, dim=-1)
            non_scalars_norm = self.norm_act(non_scalars)
            sizes = [e - s for s, e, *_ in self.non_scalar_slices]
            for (s, e, *_), chunk in zip(self.non_scalar_slices, non_scalars_norm.split(sizes, dim=-1)):
                y[..., s:e] = chunk

        return y

