"""Reducers fuse a dict of encoded tensors into a single tensor.

Every reducer exposes ``compute_output_dim(input_dims: dict[str, int]) -> int``
so that factories can size downstream layers before any forward pass.
Custom reducers must implement the same method.
"""

import torch
from einops import reduce


class Concat(torch.nn.Module):
    """Concatenate all inputs along ``dim`` in dict-insertion order."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([inputs[i] for i in inputs], dim=self._dim)

    @staticmethod
    def compute_output_dim(input_dims: dict[str, int]) -> int:
        return sum(input_dims.values())


class Reduce(torch.nn.Module):
    """Elementwise reduction across inputs via :func:`einops.reduce`.

    Stacks all input tensors along a fresh trailing axis and reduces that
    axis with the chosen operation (``"sum"``, ``"mean"``, ``"max"``,
    ``"min"``, ``"prod"``). All inputs must share the same shape.
    """

    def __init__(self, reduction: str = "sum") -> None:
        super().__init__()
        self._reduction = reduction

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([inputs[i] for i in inputs], dim=-1)
        return reduce(stacked, "... inputs -> ...", reduction=self._reduction)

    @staticmethod
    def compute_output_dim(input_dims: dict[str, int]) -> int:
        dims = list(input_dims.values())
        if any(d != dims[0] for d in dims[1:]):
            raise ValueError(
                f"Reduce requires equal input dims, got {input_dims}",
            )
        return dims[0]


class PassThrough(torch.nn.Module):
    """Return only the named stream — useful for single-modality models."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs[self._key]

    def compute_output_dim(self, input_dims: dict[str, int]) -> int:
        return input_dims[self._key]
