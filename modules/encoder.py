import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

Shape = Tuple[int]

DType = Any
Tensor = torch.tensor
TensorTree = Union[
    Tensor, Iterable["TensorTree"], Mapping[str, "TensorTree"]
]  # pytype: disable=not-supported-yet
ProcessorState = TensorTree
PRNGKey = Tensor
NestedDict = Dict[str, Any]


class CNN(nn.Module):
    """ CNN model with conv and normalization layers """

    def __init__(
        self,
        features: Sequence[int],
        kernel_size: Sequence[Tuple[int, int]],
        strides: Sequence[Tuple[int, int]],
        layer_transpose: Sequence[bool],
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        norm_type: Optional[str] = None,
        dim_name: Optional[
            str
        ] = None,  # Name of dim over which to aggregate batch stats
        output_size: Optional[int] = None,
        train: Optional[bool] = True,
    ):
        super().__init__()
        self.num_layers = len(features)
        self.features = features
        self.strides = strides
        self.kernel_size = kernel_size
        self.layer_transpose = layer_transpose
        self.activation_fn = activation_fn
        self.norm_type = norm_type
        self.dim_name = dim_name
        self.output_size = output_size
        self.training = train

        assert self.num_layers >= 1, "Need to have at least one layer."
        assert (
            len(self.kernel_size) == self.num_layers
        ), "len(kernel_size) and len(features) must match."
        assert (
            len(self.strides) == self.num_layers
        ), "len(strides) and len(features) must match."
        assert (
            len(self.layer_transpose) == self.num_layers
        ), "len(layer_transpose) and len(features) must match."

        if self.norm_type:
            assert self.norm_type in {
                "batch",
                "group",
                "instance",
                "layer",
            }, f"{self.norm_type} is not a valid normalization module."

        # Build conv net
        self.conv_layers = []
        for i, lt in enumerate(layer_transpose):
            if not lt:
                conv = nn.Conv2d
            else:
                conv = nn.ConvTranspose2d

            in_channels = 3 if i == 0 else self.features[i - 1]
            layer = conv(
                in_channels,
                self.features[i],
                kernel_size=kernel_size[i],
                stride=strides[i],
                padding="same",
                bias=False if self.norm_type else True,
            )
            self.conv_layers.append(layer)

        self.conv = nn.Sequential(*self.conv_layers)

        # Select norm type
        if self.norm_type == "batch":
            self.norm_module = nn.BatchNorm2d(
                momentum=0.1, track_running_stats=not self.training
            )
        elif self.norm_type == "group" or self.norm_type == "instance":
            self.norm_module = nn.GroupNorm(
                num_channels=self.features[-1], num_groups=32
            )
        elif self.norm_type == "layer":
            self.norm_module = nn.LayerNorm(self.features[-1])

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type:
            x = self.norm_module(x)
        x = self.activation_fn(x)

        return x
