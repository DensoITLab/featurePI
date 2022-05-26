# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Shingo Yashima from: https://github.com/google-research/sam/blob/main/sam_jax/imagenet_models/resnet.py

"""Flax implementation of ResNet V1."""


from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    projection: bool
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            if self.projection:
                residual = self.conv(
                    self.filters, (1, 1), self.strides, name="conv_proj"
                )(residual)
                residual = self.norm(name="norm_proj")(residual)
            else:
                residual = jnp.pad(
                    residual[:, :: self.strides[0], :: self.strides[1], :],
                    ((0, 0), (0, 0), (0, 0), (self.filters // 4, self.filters // 4)),
                    "constant",
                )

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    projection: bool
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    low_res: bool = False
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    split: Any = None
    projection: bool = True

    def encode(self, x, train):
        conv = partial(
            nn.Conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.variance_scaling(2.0, "fan_out", "normal"),
        )
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = conv(
            self.num_filters,
            (3, 3) if self.low_res else (7, 7),
            (1, 1) if self.low_res else (2, 2),
            padding="SAME",
            name="conv_init",
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        if not self.low_res:
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    projection=self.projection,
                )(x)
        x = jnp.mean(x, axis=(1, 2), dtype=self.dtype)
        return x

    def classify(self, x, train):
        del train
        x = nn.Dense(
            self.num_classes, dtype=self.dtype, kernel_init=dense_layer_init_fn
        )(x)
        # x = jnp.asarray(x, self.dtype)
        return x

    @nn.compact
    def __call__(self, x, train):
        if self.split == "encoder":
            x = self.encode(x, train)
        elif self.split == "classifier":
            x = self.classify(x, train)
        else:
            x = self.encode(x, train)
            x = self.classify(x, train)
        return x


def dense_layer_init_fn(
    key: jnp.ndarray, shape: Tuple[int, int], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Initializer for the final dense layer.
    Args:
        key: PRNG key to use to sample the weights.
        shape: Shape of the tensor to initialize.
        dtype: Data type of the tensor to initialize.
    Returns:
        The initialized tensor.
    """
    num_units_out = shape[1]
    unif_init_range = 1.0 / (num_units_out) ** (0.5)
    return jax.random.uniform(key, shape, dtype, -1) * unif_init_range


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)


ResNet20 = partial(
    ResNet,
    stage_sizes=[3, 3, 3],
    block_cls=ResNetBlock,
    num_filters=16,
    projection=True,
)
ResNet32 = partial(
    ResNet,
    stage_sizes=[5, 5, 5],
    block_cls=ResNetBlock,
    num_filters=16,
    projection=True,
)
ResNet44 = partial(
    ResNet,
    stage_sizes=[7, 7, 7],
    block_cls=ResNetBlock,
    num_filters=16,
    projection=True,
)
