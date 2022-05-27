#
# Copyright (C) 2022 Denso IT Laboratory, Inc.
# All Rights Reserved
#
# Denso IT Laboratory, Inc. retains sole and exclusive ownership of all
# intellectual property rights including copyrights and patents related to this
# Software.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of the Software and accompanying documentation to use, copy, modify, merge,
# publish, or distribute the Software or software derived from it for
# non-commercial purposes, such as academic study, education and personal use,
# subject to the following conditions:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright 2020 The SAM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Shingo Yashima from: https://github.com/google-research/sam/blob/main/sam_jax/imagenet_models/resnet.py

"""Flax implementation of WideResNet."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jax

ModuleDef = Any


class WideResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    projection: bool
    strides: Tuple[int, int] = (1, 1)
    activate_before_residual: bool = False

    @nn.compact
    def __call__(self, x):
        if self.activate_before_residual:
            y = self.norm()(x)
            y = self.act(y)
            residual = y
        else:
            residual = x
            y = self.norm()(x)
            y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)

        if residual.shape != y.shape:
            if self.projection:
                residual = self.conv(
                    self.filters, (1, 1), self.strides, name="conv_proj"
                )(residual)
            else:
                residual = nn.avg_pool(residual, self.strides, self.strides)
                filters_diff = y.shape[-1] - residual.shape[-1]
                residual = jnp.pad(
                    residual, ((0, 0), (0, 0), (0, 0), (0, filters_diff)), "constant"
                )

        return residual + y


class WideResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    num_classes: int
    low_res: bool = True
    num_filters: int = 16
    width_factor: int = 10
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    split: Any = None
    projection: bool = False

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
        if not self.low_res:
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                activate_before_residual = True if i == 0 and j == 0 else False
                x = WideResNetBlock(
                    self.num_filters * 2**i * self.width_factor,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    projection=self.projection,
                    activate_before_residual=activate_before_residual,
                )(x)
        x = norm(name="bn_final")(x)
        x = self.act(x)
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


WideResNet16_2 = partial(
    WideResNet, stage_sizes=[2, 2, 2], num_filters=16, width_factor=2
)
WideResNet16_4 = partial(
    WideResNet, stage_sizes=[2, 2, 2], num_filters=16, width_factor=4
)
WideResNet16_8 = partial(
    WideResNet, stage_sizes=[2, 2, 2], num_filters=16, width_factor=8
)
WideResNet28_2 = partial(
    WideResNet, stage_sizes=[4, 4, 4], num_filters=16, width_factor=2
)
WideResNet28_10 = partial(
    WideResNet, stage_sizes=[4, 4, 4], num_filters=16, width_factor=10
)
WideResNet40_4 = partial(
    WideResNet, stage_sizes=[6, 6, 6], num_filters=16, width_factor=4
)
