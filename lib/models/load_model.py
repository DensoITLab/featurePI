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

# Modified by Shingo Yashima from: https://github.com/google-research/sam/blob/main/sam_jax/models/load_model.py

"""Build FLAX models for image classification."""

from functools import partial

import jax
from jax import numpy as jnp

from lib.models import resnet, wideresnet


def create_image_model(prng_key, model, input_shape, method=None):
    @jax.jit
    def init(*args):
        return model.init_with_output(*args, train=False, method=method)

    output, variables = init({"params": prng_key}, jnp.ones(input_shape, model.dtype))
    init_state, params = variables.pop("params")
    return params, init_state, output


class ModelNameError(Exception):
    """Exception to raise when the model name is not recognized."""

    pass


def get_model(
    model_name: str,
    num_particles: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    num_channels: int,
    low_res: bool,
    prng_key: jnp.ndarray,
):
    """Returns an initialized model of the chosen architecture.

    Args:
      model_name: Name of the architecture to use. See image_classification.train
        flags for a list of available models.
      num_particles: The number of models trained.
      batch_size: The batch size that the model should expect.
      image_size: Dimension of the image (assumed to be squared).
      num_classes: Dimension of the output layer. Should be 1000, but is left as
        an argument for consistency with other load_model functions. An error will
        be raised if num_classes is not 1000.
      prng_key: PRNG key to use to sample the weights.

    Returns:
      The initialized model and its state.

    Raises:
      ModelNameError: If the name of the architecture is not recognized.
    """
    if model_name.startswith("Resnet18"):
        model = partial(resnet.ResNet18, num_classes=num_classes, low_res=low_res)
    elif model_name.startswith("Resnet34"):
        model = partial(resnet.ResNet34, num_classes=num_classes, low_res=low_res)
    elif model_name.startswith("Resnet50"):
        model = partial(resnet.ResNet50, num_classes=num_classes, low_res=low_res)
    elif model_name.startswith("Resnet20"):
        model = partial(resnet.ResNet20, num_classes=num_classes, low_res=low_res)
    elif model_name.startswith("Resnet32"):
        model = partial(resnet.ResNet32, num_classes=num_classes, low_res=low_res)
    elif model_name.startswith("Resnet44"):
        model = partial(resnet.ResNet44, num_classes=num_classes, low_res=low_res)
    elif model_name.startswith("WideResnet16-2"):
        model = partial(
            wideresnet.WideResNet16_2, num_classes=num_classes, low_res=low_res
        )
    elif model_name.startswith("WideResnet16-4"):
        model = partial(
            wideresnet.WideResNet16_4, num_classes=num_classes, low_res=low_res
        )
    elif model_name.startswith("WideResnet16-8"):
        model = partial(
            wideresnet.WideResNet16_8, num_classes=num_classes, low_res=low_res
        )
    elif model_name.startswith("WideResnet28-2"):
        model = partial(
            wideresnet.WideResNet28_2, num_classes=num_classes, low_res=low_res
        )
    elif model_name.startswith("WideResnet28-10"):
        model = partial(
            wideresnet.WideResNet28_10, num_classes=num_classes, low_res=low_res
        )
    elif model_name.startswith("WideResnet40-4"):
        model = partial(
            wideresnet.WideResNet40_4, num_classes=num_classes, low_res=low_res
        )
    else:
        raise ModelNameError("Unrecognized model name.")

    input_shape = (batch_size, image_size, image_size, num_channels)

    if model_name.endswith("_feature"):
        prng_key, cls_key = jax.random.split(prng_key, 2)
        encoder, classifier = model(split="encoder"), model(split="classifier")
        encoder_keys = jax.random.split(prng_key, num_particles)
        create_ensemble_encoder = jax.vmap(
            partial(create_image_model, model=encoder, input_shape=input_shape)
        )
        params_encoder, init_state_encoder, output = create_ensemble_encoder(
            encoder_keys
        )
        feature_shape = output.shape[1:]
        params_classifier, init_state_classifier, _ = create_image_model(
            cls_key, model=classifier, input_shape=feature_shape
        )
        model = (encoder, classifier)
        params = (params_encoder, params_classifier)
        init_state = (init_state_encoder, init_state_classifier)

    else:
        model = model()
        prng_keys = jax.random.split(prng_key, num_particles)
        create_ensemble_model = jax.vmap(
            partial(create_image_model, model=model, input_shape=input_shape)
        )
        params, init_state, _ = create_ensemble_model(prng_keys)

    return model, params, init_state
