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

# Modified by Shingo Yashima from: https://github.com/google-research/sam/blob/main/sam_jax/train.py

"""Evaluates an ensemble model on cifar10, cifar100 or imagenet."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from lib.datasets import dataset_source as dataset_source_lib
from lib.datasets import dataset_source_imagenet
from lib.models import load_model
from lib.training_utils import ensemble_evaluation
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "dataset",
    "cifar10",
    ["cifar10", "cifar100", "imagenet"],
    "Name of the in-distribution dataset.",
)
flags.DEFINE_enum(
    "model_name",
    "WideResnet16-4",
    [
        "Resnet18",
        "Resnet34",
        "Resnet50",
        "Resnet20",
        "Resnet32",
        "Resnet44",
        "WideResnet16-2",
        "WideResnet16-4",
        "WideResnet16-8",
        "WideResnet28-2",
        "WideResnet28-10",
        "WideResnet40-4",
    ],
    "Name of the model to train.",
)
flags.DEFINE_integer(
    "batch_size",
    128,
    "Global batch size. If multiple "
    "replicas are used, each replica will receive "
    "batch_size / num_replicas examples. Batch size should be divisible by "
    "the number of available devices.",
)
flags.DEFINE_enum(
    "image_level_augmentations",
    "none",
    ["none", "basic", "autoaugment"],
    "Augmentations applied to the images. Should be `none` for "
    "no augmentations, `basic` for the standard horizontal "
    "flips and random crops, and `autoaugment` for the best "
    "AutoAugment policy for cifar10. "
    "For Imagenet, setting to autoaugment will use RandAugment.",
)
flags.DEFINE_enum(
    "batch_level_augmentations",
    "none",
    ["none", "cutout", "mixup", "mixcut"],
    "Augmentations that are applied at the batch level. " "Not used by Imagenet.",
)
flags.DEFINE_string(
    "output_dir",
    "results",
    "Directory where the checkpoints and the tensorboard " "records should be saved.",
)


def main(_):

    tf.enable_v2_behavior()
    # make sure tf does not allocate gpu memory
    tf.config.experimental.set_visible_devices([], "GPU")

    tf.random.set_seed(FLAGS.run_seed)

    # Performance gains on TPU by switching to hardware bernoulli.
    def hardware_bernoulli(rng_key, p=jax.numpy.float32(0.5), shape=None):
        lax_key = jax.lax.tie_in(rng_key, 0.0)
        return jax.lax.rng_uniform(lax_key, 1.0, shape) < p

    def set_hardware_bernoulli():
        jax.random.bernoulli = hardware_bernoulli

    set_hardware_bernoulli()

    # As we gridsearch the weight decay and the learning rate, we add them to the
    # output directory path so that each model has its own directory to save the
    # results in. We also add the `run_seed` which is "gridsearched" on to
    # replicate an experiment several times.
    output_dir_suffix = os.path.join(
        FLAGS.dataset,
        FLAGS.model_name,
        "lr_" + str(FLAGS.learning_rate),
        "Adam" if FLAGS.use_adam else "SGD",
        "wd_" + str(FLAGS.weight_decay),
        "particle_" + str(FLAGS.num_particles),
        "seed_" + str(FLAGS.run_seed),
    )

    output_dir_suffix = os.path.join(output_dir_suffix, FLAGS.method)

    if FLAGS.method == "feature_wgd":
        output_dir_suffix = os.path.join(
            output_dir_suffix, FLAGS.prior, "scale_" + str(FLAGS.prior_scale)
        )

    output_dir = os.path.join(FLAGS.output_dir, output_dir_suffix)

    if not gfile.exists(output_dir):
        gfile.makedirs(output_dir)

    num_devices = jax.local_device_count() * jax.process_count()
    assert FLAGS.batch_size % num_devices == 0
    local_batch_size = FLAGS.batch_size // num_devices
    info = "Total batch size: {} ({} x {} replicas)".format(
        FLAGS.batch_size, local_batch_size, num_devices
    )
    logging.info(info)

    if FLAGS.dataset == "cifar10":
        image_size = 32
        num_channels = 3
        num_classes = 10
        low_res = True
        dataset_source = dataset_source_lib.Cifar10(
            FLAGS.batch_size // jax.process_count(),
            FLAGS.image_level_augmentations,
            FLAGS.batch_level_augmentations,
            image_size=image_size,
        )
    elif FLAGS.dataset == "cifar100":
        image_size = 32
        num_channels = 3
        num_classes = 100
        low_res = True
        dataset_source = dataset_source_lib.Cifar100(
            FLAGS.batch_size // jax.process_count(),
            FLAGS.image_level_augmentations,
            FLAGS.batch_level_augmentations,
            image_size=image_size,
        )
    elif FLAGS.dataset == "imagenet":
        image_size = 224
        num_channels = 3
        num_classes = 1000
        low_res = False
        dataset_source = dataset_source_imagenet.Imagenet(
            FLAGS.batch_size // jax.process_count(),
            image_size,
            FLAGS.image_level_augmentations,
        )
    else:
        raise ValueError("Dataset not recognized.")

    dummy_key = jax.random.PRNGKey(FLAGS.run_seed)

    if "feature" in FLAGS.method:
        FLAGS.model_name += "_feature"

    model, params, state = load_model.get_model(
        FLAGS.model_name,
        FLAGS.num_particles,
        local_batch_size,
        image_size,
        num_classes,
        num_channels,
        low_res,
        dummy_key,
    )

    # Learning rate will be overwritten by the lr schedule, we set it to zero.
    optimizer = ensemble_evaluation.create_optimizer(params, 0.0)

    ensemble_evaluation.eval(model, optimizer, state, dataset_source, output_dir)


if __name__ == "__main__":
    tf.enable_v2_behavior()
    jax.config.config_with_absl()
    app.run(main)
