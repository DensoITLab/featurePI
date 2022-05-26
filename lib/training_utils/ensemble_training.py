# Copyright 2022 The Authors.
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

# Modified by Shingo Yashima from: https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py

"""Functions to train the ensemble networks for image classification tasks."""

import functools
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import flags
from absl import logging

import jax
import jax.numpy as jnp

import flax
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import lr_schedule

from tensorflow.io import gfile

import numpy as np

from lib.optim.feature_wgd import get_feature_wgd_gradient
from lib.datasets import dataset_source as dataset_source_lib

FLAGS = flags.FLAGS

# Training hyper-parameters
flags.DEFINE_float("learning_rate", 0.1, "Initial learning rate.")
flags.DEFINE_bool(
    "use_learning_rate_schedule",
    True,
    "Whether to use a cosine schedule or keep the learning rate "
    "constant. Training on cifar should always use the schedule "
    ", this flag is mostly for testing purpose.",
)
flags.DEFINE_enum(
    "method",
    "deep_ensembles",
    ["deep_ensembles", "feature_wgd"],
    "A method to calculate ensemble gradients.",
)
flags.DEFINE_integer("num_particles", 10, "How many models should be trained.")
flags.DEFINE_integer("projection_dim", 5, "Projection dimension in feature-WGD.")
flags.DEFINE_float("weight_decay", 0.0005, "Weight decay coefficient.")
flags.DEFINE_integer(
    "run_seed",
    0,
    "Seed to use to generate pseudo random number during "
    "training (for dropout for instance). Has no influence on "
    "the dataset shuffling.",
)
flags.DEFINE_bool("use_adam", False, "If True, uses Adam instead of SGD")
flags.DEFINE_enum(
    "lr_schedule",
    "multistep",
    ["cosine", "exponential", "multistep"],
    "Learning rate schedule to use.",
)
flags.DEFINE_enum(
    "prior", "cauchy", ["gauss", "cauchy", "uniform"], "Feature prior distribution."
)
flags.DEFINE_float("prior_scale", 0.005, "Scale of feature prior.")

# Additional flags that don't affect the model.
flags.DEFINE_integer("save_progress_seconds", 3600, "Save progress every...s")
flags.DEFINE_multi_integer(
    "additional_checkpoints_at_epochs",
    [],
    "Additional epochs when we should save the model for later analysis. "
    "No matter the value of this flag, the most recent version of the model "
    "will be saved regularly to resume training if needed.",
)
flags.DEFINE_bool(
    "also_eval_on_training_set",
    False,
    "If set to true, the model will also be evaluated on the "
    "(non-augmented) training set at the end of each epoch.",
)
flags.DEFINE_bool(
    "compute_top_5_error_rate", False, "If true, will also compute top 5 error rate."
)
flags.DEFINE_integer("evaluate_every", 1, "Evaluate on the test set every n epochs.")


def local_replica_groups(inner_group_size: int) -> List[List[int]]:
    world_size = jax.device_count()
    outer_group_size, ragged = divmod(world_size, inner_group_size)
    assert not ragged, "inner group size must evenly divide global device count"

    # the last device should have maximal x and y coordinate
    def bounds_from_last_device(device):
        x, y, z = device.coords
        return (x + 1) * (device.core_on_chip + 1), (y + 1) * (z + 1)

    global_x, _ = bounds_from_last_device(jax.devices()[-1])
    per_host_x, per_host_y = bounds_from_last_device(jax.local_devices(0)[-1])
    assert inner_group_size in [
        2**i for i in range(1, 15)
    ], "inner group size must be a power of two"
    if inner_group_size <= 4:
        # inner group is Nx1 (core, chip, 2x1)
        inner_x, inner_y = inner_group_size, 1
        inner_perm = range(inner_group_size)
    else:
        if inner_group_size <= global_x * 2:
            # inner group is Nx2 (2x2 tray, 4x2 DF pod host, row of hosts)
            inner_x, inner_y = inner_group_size // 2, 2
        else:
            # inner group covers the full x dimension and must be >2 in y
            inner_x, inner_y = global_x, inner_group_size // global_x
        p = np.arange(inner_group_size)
        per_group_hosts_x = 1 if inner_x < per_host_x else inner_x // per_host_x
        p = p.reshape(
            inner_y // per_host_y,
            per_group_hosts_x,
            per_host_y,
            inner_x // per_group_hosts_x,
        )
        p = p.transpose(0, 2, 1, 3)
        p = p.reshape(inner_y // 2, 2, inner_x)
        p[:, 1, :] = p[:, 1, ::-1]
        inner_perm = p.reshape(-1)

    inner_replica_groups = [
        [o * inner_group_size + i for i in inner_perm] for o in range(outer_group_size)
    ]
    return inner_replica_groups


def restore_checkpoint(
    optimizer: flax.optim.Optimizer, model_state: Any, directory: str
) -> Tuple[flax.optim.Optimizer, Any, int]:
    train_state = dict(optimizer=optimizer, model_state=model_state, epoch=0)
    restored_state = checkpoints.restore_checkpoint(directory, train_state)
    return (
        restored_state["optimizer"],
        restored_state["model_state"],
        restored_state["epoch"],
    )


def save_checkpoint(
    optimizer: flax.optim.Optimizer, model_state: Any, directory: str, epoch: int
):
    if jax.process_index() != 0:
        return
    # Sync across replicas before saving.
    optimizer = jax.tree_map(lambda x: x[0], optimizer)
    model_state = jax.tree_map(lambda x: jnp.mean(x, axis=0), model_state)
    train_state = dict(optimizer=optimizer, model_state=model_state, epoch=epoch)
    if gfile.exists(os.path.join(directory, "checkpoint_" + str(epoch))):
        gfile.remove(os.path.join(directory, "checkpoint_" + str(epoch)))
    checkpoints.save_checkpoint(directory, train_state, epoch, keep=2)


def create_optimizer(
    params: Any, learning_rate: float, beta: float = 0.9
) -> flax.optim.Optimizer:
    if FLAGS.use_adam:
        # We set beta2 and epsilon to the values used in the efficientnet paper.
        optimizer_def = optim.Adam(learning_rate=learning_rate)
    else:
        optimizer_def = optim.Momentum(
            learning_rate=learning_rate, beta=beta, nesterov=True
        )
    optimizer = optimizer_def.create(params)
    return optimizer


def cross_entropy_loss_train(
    logits: jnp.ndarray, one_hot_labels: jnp.ndarray, mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    log_softmax_logits = jax.nn.log_softmax(logits)
    if mask is None:
        mask = jnp.ones([log_softmax_logits.shape[0]])
    mask = mask.reshape([log_softmax_logits.shape[0], 1])
    loss = -jnp.sum(one_hot_labels * log_softmax_logits * mask) / mask.sum()
    return jnp.nan_to_num(loss)


def cross_entropy_loss_eval(
    softmax_logits: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    log_softmax_logits = jnp.log(softmax_logits)
    if mask is None:
        mask = jnp.ones([log_softmax_logits.shape[0]])
    mask = mask.reshape([log_softmax_logits.shape[0], 1])
    loss = -jnp.sum(one_hot_labels * log_softmax_logits * mask) / mask.sum()
    return jnp.nan_to_num(loss)  # Set to zero if there is no non-masked samples.


def brier_score(
    softmax_logits: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    if mask is None:
        mask = jnp.ones([softmax_logits.shape[0]])
    mask = mask.reshape([softmax_logits.shape[0], 1])
    loss = jnp.sum(jnp.square(one_hot_labels - softmax_logits) * mask) / mask.sum()
    return jnp.nan_to_num(loss)  # Set to zero if there is no non-masked samples.


def error_rate_metric(
    softmax_logits: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    if mask is None:
        mask = jnp.ones([softmax_logits.shape[0]])
    mask = mask.reshape([softmax_logits.shape[0]])
    error_rate = (
        ((jnp.argmax(softmax_logits, -1) != jnp.argmax(one_hot_labels, -1))) * mask
    ).sum() / mask.sum()
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)


def top_k_error_rate_metric(
    softmax_logits: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    k: int = 5,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    if mask is None:
        mask = jnp.ones([softmax_logits.shape[0]])
    mask = mask.reshape([softmax_logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(softmax_logits, axis=-1)[:, -k:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = 1 - ((hit * mask).sum() / mask.sum())
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)


def tensorflow_to_numpy(xs):
    xs = jax.tree_map(lambda x: x._numpy(), xs)
    return xs


def shard_batch(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def load_and_shard_tf_batch(xs):
    return shard_batch(tensorflow_to_numpy(xs))


def create_exponential_learning_rate_schedule(
    base_learning_rate: float,
    steps_per_epoch: int,
    lamba: float,
    warmup_epochs: int = 0,
) -> Callable[[int], float]:
    def learning_rate_fn(step):
        t = step / steps_per_epoch
        return (
            base_learning_rate * jnp.exp(-t / lamba) * jnp.minimum(t / warmup_epochs, 1)
        )

    return learning_rate_fn


def get_cosine_schedule(
    num_epochs: int, learning_rate: float, num_training_obs: int, batch_size: int
) -> Callable[[int], float]:
    steps_per_epoch = int(math.floor(num_training_obs / batch_size))
    learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
        learning_rate,
        steps_per_epoch // jax.process_count(),
        num_epochs,
        warmup_length=0,
    )
    return learning_rate_fn


def get_exponential_schedule(
    num_epochs: int, learning_rate: float, num_training_obs: int, batch_size: int
) -> Callable[[int], float]:
    steps_per_epoch = int(math.floor(num_training_obs / batch_size))
    # At the end of the training, lr should be 1.2% of original value
    # This mimic the behavior from the efficientnet paper.
    end_lr_ratio = 0.012
    lamba = -num_epochs / math.log(end_lr_ratio)
    learning_rate_fn = create_exponential_learning_rate_schedule(
        learning_rate, steps_per_epoch // jax.process_count(), lamba
    )
    return learning_rate_fn


def get_multistep_schedule(
    num_epochs: int, learning_rate: float, num_training_obs: int, batch_size: int
) -> Callable[[int], float]:
    steps_per_epoch = int(math.floor(num_training_obs / batch_size))
    if "cifar" in FLAGS.dataset:
        step = [[150, 0.1], [225, 0.01], [250, 0.001]]
    else:
        step = [[30, 0.1], [60, 0.01]]
    learning_rate_fn = lr_schedule.create_stepped_learning_rate_schedule(
        learning_rate, steps_per_epoch // jax.process_count(), step, warmup_length=0
    )
    return learning_rate_fn


def global_norm(updates) -> jnp.ndarray:
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


def train_step(
    optimizer: flax.optim.Optimizer,
    state: Any,
    batch: Dict[str, jnp.ndarray],
    prng_key: jnp.ndarray,
    apply_fn: Any,
    learning_rate_fn: Callable[[int], float],
    l2_reg: float,
    prior_scale: float,
) -> Tuple[flax.optim.Optimizer, Any, Dict[str, float], float]:

    if FLAGS.method == "feature_wgd":
        encoder_apply_fn, classifier_apply_fn = apply_fn
        encoder_key, classifier_key = jax.random.split(prng_key)

        def encoder(params, state):
            dropout_rng = jax.random.fold_in(encoder_key, jax.lax.axis_index("batch"))
            feature, new_state = encoder_apply_fn(
                {"params": params, **state},
                rngs={"dropout": dropout_rng},
                x=batch["image"],
                mutable=["batch_stats"],
                train=True,
            )
            return feature, new_state

        def classifier(feature, state):
            dropout_rng = jax.random.fold_in(
                classifier_key, jax.lax.axis_index("batch")
            )
            logits, new_state = classifier_apply_fn(
                state,
                rngs={"dropout": dropout_rng},
                x=feature,
                mutable=["batch_stats"],
                train=True,
            )
            loss = cross_entropy_loss_train(logits, batch["label"]) * logits.shape[0]
            return loss, (new_state, logits)

        def prior(feature):
            if FLAGS.prior == "cauchy":
                feature_penalty = jnp.sum(jnp.log(1 + feature**2 * prior_scale))
            elif FLAGS.prior == "gauss":
                feature_penalty = prior_scale * 0.5 * jnp.sum(feature**2)
            elif FLAGS.prior == "uniform":
                feature_penalty = 0.0
            else:
                raise ValueError("Wrong feature prior: " + FLAGS.prior)
            return feature_penalty

        (loss, (new_state, logits)), grad = get_feature_wgd_gradient(
            encoder, classifier, prior, FLAGS.num_particles, FLAGS.projection_dim
        )(optimizer.target, state)

        grad = jax.tree_util.tree_map(
            lambda g, p: g + l2_reg * p, grad, optimizer.target
        )

    elif FLAGS.method == "deep_ensembles":

        def forward_and_loss(params, state):
            dropout_rng = jax.random.fold_in(prng_key, jax.lax.axis_index("batch"))
            logits, new_state = apply_fn(
                {"params": params, **state},
                rngs={"dropout": dropout_rng},
                x=batch["image"],
                mutable=["batch_stats"],
                train=True,
            )
            loss = cross_entropy_loss_train(logits, batch["label"])
            # We apply weight decay to all parameters, including bias and batch norm
            # parameters.
            weight_penalty_params = jax.tree_leaves(params)
            weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params])
            weight_penalty = l2_reg * 0.5 * weight_l2
            loss = loss + weight_penalty
            return loss, (new_state, logits)

        (loss, (new_state, logits)), grad = jax.vmap(
            jax.value_and_grad(forward_and_loss, argnums=0, has_aux=True)
        )(optimizer.target, state)

    else:
        raise ValueError("Wrong method: " + FLAGS.method)

    step = optimizer.state.step

    lr = learning_rate_fn(step)

    # We synchronize the gradients across replicas by averaging them.
    grad = jax.lax.pmean(grad, "batch")

    # Averages the batch norm moving averages.
    new_state = jax.lax.pmean(new_state, "batch")

    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    # Compute some norms to log on tensorboard.
    gradient_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)])
        / FLAGS.num_particles
    )
    param_norm = jnp.sqrt(
        sum(
            [
                jnp.sum(jnp.square(e))
                for e in jax.tree_util.tree_leaves(new_optimizer.target)
            ]
        )
        / FLAGS.num_particles
    )

    # Compute some metrics to monitor the training.
    metrics = {
        "train_error_rate": jnp.mean(
            jax.vmap(error_rate_metric, (0, None))(logits, batch["label"])
        ),
        "train_loss": jnp.mean(
            jax.vmap(cross_entropy_loss_train, (0, None))(logits, batch["label"])
        ),
        "gradient_norm": gradient_norm,
        "param_norm": param_norm,
    }

    return new_optimizer, new_state, metrics, lr


# Shorthand notation for typing the function defined above.
# We omit the weight decay and learning rate arguments as they will be
# passed before we pmap the function.
_TrainStep = Callable[
    [
        Any,  # model.
        Any,  # state.
        Dict[str, jnp.ndarray],  # batch.
        jnp.ndarray,  # PRNG key
    ],
    Tuple[
        flax.optim.Optimizer,
        Any,
        Dict[str, float],  # metrics.
        jnp.ndarray,  # learning rate.
    ],
]


def eval_step(params, state, batch, apply_fn):

    # Averages the batch norm moving averages.
    state = jax.lax.pmean(state, "batch")

    if FLAGS.method == "feature_wgd":
        encoder_apply_fn, classifier_apply_fn = apply_fn
        encoder_params, classifier_params = params
        encoder_state, classifier_state = state

        def forward(encoder_params, classifier_params, encoder_state, classifier_state):
            feature = encoder_apply_fn(
                {"params": encoder_params, **encoder_state},
                x=batch["image"],
                mutable=False,
                train=False,
            )
            logits = classifier_apply_fn(
                {"params": classifier_params, **classifier_state},
                x=feature,
                mutable=False,
                train=False,
            )
            softmax_logits = jax.nn.softmax(logits)
            return softmax_logits

        softmax_logits = jnp.mean(
            jax.vmap(forward, (0, None, 0, None))(
                encoder_params, classifier_params, encoder_state, classifier_state
            ),
            axis=0,
        )

    elif FLAGS.method == "deep_ensembles":

        def forward(params, state):
            logits = apply_fn(
                {"params": params, **state},
                x=batch["image"],
                mutable=False,
                train=False,
            )
            softmax_logits = jax.nn.softmax(logits)
            return softmax_logits

        softmax_logits = jnp.mean(jax.vmap(forward)(params, state), axis=0)

    else:
        raise ValueError("Wrong method: " + FLAGS.method)

    num_samples = (
        batch["image"].shape[0] if "mask" not in batch else batch["mask"].sum()
    )
    mask = batch.get("mask", None)
    labels = batch["label"]
    metrics = {
        "error_rate": error_rate_metric(softmax_logits, labels, mask) * num_samples,
        "loss": cross_entropy_loss_eval(softmax_logits, labels, mask) * num_samples,
        "brier_score": brier_score(softmax_logits, labels, mask) * num_samples,
    }
    if FLAGS.compute_top_5_error_rate:
        metrics.update(
            {
                "top_5_error_rate": top_k_error_rate_metric(
                    softmax_logits, labels, 5, mask
                )
                * num_samples
            }
        )
    metrics = jax.lax.psum(metrics, "batch")
    return metrics


def eval_on_dataset(params, state, dataset, pmapped_eval_step):
    eval_metrics = []
    total_num_samples = 0
    all_host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i")

    for eval_batch in dataset:
        # Load and shard the TF batch.
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Compute metrics and sum over all observations in the batch.
        metrics = pmapped_eval_step(params, state, eval_batch)
        eval_metrics.append(metrics)
        if "mask" not in eval_batch:
            # Number of samples seen in num_replicas * per_replica_batch_size.
            total_num_samples += (
                eval_batch["label"].shape[0]
                * eval_batch["label"].shape[1]
                * jax.process_count()
            )
        else:
            total_num_samples += all_host_psum(eval_batch["mask"])[0].sum()

    # Metrics are all the same across all replicas (since we applied psum in the
    # eval_step). The next line will fetch the metrics on one of them.
    eval_metrics = common_utils.get_metrics(eval_metrics)
    # Finally, we divide by the number of samples to get the mean error rate and
    # cross entropy.
    eval_summary = jax.tree_map(lambda x: x.sum() / total_num_samples, eval_metrics)
    return eval_summary


def train_for_one_epoch(
    dataset_source: dataset_source_lib.DatasetSource,
    optimizer: flax.optim.Optimizer,
    state: Any,
    prng_key: jnp.ndarray,
    pmapped_train_step: _TrainStep,
    summary_writer: tensorboard.SummaryWriter,
) -> Tuple[flax.optim.Optimizer, Any]:

    start_time = time.time()
    cnt = 0
    train_metrics = []
    for batch in dataset_source.get_train(use_augmentations=True):
        # Generate a PRNG key that will be rolled into the batch.
        step_key = jax.random.fold_in(prng_key, optimizer.state.step[0])
        # Load and shard the TF batch.
        batch = tensorflow_to_numpy(batch)
        batch = shard_batch(batch)
        # Shard the step PRNG key.

        sharded_keys = common_utils.shard_prng_key(step_key)
        optimizer, state, metrics, lr = pmapped_train_step(
            optimizer, state, batch, sharded_keys
        )
        cnt += 1

        train_metrics.append(metrics)
    train_metrics = common_utils.get_metrics(train_metrics)
    # Get training epoch summary for logging.
    train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
    train_summary["learning_rate"] = lr[0]
    current_step = int(optimizer.state.step[0])
    info = "Whole training step done in {} ({} steps)".format(
        time.time() - start_time, cnt
    )
    logging.info(info)
    for metric_name, metric_value in train_summary.items():
        summary_writer.scalar(metric_name, metric_value, current_step)
    summary_writer.flush()
    return optimizer, state


def train(
    model,
    optimizer: flax.optim.Optimizer,
    state: Any,
    dataset_source: dataset_source_lib.DatasetSource,
    training_dir: str,
    num_epochs: int,
    prng_key: jnp.ndarray,
):
    checkpoint_dir = os.path.join(training_dir, "checkpoints")
    summary_writer = tensorboard.SummaryWriter(training_dir)
    if jax.process_index() != 0:  # Don't log if not first host.
        summary_writer.scalar = lambda *args: None

    # Log initial results:
    if gfile.exists(checkpoint_dir):
        optimizer, state, epoch_last_checkpoint = restore_checkpoint(
            optimizer, state, checkpoint_dir
        )
        # If last checkpoint was saved at the end of epoch n, then the first
        # training epochs to do when we resume training is n+1.
        initial_epoch = epoch_last_checkpoint + 1
        info = "Resuming training from epoch {}".format(initial_epoch)
        logging.info(info)
    else:
        initial_epoch = jnp.array(0, dtype=jnp.int32)
        logging.info("Starting training from scratch.")

    optimizer = jax_utils.replicate(optimizer)
    state = jax_utils.replicate(state)

    if FLAGS.use_learning_rate_schedule:
        if FLAGS.lr_schedule == "cosine":
            learning_rate_fn = get_cosine_schedule(
                num_epochs,
                FLAGS.learning_rate,
                dataset_source.num_training_obs,
                dataset_source.batch_size,
            )
        elif FLAGS.lr_schedule == "exponential":
            learning_rate_fn = get_exponential_schedule(
                num_epochs,
                FLAGS.learning_rate,
                dataset_source.num_training_obs,
                dataset_source.batch_size,
            )
        elif FLAGS.lr_schedule == "multistep":
            learning_rate_fn = get_multistep_schedule(
                num_epochs,
                FLAGS.learning_rate,
                dataset_source.num_training_obs,
                dataset_source.batch_size,
            )
        else:
            raise ValueError("Wrong schedule: " + FLAGS.lr_schedule)
    else:
        learning_rate_fn = lambda step: FLAGS.learning_rate

    if "feature" in FLAGS.method:
        encoder, classifier = model
        apply_fn = (encoder.apply, classifier.apply)
    else:
        apply_fn = model.apply

    # pmap the training and evaluation functions.
    pmapped_train_step = jax.pmap(
        functools.partial(
            train_step,
            apply_fn=apply_fn,
            learning_rate_fn=learning_rate_fn,
            l2_reg=FLAGS.weight_decay,
            prior_scale=FLAGS.prior_scale,
        ),
        axis_name="batch",
    )
    pmapped_eval_step = jax.pmap(
        functools.partial(eval_step, apply_fn=apply_fn), axis_name="batch"
    )

    time_at_last_checkpoint = time.time()
    for epochs_id in range(initial_epoch, num_epochs):
        if epochs_id in FLAGS.additional_checkpoints_at_epochs:
            # To save additional checkpoints that will not be erase by later version,
            # we save them in a new directory.
            c_path = os.path.join(checkpoint_dir, "additional_ckpt_" + str(epochs_id))
            save_checkpoint(optimizer, state, c_path, epochs_id)
        tick = time.time()

        optimizer, state = train_for_one_epoch(
            dataset_source,
            optimizer,
            state,
            prng_key,
            pmapped_train_step,
            summary_writer,
        )

        tock = time.time()
        info = "Epoch {} finished in {:.2f}s.".format(epochs_id, tock - tick)
        logging.info(info)

        # Evaluate the model on the test set, and optionally the training set.
        if (epochs_id + 1) % FLAGS.evaluate_every == 0:
            info = "Evaluating at end of epoch {} (0-indexed)".format(epochs_id)
            logging.info(info)
            tick = time.time()
            current_step = int(optimizer.state.step[0])
            if FLAGS.also_eval_on_training_set:
                train_ds = dataset_source.get_train(use_augmentations=False)
                train_metrics = eval_on_dataset(
                    optimizer.target, state, train_ds, pmapped_eval_step
                )
                for metric_name, metric_value in train_metrics.items():
                    summary_writer.scalar(
                        "eval_on_train_" + metric_name, metric_value, current_step
                    )
                summary_writer.flush()

            test_ds = dataset_source.get_test()
            test_metrics = eval_on_dataset(
                optimizer.target, state, test_ds, pmapped_eval_step
            )
            for metric_name, metric_value in test_metrics.items():
                summary_writer.scalar("test_" + metric_name, metric_value, current_step)
            summary_writer.flush()

            tock = time.time()
            info = "Evaluated model in {:.2f}.".format(tock - tick)
            logging.info(info)

        # Save new checkpoint if the last one was saved more than
        # `save_progress_seconds` seconds ago.
        sec_from_last_ckpt = time.time() - time_at_last_checkpoint
        if sec_from_last_ckpt > FLAGS.save_progress_seconds:
            save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
            time_at_last_checkpoint = time.time()
            logging.info("Saved checkpoint.")

    # Always save final checkpoint
    save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
