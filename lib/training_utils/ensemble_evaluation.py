import functools
import os
import time
from typing import Any, List, Tuple

from absl import flags
from absl import logging

import jax
import jax.numpy as jnp

import flax
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints

from tensorflow.io import gfile

import numpy as np
import scipy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from lib.datasets import dataset_source as dataset_source_lib

FLAGS = flags.FLAGS


flags.DEFINE_float("learning_rate", 0.1, "Initial learning rate.")
flags.DEFINE_enum(
    "method",
    "deep_ensembles",
    ["deep_ensembles", "feature_wgd"],
    "Whether to use svgd or not to calculate gradients.",
)
flags.DEFINE_integer(
    "num_epochs", 300, "How many epochs the model should be trained for."
)
flags.DEFINE_integer("num_particles", 4, "How many models should be trained.")
flags.DEFINE_float("weight_decay", 0.0005, "Weight decay coefficient.")
flags.DEFINE_integer(
    "run_seed",
    0,
    "Seed to use to generate pseudo random number during "
    "training (for dropout for instance). Has no influence on "
    "the dataset shuffling.",
)
flags.DEFINE_bool("use_adam", False, "If True, uses Adam instead of SGD")
flags.DEFINE_bool(
    "compute_top_5_error_rate", False, "If true, will also compute top 5 error rate."
)
flags.DEFINE_integer("ece_bins", 20, "The number of bins used in calculation of ECE.")
flags.DEFINE_enum(
    "prior", "cauchy", ["gauss", "cauchy", "uniform"], "Feature prior distribution."
)
flags.DEFINE_float("prior_scale", 0.005, "Scale of feature prior.")


def local_replica_groups(inner_group_size: int) -> List[List[int]]:
    """Constructs local nearest-neighbor rings given the JAX device assignment.

    For inner_group_size=8, each inner group is a tray with replica order:

    0/1 2/3
    7/6 5/4

    Args:
          inner_group_size: Number of replica in each group.

    Returns:
          A list of replica id groups.
    """
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
    """Restores a model and its state from a given checkpoint.

    If several checkpoints are saved in the checkpoint directory, the latest one
    will be loaded (based on the `epoch`).

    Args:
          optimizer: The optimizer containing the model that we are training.
          model_state: Current state associated with the model.
          directory: Directory where the checkpoints should be saved.

    Returns:
          The restored optimizer and model state, along with the number of epochs the
            model was trained for.
    """
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
    """Saves a model and its state.

    Removes a checkpoint if it already exists for a given epoch. For multi-host
    training, only the first host will save the checkpoint.

    Args:
          optimizer: The optimizer containing the model that we are training.
          model_state: Current state associated with the model.
          directory: Directory where the checkpoints should be saved.
          epoch: Number of epochs the model has been trained for.
    """
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
    """Creates an optimizer.

    Learning rate will be ignored when using a learning rate schedule.

    Args:
          model: The FLAX model to optimize.
          learning_rate: Learning rate for the gradient descent.
          beta: Momentum parameter.

    Returns:
          A SGD (or RMSProp) optimizer that targets the model.
    """
    if FLAGS.use_adam:
        # We set beta2 and epsilon to the values used in the efficientnet paper.
        optimizer_def = optim.Adam(learning_rate=learning_rate)
    else:
        optimizer_def = optim.Momentum(
            learning_rate=learning_rate, beta=beta, nesterov=True
        )
    optimizer = optimizer_def.create(params)
    return optimizer


def apply_t(softmax_logits, t):
    return jax.nn.softmax(jnp.log(softmax_logits) / t)


def cross_entropy_loss(
    softmax_logits: jnp.ndarray, one_hot_labels: jnp.ndarray
) -> jnp.ndarray:
    loss = jnp.sum(-jax.scipy.special.xlogy(one_hot_labels, softmax_logits), axis=-1)
    return loss


def brier_score(
    softmax_logits: jnp.ndarray, one_hot_labels: jnp.ndarray
) -> jnp.ndarray:
    loss = jnp.sum(jnp.square(one_hot_labels - softmax_logits), axis=-1)
    return loss


def entropy(softmax_logits: jnp.ndarray) -> jnp.ndarray:
    entropy = -jnp.sum(jax.scipy.special.xlogy(softmax_logits, softmax_logits), axis=-1)
    return entropy


def model_variance(each_softmax_logits: jnp.ndarray) -> jnp.ndarray:
    model_variance = jnp.mean(jnp.var(each_softmax_logits, axis=0), axis=-1)
    return model_variance


def auroc(in_scores, out_scores) -> float:
    in_labels = jnp.zeros_like(in_scores)
    out_labels = jnp.ones_like(out_scores)
    scores = jnp.concatenate((in_scores, out_scores))
    labels = jnp.concatenate((in_labels, out_labels))
    auroc = roc_auc_score(labels, scores)
    return auroc


def ece(softmax_logits: np.ndarray, one_hot_labels: np.ndarray, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(softmax_logits, -1), np.argmax(softmax_logits, -1)
    accuracies = predictions == np.argmax(one_hot_labels, -1)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)

    return ece


def error_rate_metric(
    softmax_logits: jnp.ndarray, one_hot_labels: jnp.ndarray
) -> jnp.ndarray:
    error_rate = jnp.argmax(softmax_logits, -1) != jnp.argmax(one_hot_labels, -1)
    # Set to zero if there is no non-masked samples.
    return error_rate


def top_k_error_rate_metric(
    softmax_logits: jnp.ndarray, one_hot_labels: jnp.ndarray, k: int = 5
) -> jnp.ndarray:
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(softmax_logits, axis=-1)[:, -k:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = 1 - hit
    # Set to zero if there is no non-masked samples.
    return error_rate


def tensorflow_to_numpy(xs):
    """Converts a tree of tensorflow tensors to numpy arrays.

    Args:
          xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
            tensorflow tensors.

    Returns:
          A pytree with the same structure as xs, where the leaves have been converted
            to jax numpy ndarrays.
    """
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    xs = jax.tree_map(lambda x: x._numpy(), xs)  # pylint: disable=protected-access
    return xs


def shard_batch(xs):
    """Shards a batch across all available replicas.

    Assumes that the number of samples (first dimension of xs) is divisible by the
    number of available replicas.

    Args:
          xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
            numpy ndarrays.

    Returns:
          A pytree with the same structure as xs, where the leaves where added a
            leading dimension representing the replica the tensor is on.
    """
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def load_and_shard_tf_batch(xs):
    """Converts to numpy arrays and distribute a tensorflow batch.

    Args:
          xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
            tensorflow tensors.

    Returns:
          A pytree of numpy ndarrays with the same structure as xs, where the leaves
            where added a leading dimension representing the replica the tensor is on.
    """
    return shard_batch(tensorflow_to_numpy(xs))


def eval_step(params, state, batch, apply_fn):
    """Evaluates the model on a single batch.

    Args:
      model: The model to evaluate.
      state: Current state associated with the model (contains the batch norm MA).
      batch: Batch on which the model should be evaluated. Must have an `image`
            and `label` key.

    Returns:
      A dictionary containing the loss and error rate on the batch. These metrics
      are summed over the samples (and not averaged).
    """

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

        each_softmax_logits = jax.vmap(forward, (0, None, 0, None))(
            encoder_params, classifier_params, encoder_state, classifier_state
        )
        softmax_logits = jnp.mean(each_softmax_logits, axis=0)

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

        each_softmax_logits = jax.vmap(forward)(params, state)
        softmax_logits = jnp.mean(each_softmax_logits, axis=0)

    else:
        raise ValueError("Wrong method: " + FLAGS.method)

    labels = batch["label"]
    # model_var = model_variance(each_softmax_logits)
    # return labels, softmax_logits, model_var
    return labels, softmax_logits


def collect_metrics(device_metrics):
    def concat_forest(forest):
        concat_args = lambda *args: np.concatenate(args)
        return jax.tree_multimap(concat_args, *forest)

    device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
    metrics_np = jax.device_get(device_metrics)
    return concat_forest(metrics_np)


def eval_forward(params, state, dataset, pmapped_eval_step):
    eval_labels = []
    eval_softmax_logits = []

    for eval_batch in dataset:
        # Load and shard the TF batch.
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Compute metrics and sum over all observations in the batch.
        labels, softmax_logits = pmapped_eval_step(params, state, eval_batch)
        eval_labels.append(labels.reshape(-1, labels.shape[-1]))
        eval_softmax_logits.append(softmax_logits.reshape(-1, softmax_logits.shape[-1]))

    # Metrics are all the same across all replicas (since we applied psum in the
    # eval_step). The next line will fetch the metrics on one of them.
    eval_labels = jax.device_get(jnp.concatenate(eval_labels))
    eval_softmax_logits = jax.device_get(jnp.concatenate(eval_softmax_logits))

    return eval_labels, eval_softmax_logits


def temp_scaling(softmax_logits, labels):
    obj_fn = lambda t: jnp.mean(cross_entropy_loss(apply_t(softmax_logits, t), labels))
    res = scipy.optimize.minimize(
        obj_fn, 1, method="nelder-mead", options={"xtol": 1e-3}
    )
    return res.x[0]


def eval_on_dataset(params, state, dataset, pmapped_eval_step, n_splits=2, n_runs=5):
    """Evaluates the model on the whole dataset.

    Args:
      model: The model to evaluate.
      state: Current state associated with the model (contains the batch norm MA).
      dataset: Dataset on which the model should be evaluated. Should already
            being batched.
      pmapped_eval_step: A pmapped version of the `eval_step` function (see its
            documentation for more details).

    Returns:
      A dictionary containing the loss and error rate on the batch. These metrics
      are averaged over the samples.
    """
    eval_labels, eval_softmax_logits = eval_forward(
        params, state, dataset, pmapped_eval_step
    )

    # Metrics are all the same across all replicas (since we applied psum in the
    # eval_step). The next line will fetch the metrics on one of them.

    eval_metrics = {
        "error_rate": 0,
        "loss": 0,
        "brier_score": 0,
        "ece": 0,
    }

    for _ in range(n_runs):
        for tr_idx, te_idx in KFold(n_splits=n_splits, shuffle=True).split(
            eval_softmax_logits
        ):
            train_t = temp_scaling(eval_softmax_logits[tr_idx], eval_labels[tr_idx])
            test_softmax_logits = apply_t(eval_softmax_logits[te_idx], train_t)
            test_labels = eval_labels[te_idx]

            test_metrics = {
                "error_rate": jnp.mean(
                    error_rate_metric(test_softmax_logits, test_labels)
                ),
                "loss": jnp.mean(cross_entropy_loss(test_softmax_logits, test_labels)),
                "brier_score": jnp.mean(brier_score(test_softmax_logits, test_labels)),
                "ece": ece(np.array(test_softmax_logits), np.array(test_labels)),
            }

            for k, v in test_metrics.items():
                eval_metrics[k] += v / (n_splits * n_runs)

    return eval_metrics


def eval(
    model,
    optimizer: flax.optim.Optimizer,
    state: Any,
    dataset_source: dataset_source_lib.DatasetSource,
    training_dir: str,
):
    """Trains the model.

    Args:
      optimizer: The optimizer targeting the model to train.
      state: Current state associated with the model (contains the batch norm MA).
      dataset_source: Container for the training dataset.
      training_dir: Parent directory where the tensorboard logs and model
            checkpoints should be saved.
    num_epochs: Number of epochs for which we want to train the model.
    """
    checkpoint_dir = os.path.join(training_dir, "checkpoints")
    results_dir = os.path.join(training_dir, "evaluation")
    summary_writer = tensorboard.SummaryWriter(results_dir)
    if jax.process_index() != 0:  # Don't log if not first host.
        summary_writer.scalar = lambda *args: None

    # Log initial results:
    if gfile.exists(checkpoint_dir):

        optimizer, state, epoch_last_checkpoint = restore_checkpoint(
            optimizer, state, checkpoint_dir
        )
        # If last checkpoint was saved at the end of epoch n, then the first
        # training epochs to do when we resume training is n+1.
        info = "Restore model from epoch {}".format(epoch_last_checkpoint)
        logging.info(info)
    else:
        raise ValueError("checkpoint does not exist.")

    optimizer = jax_utils.replicate(optimizer)
    state = jax_utils.replicate(state)

    if "feature" in FLAGS.method:
        encoder, classifier = model
        apply_fn = (encoder.apply, classifier.apply)
    else:
        apply_fn = model.apply

    # pmap the evaluation functions.
    pmapped_eval_step = jax.pmap(
        functools.partial(eval_step, apply_fn=apply_fn), axis_name="batch"
    )

    tick = time.time()
    current_step = int(optimizer.state.step[0])

    test_ds = dataset_source.get_test()
    test_metrics = eval_on_dataset(optimizer.target, state, test_ds, pmapped_eval_step)

    for metric_name, metric_value in test_metrics.items():
        summary_writer.scalar("test_" + metric_name, metric_value, current_step)

    summary_writer.flush()

    tock = time.time()
    info = "Evaluated model in {:.2f}.".format(tock - tick)
    logging.info(info)
