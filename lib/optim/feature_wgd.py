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

"""Calculate gradients using feature-WGD."""


import jax.numpy as jnp
import jax
from jax.tree_util import tree_unflatten, tree_flatten, tree_leaves


def squared_norm(x):
    return jnp.sum(jnp.square(x))


@jax.jit
def rbf(dist, h):
    return jnp.exp(-sum(tree_leaves(jax.tree_multimap(jnp.divide, dist, h))))


def tree_divide(tree, x):
    return jax.tree_map(lambda t: jnp.divide(t, x), tree)


def get_projection(data):
    cov = jnp.einsum("pijc,qijc->pq", data, data)
    w, u = jnp.linalg.eigh(cov)
    proj_basis = jnp.einsum(
        "i,i...->i...", 1 / jnp.sqrt(w), jnp.einsum("ij,i...->j...", u, data)
    )
    return proj_basis


def get_feature_wgd_gradient(encoder, classifier, prior, num_particles, proj_dim):
    def get_wgd_gradient(target_likelihood, num_particles):
        target_likelihood_grads = jax.vmap(
            jax.value_and_grad(target_likelihood, argnums=0, has_aux=True), (0, None)
        )
        target_prior_grads = jax.vmap(jax.value_and_grad(prior, argnums=0))
        kernel_val_grad = jax.value_and_grad(rbf, argnums=0)
        pairwise_dist_val_grad = jax.vmap(
            jax.vmap(
                jax.value_and_grad(lambda p1, p2: squared_norm(p1 - p2), argnums=0),
                (0, None),
            ),
            (None, 0),
        )

        def kernel_variational_grad(dist_vals, dist_grads, h):
            kernel_vals, kernel_grads = jax.vmap(kernel_val_grad, (0, None))(
                dist_vals, h
            )
            kernel_val_sum = jnp.sum(kernel_vals, axis=0)
            kernel_grad_sum = jax.tree_map(
                lambda x: jnp.sum(x, axis=0),
                jax.tree_multimap(
                    lambda x, y: jnp.einsum("i,i...->i...", x, y),
                    kernel_grads,
                    dist_grads,
                ),
            )
            return tree_divide(kernel_grad_sum, kernel_val_sum)

        def wgd_gradient(params, state):
            _, treedef = tree_flatten(params)
            (loss, (new_state, logits)), likelihood_grads = target_likelihood_grads(
                params, state
            )
            _, prior_grads = target_prior_grads(params)

            new_state = jax.tree_map(lambda x: jnp.mean(x, axis=0), new_state)
            params = jax.lax.all_gather(params, "batch", axis=1)
            likelihood_grads = jax.lax.all_gather(likelihood_grads, "batch", axis=1)

            proj_basis = get_projection(likelihood_grads)[:proj_dim]
            proj_params = jnp.einsum("pijc,qijc->pq", params, proj_basis)

            pairwise_dist_val_grads = treedef.flatten_up_to(
                jax.tree_multimap(pairwise_dist_val_grad, proj_params, proj_params)
            )
            pairwise_dist_vals, pairwise_dist_grads = tree_unflatten(
                treedef, [p[0] for p in pairwise_dist_val_grads]
            ), tree_unflatten(treedef, [p[1] for p in pairwise_dist_val_grads])
            h = jax.tree_map(
                lambda x: jnp.median(x, axis=0) ** 2 / jnp.log(num_particles) + 1e-12,
                pairwise_dist_vals,
            )
            vmap_kernel_variational_grad = jax.vmap(kernel_variational_grad, (1, 1, 0))
            proj_grads = vmap_kernel_variational_grad(
                pairwise_dist_vals, pairwise_dist_grads, h
            )
            repulsive_grads = jnp.einsum("pq,qijc->pijc", proj_grads, proj_basis)[
                :, jax.lax.axis_index("batch")
            ]
            grads = (
                repulsive_grads
                + prior_grads
                + likelihood_grads[:, jax.lax.axis_index("batch")]
            )
            grads = grads / grads.shape[1]

            return (loss, (new_state, logits)), grads

        return wgd_gradient

    def functional_wgd_gradient(params, state):
        (encoder_params, classifier_params) = params
        (encoder_state, classifier_state) = state
        feature_wgd_gradient = get_wgd_gradient(classifier, num_particles)
        feature, vjp_fun, new_encoder_state = jax.vmap(
            lambda params, state: jax.vjp(
                lambda params: encoder(params, state), params, has_aux=True
            )
        )(encoder_params, encoder_state)
        (loss, (new_classifier_state, logits)), feature_grads = feature_wgd_gradient(
            feature, {"params": classifier_params, **classifier_state}
        )
        (encoder_grads,) = jax.vmap(lambda vjp_fun, v: vjp_fun(v))(
            vjp_fun, feature_grads
        )
        classifier_grads = jax.grad(
            lambda params, feature, state: jnp.mean(
                jax.vmap(classifier, (0, None))(feature, {"params": params, **state})[0]
            )
            / logits.shape[1]
        )(classifier_params, feature, classifier_state)
        new_state = (new_encoder_state, new_classifier_state)
        grads = (encoder_grads, classifier_grads)
        return (loss, (new_state, logits)), grads

    return functional_wgd_gradient
