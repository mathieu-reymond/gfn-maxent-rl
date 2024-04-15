import jax.numpy as jnp
import jax
import haiku as hk

from gfn_maxent_rl.nets.gnn import GNNBackbone


def log_policy(logits, masks):
    # Mask out invalid actions
    logits = jnp.where(masks, logits, -jnp.inf)
    return jax.nn.log_softmax(logits, axis=1)


def uniform_log_policy(masks):
    logits = jnp.where(masks, 0., -jnp.inf)
    return jax.nn.log_softmax(logits, axis=-1)


def policy_network_mlp():
    def network(observations):
        batch_size = observations['nodes'].shape[0]
        output_size = observations['mask'].shape[-1]

        inputs = observations['nodes'].reshape(batch_size, -1)

        logits = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(inputs)

        # Mask out the invalid actions
        norm = hk.get_state('normalization', (), init=jnp.ones)
        return log_policy(logits * norm, observations['mask'])

    return network


def q_network_mlp():
    def network(observations):
        batch_size = observations['nodes'].shape[0]
        output_size = observations['mask'].shape[-1]

        inputs = observations['nodes'].reshape(batch_size, -1)

        q_s = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(inputs)

        # Mask the Q-value
        outputs = jnp.where(observations['mask'], q_s, -jnp.inf)
        return outputs

    return network


def f_network_mlp():
    def network(observations):
        batch_size = observations['nodes'].shape[0]
        output_size = 1

        inputs = observations['nodes'].reshape(batch_size, -1)

        outputs = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(inputs)

        outputs = jnp.squeeze(outputs, axis=-1)
        # Set the flow at terminating states to 0
        # /!\ This is assuming that the terminal state is the *only* child of
        # any terminating state, which is true for the TreeSample environments.
        is_intermediate = jnp.any(observations['nodes'], axis=-1)
        outputs = jnp.where(is_intermediate, outputs, 0.)
        return outputs

    return network