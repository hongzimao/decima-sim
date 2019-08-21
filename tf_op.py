import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def expand_act_on_state(state, sub_acts):
    # expand the state by explicitly adding in actions
    batch_size = tf.shape(state)[0]
    num_nodes = tf.shape(state)[1]
    num_features = state.shape[2].value  # deterministic
    expand_dim = len(sub_acts)

    # replicate the state
    state = tf.tile(state, [1, 1, expand_dim])
    state = tf.reshape(state,
        [batch_size, num_nodes * expand_dim, num_features])

    # prepare the appended sub-actions
    sub_acts = tf.constant(sub_acts, dtype=tf.float32)
    sub_acts = tf.reshape(sub_acts, [1, 1, expand_dim])
    sub_acts = tf.tile(sub_acts, [1, 1, num_nodes])
    sub_acts = tf.reshape(sub_acts, [1, num_nodes * expand_dim, 1])
    sub_acts = tf.tile(sub_acts, [batch_size, 1, 1])

    # concatenate expanded state with sub-action features
    concat_state = tf.concat([state, sub_acts], axis=2)

    return concat_state


def glorot(shape, dtype=tf.float32, scope='default'):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    with tf.variable_scope(scope):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        init = tf.random_uniform(
            shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init)


def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)


def masked_outer_product(a, b, mask):
    """
    combine two probability distribution together
    a: batch_size * num_nodes
    b: batch_size * (num_executor_limit * num_jobs)
    """
    batch_size = tf.shape(a)[0]
    num_nodes = tf.shape(a)[1]
    num_limits = tf.shape(b)[1]

    a = tf.reshape(a, [batch_size, num_nodes, 1])
    b = tf.reshape(b, [batch_size, 1, num_limits])

    # outer matrix product
    outer_product = a * b
    outer_product = tf.reshape(outer_product, [batch_size, -1])

    # mask
    outer_product = tf.transpose(outer_product)
    outer_product = tf.boolean_mask(outer_product, mask)
    outer_product = tf.transpose(outer_product)

    return outer_product


def ones(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.ones(shape, dtype=dtype)
        return tf.Variable(init)


def zeros(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.zeros(shape, dtype=dtype)
        return tf.Variable(init)
