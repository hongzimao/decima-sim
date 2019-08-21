"""
Graph Summarization Network

Summarize node features globally
via parameterized aggregation scheme
"""

import copy
import numpy as np
import tensorflow as tf
from tf_op import glorot, ones, zeros


class GraphSNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim, act_fn, scope='gsn'):
        # on each transformation, input_dim -> (multiple) hid_dims -> output_dim
        # the global level summarization will use output from DAG level summarizaiton

        self.inputs = inputs

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims

        self.act_fn = act_fn
        self.scope = scope

        # DAG level and global level summarization
        self.summ_levels = 2

        # graph summarization, hierarchical structure
        self.summ_mats = [tf.sparse_placeholder(
            tf.float32, [None, None]) for _ in range(self.summ_levels)]

        # initialize summarization parameters for each hierarchy
        self.dag_weights, self.dag_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)

        self.global_weights, self.global_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)

        # graph summarization operation
        self.summaries = self.summarize()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def summarize(self):
        # summarize information in each hierarchy
        # e.g., first level summarize each individual DAG
        # second level globally summarize all DAGs
        x = self.inputs

        summaries = []

        # DAG level summary
        s = x
        for i in range(len(self.dag_weights)):
            s = tf.matmul(s, self.dag_weights[i])
            s += self.dag_bias[i]
            s = self.act_fn(s)

        s = tf.sparse_tensor_dense_matmul(self.summ_mats[0], s)
        summaries.append(s)

        # global level summary
        for i in range(len(self.global_weights)):
            s = tf.matmul(s, self.global_weights[i])
            s += self.global_bias[i]
            s = self.act_fn(s)

        s = tf.sparse_tensor_dense_matmul(self.summ_mats[1], s)
        summaries.append(s)

        return summaries
