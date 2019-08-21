import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
import bisect
from param import *
from utils import *
from tf_op import *
from msg_passing_path import *
from gcn import GraphCNN
from gsn import GraphSNN
from agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node


class ActorAgent(Agent):
    def __init__(self, sess, node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent'):

        Agent.__init__(self)

        self.sess = sess
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        # for computing and storing message passing path
        self.postman = Postman()

        # node input dimension: [total_num_nodes, num_features]
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])

        # job input dimension: [total_num_jobs, num_features]
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])

        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)

        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn, self.scope)

        # valid mask for node action ([batch_size, total_num_nodes])
        self.node_valid_mask = tf.placeholder(tf.float32, [None, None])

        # valid mask for executor limit on jobs ([batch_size, num_jobs * num_exec_limits])
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])

        # map back the dag summeraization to each node ([total_num_nodes, num_dags])
        self.dag_summ_backward_map = tf.placeholder(tf.float32, [None, None])

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        self.node_act_probs, self.job_act_probs = self.actor_network(
            self.node_inputs, self.gcn.outputs, self.job_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1],
            self.node_valid_mask, self.job_valid_mask,
            self.dag_summ_backward_map, self.act_fn)

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        logits = tf.log(self.node_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.node_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # job_acts [batch_size, num_jobs, 1]
        logits = tf.log(self.job_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.job_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])
        self.job_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # select node action probability
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.node_act_probs, self.node_act_vec),
            reduction_indices=1, keep_dims=True)

        # select job action probability
        self.selected_job_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.job_act_probs, self.job_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        # actor loss due to advantge (negated)
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_node_prob * self.selected_job_prob + \
            self.eps), -self.adv))

        # node_entropy
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.node_act_probs, tf.log(self.node_act_probs + self.eps)))

        # prob on each job
        self.prob_each_job = tf.reshape(
            tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0],
                tf.reshape(self.node_act_probs, [-1, 1])),
                [tf.shape(self.node_act_probs)[0], -1])

        # job entropy
        self.job_entropy = \
            tf.reduce_sum(tf.multiply(self.prob_each_job,
            tf.reduce_sum(tf.multiply(self.job_act_probs,
                tf.log(self.job_act_probs + self.eps)), reduction_indices=2)))

        # entropy loss
        self.entropy_loss = self.node_entropy + self.job_entropy

        # normalize entropy
        self.entropy_loss /= \
            (tf.log(tf.cast(tf.shape(self.node_act_probs)[1], tf.float32)) + \
            tf.log(float(len(self.executor_levels))))
            # normalize over batch size (note: adv_loss is sum)
            # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, node_inputs, gcn_outputs, job_inputs,
                      gsn_dag_summary, gsn_global_summary,
                      node_valid_mask, job_valid_mask,
                      gsn_summ_backward_map, act_fn):

        # takes output from graph embedding and raw_input from environment

        batch_size = tf.shape(node_valid_mask)[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = tf.reshape(
            node_inputs, [batch_size, -1, self.node_input_dim])

        # (2) reshape job inputs to batch format
        job_inputs_reshape = tf.reshape(
            job_inputs, [batch_size, -1, self.job_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = tf.reshape(
            gcn_outputs, [batch_size, -1, self.output_dim])

        # (5) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = tf.reshape(
            gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(
            tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_dag_summ_extend = tf.matmul(
            gsn_summ_backward_map_extend, gsn_dag_summ_reshape)

        # (6) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = tf.reshape(
            gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])
        gsn_global_summ_extend_node = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_extend)[1], 1])

        # (4) actor neural network
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            merge_node = tf.concat([
                node_inputs_reshape, gcn_outputs_reshape,
                gsn_dag_summ_extend,
                gsn_global_summ_extend_node], axis=2)

            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_num_nodes)
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])

            # valid mask on node
            node_valid_mask = (node_valid_mask - 1) * 10000.0

            # apply mask
            node_outputs = node_outputs + node_valid_mask

            # do masked softmax over nodes on the graph
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)

            # -- part B, the distribution over executor limits --
            merge_job = tf.concat([
                job_inputs_reshape,
                gsn_dag_summ_reshape,
                gsn_global_summ_extend_job], axis=2)

            expanded_state = expand_act_on_state(
                merge_job, [l / 50.0 for l in self.executor_levels])

            job_hid_0 = tl.fully_connected(expanded_state, 32, activation_fn=act_fn)
            job_hid_1 = tl.fully_connected(job_hid_0, 16, activation_fn=act_fn)
            job_hid_2 = tl.fully_connected(job_hid_1, 8, activation_fn=act_fn)
            job_outputs = tl.fully_connected(job_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])

            # valid mask on job
            job_valid_mask = (job_valid_mask - 1) * 10000.0

            # apply mask
            job_outputs = job_outputs + job_valid_mask

            # reshape output dimension for softmaxing the executor limits
            # (batch_size, num_jobs, num_exec_limits)
            job_outputs = tf.reshape(
                job_outputs, [batch_size, -1, len(self.executor_levels)])

            # do masked softmax over jobs
            job_outputs = tf.nn.softmax(job_outputs, dim=-1)

            return node_outputs, job_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run([self.gsn.summaries],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + self.gsn.summ_mats,
                [node_inputs] + summ_mats)
        })

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self, node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map,
            node_act_vec, job_act_vec, adv, entropy_weight):

        return self.sess.run([self.act_gradients,
            [self.adv_loss, self.entropy_loss]],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.job_inputs] + \
                [self.node_valid_mask] + [self.job_valid_mask] + \
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map] + [self.node_act_vec] + \
                [self.job_act_vec] + [self.adv] + [self.entropy_weight], \
                [node_inputs] + [job_inputs] + \
                [node_valid_mask] + [job_valid_mask] + \
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map] + [node_act_vec] + \
                [job_act_vec] + [adv] + [entropy_weight])
        })

    def predict(self, node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map):
        return self.sess.run([self.node_act_probs, self.job_act_probs,
            self.node_acts, self.job_acts], \
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.job_inputs] + \
                [self.node_valid_mask] + [self.job_valid_mask] + \
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map], \
                [node_inputs] + [job_inputs] + \
                [node_valid_mask] + [job_valid_mask] +  \
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map])
        })

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        """
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        # compute total number of nodes
        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags), self.job_input_dim])

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # gather job level inputs
        job_idx = 0
        for job_dag in job_dags:
            # number of executors in the job
            job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
            # the current executor belongs to this job or not
            if job_dag is source_job:
                job_inputs[job_idx, 1] = 2
            else:
                job_inputs[job_idx, 1] = -2
            # number of source executors
            job_inputs[job_idx, 2] = num_source_exec / 20.0

            job_idx += 1

        # gather node level inputs
        node_idx = 0
        job_idx = 0
        for job_dag in job_dags:
            for node in job_dag.nodes:

                # copy the feature from job_input first
                node_inputs[node_idx, :3] = job_inputs[job_idx, :3]

                # work on the node
                node_inputs[node_idx, 3] = \
                    (node.num_tasks - node.next_task_idx) * \
                    node.tasks[-1].duration / 100000.0

                # number of tasks left
                node_inputs[node_idx, 4] = \
                    (node.num_tasks - node.next_task_idx) / 200.0

                node_idx += 1

            job_idx += 1

        return node_inputs, job_inputs, \
               job_dags, source_job, num_source_exec, \
               frontier_nodes, executor_limits, \
               exec_commit, moving_executors, \
               exec_map, action_map

    def get_valid_masks(self, job_dags, frontier_nodes,
            source_job, num_source_exec, exec_map, action_map):

        job_valid_mask = np.zeros([1, \
            len(job_dags) * len(self.executor_levels)])

        job_valid = {}  # if job is saturated, don't assign node

        base = 0
        for job_dag in job_dags:
            # new executor level depends on the source of executor
            if job_dag is source_job:
                least_exec_amount = \
                    exec_map[job_dag] - num_source_exec + 1
                    # +1 because we want at least one executor
                    # for this job
            else:
                least_exec_amount = exec_map[job_dag] + 1
                # +1 because of the same reason above

            assert least_exec_amount > 0
            assert least_exec_amount <= self.executor_levels[-1] + 1

            # find the index for first valid executor limit
            exec_level_idx = bisect.bisect_left(
                self.executor_levels, least_exec_amount)

            if exec_level_idx >= len(self.executor_levels):
                job_valid[job_dag] = False
            else:
                job_valid[job_dag] = True

            for l in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, base + l] = 1

            base += self.executor_levels[-1]

        total_num_nodes = int(np.sum(
            job_dag.num_nodes for job_dag in job_dags))

        node_valid_mask = np.zeros([1, total_num_nodes])

        for node in frontier_nodes:
            if job_valid[node.job_dag]:
                act = action_map.inverse_map[node]
                node_valid_mask[0, act] = 1

        return node_valid_mask, job_valid_mask

    def invoke_model(self, obs):
        # implement this module here for training
        # (to pick up state and action to record)
        node_inputs, job_inputs, \
            job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, \
            exec_commit, moving_executors, \
            exec_map, action_map = self.translate_state(obs)

        # get message passing path (with cache)
        gcn_mats, gcn_masks, dag_summ_backward_map, \
            running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags)

        # get node and job valid masks
        node_valid_mask, job_valid_mask = \
            self.get_valid_masks(job_dags, frontier_nodes,
                source_job, num_source_exec, exec_map, action_map)

        # get summarization path that ignores finished nodes
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        # invoke learning model
        node_act_probs, job_act_probs, node_acts, job_acts = \
            self.predict(node_inputs, job_inputs,
                node_valid_mask, job_valid_mask, \
                gcn_mats, gcn_masks, summ_mats, \
                running_dags_mat, dag_summ_backward_map)

        return node_acts, job_acts, \
               node_act_probs, job_act_probs, \
               node_inputs, job_inputs, \
               node_valid_mask, job_valid_mask, \
               gcn_mats, gcn_masks, summ_mats, \
               running_dags_mat, dag_summ_backward_map, \
               exec_map, job_dags_changed

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        if len(frontier_nodes) == 0:
            # no action to take
            return None, num_source_exec

        # invoking the learning model
        node_act, job_act, \
            node_act_probs, job_act_probs, \
            node_inputs, job_inputs, \
            node_valid_mask, job_valid_mask, \
            gcn_mats, gcn_masks, summ_mats, \
            running_dags_mat, dag_summ_backward_map, \
            exec_map, job_dags_changed = self.invoke_model(obs)

        if sum(node_valid_mask[0, :]) == 0:
            # no node is valid to assign
            return None, num_source_exec

        # node_act should be valid
        assert node_valid_mask[0, node_act[0]] == 1

        # parse node action
        node = action_map[node_act[0]]

        # find job index based on node
        job_idx = job_dags.index(node.job_dag)

        # job_act should be valid
        assert job_valid_mask[0, job_act[0, job_idx] + \
            len(self.executor_levels) * job_idx] == 1

        # find out the executor limit decision
        if node.job_dag is source_job:
            agent_exec_act = self.executor_levels[
                job_act[0, job_idx]] - \
                exec_map[node.job_dag] + \
                num_source_exec
        else:
            agent_exec_act = self.executor_levels[
                job_act[0, job_idx]] - exec_map[node.job_dag]

        # parse job limit action
        use_exec = min(
            node.num_tasks - node.next_task_idx - \
            exec_commit.node_commit[node] - \
            moving_executors.count(node),
            agent_exec_act, num_source_exec)

        return node, use_exec
