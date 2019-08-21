"""
Compute the message passing path in O(num_total_nodes),
represent the path with sparse adjacency matrices (parent-
child pairs at each iteration) and frontier masks (aggregation
node points at each iteration)
"""

import numpy as np
import tensorflow as tf
from utils import OrderedSet
from sparse_op import *
from param import *


class Postman(object):
    """
    Check if the set of DAGs changes and then compute the
    message passing path, to save computation
    """
    def __init__(self):
        self.reset()

    def get_msg_path(self, job_dags):
        if len(self.job_dags) != len(job_dags):
            job_dags_changed = True
        else:
            job_dags_changed = not(all(i is j for \
                (i, j) in zip(self.job_dags, job_dags)))

        if job_dags_changed:
            self.msg_mats, self.msg_masks = get_msg_path(job_dags)
            self.dag_summ_backward_map = get_dag_summ_backward_map(job_dags)
            self.running_dag_mat = get_running_dag_mat(job_dags)
            self.job_dags = OrderedSet(job_dags)

        return self.msg_mats, self.msg_masks, \
               self.dag_summ_backward_map, self.running_dag_mat, \
               job_dags_changed

    def reset(self):
        self.job_dags = OrderedSet()
        self.msg_mats = []
        self.msg_masks = []
        self.dag_summ_backward_map = None
        self.running_dag_mat = None


def get_msg_path(job_dags):
    """
    matrix: parent-children relation in each message passing step
    mask: set of nodes doing message passing at each step
    """
    msg_mats, msg_masks = [], []

    for job_dag in job_dags:
        msg_mat, msg_mask = get_bottom_up_paths(job_dag)
        msg_mats.append(msg_mat)
        msg_masks.append(msg_mask)

    if len(job_dags) > 0:
        msg_mats = absorb_sp_mats(
            msg_mats, args.max_depth)
        msg_masks = merge_masks(msg_masks)

    return msg_mats, msg_masks


def get_init_frontier(job_dag, depth):
    """
    Get the initial set of frontier nodes, based on the depth
    """
    sources = set(job_dag.nodes)

    for d in range(depth):
        new_sources = set()
        for n in sources:
            if len(n.child_nodes) == 0:
                new_sources.add(n)
            else:
                new_sources.update(n.child_nodes)
        sources = new_sources

    frontier = sources
    return frontier


def get_bottom_up_paths(job_dag):
    """
    The paths start from all leaves and end with
    frontier (parents all finished) unfinished nodes
    """
    num_nodes = job_dag.num_nodes

    msg_mats = []
    msg_masks = np.zeros([args.max_depth, num_nodes])

    # get set of frontier nodes in the beginning
    # this is constrained by the message passing depth
    frontier = get_init_frontier(job_dag, args.max_depth)
    msg_level = {}

    # initial nodes are all message passed
    for n in frontier:
        msg_level[n] = 0

    # pass messages
    for depth in range(args.max_depth):
        new_frontier = set()
        parent_visited = set()  # save some computation
        for n in frontier:
            for parent in n.parent_nodes:
                if parent not in parent_visited:
                    curr_level = 0
                    children_all_in_frontier = True
                    for child in parent.child_nodes:
                        if child not in frontier:
                            children_all_in_frontier = False
                            break
                        if msg_level[child] > curr_level:
                            curr_level = msg_level[child]
                    # children all ready
                    if children_all_in_frontier:
                        if parent not in msg_level or \
                           curr_level + 1 > msg_level[parent]:
                            # parent node has deeper message passed
                            new_frontier.add(parent)
                            msg_level[parent] = curr_level + 1
                    # mark parent as visited
                    parent_visited.add(parent)

        if len(new_frontier) == 0:
            break  # some graph is shallow

        # assign parent-child path in current iteration
        sp_mat = SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes))
        for n in new_frontier:
            for child in n.child_nodes:
                sp_mat.add(row=n.idx, col=child.idx, data=1)
            msg_masks[depth, n.idx] = 1
        msg_mats.append(sp_mat)

        # Note: there might be residual nodes that
        # can directly pass message to its parents
        # it needs two message passing steps
        # (e.g., TPCH-17, node 0, 2, 4)
        for n in frontier:
            parents_all_in_frontier = True
            for p in n.parent_nodes:
                if not p in msg_level:
                    parents_all_in_frontier = False
                    break
            if not parents_all_in_frontier:
                new_frontier.add(n)

        # start from new frontier
        frontier = new_frontier

    # deliberately make dimension the same, for batch processing
    for _ in range(depth, args.max_depth):
        msg_mats.append(SparseMat(dtype=np.float32,
            shape=(num_nodes, num_nodes)))

    return msg_mats, msg_masks


def get_dag_summ_backward_map(job_dags):
    # compute backward mapping from node idx to dag idx
    total_num_nodes = \
        int(np.sum([job_dag.num_nodes for job_dag in job_dags]))

    dag_summ_backward_map = \
        np.zeros([total_num_nodes, len(job_dags)])

    base = 0
    j_idx = 0
    for job_dag in job_dags:
        for node in job_dag.nodes:
            dag_summ_backward_map[base + node.idx, j_idx] = 1
        base += job_dag.num_nodes
        j_idx += 1

    return dag_summ_backward_map


def get_running_dag_mat(job_dags):
    # this is from the legacy code
    # now all the jobs in job_dags should be unfinished
    running_dag_row_idx = []
    running_dag_col_idx = []
    running_dag_data = []
    running_dag_shape = (1, len(job_dags))

    j_idx = 0
    for job_dag in job_dags:
        if not job_dag.completed:
            running_dag_row_idx.append(0)
            running_dag_col_idx.append(j_idx)
            running_dag_data.append(1)

        j_idx += 1

    running_dag_indices = np.mat(
        [running_dag_row_idx, running_dag_col_idx]).transpose()
    running_dag_mat = tf.SparseTensorValue(
        running_dag_indices, running_dag_data, running_dag_shape)

    return running_dag_mat


def merge_masks(masks):
    """
    e.g.,

    [0, 1, 0]  [0, 1]  [0, 0, 0, 1]
    [0, 0, 1]  [1, 0]  [1, 0, 0, 0]
    [1, 0, 0]  [0, 0]  [0, 1, 1, 0]

    to

    a list of
    [0, 1, 0, 0, 1, 0, 0, 0, 1]^T,
    [0, 0, 1, 1, 0, 1, 0, 0, 0]^T,
    [1, 0, 0, 0, 0, 0, 1, 1, 0]^T

    Note: mask dimension d is pre-determined
    """
    merged_masks = []

    for d in range(args.max_depth):

        merged_mask = []
        for mask in masks:
            merged_mask.append(mask[d:d+1, :].transpose())

        if len(merged_mask) > 0:
            merged_mask = np.vstack(merged_mask)

        merged_masks.append(merged_mask)

    return merged_masks


def get_unfinished_nodes_summ_mat(job_dags):
    # 1. connect the unfinished nodes to "summarized node"
    # 2. silent out all the nodes that's already done
    # O(num_total_nodes)

    total_num_nodes = \
        np.sum([job_dag.num_nodes for job_dag in job_dags])

    summ_row_idx = []
    summ_col_idx = []
    summ_data = []
    summ_shape = (len(job_dags), total_num_nodes)

    base = 0
    j_idx = 0
    for job_dag in job_dags:

        for node in job_dag.nodes:
            if not node.tasks_all_done:
                summ_row_idx.append(j_idx)
                summ_col_idx.append(base + node.idx)
                summ_data.append(1)

        base += job_dag.num_nodes
        j_idx += 1

    summ_indices = np.mat([summ_row_idx, summ_col_idx]).transpose()
    summerize_mat = tf.SparseTensorValue(
        summ_indices, summ_data, summ_shape)

    return summerize_mat
