import numpy as np
from param import *
from agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node


class DynamicPartitionAgent(Agent):
    # dynamically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self):
        Agent.__init__(self)

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum([any(n.next_task_idx + \
            exec_commit.node_commit[n] + moving_executors.count(n) \
            < n.num_tasks for n in job_dag.nodes) \
            for job_dag in job_dags])

        # compute the executor cap
        exec_cap = int(np.ceil(args.exec_cap / max(1, num_unfinished_jobs)))

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

        scheduled = False
        # first assign executor to the same job
        if source_job is not None:
            # immediately scheduable nodes
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    return node, num_source_exec
            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    return node, num_source_exec

        # the source job is finished or does not exist
        for job_dag in job_dags:
            if exec_map[job_dag] < exec_cap:
                next_node = None
                # immediately scheduable node first
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        next_node = node
                        break
                # then schedulable node in the job
                if next_node is None:
                    for node in frontier_nodes:
                        if node in job_dag.nodes:
                            next_node = node
                            break
                # node is selected, compute limit
                if next_node is not None:
                    use_exec = min(
                        node.num_tasks - node.next_task_idx - \
                        exec_commit.node_commit[node] - \
                        moving_executors.count(node),
                        exec_cap - exec_map[job_dag],
                        num_source_exec)
                    return node, use_exec

        # there is more executors than tasks in the system
        return None, num_source_exec
