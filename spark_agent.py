import numpy as np
from agent import Agent


class SparkAgent(Agent):
    # statically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, exec_cap):
        Agent.__init__(self)

        # executor limit set to each job
        self.exec_cap = exec_cap

        # map for executor assignment
        self.exec_map = {}

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        # sort out the new exec_map
        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]

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
            if self.exec_map[job_dag] < self.exec_cap:
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
                        self.exec_cap - self.exec_map[job_dag],
                        num_source_exec)
                    self.exec_map[job_dag] += use_exec
                    return node, use_exec

        # there is more executors than tasks in the system
        return None, num_source_exec
