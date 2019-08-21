from param import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def visualize_executor_usage(job_dags, file_path):
    exp_completion_time = int(np.ceil(np.max([
        j.completion_time for j in job_dags])))

    job_durations = \
        [job_dag.completion_time - \
        job_dag.start_time for job_dag in job_dags]

    executor_occupation = np.zeros(exp_completion_time)
    executor_limit = np.ones(exp_completion_time) * args.exec_cap

    num_jobs_in_system = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                executor_occupation[
                    int(task.start_time) : \
                    int(task.finish_time)] += 1
        num_jobs_in_system[
            int(job_dag.start_time) : \
            int(job_dag.completion_time)] += 1

    executor_usage = \
        np.sum(executor_occupation) / np.sum(executor_limit)

    fig = plt.figure()

    plt.subplot(2, 1, 1)
    # plt.plot(executor_occupation)
    # plt.fill_between(range(len(executor_occupation)), 0,
    #                  executor_occupation)
    plt.plot(moving_average(executor_occupation, 10000))

    plt.ylabel('Number of busy executors')
    plt.title('Executor usage: ' + str(executor_usage) + \
              '\n average completion time: ' + \
              str(np.mean(job_durations)))

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_system)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(file_path)
    plt.close(fig)


def visualize_dag_time(job_dags, executors, plot_total_time=None, plot_type='stage'):

    dags_makespan = 0
    all_tasks = []
    # 1. compute each DAG's finish time
    # so that we can visualize it later
    dags_finish_time = []
    dags_duration = []
    for dag in job_dags:
        dag_finish_time = 0
        for node in dag.nodes:
            for task in node.tasks:
                all_tasks.append(task)
                if task.finish_time > dag_finish_time:
                    dag_finish_time = task.finish_time
        dags_finish_time.append(dag_finish_time)
        assert dag_finish_time == dag.completion_time
        dags_duration.append(dag_finish_time - dag.start_time)

    # 2. visualize them in a canvas
    if plot_total_time is None:
        canvas = np.ones([len(executors), int(max(dags_finish_time))]) * args.canvas_base
    else:
        canvas = np.ones([len(executors), int(plot_total_time)]) * args.canvas_base

    base = 0
    bases = {}  # job_dag -> base

    for job_dag in job_dags:
        bases[job_dag] = base
        base += job_dag.num_nodes

    for task in all_tasks:

        start_time = int(task.start_time)
        finish_time = int(task.finish_time)
        exec_id = task.executor.idx

        if plot_type == 'stage':

            canvas[exec_id, start_time : finish_time] = \
                bases[task.node.job_dag] + task.node.idx

        elif plot_type == 'app':
            canvas[exec_id, start_time : finish_time] = \
                job_dags.index(task.node.job_dag)

    return canvas, dags_finish_time, dags_duration


def visualize_dag_time_save_pdf(
        job_dags, executors, file_path, plot_total_time=None, plot_type='stage'):
    
    canvas, dag_finish_time, dags_duration = \
        visualize_dag_time(job_dags, executors, plot_total_time, plot_type)

    fig = plt.figure()

    # canvas
    plt.imshow(canvas, interpolation='nearest', aspect='auto')
    # plt.colorbar()
    # each dag finish time
    for finish_time in dag_finish_time:
        plt.plot([finish_time, finish_time],
                 [- 0.5, len(executors) - 0.5], 'r')
    plt.title('average DAG completion time: ' + str(np.mean(dags_duration)))
    fig.savefig(file_path)
    plt.close(fig)

