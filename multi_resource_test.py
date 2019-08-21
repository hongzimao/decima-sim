import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from multi_resource_env.env import MultiResEnvironment as Env
from multi_resource_agents.multi_resource_heuristic_agent import *
from multi_resource_agents.actor_agent import MultiResActorAgent
from spark_env.canvas import *
from param import *
from utils import *


# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflo seeding
tf.set_random_seed(args.seed)

# set up environment
env = Env()

# set up agents
agents = {}

for scheme in args.test_schemes:
    if scheme == 'learn':
        sess = tf.Session()
        agents[scheme] = MultiResActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, sum(args.exec_group_num) + 1), args.exec_mems)
    elif scheme == 'dynamic_partition':
        agents[scheme] = MultiResDynamicPartitionAgent()
    elif scheme == 'packer':
        agents[scheme] = MultiResPackingAgent()
    elif scheme == 'graphene':
        agents[scheme] = MultiResGrapheneAgent()
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

# store info for all schemes
all_total_reward = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []


for exp in range(args.num_exp):
    print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

    for scheme in args.test_schemes:
        # reset environment with seed
        env.seed(args.num_ep + exp)
        env.reset()

        # load an agent
        agent = agents[scheme]

        # start experiment
        obs = env.observe()

        total_reward = 0
        done = False

        while not done:
            node, exec_type, use_exec = agent.get_action(obs)
            obs, reward, done = env.step(node, exec_type, use_exec)
            total_reward += reward

        all_total_reward[scheme].append(total_reward)

        if args.canvs_visualization:
            visualize_dag_time_save_pdf(
                env.finished_job_dags, env.executors,
                args.result_folder + 'visualization_exp_' + \
                str(exp) + '_scheme_' + scheme + \
                '.png', plot_type='app')
        else:
            visualize_executor_usage(env.finished_job_dags,
                args.result_folder + 'visualization_exp_' + \
                str(exp) + '_scheme_' + scheme + '.png')


    # plot CDF of performance

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in args.test_schemes:
        x, y = compute_CDF(all_total_reward[scheme])
        ax.plot(x, y)

    plt.xlabel('Total reward')
    plt.ylabel('CDF')
    plt.legend(args.test_schemes)
    fig.savefig(args.result_folder + 'total_reward.png')

    plt.close(fig)
