import os
# os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from param import *
from utils import *
from spark_env.env import Environment
from average_reward import *
from compute_baselines import *
from compute_gradients import *
from actor_agent import ActorAgent
from tf_logger import TFLogger


def invoke_model(actor_agent, obs, exp):
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
        exec_map, job_dags_changed = \
            actor_agent.invoke_model(obs)

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
        len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[
            job_act[0, job_idx]] - \
            exec_map[node.job_dag] + \
            num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[
            job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(
        node.num_tasks - node.next_task_idx - \
        exec_commit.node_commit[node] - \
        moving_executors.count(node),
        agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp['node_inputs'].append(node_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['job_act_vec'].append(job_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['job_valid_mask'].append(job_valid_mask)
    exp['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # model evaluation seed
    tf.set_random_seed(agent_id)

    # set up environment
    env = Environment()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, max_time, entropy_weight) = \
            param_queue.get()
        
        # synchronize model
        actor_agent.set_params(actor_params)

        # reset environment
        env.seed(seed)
        env.reset(max_time=max_time)

        # set up storage for experience
        exp = {'node_inputs': [], 'job_inputs': [], \
               'gcn_mats': [], 'gcn_masks': [], \
               'summ_mats': [], 'running_dag_mat': [], \
               'dag_summ_back_mat': [], \
               'node_act_vec': [], 'job_act_vec': [], \
               'node_valid_mask': [], 'job_valid_mask': [], \
               'reward': [], 'wall_time': [],
               'job_state_change': []}

        try:
            # The masking functions (node_valid_mask and
            # job_valid_mask in actor_agent.py) has some
            # small chance (once in every few thousand
            # iterations) to leave some non-zero probability
            # mass for a masked-out action. This will
            # trigger the check in "node_act and job_act
            # should be valid" in actor_agent.py
            # Whenever this is detected, we throw out the
            # rollout of that iteration and try again.

            # run experiment
            obs = env.observe()
            done = False

            # initial time
            exp['wall_time'].append(env.wall_time.curr_time)

            while not done:
                
                node, use_exec = invoke_model(actor_agent, obs, exp)

                obs, reward, done = env.step(node, use_exec)

                if node is not None:
                    # valid action, store reward and time
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)
                elif len(exp['reward']) > 0:
                    # Note: if we skip the reward when node is None
                    # (i.e., no available actions), the sneaky
                    # agent will learn to exhaustively pick all
                    # nodes in one scheduling round, in order to
                    # avoid the negative reward
                    exp['reward'][-1] += reward
                    exp['wall_time'][-1] = env.wall_time.curr_time

            # report reward signals to master
            assert len(exp['node_inputs']) == len(exp['reward'])
            reward_queue.put(
                [exp['reward'], exp['wall_time'],
                len(env.finished_job_dags),
                np.mean([j.completion_time - j.start_time \
                         for j in env.finished_job_dags]),
                env.wall_time.curr_time >= env.max_time])

            # get advantage term from master
            batch_adv = adv_queue.get()

            if batch_adv is None:
                # some other agents panic for the try and the
                # main thread throw out the rollout, reset and
                # try again now
                continue

            # compute gradients
            actor_gradient, loss = compute_actor_gradients(
                actor_agent, exp, batch_adv, entropy_weight)

            # report gradient to master
            gradient_queue.put([actor_gradient, loss])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            reward_queue.put(None)
            # need to still get from adv_queue to 
            # prevent blocking
            adv_queue.get()


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # tensorboard logging
    tf_logger = TFLogger(sess, [
        'actor_loss', 'entropy', 'value_loss', 'episode_length',
        'average_reward_per_second', 'sum_reward', 'reset_probability',
        'num_jobs', 'reset_hit', 'average_job_duration',
        'entropy_weight'])

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([
                actor_params, args.seed + ep,
                max_time, entropy_weight])

        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
        all_num_finished_jobs, all_avg_job_duration, \
        all_reset_hit, = [], [], [], [], [], []

        t1 = time.time()

        # get reward from agents
        any_agent_panic = False

        for i in range(args.num_agents):
            result = reward_queues[i].get()

            if result is None:
                any_agent_panic = True
                continue
            else:
                batch_reward, batch_time, \
                    num_finished_jobs, avg_job_duration, \
                    reset_hit = result

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_list_filter_zero(
                batch_reward, diff_time)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            for i in range(args.num_agents):
                adv_queues[i].put(None)
            continue

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                    (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                    (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            adv_queues[i].put(batch_adv)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / \
                float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(
            aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        tf_logger.log(ep, [
            np.mean(all_action_loss),
            np.mean(all_entropy),
            np.mean(all_value_loss),
            np.mean([len(b) for b in baselines]),
            avg_per_step_reward * args.reward_scale,
            np.mean([cr[0] for cr in all_cum_reward]),
            reset_prob,
            np.mean(all_num_finished_jobs),
            np.mean(all_reset_hit),
            np.mean(all_avg_job_duration),
            entropy_weight])

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
            args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob,
            args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + \
                'model_ep_' + str(ep))

    sess.close()


if __name__ == '__main__':
    main()