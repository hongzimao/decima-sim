# Decima

Simulator part of Decima (SIGCOMM '19) https://web.mit.edu/decima

Example:

Train Decima with 50 executors, 200 streaming jobs, 25 second Poisson job arrival interval (load ~85%), stochastic termination, input-dependent baseline and average reward, run
```
python3 train.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 200 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --num_agents 16 --model_save_interval 100 --model_folder ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/
```

Use `tensorboard` to monitor the training process, some screenshots of the results are in `results/`

Test Decima after 10,000 iterations with 50 executors, 5000 streaming jobs (>10x longer than training), run
```
python3 test.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 5000 --canvs_visualization 0 --test_schemes dynamic_partition learn --num_exp 1 --saved_model ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/model_ep_10000
```

Some example output are in `results/`

We are currently in the process of refactoring the Spark implementation.
