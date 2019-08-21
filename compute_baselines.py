import numpy as np
import bisect
import itertools


def get_ployfit_baseline(all_cum_rewards, all_wall_time, polyfit_order=5):
    # do a nth order polynomial fit over time
    # all_cum_rewards: list of lists of cumulative rewards
    # all_wall_time:   list of lists of physical time
    assert len(all_cum_rewards) == len(all_wall_time)

    # build one list of all values
    list_cum_rewards = list(itertools.chain.from_iterable(all_cum_rewards))
    list_wall_time = list(itertools.chain.from_iterable(all_wall_time))
    assert len(list_cum_rewards) == len(list_wall_time)

    # normalize the time by the max time
    max_time = float(max(list_wall_time))
    max_time = max(1, max_time)
    list_wall_time = [t / max_time for t in list_wall_time]
    polyfit_model = np.polyfit(list_wall_time, list_cum_rewards, polyfit_order)

    # use nth order polynomial to get a baseline
    # normalize the time
    max_time = float(max([max(wall_time) for wall_time in all_wall_time]))
    max_time = max(1, max_time)
    baselines = []
    for i in range(len(all_wall_time)):
        normalized_time = [t / max_time for t in all_wall_time[i]]
        baseline = sum(polyfit_model[o] * \
                       np.power(normalized_time, polyfit_order - o) \
                       for o in range(polyfit_order + 1))
        baselines.append(baseline)

    return baselines

def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    # do a piece-wise linear fit baseline
    # all_cum_rewards: list of lists of cumulative rewards
    # all_wall_time:   list of lists of physical time
    assert len(all_cum_rewards) == len(all_wall_time)

    # all unique wall time
    unique_wall_time = np.unique(np.hstack(all_wall_time))

    # for find baseline value for all unique time points
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(len(all_wall_time))

    # output n baselines
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)

    return baselines
