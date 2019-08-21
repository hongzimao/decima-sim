import os
import sys
import itertools
import numpy as np
from collections import OrderedDict


def aggregate_gradients(gradients):

    ground_gradients = [np.zeros(g.shape) for g in gradients[0]]
    for gradient in gradients:
        for i in range(len(ground_gradients)):
            ground_gradients[i] += gradient[i]
    return ground_gradients


def compute_CDF(arr, num_bins=100):
    """
    usage: x, y = compute_CDF(arr):
           plt.plot(x, y)
    """
    values, base = np.histogram(arr, bins=num_bins)
    cumulative = np.cumsum(values)
    return base[:-1], cumulative / float(cumulative[-1])


def convert_indices_to_mask(indices, mask_len):
    mask = np.zeros([1, mask_len])
    for idx in indices:
        mask[0, idx] = 1
    return mask


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def generate_coin_flips(p):
    # generate coin flip until first head, with Pr(head) = p
    # this follows a geometric distribution
    if p == 0:
        # infinite sequence
        return np.inf

    # use geometric distribution
    flip_counts = np.random.geometric(p)

    return flip_counts


def get_outer_product_boolean_mask(job_dags, executor_limits):
    num_nodes = sum([j.num_nodes for j in job_dags])
    num_jobs = len(job_dags)
    num_exec_limits = len(executor_limits)
    mask = np.zeros([num_nodes, num_jobs * num_exec_limits], dtype=np.bool)

    # fill in valid entries
    base = 0
    for i in range(len(job_dags)):
        job_dag = job_dags[i]
        mask[base : base + job_dag.num_nodes,
             i * num_exec_limits : (i + 1) * num_exec_limits] = True
        base += job_dag.num_nodes

    # reshape into 1D array
    mask = np.reshape(mask, [-1])

    return mask


def get_poly_baseline(polyfit_model, all_wall_time):
    # use 5th order polynomial to get a baseline
    # normalize the time
    max_time = float(max([max(wall_time) for wall_time in all_wall_time]))
    max_time = max(1, max_time)
    baselines = []
    for i in range(len(all_wall_time)):
        normalized_time = [t / max_time for t in all_wall_time[i]]
        baseline = polyfit_model[0] * np.power(normalized_time, 5) + \
                   polyfit_model[1] * np.power(normalized_time, 4) + \
                   polyfit_model[2] * np.power(normalized_time, 3) + \
                   polyfit_model[3] * np.power(normalized_time, 2) + \
                   polyfit_model[4] * np.power(normalized_time, 1) + \
                   polyfit_model[5]
        baselines.append(baseline)
    return baselines


def get_wall_time_baseline(all_cum_rewards, all_wall_time):
    # do a 5th order polynomial fit over time
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
    polyfit_model = np.polyfit(list_wall_time, list_cum_rewards, 5)
    baselines = get_poly_baseline(polyfit_model, all_wall_time)
    return baselines


def increase_var(var, max_var, increase_rate):
    if var + increase_rate <= max_var:
        var += increase_rate
    else:
        var = max_var
    return var


def list_to_str(lst):
    """
    convert list of number of a string with space
    """
    return ' '.join([str(e) for e in lst])


def min_nonzero(x):
    min_val = np.inf
    for i in x:
        if i != 0 and i < min_val:
            min_val = i
    return min_val


def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


class OrderedSet(object):
    def __init__(self, contents=()):
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, item):
        return item in self.set

    def __iter__(self):
        return iter(self.set.keys())

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set[item] = None

    def clear(self):
        self.set.clear()

    def index(self, item):
        idx = 0
        for i in self.set.keys():
            if item == i:
                break
            idx += 1
        return idx

    def pop(self):
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, item):
        del self.set[item]

    def to_list(self):
        return [k for k in self.set]

    def update(self, contents):
        for c in contents:
            self.add(c)


def progress_bar(count, total, status='', pattern='|', back='-'):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = pattern * filled_len + back * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

    if count == total:
        print('')


class SetWithCount(object):
    """
    allow duplication in set
    """
    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]


def truncate_experiences(lst):
    """
    truncate experience based on a boolean list
    e.g.,    [True, False, False, True, True, False]
          -> [0, 3, 4, 6]  (6 is dummy)
    """
    batch_points = [i for i, x in enumerate(lst) if x]
    batch_points.append(len(lst))

    return batch_points
