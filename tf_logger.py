import tensorflow as tf
from time import gmtime, strftime
from param import *


class TFLogger(object):
    def __init__(self, sess, var_list):
        self.sess = sess

        self.summary_vars = []

        for var in var_list:
            tf_var = tf.Variable(0.)
            tf.summary.scalar(var, tf_var)
            self.summary_vars.append(tf_var)

        self.summary_ops = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            args.result_folder + args.model_folder + \
            strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    def log(self, ep, values):
        assert len(self.summary_vars) == len(values)

        feed_dict = {self.summary_vars[i]: values[i] \
            for i in range(len(values))}

        summary_str = self.sess.run(
            self.summary_ops, feed_dict=feed_dict)

        self.writer.add_summary(summary_str, ep)
        self.writer.flush()
