import numpy as np
import tensorflow as tf


class SparseMat(object):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row = []
        self.col = []
        self.data = []

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_col(self):
        return np.array(self.col)

    def get_row(self):
        return np.array(self.row)

    def get_data(self):
        return np.array(self.data)

    def to_tfsp_matrix(self):
        indices = np.mat([row, col]).transpose()
        return tf.SparseTensorValue(indices, data, self.shape)


def absorb_sp_mats(in_mats, depth):
    """
    Merge multiple sparse matrices to 
    a giant one on its diagonal

    e.g., 
    
    [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
    [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
    [0, 0, 1]    [1, 0, 0]    [0, 1, 0]
    
    to 
    
    [0, 1, 0]
    [1, 0, 0]   ..  ..    ..  ..
    [0, 0, 1]
              [0, 1, 0]
     ..  ..   [0, 0, 1]   ..  ..
              [1, 0, 0]
                        [0, 0, 1]
     ..  ..    ..  ..   [0, 1, 0]
                        [0, 1, 0]

    where ".." are all zeros

    depth is on the 3rd dimension,
    which is orthogonal to the planar 
    operations above

    output SparseTensorValue from tensorflow
    """
    sp_mats = []

    for d in range(depth):
        row_idx = []
        col_idx = []
        data = []
        shape = 0
        base = 0
        for m in in_mats:
            row_idx.append(m[d].get_row() + base)
            col_idx.append(m[d].get_col() + base)
            data.append(m[d].get_data())
            shape += m[d].shape[0]
            base += m[d].shape[0]

        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)

        indices = np.mat([row_idx, col_idx]).transpose()
        sp_mats.append(tf.SparseTensorValue(
            indices, data, (shape, shape)))

    return sp_mats


def expand_sp_mat(sp, exp_step):
    """
    Make a stack of same sparse matrix to 
    a giant one on its diagonal

    The input is tf.SparseTensorValue

    e.g., expand dimension 3

    [0, 1, 0]    [0, 1, 0]
    [1, 0, 0]    [1, 0, 0]  ..  ..   ..  ..
    [0, 0, 1]    [0, 0, 1]
                          [0, 1, 0]
              to  ..  ..  [1, 0, 0]  ..  ..
                          [0, 0, 1]
                                   [0, 1, 0]
                  ..  ..   ..  ..  [1, 0, 0]
                                   [0, 0, 1]
    
    where ".." are all zeros

    depth is on the 3rd dimension,
    which is orthogonal to the planar 
    operations above

    output SparseTensorValue from tensorflow
    """

    extended_mat = []

    depth = len(sp)

    for d in range(depth):
        row_idx = []
        col_idx = []
        data = []
        shape = 0
        base = 0
        for i in range(exp_step):
            indices = sp[d].indices.transpose()
            row_idx.append(np.squeeze(np.asarray(indices[0, :]) + base))
            col_idx.append(np.squeeze(np.asarray(indices[1, :]) + base))
            data.append(sp[d].values)
            shape += sp[d].dense_shape[0]
            base += sp[d].dense_shape[0]

        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)

        indices = np.mat([row_idx, col_idx]).transpose()
        extended_mat.append(tf.SparseTensorValue(
            indices, data, (shape, shape)))
    
    return extended_mat


def merge_and_extend_sp_mat(sp):
    """
    Transform a stack of sparse matrix into a giant diagonal matrix
    These sparse matrices should have same shape

    e.g.,

    list of
    [1, 0, 1, 1] [0, 0, 0, 1]
    [1, 1, 1, 1] [0, 1, 1, 1]
    [0, 0, 1, 1] [1, 1, 1, 1]

    to

    [1, 0, 1, 1]
    [1, 1, 1, 1]    ..  ..
    [0, 0, 1, 1]
                 [0, 0, 0, 1]
       ..  ..    [0, 1, 1, 1]
                 [1, 1, 1, 1]
    """

    batch_size = len(sp)
    row_idx = []
    col_idx = []
    data = []
    shape = (sp[0].dense_shape[0] * batch_size, sp[0].dense_shape[1] * batch_size)

    row_base = 0
    col_base = 0
    for b in range(batch_size):
        indices = sp[b].indices.transpose()
        row_idx.append(np.squeeze(np.asarray(indices[0, :]) + row_base))
        col_idx.append(np.squeeze(np.asarray(indices[1, :]) + col_base))
        data.append(sp[b].values)
        row_base += sp[b].dense_shape[0]
        col_base += sp[b].dense_shape[1]

    row_idx = np.hstack(row_idx)
    col_idx = np.hstack(col_idx)
    data = np.hstack(data)

    indices = np.mat([row_idx, col_idx]).transpose()
    extended_mat = tf.SparseTensorValue(indices, data, shape)

    return extended_mat
