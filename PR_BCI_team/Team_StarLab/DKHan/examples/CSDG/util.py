import torch as th
from torch.autograd import Variable
import numpy as np
import random

def np_to_var(X, requires_grad=False, dtype=None, pin_memory=False,
              **tensor_kwargs):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, '__len__'):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor
#
#
# from sklearn.model_selection import KFold
# allsubj = np.r_[0:54]
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
# kf = KFold(n_splits=10,random_state=None,shuffle=True)
# kf.get_n_splits(allsubj)
#
# from datasets import load_smt
# x_data, y_data = load_smt()
# #get subject number
# y_subj = np.zeros([108, 200])
# for i in range(108):
#     y_subj[i, :] = i * 2
# y_subj = y_data.reshape(108, 200) + y_subj
# y_subj = y_subj.reshape(21600)
# #y_subj = np.concatenate([y_data,y_subj],axis=1)
#
# # For classification data
# valtype='subj'
#
#
#
#
# print(kf)
# for train_index, test_index in kf.split(allsubj):
#     print("TRAIN:", train_index, "TEST:", test_index)
#
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
