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


import torch

def extract_embeddings(dataloader,model, cuda):
    with torch.no_grad():
        model.eval()
        embedding_net = model.embedding_net
        num_hidden = model.embedding_net.num_hidden
        clf_net = model.clf_net

        # num_ftrs = model.embedding_net.fc.out_features
        embeddings = np.zeros((len(dataloader.dataset), num_hidden))
        labels = np.zeros(len(dataloader.dataset))
        preds = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()

            embeddings[k:k + len(images)] = embedding_net(images).data.cpu().numpy()
            output = clf_net(images)
            pred = output.argmax(dim=1, keepdim=True)

            labels[k:k + len(images)] = target.numpy()
            preds[k:k + len(images)] = pred.squeeze().cpu().numpy()

            k += len(images)
    return embeddings, labels, preds
