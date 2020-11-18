import scipy.io as sio
import numpy as np
import os
import mne
import gigadata
from mayavi import mlab

import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from pyriemann.estimation import Covariances

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from datetime import datetime

from sklearn.preprocessing import StandardScaler
import pickle

with open('cov_data_train.pkl', 'rb') as f:
    x_data = pickle.load(f)


scaler = StandardScaler()
i=0
for i in range(x_data.shape[0]):
    if i == 0:
        cov_fit = scaler.fit_transform(x_data[i, :, :])
        cov_data_train = np.expand_dims(cov_fit, axis=0)
    else:
        cov_fit = np.expand_dims(scaler.fit_transform(x_data[i, :, :]), axis=0)
        cov_data_train = np.append(cov_data_train, cov_fit, axis=0)

    print(i)


with open('cov_data_train_scale.pkl', 'wb') as f:
    pickle.dump(cov_data_train, f)
