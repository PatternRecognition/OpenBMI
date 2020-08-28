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
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

import pickle

with open('C:/Users/dk/PycharmProjects/giga_cnn/convert/smt100_ica_2.pkl', 'rb') as f:
    x_data = pickle.load(f)


scaler = StandardScaler()
i=0

raw_data_train = np.zeros_like(x_data)
for i in range(x_data.shape[0]):
    raw_fit = maxabs_scale(x_data[i, :, :])
    raw_data_train[i,:,:] = raw_fit[:,:]
    print(i)


with open('C:/Users/dk/PycharmProjects/giga_cnn/convert/smt100_ica_maxabs_2.pkl', 'wb') as f:
    pickle.dump(raw_data_train, f)
