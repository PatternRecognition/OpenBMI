import scipy.io as sio
import numpy as np
import os
import mne
import gigadata

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

import pickle

with open('MI_62ch_100Hz_4-50.pkl', 'rb') as f:
    data = pickle.load(f)

sess = 2
sub= 43

for sess in [1,2]:
    print("session:",sess)
    for sub in range(1,55):
        print("subject#",sub)
        if sess == 1 :
            epochs = data[sub-1]
        else :
            epochs = data[sub+53]


        epochs_train = epochs.copy()


        if sess == 1 and sub ==1:
            epochs_data_train = epochs_train.get_data()
            labels = epochs.events[:, -1] - 1
        else:
            epoch_temp = epochs_train.get_data()
            epochs_data_train = np.append(epochs_data_train, epoch_temp,axis=0)
            label_temp = epochs.events[:, -1] - 1
            labels = np.hstack((labels, label_temp))

        print(epochs_data_train.shape)


np.save('x_data_450',epochs_data_train)
np.save('y_data',labels)

