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

import pickle

with open('MI_62ch_100Hz.pkl', 'rb') as f:
    data = pickle.load(f)

#f = open("MDM_62ch_100hz" + datetime.today().strftime("%m_%d_%H_%M") + ".txt", 'w')

sess = 1
sub= 1

for sess in [1,2]:
    print("session:",sess)
    for sub in range(1,55):
        print("subject#",sub)
        if sess == 1 :
            epochs = data[sub-1]
        else :
            epochs = data[sub+53]

        epochs.filter(l_freq=8, h_freq=30)

        idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
        #chans = np.array(epochs.ch_names)[idx].tolist()
        #epochs.pick_channels(chans)

        epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

        scores = []
        epochs_data = epochs.get_data()

        epochs_data_train = epochs_train.get_data()


        #covs = Covariances(estimator='lwf').transform(epochs_data_train)
        if sess == 1 and sub ==1:
            cov_data_train = Covariances(estimator='lwf').transform(epochs_data_train)
            labels = epochs.events[:, -1] - 1
        else:
            covs = Covariances(estimator='lwf').transform(epochs_data_train)
            cov_data_train = np.append(cov_data_train, covs,axis=0)
            label_temp = epochs.events[:, -1] - 1
            labels = np.hstack((labels, label_temp))

        print(cov_data_train.shape)

#f.close()



with open('cov_data_train.pkl', 'wb') as f:
    pickle.dump(cov_data_train, f)

with open('cov_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)