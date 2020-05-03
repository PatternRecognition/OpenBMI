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

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from datetime import datetime


def csp_plot(epochs):
    # idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    # chans = np.array(epochs.ch_names)[idx].tolist()
    # epochs.pick_channels(chans)

    epochs_train = epochs.copy().crop(tmin=0.0, tmax=1.5)
    labels = epochs.events[:, -1] - 1

    scores = []

    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    m_score = np.mean(scores)
    print("Classification accuracy: %f / Chance level: %f" % (m_score,
                                                              class_balance))
    #


    epochs_data = epochs.get_data()
    # epochs.plot_sensors(ch_type='eeg')
    csp.fit_transform(epochs_data, labels)
    layout = read_layout('EEG1005')

    # layout.plot()
    montage = mne.channels.make_standard_montage('standard_1005')
    epochs.set_montage(montage)
    csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                      units='Patterns (AU)', size=1.5)


import pickle

with open('MI_62ch_100Hz_4-50.pkl', 'rb') as f:
    data = pickle.load(f)

sess = 2
sub= 43

# target = np.r_[1, 2, 4, 5, 8, 17, 18, 20, 21, 27, 28, 32, 35, 36, 42, 43, 44, 51]+1

target = [30]
for sub in target:
    print("subject#", sub)
    if sess == 1:
        epochs = data[sub - 1]
    else:
        epochs = data[sub + 53]

    # epochs.filter(l_freq=8, h_freq=80) #원래 8-30이었음

    # idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    # chans = np.array(epochs.ch_names)[idx].tolist()
    # epochs.pick_channels(chans)

    epochs_train = epochs.copy()

    montage = mne.channels.make_standard_montage('standard_1005')
    epochs_train.set_montage(montage)
    csp_plot(epochs_train)
    epochs_train.plot_psd_topomap()
    epochs_train.plot_psd()

    # epochs.plot_psd_topomap()

# montage = mne.channels.make_standard_montage('standard_1005')
# epochs_train.set_montage(montage)
# epochs_train.plot_psd()
# csp_plot(epochs_train)

#numpy로  저장