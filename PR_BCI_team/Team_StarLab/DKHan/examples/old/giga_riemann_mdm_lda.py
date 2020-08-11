import scipy.io as sio
import numpy as np
import os
import mne
import gigadata
from mayavi import mlab

import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from pyriemann.classification import MDM, TSclassifier
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

f = open("MDM_62ch_100hz" + datetime.today().strftime("%m_%d_%H_%M") + ".txt", 'w')

sess = 1
sub= 27

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

        labels = epochs.events[:, -1] - 1

        scores = []
        epochs_data = epochs.get_data()

        epochs_data_train = epochs_train.get_data()



        cov_data_train = Covariances(estimator='lwf').transform(epochs_data_train)

        cv = ShuffleSplit(10, random_state=42)
        cv_split = cv.split(cov_data_train)
        # Assemble a classifier
        #csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        # Use scikit-learn Pipeline with cross_val_score function
        mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

        scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=-1)

        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)
        m_score = np.mean(scores)
        print("Classification accuracy: %f / Chance level: %f" % (m_score,
                                                                  class_balance))
        f.write(str(m_score) + '\n')

f.close()
