import scipy.io as sio
import numpy as np
import os
import mne
import gigadata
from mayavi import mlab

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from datetime import datetime

import pickle

with open('MI_8_30.pkl', 'rb') as f:
    data = pickle.load(f)

f = open("csp_cv_sess1_10fold" + datetime.today().strftime("%m_%d_%H_%M") + ".txt", 'w')

for sess in [1,2]:
    print("session:",sess)
    for sub in range(1,55):
        print("subject#",sub)
        if sess == 1 :
            epochs = data[sub-1]
        else :
            epochs = data[sub+53]
        epochs_train = epochs.copy().crop(tmin=1., tmax=3.5)

        labels = epochs.events[:, -1] - 1

        scores = []
        epochs_data = epochs.get_data()

        epochs_data_train = epochs_train.get_data()
        cv = ShuffleSplit(10, random_state=42)
        cv_split = cv.split(epochs_data_train)
        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=-1, )

        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)
        m_score = np.mean(scores)
        print("Classification accuracy: %f / Chance level: %f" % (m_score,
                                                                  class_balance))
        f.write(str(m_score) + '\n')

f.close()