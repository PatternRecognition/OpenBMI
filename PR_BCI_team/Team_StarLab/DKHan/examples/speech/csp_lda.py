import scipy.io as sio
import numpy as np
import os
import mne
from mayavi import mlab

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
from sklearn.model_selection import train_test_split


from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from datetime import datetime

from datasets import *

import pickle

allsubj = list()
for subj in range(1,17):
    print("subject#",subj)

    dataname = 's%d_sess3' % (subj)

    SP = sio.loadmat('C:\\Data_speech\\' + dataname
                     , struct_as_record=False, squeeze_me=True)
    temp = SP['epo']
    x = temp.x
    x = np.transpose(x, [2, 1, 0])
    y = temp.y

    chan = temp.clab.tolist()

    n_channels = 64
    sfreq = 250
    info = mne.create_info(ch_names=chan, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(x, info)
   # epochs.filter(l_freq=30,h_freq=100)


    # idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    # chans = np.array(epochs.ch_names)[idx].tolist()
    # epochs.pick_channels(chans)

    epochs_train = epochs.copy().crop(tmin=0.2, tmax=2.2)

    labels = y

    scores = []

    epochs_data = epochs_train.get_data()
    epochs_data_train = epochs_train.get_data()

    # X_train, X_test, y_train, y_test = train_test_split(epochs_data_train, labels, test_size=0.3, random_state=42)

    csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
    svc = SVC(gamma=1.5, C=10)

    # X_train_csp = csp.fit_transform(X_train, y_train)
    # X_test_csp = csp.transform(X_test)
    #
    # svc.fit(X_train_csp, y_train)
    # score = svc.score(X_test_csp, y_test)
    # print(score)



    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('SVC', svc)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

    allsubj.append(np.mean(scores))

print(allsubj)
























#     epochs_data_train, epochs_data_test, labels_train, labels_test = train_test_split(epochs_train, labels, test_size=0.3, random_state=42)
#
#
#
#
#
#
#     csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
#     X_train = csp.fit_transform(epochs_data_train, labels_train)
#     X_test = csp.transform(epochs_data_test)
#
#     # fit classifier
#     lda.fit(X_train, labels_train)
#     print(lda.score(X_test, labels_test))
#     csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
#                       units='Patterns (AU)', size=1.5)
#
#
#
#     evoked = epochs.average()
#     evoked.data = csp.patterns_.T
#     evoked.times = np.arange(evoked.data.shape[0])
#     evoked.plot_topomap()
#
#     cv = ShuffleSplit(1, random_state=42)
#     cv_split = cv.split(epochs_train)
#     # Assemble a classifier
#     lda = SVC()
#     csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
#     # Use scikit-learn Pipeline with cross_val_score function
#
# #####################윈도우##########################
#     sfreq = epochs.info['sfreq']
#     w_length = int(sfreq * 3)  # running classifier: window length
#     w_step = int(sfreq * 0.1)  # running classifier: window step size
#     w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
#
#     scores_windows = []
#
#
#     # fit classifier
#     lda.fit(X_train, labels_train)
#
#     # running classifier: test classifier on sliding window
#     score_this_window = []
#     for n in w_start:
#         epochs_data_train = epochs_train.get_data()[0:100, :, n:(n + w_length)]
#         epochs_data_test = epochs_train.get_data()[100:200, :, n:(n + w_length)]
#         X_train = csp.fit_transform(epochs_data_train, labels_train)
#         X_test = csp.transform(epochs_data_test)
#         lda.fit(X_train, labels_train)
#         score_this_window.append(lda.score(X_test, labels_test))
#     scores_windows.append(score_this_window)
#
#     # Plot scores over time
#     w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
#
#     plt.figure()
#     plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
#     plt.axvline(0, linestyle='--', color='k', label='Onset')
#     plt.axhline(0.5, linestyle='-', color='k', label='Chance')
#     plt.xlabel('time (s)')
#     plt.ylabel('classification accuracy')
#     plt.title('Classification score over time')
#     plt.legend(loc='lower right')
#     plt.show()
#
#
#
#
#     clf = Pipeline([('CSP', csp), ('LDA', lda)])
#     scores = cross_val_score(clf, epochs_train, labels, cv=cv, n_jobs=-1, )
#
#
#     csp.fit_transform(epochs_data_test, labels_test)
#
#     layout = read_layout('EEG1005')
#     csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
#                       units='Patterns (AU)', size=1.5)
#
#     class_balance = np.mean(labels == labels[0])
#     class_balance = max(class_balance, 1. - class_balance)
#     m_score = np.mean(scores)
#     print("Classification accuracy: %f / Chance level: %f" % (m_score,
#                                                               class_balance))


