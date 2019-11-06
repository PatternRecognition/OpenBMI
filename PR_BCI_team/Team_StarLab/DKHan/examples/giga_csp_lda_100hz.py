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

#giga science mi data csp-lda example

with open('C:\\Users\dk\PycharmProjects\giga_cnn\convert\MI_62ch_250Hz.pkl', 'rb') as f:
    data = pickle.load(f)

f = open("CSP_62ch_100hz" + datetime.today().strftime("%m_%d_%H_%M") + ".txt", 'w')

for sess in [1,2]:
    print("session:",sess)
    for sub in range(1,55):
        print("subject#",sub)
        if sess == 1 :
            epochs = data[sub-1]
        else :
            epochs = data[sub+53]

        sess= 2
        sub = 2
        sub2epo = data[51+54-1].copy()
        epochs = sub2epo.copy()
        epochs.filter(l_freq=8, h_freq=30)

        idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
        chans = np.array(epochs.ch_names)[idx].tolist()
        epochs.pick_channels(chans)

        epochs_train = epochs.copy().crop(tmin=0, tmax=4.0)

        labels = epochs.events[:, -1] - 1

        scores = []
        epochs_data = epochs.get_data()

        epochs_data_train = epochs_train.get_data()[0:100,:,:]
        epochs_data_test = epochs_train.get_data()[100:200,:,:]
        labels_train = labels[0:100]
        labels_test = labels[100:200]
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        X_train = csp.fit_transform(epochs_data_train, labels_train)
        X_test = csp.transform(epochs_data_test)

        # fit classifier
        lda.fit(X_train, labels_train)
        print(lda.score(X_test, labels_test))
        csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                          units='Patterns (AU)', size=1.5)



        evoked = epochs.average()
        evoked.data = csp.patterns_.T
        evoked.times = np.arange(evoked.data.shape[0])
        evoked.plot_topomap()

        cv = ShuffleSplit(1, random_state=42)
        cv_split = cv.split(epochs_data_train)
        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        # Use scikit-learn Pipeline with cross_val_score function

#####################윈도우##########################
        sfreq = epochs.info['sfreq']
        w_length = int(sfreq * 3)  # running classifier: window length
        w_step = int(sfreq * 0.1)  # running classifier: window step size
        w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

        scores_windows = []


        # fit classifier
        lda.fit(X_train, labels_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            epochs_data_train = epochs_train.get_data()[0:100, :, n:(n + w_length)]
            epochs_data_test = epochs_train.get_data()[100:200, :, n:(n + w_length)]
            X_train = csp.fit_transform(epochs_data_train, labels_train)
            X_test = csp.transform(epochs_data_test)
            lda.fit(X_train, labels_train)
            score_this_window.append(lda.score(X_test, labels_test))
        scores_windows.append(score_this_window)

        # Plot scores over time
        w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

        plt.figure()
        plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
        plt.axvline(0, linestyle='--', color='k', label='Onset')
        plt.axhline(0.5, linestyle='-', color='k', label='Chance')
        plt.xlabel('time (s)')
        plt.ylabel('classification accuracy')
        plt.title('Classification score over time')
        plt.legend(loc='lower right')
        plt.show()





























        # clf = Pipeline([('CSP', csp), ('LDA', lda)])
        # scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=-1, )


        csp.fit_transform(epochs_data_test, labels_test)

        layout = read_layout('EEG1005')
        csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                          units='Patterns (AU)', size=1.5)

        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)
        m_score = np.mean(scores)
        print("Classification accuracy: %f / Chance level: %f" % (m_score,
                                                                  class_balance))
        f.write(str(m_score) + '\n')

f.close()