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

sess = 1
sub = 1
allsub_scores=[]
f = open("csp_cv_sess1_10fold"+datetime.today().strftime("%m_%d_%H_%M")+".txt", 'w')
for sub in range(1,55):
    dataname = 'sess%02d_subj%02d_EEG_MI.mat' % (sess, sub)
    path = 'C:/Data_MI/' + dataname
    MI = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    raw1, events1 = gigadata.load_gigadata(MI)
    raw2, events2 = gigadata.load_gigadata(MI, istrain=False)

    # bandpass [8-30Hz]
    raw1.filter(l_freq=8, h_freq=30, fir_design='firwin', skip_by_annotation='edge')
    raw2.filter(l_freq=8, h_freq=30, fir_design='firwin', skip_by_annotation='edge')

    idx = np.array(list(range(7,11))+list(range(12,15))+list(range(17,21))+list(range(32,41)))
    chans = np.array(raw1.ch_names)[idx].tolist()

    raw1.pick_channels(chans)
    raw2.pick_channels(chans)

    epochs1 = gigadata.gigadata_epochs(raw1, events1, tmin=-1., tmax=4.)
    epochs2 = gigadata.gigadata_epochs(raw2, events2, tmin=-1., tmax=4.)
    epochs = mne.concatenate_epochs([epochs1,epochs2])
    epochs.resample(sfreq=100)
##############################################################
#########################CSP-CV###############################

    epochs_train = epochs.copy().crop(tmin=1., tmax=3.5)

    labels = epochs.events[:, -1]-1

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
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=-1,)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    m_score = np.mean(scores)
    print("Classification accuracy: %f / Chance level: %f" % (m_score,
                                                              class_balance))


    allsub_scores= allsub_scores + [m_score]
    f.write(str(m_score) + '\n')
f.close()
'''
    csp.fit_transform(epochs_data, labels)

    layout = read_layout('EEG1005')
    csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                      units='Patterns (AU)', size=1.5)

'''

