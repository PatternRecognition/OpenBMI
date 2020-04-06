import scipy.io as sio
import numpy as np
import os
import mne
from gigadata import *
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

import matplotlib.pyplot as plt

import pickle

allepo = list()
for sess in [1,2]:
    print("session:",sess)
    for sub in range(1,55):
        print("subject#",sub)
        dataname = 'sess%02d_subj%02d_EEG_MI.mat' % (sess, sub)
        path = 'C:/data/Data(MIonly)/' + dataname
        MI = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        raw1, events1 = load_gigadata(MI)
        raw2, events2 = load_gigadata(MI, istrain=False)

        idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
        chans = np.array(epochs.ch_names)[idx].tolist()
        epochs.pick_channels(chans)

        epochs1 = gigadata_epochs(raw1, events1, tmin=-1., tmax=8.)
        epochs2 = gigadata_epochs(raw2, events2, tmin=-1., tmax=8.)
        epochs = mne.concatenate_epochs([epochs1,epochs2])
        epochs.filter(l_freq=8, h_freq=30)
        epochs.resample(100)

        allepo.append(epochs.copy())


with open('MI_20ch_100Hz_8-30.pkl', 'wb') as f:
    pickle.dump(allepo, f,protocol=4)
