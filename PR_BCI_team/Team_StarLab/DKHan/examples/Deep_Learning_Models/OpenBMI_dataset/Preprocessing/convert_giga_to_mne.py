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

        raw1.filter(l_freq=4, h_freq=50, fir_design='firwin', skip_by_annotation='edge')
        raw2.filter(l_freq=4, h_freq=50, fir_design='firwin', skip_by_annotation='edge')

        epochs1 = gigadata_epochs(raw1, events1, tmin=-1., tmax=4.)
        epochs2 = gigadata_epochs(raw2, events2, tmin=-1., tmax=4.)
        epochs = mne.concatenate_epochs([epochs1,epochs2])
        epochs.resample(100)
        allepo.append(epochs.copy())


with open('MI_62ch_100Hz_4-50.pkl', 'wb') as f:
    pickle.dump(allepo, f,protocol=4)
