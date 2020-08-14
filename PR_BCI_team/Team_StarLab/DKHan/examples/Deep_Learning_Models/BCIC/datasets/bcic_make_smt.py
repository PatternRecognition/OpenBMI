from moabb.datasets import BNCI2014001
from mne import Epochs, pick_types, events_from_annotations, annotations_from_events
from mne import find_events
import mne
import dill
import numpy as np
import pandas as pd
import copy

from datasets.bcic import BcicDataset

#download and load dataset
dataset = BNCI2014001()
dataset.subject_list = dataset.subject_list[:10]
data= dataset.get_data()

df = pd.DataFrame()

l_freq = 0.; h_freq = 38.

#get Epochs
datasets = []
for subj_id, subj_data in data.items():
    epochs_list = []
    for sess_id, sess_data in subj_data.items():
        for run_id, raw in sess_data.items():
            events = find_events(raw)
            onset,offset = dataset.interval
            events[:,0] += onset*250
            raw2 = raw.pick_types(eeg=True, meg=False, stim=False)
            raw2 = raw2.filter(l_freq,h_freq)

            epoch = Epochs(raw2, events, event_id=dataset.event_id, tmin=-0.5, tmax=4)
            epoch._data = epoch.get_data()
            #epoch.load_data()
            epochs_list.append(epoch)
    epochs = mne.concatenate_epochs(epochs_list)
    #epoch to torch.Dataset
    datasets.append(BcicDataset(epochs,subj_id))



with open('bcic_datasets_[0,38].pkl', 'wb', ) as f:
    dill.dump(datasets, f)
