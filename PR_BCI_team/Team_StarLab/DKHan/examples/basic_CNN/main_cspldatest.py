import scipy.io as sio
import numpy as np
import os
import mne
from mayavi import mlab

sess = 1
sub = 1
dataname = 'sess%02d_subj%02d_EEG_MI.mat' % (sess, sub)
path = 'C:/Data_MI/' + dataname
MI_s1 = sio.loadmat(path,struct_as_record=False,squeeze_me=True)

temp = MI_s1['EEG_MI_train']

sfreq = 1000  # Sampling frequency
chan = temp.chan.tolist()

#채널정보 추가해야함
info = mne.create_info(ch_names=chan, sfreq=sfreq, ch_types='eeg')

data2 =  temp.x.T

raw = mne.io.RawArray(data2, info)

scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw.plot(n_channels=15, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)

event_id = [1,2]

t = np.hstack((temp.t.reshape(100,1), np.zeros((100,1))))
y_label= temp.y_dec.reshape(100,1)
events = np.hstack((t,y_label)).astype('int')

tmin = -1.
tmax = 3  # inclusive tmax, 1 second epochs

# create :class:`Epochs <mne.Epochs>` object
epochs = mne.Epochs(raw, events=events, event_id=[1,2], tmin=tmin,
                    tmax=tmax, baseline=None, verbose=True, preload=True)
epochs.plot(scalings='auto', block=True)

epochs_train = epochs.copy().crop(tmin=-1., tmax=2.)
labels = epochs.events[:, -1] - 1

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP



scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

layout = read_layout('EEG1005')
csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                  units='Patterns (AU)', size=1.5)


####################################################################
import gigadata

raw = gigadata.load_gigadata(path, plot=True)