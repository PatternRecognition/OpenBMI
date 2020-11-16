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
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap


def csp_plot(epochs, sel_ch = False):

    if sel_ch is True:
        idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
        chans = np.array(epochs.ch_names)[idx].tolist()
        epochs.pick_channels(chans)
    # epochs.pick_channels(["C3", "Cz", "C4"])
    epochs.filter(l_freq=70, h_freq=80)
    epochs_train = epochs.copy().crop(tmin=0., tmax=1)


    labels = epochs.events[:, -1] - 1

    scores = []
    # epochs_train = epochs_train.filter(8,30)
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

def erds_plot(epochs):
    # compute ERDS maps ###########################################################
    # idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    # chans = np.array(epochs.ch_names)[idx].tolist()
    epochs.pick_channels(["C3","Cz","C4"])


    event_ids = epochs.event_id
    tmin, tmax = -1, 4
    freqs = np.arange(60, 90, 1)  # frequencies from 2-35Hz
    n_cycles = freqs  # use constant t/f resolution
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    # vmin, vmax = -0.5, 0.5  # set min and max ERDS values in plot
    baseline = [-1, 0.3]  # baseline interval (in s)
    cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test

    # Run TF decomposition overall epochs
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                         use_fft=True, return_itc=False, average=False,
                         decim=2)
    tfr.crop(tmin, tmax)
    tfr.apply_baseline(baseline, mode="percent")
    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                                 gridspec_kw={"width_ratios": [10, 10, 10, 1]})
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                         **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                                  axes=ax, colorbar=False, show=False, mask=mask,
                                  mask_style="mask")

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if not ax.is_first_col():
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle("ERDS ({})".format(event))
        fig.show()


import pickle

with open('MI_62ch_250Hz_raw.pkl', 'rb') as f:
    data = pickle.load(f)

sess = 2
sub= 28

# target = np.r_[1, 2, 4, 5, 8, 17, 18, 20, 21, 27, 28, 32, 35, 36, 42, 43, 44, 51]+1

target = [36] #1부터 시작하는 인덱스로
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
    csp_plot(epochs_train,True)
    erds_plot(epochs_train)
    epochs_train.plot_psd_topomap()
    epochs_train.plot_psd()

    # epochs.plot_psd_topomap()

# montage = mne.channels.make_standard_montage('standard_1005')
# epochs_train.set_montage(montage)
# epochs_train.plot_psd()
# csp_plot(epochs_train)

#numpy로  저장