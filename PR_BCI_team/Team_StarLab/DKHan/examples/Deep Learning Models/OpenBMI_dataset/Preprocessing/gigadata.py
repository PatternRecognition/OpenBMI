import scipy.io as sio
import numpy as np
import mne

def load_gigadata(MI, plot=False, istrain=True):
    if istrain == True:
        temp = MI['EEG_MI_train']
    else:
        temp = MI['EEG_MI_test']
    sfreq = 1000  # Sampling frequency
    chan = temp.chan.tolist()

    # 채널정보 추가해야함
    info = mne.create_info(ch_names=chan, sfreq=sfreq, ch_types='eeg')

    t = np.hstack((temp.t.reshape(100, 1), np.zeros((100, 1))))
    y_label = temp.y_dec.reshape(100, 1)
    events = np.hstack((t, y_label)).astype('int')

    raw = mne.io.RawArray(temp.x.T, info)
    if plot == True:
        scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
        raw.plot(n_channels=62, scalings=scalings, title='Auto-scaled Data from arrays',
                 show=True, block=True)

    return raw, events

def gigadata_epochs(raw,events,tmin=-1,tmax=3,plot=False):
    epochs = mne.Epochs(raw, events=events, event_id=[1, 2], tmin=tmin,
                        tmax=tmax, baseline=None, verbose=True, preload=True)
    if plot == True:
        epochs.plot(scalings='auto', block=True)
    return epochs

#기가  데이터 mat파일을  로드해서  mne형식으로  변환