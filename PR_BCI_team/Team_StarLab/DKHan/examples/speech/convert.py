import scipy.io as sio
import numpy as np
import mne
import pickle
subj=1
subj_blocks = list()
for subj in range(1,17):
    print(subj)

    dataname = 's%d_sess3' % (subj)
    SP = sio.loadmat('C:\\Data_speech\\'+dataname
                     , struct_as_record=False, squeeze_me=True)
    temp = SP['epo']
    x = temp.x
    x = np.transpose(x,[2,1,0])
    y = temp.y

    chan = temp.clab.tolist()

    # 채널정보 추가해야함


    n_channels = 64
    sfreq = 250
    info = mne.create_info(ch_names=chan, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(x,info)
    # epochs.filter(l_freq=30,h_freq=100)

    subj_blocks.append(y.shape[0])
    if subj == 1:
        epochs_data_train = epochs.get_data()
        labels = y
    else:
        epoch_temp = epochs.get_data()
        epochs_data_train = np.append(epochs_data_train, epoch_temp, axis=0)
        label_temp = y
        labels = np.hstack((labels, label_temp))



    print(epochs_data_train.shape)


with open('epoch_sess3.pkl', 'wb') as f:
    pickle.dump(epochs_data_train, f, protocol=4)

with open('epoch_sess3_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

with open('epoch_sess3_sizes.pkl', 'wb') as f:
    pickle.dump(subj_blocks, f)