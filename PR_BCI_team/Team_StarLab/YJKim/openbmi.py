import numpy as np
import pandas as pd
import random
import copy
import scipy
from scipy.signal import butter, lfilter, filtfilt,resample, iirdesign, sosfiltfilt, zpk2sos, cheb2ord, buttord, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mutual_info_score
from sklearn.neighbors.kde import KernelDensity
import sklearn.feature_selection
import math


######################################### Option
def ismember(dat, list_index):
    bind = {}
    for i, elt in enumerate(list_index):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in dat]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    # y = filtfilt(b, a, data)
    # y = sosfiltfilt([b, a], data)
    return y

def chebyBandpassFilter(data, cutoff, gstop=40, gpass=1, fs=2048.):
    """
    Design a filter with scipy functions avoiding unstable results (when using
    ab output and filtfilt(), lfilter()...).
    Cf. ()[]
    Parameters
    ----------
    data : instance of numpy.array | instance of pandas.core.DataFrame
        Data to be filtered. Each column will be filtered if data is a
        dataframe.
    cutoff : array-like of float
        Pass and stop frequencies in order:
            - the first element is the stop limit in the lower bound
            - the second element is the lower bound of the pass-band
            - the third element is the upper bound of the pass-band
            - the fourth element is the stop limit in the upper bound
        For instance, [0.9, 1, 45, 48] will create a band-pass filter between
        1 Hz and 45 Hz.
    gstop : int
        The minimum attenuation in the stopband (dB).
    gpass : int
        The maximum loss in the passband (dB).
    Returns:
    zpk :
    filteredData : instance of numpy.array | instance of pandas.core.DataFrame
        The filtered data.
    """

    wp = [cutoff[1]/(fs/2), cutoff[2]/(fs/2)]
    ws = [cutoff[0]/(fs/2), cutoff[3]/(fs/2)]

    z, p, k = iirdesign(wp = wp, ws= ws, gstop=gstop, gpass=gpass,
        ftype='butter', output='zpk')
    zpk = [z, p, k]
    sos = zpk2sos(z, p, k)

    order, Wn = buttord(wp = wp, ws= ws, gstop=gstop, gpass=gpass, analog=False)
    print ('Creating cheby filter of order %d...' % order)

    if (data.ndim == 2):
        # print ('Data contain multiple columns. Apply filter on each columns.')
        filteredData = np.zeros(data.shape)
        for electrode in range(data.shape[1]):
            # print 'Filtering electrode %s...' % electrode
            filteredData[:, electrode] = sosfiltfilt(sos, data[:, electrode])
            # filteredData[:, electrode] = lfilter(sos, data[:, electrode])
    else:
        # Use sosfiltfilt instead of filtfilt fixed the artifacts at the beggining
        # of the signal
        filteredData = sosfiltfilt(sos, data)
        # filteredData = lfilter(sos, data)
    return filteredData # , zpk

######################################### Data load
#########################################
def data_structure(matt,dataset_name):

    if dataset_name == 'Giga_Science':
        x = matt['x'][0][0]
        t = matt['t'][0][0][0]
        fs = matt['fs'][0][0][0][0]
        y_dec = matt['y_dec'][0][0][0]
        y_logic = matt['y_logic'][0][0]
        class_name = matt['class'][0][0]
        chan = matt['chan'][0][0]

        class_list = []
        for ii in range(class_name.shape[0]):
            class_list.append(class_name[ii][1][0])
        class_list = np.asarray(class_list)

        chan_list = []
        for ii in range(chan.shape[1]):
            chan_list.append(chan[0][ii][0])
        chan_list = np.asarray(chan_list)

        CNT = {'x': x}
        CNT['t']=t.astype(int)
        CNT['fs'] = fs
        CNT['y']=y_dec.astype(int)
        CNT['y_logic'] = y_logic.astype(int)
        CNT['class_name'] = class_list
        CNT['channel'] = chan_list

    else:
        x = matt['x']
        y_dec = matt['y_dec'][0]
        t = matt['t'][0]
        fs = matt['fs'][0][0]
        class_name = matt['class']
        chan = matt['chan'][0]

        class_list = []
        for ii in range(len(class_name)):
            class_list.append(class_name[ii][1][0])
        class_list = np.asarray(class_list)

        chan_list = []
        for ii in range(chan.size):
            chan_list.append(chan[ii][0])
        chan_list = np.asarray(chan_list)

        CNT = {'x': x}
        CNT['y'] = y_dec
        for ii in range(len(y_dec)):
            if y_dec[ii] == 2:
                CNT['y'][ii] = 0
        CNT['t'] = t
        CNT['fs'] = fs
        CNT['channel'] = chan_list
        CNT['class_name'] = class_list

    return CNT


######################################### Cross-validation partitioning
#########################################
def random_order(CNT):
    n_total = int(CNT['y'].size / 2)
    r_order = list(range(0, n_total))
    random.shuffle(r_order)
    r_order = np.asarray(r_order)
    return r_order


def partitioning(k_fold, random_order):
    len = random_order.size
    cv_len = int(len / k_fold)

    part_cv = ()
    for i in range(0, k_fold):
        part_cv = part_cv + (random_order[i * cv_len:(i + 1) * cv_len],)
    part_cv = np.asarray(part_cv)

    index_set = []
    for i in range(0, k_fold):
        k_set = list(range(0, k_fold))
        index_te = k_set[i]
        k_set.remove(i)
        index_tr = k_set
        index_set = index_set + [(index_tr, index_te)]

    index_set = np.asarray(index_set)
    true_index = []
    for i in range(0, k_fold):
        index_tr = part_cv[index_set[i][0]]
        index_te = part_cv[index_set[i][1]]
        index_tr = np.reshape(index_tr, cv_len * (k_fold - 1))
        true_index = true_index + [(index_tr, index_te)]

    return true_index


######################################### Preprocessing
#########################################
def downsample(CNT, dfs):
    CNT_out = copy.deepcopy(CNT)
    x = CNT_out['x']
    t = CNT_out['t']
    fs = CNT_out['fs']

    n_sample = int(((x.shape[0])/fs)*dfs)
    new_sample = np.ndarray((n_sample,x.shape[1]))
    for ii in range(x.shape[1]):
        new_sample[:,ii] = resample(x[:,ii], n_sample)

    new_t = t/fs*dfs
    new_t = new_t.round(0).astype('int')

    CNT_out['x'] = new_sample
    CNT_out['fs'] = dfs
    CNT_out['t'] = new_t

    return CNT_out


def class_selection(CNT, name):
    shape_ = CNT['x'].shape
    shape = len(shape_)

    CNT_out = copy.deepcopy(CNT)
    x = CNT_out['x']
    t = CNT_out['t']
    y = CNT_out['y']

    class_all = np.asarray(CNT['class_name'])
    class_list = np.asarray(name)
    selected_idx = ismember(class_list, class_all)
    selected_logic = CNT['y_logic'][selected_idx, :]
    N_tri_class = int(shape_[1] / len(class_all))
    if shape == 2:
        t_out = []
        y_out = []
        for ii in range(len(name)):
            selected_marker = np.where(selected_logic[ii,:]==1)[0]
            t_out = np.append(t_out, np.take(t, selected_marker))
            y_out = np.append(y_out, np.take(y, selected_marker))

        class_selected = np.take(class_all, selected_idx, axis=0)

        CNT_out['t'] = t_out
        CNT_out['y'] = y_out
        CNT_out['class_name'] = class_selected
    else:
        t_out = []
        y_out = []
        x_out = np.zeros((shape_[0],int(N_tri_class*len(class_list)),shape_[2]))
        for ii in range(len(name)):
            selected_marker = np.where(selected_logic[ii, :] == 1)[0]
            t_out = np.append(t_out, np.take(t, selected_marker))
            y_out = np.append(y_out, np.take(y, selected_marker))
            x_out[:,ii*N_tri_class:(ii+1)*N_tri_class,:] = np.take(x, selected_marker,axis=1)

        class_selected = np.take(class_all, selected_idx, axis=0)

        CNT_out['t'] = t_out.astype(int)
        CNT_out['y'] = y_out.astype(int)
        CNT_out['class_name'] = class_selected
        CNT_out['x'] = x_out

    return CNT_out


def channel_selection(CNT, channel_name):
    CNT_out = copy.deepcopy(CNT)
    x = CNT_out['x']

    # find index
    channel_all = np.asarray(CNT['channel'])
    channel_list = np.asarray(channel_name)
    selected_idx = ismember(channel_list, channel_all)

    xx = [x for x in selected_idx if x is not None]

    # select
    channel_selected = np.take(channel_all, xx)
    x_selected = x.take(xx, axis=1)

    CNT_out['channel'] = channel_selected
    CNT_out['x'] = x_selected
    return CNT_out


def bandpass_filter(CNT, f_range, filter_order, type_filter):
    cnt_out = copy.deepcopy(CNT)
    x = CNT['x'];
    fs = CNT['fs']
    lowcut = f_range[0];
    highcut = f_range[1]
    order = filter_order

    if type_filter == 'butter':
        dat_out = np.zeros(x.shape[::-1])
        for ii in range(0, x.shape[1]):
            dat = x.take(ii, axis=1)
            dat_out[ii,] = butter_bandpass_filter(dat, lowcut, highcut, fs, order=filter_order)
        dat_out = dat_out.transpose()
    else:
        dat_out = chebyBandpassFilter(x,f_range)

    cnt_out['x'] = dat_out
    return cnt_out

def filter_bank(CNT, f_range, filter_order, type_filter):
    cnt_copy = copy.deepcopy(CNT)
    cnt = bandpass_filter(cnt_copy, f_range[0, :], filter_order, type_filter)
    x = cnt['x']

    if type_filter == 'butter':
        for ii in range(np.shape(f_range)[0] - 1):
            cnt = bandpass_filter(cnt_copy, f_range[ii + 1, :], filter_order, type_filter)
            x = np.concatenate((x, cnt['x']), axis=1)
    else:
        for ii in range(np.shape(f_range)[0] - 1):
            cnt = bandpass_filter(cnt_copy, f_range[ii + 1, :], filter_order, type_filter)
            x = np.concatenate((x, cnt['x']), axis=1)

    cnt_copy['x'] = x

    return cnt_copy


def segmentation(CNT, t_interval):
    smt = copy.deepcopy(CNT)
    x = CNT['x']
    fs = CNT['fs']
    t = CNT['t']
    chan = CNT['channel']
    ival = t_interval;
    n_chan = len(chan)
    try:
        n_events = np.shape(t)[1]
    except:
        n_events = len(t)

    idc = np.array(range(int(np.floor(ival[0] * fs / 1000)), int(np.ceil(int(ival[1]) * fs / 1000))))
    T = len(idc)

    IV = (np.ones((n_events, 1), dtype=int) * idc).transpose() + np.ones((T, 1), dtype=int) * t
    IV = IV.astype(int).transpose()
    smt['x'] = x.take(IV.flatten(), axis=0).reshape([T, n_events, n_chan],order='F')

    return smt


def segment_fb(CNT, t_interval):
    cnt_copy = copy.deepcopy(CNT)
    x = cnt_copy['x'][:, 0:len(cnt_copy['channel'])]
    cnt_copy['x'] = x
    smt = segmentation(cnt_copy, t_interval)
    x_smt = smt['x']

    len_tot = np.shape(CNT['x'])[1]
    len_one = len(CNT['channel'])

    for ii in range(int(len_tot / len_one - 1)):
        cnt_copy = copy.deepcopy(CNT)
        x = cnt_copy['x'][:, (ii + 1) * len(cnt_copy['channel']):(ii + 2) * len(cnt_copy['channel'])]
        cnt_copy['x'] = x
        smt = segmentation(cnt_copy, t_interval)
        x = smt['x']

        x_smt = np.concatenate((x_smt, x), axis=2)

    smt['x'] = x_smt

    return smt


def select_trial_samsung(CNT, index):
    cnt_out = copy.deepcopy(CNT)
    y = cnt_out['y']
    t = cnt_out['t']

    idx = np.unique(y)

    idx_class1 = np.where(y == idx[0])[1]
    idx_class2 = np.where(y == idx[1])[1]
    # idx_class1 = idx_class1[1]
    # idx_class2 = idx_class2[1]
    idx_class1 = idx_class1[index]
    idx_class2 = idx_class2[index]

    idxx = np.append(idx_class1, idx_class2)[np.newaxis, :]

    cnt_out['t'] = t[0][idxx]
    cnt_out['y'] = y[0][idxx]

    return cnt_out


def select_trial(CNT, index):
    cnt_out = copy.deepcopy(CNT)
    y = cnt_out['y']
    t = cnt_out['t']

    idx = np.unique(y)

    idx_class1, = np.where(y == idx[0])
    idx_class2, = np.where(y == idx[1])
    # idx_class1 = idx_class1[1]
    # idx_class2 = idx_class2[1]
    idx_class1 = idx_class1[index]
    idx_class2 = idx_class2[index]
    idx = np.append(idx_class1, idx_class2)

    cnt_out['t'] = t[idx]
    cnt_out['y'] = y[idx]

    return cnt_out


def time_delay_embed(SMT, CNT, tau, t_interval):
    smt_out = copy.deepcopy(SMT)
    x = smt_out['x']
    tau = tau * 10

    smt_delay = segmentation(CNT, t_interval - tau)
    x_delay = smt_delay['x']

    smt_out['x'] = np.concatenate((x, x_delay), axis=2)

    return smt_out


######################################### Feature extraction
#########################################
def covariance_matrix(SMT):
    SMT_out = copy.deepcopy(SMT)
    x = SMT['x']
    j, k, q = x.shape
    cov = np.zeros((k, q, q), dtype=float)
    for ii in range(k):
        xx = x[:, ii, :]
        cov[ii, :, :] = np.cov(xx.T)
    # cov[ii,:,:] = np.dot(np.transpose(xx), xx) / np.trace(np.dot(np.transpose(xx), xx))

    SMT_out['x'] = cov

    return SMT_out


def common_spatial_pattern(SMT, n_pattern):
    SMT_out = copy.deepcopy(SMT)
    x = SMT['x']
    j, k, q = x.shape
    N_trial = int(k / 2)
    cov1 = np.zeros((q, q), dtype=float)
    cov2 = np.zeros((q, q), dtype=float)

    for ii in range(0, N_trial):
        x_c1 = x[:, ii, :]
        x_c2 = x[:, ii + N_trial, :]
        cov1 = cov1 + np.cov(x_c1.T)
        cov2 = cov2 + np.cov(x_c2.T)

    # cov1 = cov1 + np.dot(np.transpose(x_c1), x_c1) / np.trace(np.dot(np.transpose(x_c1), x_c1))
    # cov2 = cov2 + np.dot(np.transpose(x_c2), x_c2) / np.trace(np.dot(np.transpose(x_c2), x_c2))

    cov1 = cov1 / N_trial
    cov2 = cov2 / N_trial

    D, W = scipy.linalg.eigh(cov1, cov1 + cov2)
    CSP_W = W.take(list(range(0, n_pattern)) + list(range(q - n_pattern, q)), axis=1)

    x_out = np.zeros([j, k, n_pattern * 2], dtype=float)
    for ii in range(0, k):
        x_out[:, ii, :] = np.dot(x[:, ii, :], CSP_W)

    SMT_out['x'] = x_out

    return SMT_out, CSP_W


def csp_fb(SMT, n_pattern):
    len_chan = len(SMT['channel'])
    len_tot = np.shape(SMT['x'])[2]

    smt_copy = copy.deepcopy(SMT)
    smt_copy['x'] = SMT['x'][:, :, 0:len_chan]
    smt_filt, w_filter = common_spatial_pattern(smt_copy, n_pattern)

    for ii in range(int(len_tot / len_chan - 1)):
        smt_copy = copy.deepcopy(SMT)
        smt_copy['x'] = SMT['x'][:, :, (ii + 1) * len_chan:(ii + 2) * len_chan]
        smt_filt_, w_filter_ = common_spatial_pattern(smt_copy, n_pattern)

        smt_filt['x'] = np.concatenate((smt_filt['x'], smt_filt_['x']), axis=2)
        w_filter = np.dstack((w_filter, w_filter_))

    return smt_filt, w_filter


def project_CSP(SMT, CSP):
    SMT_out = copy.deepcopy(SMT)
    x = SMT['x']
    j, k, q = x.shape
    n_pattern = int(CSP.shape[1] / 2)

    x_out = np.zeros([j, k, n_pattern * 2], dtype=float)
    for ii in range(0, k):
        x_out[:, ii, :] = np.dot(x[:, ii, :], CSP)

    SMT_out['x'] = x_out
    return SMT_out


def project_CSP_fb(smt, w):
    len_fb = np.shape(w)[2]
    len_ = int(np.shape(smt['x'])[2] / len_fb)
    smt_copy = copy.deepcopy(smt)
    smt_copy['x'] = smt['x'][:, :, 0:len_]
    smt_out = project_CSP(smt_copy, w[:, :, 0])
    x = smt_out['x']
    for ii in range(len_fb - 1):
        smt_copy = copy.deepcopy(smt)
        smt_copy['x'] = smt['x'][:, :, (ii + 1) * len_:(ii + 2) * len_]
        smt_out = project_CSP(smt_copy, w[:, :, ii + 1])
        xx = smt_out['x']

        x = np.dstack((x, xx))

    smt_out['x'] = x

    return smt_out


def log_variance(SMT):
    SMT_out = copy.deepcopy(SMT)
    x = SMT['x']
    j, k, q = x.shape

    x_out = np.zeros([q, k], dtype=float)
    for ii in range(0, k):
        x_out[:, ii] = np.var(x[:, ii, :], axis=0)

    SMT_out['x'] = x_out

    return SMT_out


def corr_matrix(SMT):
    smt_out = copy.deepcopy(SMT)
    x = SMT['x'];
    out = np.zeros([x.shape[1], x.shape[2], x.shape[2]], dtype=float)
    for ii in range(0, x.shape[1]):
        aa = x[:, ii, :]
        out[ii, :, :] = np.corrcoef(x[:, ii, :].transpose())

    smt_out['x'] = out
    return smt_out


######################################### Feature selection
#########################################
def mutual_info(FT):
    # ft = copy.deepcopy(FT)
    n_neighbor = 10
    mi = np.zeros(np.shape(FT['x'])[0])
    for ii in range(np.shape(FT['x'])[0]):
        X = FT['x'][ii, :]
        Y = np.reshape(FT['y'], (len(X),))
        mi[ii] = max(0, sklearn.feature_selection.mutual_info_._compute_mi_cd(X, Y, n_neighbor))

    # X = FT['x'][ii,:][:,np.newaxis]
    # X_C1 = X[0:int(len(X)/2),:]
    # X_C2 = X[int(len(X)/2)+0:int(len(X) / 2)+int(len(X) / 2), :]
    # band_wid_X = suitable_bandwidth(X)
    # band_wid_X1 = suitable_bandwidth(X_C1)
    # band_wid_X2 = suitable_bandwidth(X_C2)
    #
    # kde_X = KernelDensity(kernel='gaussian', bandwidth=band_wid_X).fit(X).score_samples(X)
    # kde_X1 = KernelDensity(kernel='gaussian', bandwidth=band_wid_X1).fit(X_C1).score_samples(X_C1)
    # kde_X2 = KernelDensity(kernel='gaussian', bandwidth=band_wid_X2).fit(X_C2).score_samples(X_C2)
    #
    # entropy_X = -1*np.sum(kde_X)/len(kde_X)
    # entropy_X1 = -1 * np.sum(kde_X1) / len(kde_X1)
    # entropy_X2 = -1 * np.sum(kde_X1) / len(kde_X2)
    #
    # entropy_cond = (entropy_X1 + entropy_X2)/2
    #
    # mi[ii] = entropy_X - entropy_cond

    return mi[:, np.newaxis]


def suitable_bandwidth(X):
    band_wid = math.pow(1.06 * np.std(X) * len(X), -1 / 5)
    return band_wid


def feat_select(mi):
    k = 4
    indexx = np.argsort(mi, axis=0)[::-1][0:k, :]
    index_pair = np.zeros((len(indexx), 1))

    for ii in range(k):
        if np.mod(indexx[ii], 6) == 0:
            index_pair[ii] = indexx[ii] + 5
        elif np.mod(indexx[ii], 6) == 1:
            index_pair[ii] = indexx[ii] + 3
        elif np.mod(indexx[ii], 6) == 2:
            index_pair[ii] = indexx[ii] + 1
        elif np.mod(indexx[ii], 6) == 3:
            index_pair[ii] = indexx[ii] - 1
        elif np.mod(indexx[ii], 6) == 4:
            index_pair[ii] = indexx[ii] - 3
        else:
            index_pair[ii] = indexx[ii] - 5

    # for ii in range(k):
    #     if np.mod(indexx[ii],4) == 0:
    #         index_pair[ii] = indexx[ii]+3
    #     elif np.mod(indexx[ii],4) == 1:
    #         index_pair[ii] = indexx[ii] + 1
    #     elif np.mod(indexx[ii], 4) == 2:
    #         index_pair[ii] = indexx[ii] -1
    #     else:
    #         index_pair[ii] = indexx[ii] - 3

    index = np.concatenate((indexx, index_pair), axis=0).tolist()
    index = np.unique(index)
    index = index.astype(int)

    return index


######################################### Classification
#########################################
def shrinkage_LDA(FT):
    LDA_out = copy.deepcopy(FT)
    x = np.transpose(FT['x'])
    y = np.transpose(FT['y'])

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(x, y)
    return clf


def project_shLDA(FT, model):
    x = np.transpose(FT['x'])
    y = np.transpose(FT['y'])

    score = model.score(x, y)

    return score


######################################### Evaluation
#########################################
def evaluation(out_classifier):
    return 0


######################################### Visualization
#########################################
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels = ['Sex', 'Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings', ]
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()