import openbmi as ob
import scipy.io as sio
import numpy as np
import os

###################################################################################################
# Data_load setting
###################################################################################################
dataset_name = 'Giga_Science'
session = ['session1','session1']
subject = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
           's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',
           's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',
           's31','s32','s33','s34','s35','s36','s37','s38','s39','s40',
           's41','s42','s43','s44','s45','s46','s47','s48','s49','s50',
           's51','s52','s53','s54']
sess1_sub_paradigm = ['sess01_subj01_EEG_MI','sess01_subj02_EEG_MI','sess01_subj03_EEG_MI','sess01_subj04_EEG_MI','sess01_subj05_EEG_MI',
            'sess01_subj06_EEG_MI','sess01_subj07_EEG_MI','sess01_subj08_EEG_MI','sess01_subj09_EEG_MI','sess01_subj10_EEG_MI',
            'sess01_subj11_EEG_MI','sess01_subj12_EEG_MI','sess01_subj13_EEG_MI','sess01_subj14_EEG_MI','sess01_subj15_EEG_MI',
            'sess01_subj16_EEG_MI','sess01_subj17_EEG_MI','sess01_subj18_EEG_MI','sess01_subj19_EEG_MI','sess01_subj20_EEG_MI',
            'sess01_subj21_EEG_MI','sess01_subj22_EEG_MI','sess01_subj23_EEG_MI','sess01_subj24_EEG_MI','sess01_subj25_EEG_MI',
            'sess01_subj26_EEG_MI','sess01_subj27_EEG_MI','sess01_subj28_EEG_MI','sess01_subj29_EEG_MI','sess01_subj30_EEG_MI',
            'sess01_subj31_EEG_MI','sess01_subj32_EEG_MI','sess01_subj33_EEG_MI','sess01_subj34_EEG_MI','sess01_subj35_EEG_MI',
            'sess01_subj36_EEG_MI','sess01_subj37_EEG_MI','sess01_subj38_EEG_MI','sess01_subj39_EEG_MI','sess01_subj40_EEG_MI',
            'sess01_subj41_EEG_MI','sess01_subj42_EEG_MI','sess01_subj43_EEG_MI','sess01_subj44_EEG_MI','sess01_subj45_EEG_MI',
            'sess01_subj46_EEG_MI','sess01_subj47_EEG_MI','sess01_subj48_EEG_MI','sess01_subj49_EEG_MI','sess01_subj50_EEG_MI',
            'sess01_subj51_EEG_MI','sess01_subj52_EEG_MI','sess01_subj53_EEG_MI','sess01_subj54_EEG_MI',]

sess2_sub_paradigm = ['sess02_subj01_EEG_MI','sess02_subj02_EEG_MI','sess02_subj03_EEG_MI','sess02_subj04_EEG_MI','sess02_subj05_EEG_MI',
            'sess02_subj06_EEG_MI','sess02_subj07_EEG_MI','sess02_subj08_EEG_MI','sess02_subj09_EEG_MI','sess02_subj10_EEG_MI',
            'sess02_subj11_EEG_MI','sess02_subj12_EEG_MI','sess02_subj13_EEG_MI','sess02_subj14_EEG_MI','sess02_subj15_EEG_MI',
            'sess02_subj16_EEG_MI','sess02_subj17_EEG_MI','sess02_subj18_EEG_MI','sess02_subj19_EEG_MI','sess02_subj20_EEG_MI',
            'sess02_subj21_EEG_MI','sess02_subj22_EEG_MI','sess02_subj23_EEG_MI','sess02_subj24_EEG_MI','sess02_subj25_EEG_MI',
            'sess02_subj26_EEG_MI','sess02_subj27_EEG_MI','sess02_subj28_EEG_MI','sess02_subj29_EEG_MI','sess02_subj30_EEG_MI',
            'sess02_subj31_EEG_MI','sess02_subj32_EEG_MI','sess02_subj33_EEG_MI','sess02_subj34_EEG_MI','sess02_subj35_EEG_MI',
            'sess02_subj36_EEG_MI','sess02_subj37_EEG_MI','sess02_subj38_EEG_MI','sess02_subj39_EEG_MI','sess02_subj40_EEG_MI',
            'sess02_subj41_EEG_MI','sess02_subj42_EEG_MI','sess02_subj43_EEG_MI','sess02_subj44_EEG_MI','sess02_subj45_EEG_MI',
            'sess02_subj46_EEG_MI','sess02_subj47_EEG_MI','sess02_subj48_EEG_MI','sess02_subj49_EEG_MI','sess02_subj50_EEG_MI',
            'sess02_subj51_EEG_MI','sess02_subj52_EEG_MI','sess02_subj53_EEG_MI','sess02_subj54_EEG_MI',]
###################################################################################################
# Parameter for Brain signal - channel to be used, bpf range, segment length, number of csp pattern
###################################################################################################
# name_index_motor = ['FT9','T7', 'TP7', 'TP9', 'FC5', 'C5', 'CP5', 'FC3', 'C3', 'CP3', 'FC1', 'C1', 'CP1', 'FCz', 'CPz',
#                     'FC2', 'C2', 'CP2', 'FC4', 'C4', 'CP4', 'FC6', 'C6', 'CP6', 'FT10', 'T8', 'TP8', 'TP10'] # Cz 빠짐
name_index_motor = ['FC5', 'C5', 'CP5', 'FC3', 'C3', 'CP3', 'FC1', 'C1', 'CP1', 'FCz',
                    'CPz', 'FC2', 'C2', 'CP2', 'FC4', 'C4', 'CP4', 'FC6', 'C6', 'CP6']

filter_order = 5
f_range = [8, 30]
# f_range = np.array([[8,12],[12,16],[16,20],[20,24],[24,28],[28,32]])
t_interval = [500, 3500]
n_pattern = 3
down_fs = 100

###################################################################################################
#
###################################################################################################

accuracy = np.zeros(54)

for ii in range(len(subject)):
    file_dir = 'D:\data'
    file_dir_ = os.path.join(file_dir, dataset_name, session[1], subject[ii], sess1_sub_paradigm[ii]+'.mat')
    mat = sio.loadmat(file_dir_)
    CNT_tr = ob.data_structure(mat['EEG_MI_train'],dataset_name)
    CNT_te = ob.data_structure(mat['EEG_MI_test'], dataset_name)
    count = 0

    #  preprocessing
    # downsampling
    CNT_tr = ob.downsample(CNT_tr, down_fs)
    CNT_te = ob.downsample(CNT_te, down_fs)

    # channel selection
    CNT_tr = ob.channel_selection(CNT_tr, name_index_motor)
    CNT_te = ob.channel_selection(CNT_te, name_index_motor)

    # band-pass filter
    CNT_tr = ob.bandpass_filter(CNT_tr, f_range, filter_order)
    CNT_te = ob.bandpass_filter(CNT_te, f_range, filter_order)

    # segmentation
    SMT_tr = ob.segmentation(CNT_tr, t_interval)
    SMT_te = ob.segmentation(CNT_te, t_interval)

    # class selection
    SMT_tr = ob.class_selection(SMT_tr, ['right', 'left'])

    # feature extraction
    SMT_tr, CSP_W= ob.common_spatial_pattern(SMT_tr, n_pattern)
    SMT_te = ob.project_CSP(SMT_te, CSP_W)

    FT_tr = ob.log_variance(SMT_tr)
    FT_te = ob.log_variance(SMT_te)

    # classification
    sh_LDA_motor = ob.shrinkage_LDA(FT_tr)
    OUT_lda_motor = ob.project_shLDA(FT_te, sh_LDA_motor)

    accuracy[ii] = OUT_lda_motor
    count = count + 1

    print(OUT_lda_motor)

    del CNT_te, CNT_tr, SMT_te, SMT_tr

sio.savemat('D:\data\BCI_comp_III_IVa/accuracy_fbcsp_k10.mat', {'accuracy': accuracy})