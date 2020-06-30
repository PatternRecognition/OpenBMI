import numpy as np
import sys
import os
import pickle
from  datasets.gigadataset import GigaDataset, GigaDataset_gpu
import torch
def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def get_data_eeg(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    x_data = np.load(args.data_root + '/x_data_raw.npy', mmap_mode='r')
    y_data = np.load(args.data_root + '/y_data_raw.npy')

    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = x_data.shape[2]
    args.n_time = x_data.shape[3]

    datasets = []
    for s_id in range(0, 108):
        datasets.append(GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id))

    test_subj = np.r_[fold_idx * 9:fold_idx * 9 + 9, fold_idx * 9 + 54:fold_idx * 9 + 9 + 54]
    print(test_subj)
    train_subj = np.setdiff1d(np.r_[0:108], test_subj)

    train_set = [datasets[i] for i in train_subj]
    test_set =[datasets[i] for i in test_subj]


    train_set = torch.utils.data.ConcatDataset(train_set)
    train_len = int(0.9 * len(train_set))
    valid_len = len(train_set) - train_len

    train_set, valid_set = torch.utils.data.dataset.random_split(train_set,[train_len, valid_len])
    test_set = torch.utils.data.ConcatDataset(test_set)

    return train_set, valid_set, test_set, args

def get_data_eeg_subject_subset(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    x_data = np.load(args.data_root + '/x_data.npy', mmap_mode='r')
    y_data = np.load(args.data_root + '/y_data.npy')


    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = 62
    args.n_time = x_data.shape[3]

    datasets = []
    for s_id in range(0, 108):
        datasets.append(GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id))

    subject_list = np.r_[0:54]
    subject_list_sess2 = subject_list+54
    all_subject_list = np.concatenate([subject_list, subject_list_sess2])

    test_subj =  np.r_[fold_idx, fold_idx+54]
    train_subj = np.setdiff1d(all_subject_list, test_subj)

    train_set = [datasets[i] for i in train_subj]
    test_set = [datasets[i] for i in test_subj]

    train_set = torch.utils.data.ConcatDataset(train_set)
    train_len = int(0.9 * len(train_set))
    valid_len = len(train_set) - train_len

    train_set, valid_set = torch.utils.data.dataset.random_split(train_set, [train_len, valid_len])
    test_set = torch.utils.data.ConcatDataset(test_set)
    print(train_set.indices)

    return train_set, valid_set, test_set, args

