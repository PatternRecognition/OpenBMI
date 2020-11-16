import numpy as np
import sys
import os
import pickle
from datasets.gigadataset import GigaDataset, GigaDataset_gpu
import torch
import copy
from sklearn.model_selection import train_test_split
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

    with open(args.data_root + '/openbmi_mi_raw_smt_data.pkl', 'rb') as f:
        x_data = pickle.load(f)
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = x_data.shape[2]
    args.n_time = x_data.shape[3]

    datasets = []
    for s_id in range(0, 108):
        datasets.append(GigaDataset_gpu([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id,'cuda:0'))

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

def get_data_eeg_4fold(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(args.data_root + '/openbmi_mi_filt830_smt_data.pkl', 'rb') as f:
        x_data = pickle.load(f)
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    ch_idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    x_data = x_data[:, :, ch_idx, 125:]

    args.n_class = len(np.unique(y_data))
    args.n_ch = x_data.shape[2]
    args.n_time = x_data.shape[3]

    datasets = []
    for s_id in range(0, 108):
        datasets.append(GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id))

    train_subj = np.r_[0:36, 54:90]  # 0~35번
    test_subj = np.r_[36:45, 54 + 36:54 + 45]  # 36~44
    valid_subj = np.r_[45:54, 54 + 45:108]  # 45~54

    train_set = [datasets[i] for i in train_subj]
    valid_set = [datasets[i] for i in valid_subj]
    test_set = [datasets[i] for i in test_subj]

    train_set = torch.utils.data.ConcatDataset(train_set)
    valid_set = torch.utils.data.ConcatDataset(valid_set)
    test_set = torch.utils.data.ConcatDataset(test_set)

    return train_set, valid_set, test_set, args

import random
def get_data_eeg_single_subject(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    with open(args.data_root + '/openbmi_mi_raw_smt_data.pkl', 'rb') as f:
        x_data = pickle.load(f)
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

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

import torch.cuda as cuda
import hyperparameter as hp
def get_data_eeg_subject_subset(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = hp.data_path
    args.data = data_path
    print(f'data path : {data_path}')

    x_data = np.load(args.data_root + data_path, mmap_mode='r')
    # y_data = np.load(args.data_root + '/y_data_raw.pkl', mmap_mode='r')
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    # np.save("C:/data/x_data.npy", x_data)


    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = 20
    args.n_time = 250

    device = 'cuda' if cuda else 'cpu'
    datasets = []
    for s_id in range(0, 108):
        # datasets.append(GigaDataset_gpu([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id,'cuda:0'))
        datasets.append(GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id))

    # subject_list = np.r_[0, 1, 2, 4, 5, 8, 17, 18, 20, 27, 28, 32, 35, 36, 42, 43, 44, 51]
    subject_list = np.r_[0:54]
    subject_list_sess2 = subject_list+54
    all_subject_list = np.concatenate([subject_list, subject_list_sess2])

    test_subj =  np.r_[fold_idx, fold_idx+54]
    train_subj = np.setdiff1d(all_subject_list, test_subj)

    # train_set = [datasets[i] for i in train_subj]
    test_set = [datasets[i] for i in test_subj]
    for t in test_set:
        t.istrain = False

    train_set = []
    valid_set = []
    for i in train_subj:
        train_len = int(0.90 * len(datasets[i]))
        valid_len = len(datasets[i]) - train_len
        # print(f'{train_len},{valid_len}')
        train_set_temp, valid_set_temp = torch.utils.data.dataset.random_split(datasets[i], [train_len, valid_len])
        temp = copy.deepcopy(valid_set_temp)
        temp.dataset.istrain = False
        train_set.append(train_set_temp)
        valid_set.append(temp)

    train_set = torch.utils.data.ConcatDataset(train_set)
    valid_set = torch.utils.data.ConcatDataset(valid_set)
    # train_set = torch.utils.data.ConcatDataset(train_set)
    # train_len = int(0.9 * len(train_set))
    # valid_len = len(train_set) - train_len
    #
    # train_set, valid_set = torch.utils.data.dataset.random_split(train_set, [train_len, valid_len])
    test_set = torch.utils.data.ConcatDataset(test_set)
    # print(train_set.indices)

    return train_set, valid_set, test_set, args

def get_data_eeg_subject_subset_inference(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = hp.data_path
    args.data = data_path
    print(f'data path : {data_path}')

    x_data = np.load(args.data_root + data_path, mmap_mode='r')
    # y_data = np.load(args.data_root + '/y_data_raw.pkl', mmap_mode='r')
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    # np.save("C:/data/x_data.npy", x_data)


    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = 20
    args.n_time = 250

    device = 'cuda' if cuda else 'cpu'
    datasets = []
    for s_id in range(0, 108):
        # datasets.append(GigaDataset_gpu([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id,'cuda:0'))
        datasets.append(GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id,False))

    # subject_list = np.r_[0, 1, 2, 4, 5, 8, 17, 18, 20, 27, 28, 32, 35, 36, 42, 43, 44, 51]
    subject_list = np.r_[0:54]
    subject_list_sess2 = subject_list+54
    all_subject_list = np.concatenate([subject_list, subject_list_sess2])

    test_subj =  np.r_[fold_idx, fold_idx+54]
    train_subj = np.setdiff1d(all_subject_list, test_subj)

    # train_set = [datasets[i] for i in train_subj]
    test_set = [datasets[i] for i in test_subj]
    for t in test_set:
        t.istrain = False

    train_set = []
    valid_set = []
    for i in train_subj:
        train_len = int(0.9 * len(datasets[i]))
        valid_len = len(datasets[i]) - train_len
        # print(f'{train_len},{valid_len}')
        train_set_temp, valid_set_temp = torch.utils.data.dataset.random_split(datasets[i], [train_len, valid_len])
        # temp = copy.deepcopy(valid_set_temp)
        # temp.dataset.istrain = False
        train_set.append(train_set_temp)
        valid_set.append(valid_set_temp)

    train_set = torch.utils.data.ConcatDataset(train_set)
    valid_set = torch.utils.data.ConcatDataset(valid_set)
    # train_set = torch.utils.data.ConcatDataset(train_set)
    # train_len = int(0.9 * len(train_set))
    # valid_len = len(train_set) - train_len
    #
    # train_set, valid_set = torch.utils.data.dataset.random_split(train_set, [train_len, valid_len])
    test_set = torch.utils.data.ConcatDataset(test_set)
    # print(train_set.indices)

    return train_set, valid_set, test_set, args

def get_data_eeg_subject_subset_dev(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = hp.data_path
    args.data = data_path
    print(f'data path : {data_path}')

    x_data = np.load(args.data_root + data_path, mmap_mode='r')
    # y_data = np.load(args.data_root + '/y_data_raw.pkl', mmap_mode='r')
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    # np.save("C:/data/x_data.npy", x_data)


    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = 20
    args.n_time = 250

    device = 'cuda' if cuda else 'cpu'

    #사용자 세팅
    subject_list = np.r_[0:54]
    subject_list_sess2 = subject_list + 54
    all_subject_list = np.concatenate([subject_list, subject_list_sess2])

    test_subj = np.r_[fold_idx, fold_idx + 54]
    train_subj = np.setdiff1d(all_subject_list, test_subj)


    #test셋 먼저 세팅
    test_set = [GigaDataset([np.expand_dims(x_data[i, :, :, :], axis=1), y_data[i, :]], i, False) for i in test_subj]
    train_set = []
    valid_set = []

    for i in train_subj:
        train_len = int(0.9 * len(x_data[i, :, :, :]))
        valid_len = len(x_data[i, :, :, :]) - train_len
        # print(f'{train_len},{valid_len}')
        idx = np.arange(len(x_data[i, :, :, :]))
        tr_idx,val_idx = train_test_split(idx,test_size=0.1)
        train_set.append(GigaDataset([np.expand_dims(x_data[i, tr_idx, :, :], axis=1), y_data[i, :]], i, True))
        valid_set.append(GigaDataset([np.expand_dims(x_data[i, val_idx, :, :], axis=1), y_data[i, :]], i, False))

    train_set = torch.utils.data.ConcatDataset(train_set)
    valid_set = torch.utils.data.ConcatDataset(valid_set)
    test_set = torch.utils.data.ConcatDataset(test_set)

    return train_set, valid_set, test_set, args

def get_data_eeg_subject_subset_woval(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = '/x_data_380.npy'
    args.data = data_path
    print(f'data path : {data_path}')

    x_data = np.load(args.data_root + data_path, mmap_mode='r')
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    # np.save("C:/data/x_data.npy", x_data)


    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)



    args.n_class = len(np.unique(y_data))
    args.n_ch = 20
    args.n_time = 250

    device = 'cuda' if cuda else 'cpu'
    datasets = []
    for s_id in range(0, 108):
        datasets.append(GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id))

    # subject_list = np.r_[0, 1, 2, 4, 5, 8, 17, 18, 20, 27, 28, 32, 35, 36, 42, 43, 44, 51]
    subject_list = np.r_[0:54]
    subject_list_sess2 = subject_list+54
    all_subject_list = np.concatenate([subject_list, subject_list_sess2])

    test_subj =  np.r_[fold_idx, fold_idx+54]
    train_subj = np.setdiff1d(all_subject_list, test_subj)

    train_set = [datasets[i] for i in train_subj]
    test_set = [datasets[i] for i in test_subj]

    train_set = torch.utils.data.ConcatDataset(train_set)
    test_set = torch.utils.data.ConcatDataset(test_set)

    return train_set, test_set, args

def get_data_eeg_subject_subset2(args,device):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = '/openbmi_mi_filt830_smt_data.pkl'
    args.data = data_path
    print(f'data path : {data_path}')
    with open(args.data_root + data_path, 'rb') as f:
        x_data = pickle.load(f)
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    x_data_t = torch.Tensor(x_data).to('cuda:0')
    # torch.save(x_data_t,'openbmi_mi_filt830_smt_data_tensor.pt')
    y_data_t = torch.Tensor(y_data)
    # torch.save(y_data_t, 'epoch_labels.pt')
    in_chans = x_data.shape[2]


    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    ch_idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    x_data = x_data[:, :, ch_idx, 125:]

    args.n_class = len(np.unique(y_data))
    args.n_ch = x_data.shape[2]
    args.n_time = x_data.shape[3]

    datasets = []
    for s_id in range(0, 108):
        datasets.append(GigaDataset_gpu([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id,device))

    return datasets, args

def get_tensor(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = '/openbmi_mi_filt830_smt_data_tensor.pt'
    args.data = data_path
    print(f'data path : {data_path}')


    x_data = torch.load(args.data_root+data_path, map_location='cpu')
    y_data = torch.load(args.data_root + '/epoch_labels.pt')

    # x_data = np.expand_dims(x_data, axis=1)
    in_chans = x_data.shape[2]


    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    ch_idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    x_data = x_data[:, :, ch_idx, 125:]

    args.n_class = len(np.unique(y_data))
    args.n_ch = x_data.shape[2]
    args.n_time = x_data.shape[3]

    return x_data, y_data, args

def get_data_eeg_chsel(args,fold_idx):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    data_path = '/openbmi_mi_filt830_smt_data.pkl'
    args.data = data_path
    print(f'data path : {data_path}')
    with open(args.data_root + data_path, 'rb') as f:
        x_data = pickle.load(f)
    with open(args.data_root + '/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)

    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    ch_idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))
    x_data = x_data[:,:,ch_idx,125+250:125+250+625]

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
    print(train_set.indices)

    return train_set, valid_set, test_set, args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


import pandas as pd
import train_eval as te

def get_testset_accuracy(model, device , test_set, args):
    subj_num = len(test_set.datasets)//2
    all_test_score = []
    for subj in range(subj_num):
        print(subj)
        for sess in range(2):
            for onoff in range(2):
                if sess == 0:
                    subj_id = subj
                else:
                    subj_id = subj + subj_num

                if onoff == 0:
                    data = torch.utils.data.Subset(test_set, range(subj_id*200,subj_id*200+100))
                else:
                    data = torch.utils.data.Subset(test_set, range(subj_id*200+100,subj_id*200+200))
                print(f'subject:{subj+1}, session:{sess}, onoff:{onoff}')
                test_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
                blockPrint()
                test_loss, test_score = te.eval(model, device, test_loader)
                all_test_score.append(test_score)
                enablePrint()
                print(f"subject:{subj+1}, acc:{test_score}")


    df = pd.DataFrame(np.array(all_test_score).reshape(-1,4),columns=['sess1-off','sess1-on','sess2-off','sess2-on'])
    print(f"all acc: {np.mean(all_test_score):.4f}")


    print(df)

    return df