from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import os

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

from train_eval import *

# from torch.utils.tensorboard import SummaryWriter
from datasets import *

from trainer import fit

class MIDataset(Dataset):
    def __init__(self,x,y):
        x_data = x.copy()
        y_data = y.copy()

        self.in_chans = x_data.shape[2]
        self.input_time_length = x_data.shape[3]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)

        y_data = torch.from_numpy(y_data)
        self.x_data = x_data.type(torch.FloatTensor)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index]

        return x, y

    def __len__(self):
        return self.len

def load_midata(path='data'):
    with open(path+'/mi_train_label.pkl', 'rb') as f:
        y_data_train = pickle.load(f)
    with open(path+'/mi_test_label.pkl', 'rb') as f:
        y_data_test = pickle.load(f)

    with open(path+'/mi_train.pkl', 'rb') as f:
        x_data_train = pickle.load(f)
    with open(path+'/mi_test.pkl', 'rb') as f:
        x_data_test = pickle.load(f)


    x_data_train = np.expand_dims(x_data_train, axis=1)
    x_data_train = x_data_train[:, :, :, 101:]

    x_data_test = np.expand_dims(x_data_test, axis=1)
    x_data_test = x_data_test[:, :, :, 101:]


    return x_data_train, x_data_test, y_data_train, y_data_test
import scipy.io as sio
def load_midata_from_mat(path='data'):

    MI_struct = sio.loadmat(path + 'midata', struct_as_record=False, squeeze_me=True)
    MI_data = MI_struct['data']


    x_data_train = MI_data.x_train
    x_data_test = MI_data.x_test
    y_data_train = MI_data.y_train-1
    y_data_test = MI_data.y_test-1

    x_data_train = np.expand_dims(x_data_train, axis=1)
    x_data_train = x_data_train[:, :, :, 1:]

    x_data_test = np.expand_dims(x_data_test, axis=1)
    x_data_test = x_data_test[:, :, :, 1:]


    return x_data_train, x_data_test, y_data_train, y_data_test







def experiment(args):
    fold_idx = args.fold_idx
    startepoch = 0
    folder_name = args.folder_name
    comment = args.comment

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = args.device


    #data load
    x_data_train1, x_data_test1, y_data_train1, y_data_test1 = load_midata_from_mat(path='data/midata/')
    x_data_train, x_data_test, y_data_train, y_data_test = load_midata(path='data/midata/')
    #
    # 필요시 데이터 시각화해서 확인
    # sfreq = 100  # Sampling frequency
    # ch_types = ['eeg']
    # ch_names = [str(i) for i in range(0,62)]
    # import mne
    #
    # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    # x_data_train2 = x_data_train.reshape(5100, 62, 400)
    # temp = mne.EpochsArray(x_data_train2[0:10,:,:],info)
    #
    #
    #
    # temp.plot()
    # scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    # temp.plot(scalings='auto', n_epochs=3, show=True, block=True)
    #
    #
    # plt.imshow(x_temp)
    # plt.show()
    # # plt.plot(x_data[0,0,1,:])
    # valtype = 'subj'
    #
    # test_subj = np.r_[fold_idx * 9:fold_idx * 9 + 9, fold_idx * 9 + 54:fold_idx * 9 + 9 + 54]
    #
    # train_subj = np.setdiff1d(np.r_[0:108], test_subj)


    dataset_train = MIDataset(x=x_data_train, y=y_data_train)
    dataset_test = MIDataset(x=x_data_test, y=y_data_test)



    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    #create model
    from networks import Deep4Net_origin, ConvClfNet, TripletNet, FcClfNet
    import networks as nets
    from torch.optim import lr_scheduler
    import torch.optim as optim

    embedding_net = nets.Deep4Net_origin(2, 62, 400)
    model = FcClfNet(embedding_net)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.5 * 0.001)

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    if cuda:
        model.cuda(device)


    print(model)

    # if startepoch > 0:
    #     #     load_model_path = model_save_path + 'danet_' + str(args.gamma) + '_' + str(startepoch) + '.pt'
    #     #     model_save_path = model_save_path + '(cont)'
    #     # else:
    load_model_path = None
    # load_model_path ="C:\\Users\\Starlab\\PycharmProjects\\csdg\\exp0719\\Sep25_01-59-47_DESKTOP-186GIONsubj_sim_g_0.7_m_1.0danet_0.7_99.pt" #기존모델  있으면 경로
    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))

    acc_all = np.empty((1, 18))
    max = 0
    for epochidx in range(1, 100):
        print(epochidx)
        train(args, model, device, train_loader,optimizer,scheduler,cuda, args.gpuidx)
        eval_loss, eval_score = eval(args, model, device, train_loader)
        eval_loss, eval_score = eval(args, model, device, test_loader)

    #
    #     eval_temp = np.array(eval_score)
    #     eval_temp = eval_temp.reshape(4, 18)  # 한폴드 9명 2세션 =18
    #     acc = eval_temp.mean(0) / args.batch_size
    #     acc_m = acc.mean(0)
    #     if acc_m > max:
    #         max = acc_m
    #     print("highest acc : ", max)
    #
    #     acc_all = np.vstack([acc_all, acc])
    #     np.save('[DeepALL]acc_all_' + str(args.fold_idx), acc_all)
    #
    #     if args.use_tensorboard:
    #         for subj in range(18):
    #             writer.add_scalar('eachsubj/'+str(subj), np.sum(eval_score[subj*2:subj*2+2]) / 200, epochidx)
    #         writer.add_scalar('Train/Loss', np.mean(train_loss) / args.batch_size, epochidx)
    #         writer.add_scalar('Train/Acc', np.mean(train_score) / args.batch_size, epochidx)
    #         writer.add_scalar('Eval/Loss', np.mean(eval_loss) / args.batch_size, epochidx)
    #         writer.add_scalar('Eval/Acc', np.mean(eval_score) / args.batch_size, epochidx)
    #         writer.close()
    #     if args.save_model:
    #         torch.save(model.state_dict(), model_save_path + 'danet_' + str(args.gamma) + '_' + str(epochidx) + '.pt')
    # acc_all = np.delete(acc_all, [0, 0], axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cross subject domain adaptation')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    args.cuda =  torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    args.gpuidx = 0
    args.seed = 0
    args.use_tensorboard = False
    args.save_model = False
    args.log_interval = 10
    # args.fold_idx = 4

    args.gamma = 0.7
    args.margin = 1.0
    args.DAsetting = False

    args.folder_name = 'subj_sim'
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.comment = 'dgnet/Deep4Net_origin_tripletloss_choicefix' + '_g_' + str(args.gamma) + '_m_' + str(
        args.margin)
    args.log_dir = os.path.join('subjsim0926', current_time + '_' + socket.gethostname() + args.comment)

    for fold in range(0,6) :
        print(fold)
        args.fold_idx = fold
        experiment(args)