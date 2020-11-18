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

from torch.utils.tensorboard import SummaryWriter
from datasets import *

from trainer import fit


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
    x_data, y_data = load_smt(fs=250)
    # get subject number
    y_subj = np.zeros([108, 200])
    for i in range(108):
        y_subj[i, :] = i * 2
    y_subj = y_data.reshape(108, 200) + y_subj
    y_subj = y_subj.reshape(21600)

    valtype = 'subj'

    test_subj = np.r_[fold_idx * 9:fold_idx * 9 + 9, fold_idx * 9 + 54:fold_idx * 9 + 9 + 54]
    print('test subj:' + str(test_subj))
    train_subj = np.setdiff1d(np.r_[0:108], test_subj)

    trial_train = (0, 200)
    trial_val = (0, 200)

    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, 1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)



    from sklearn.model_selection import train_test_split

    x_data_test = x_data[test_subj, :, :, :, :]

    cnt = 0
    for subj in range(0,90):
        if cnt == 0:
            X_train, X_val, y_train, y_val = train_test_split(x_data[subj], y_data[subj], test_size = 0.1, random_state = 42)
            X_train = np.expand_dims(X_train, axis=0)
            X_val = np.expand_dims(X_val, axis=0)
        else:
            X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(x_data[subj], y_data[subj], test_size = 0.1, random_state = 42)
            X_train_temp = np.expand_dims(X_train_temp, axis=0)
            X_val_temp = np.expand_dims(X_val_temp, axis=0)


            X_train = np.concatenate((X_train, X_train_temp),axis=0)
            X_val = np.concatenate((X_val, X_val_temp),axis=0)
            y_train = np.concatenate((y_train, y_train_temp), axis=0)
            y_val = np.concatenate((y_val, y_val_temp), axis=0)
        cnt = cnt+1







    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 42)



    for subj in range(0,100):
        x_data_sub = x_data[0]
        y_data_sub = y_data[0]
        dataset_train = GigaDataset2(x=x_data_sub, y=y_data_sub, valtype=valtype, istrain=True, subj=0,
                                trial=trial_train)


        source_data.append(dataset_train)

    from torch.utils.data import Dataset, DataLoader
    temp = torch.utils.data.ConcatDataset(source_data)

    train_data, train_labels = temp.dataset.__getitem__(temp.indices)

    dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj, trial=trial_val)














    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    #create model
    from networks import Deep4Net_origin, ConvClfNet, TripletNet, FcClfNet
    import networks as nets
    from torch.optim import lr_scheduler
    import torch.optim as optim

    embedding_net = nets.Deep4Net_origin(2, 62, 1000)
    model = ConvClfNet(embedding_net)
    optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.5 * 0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    if cuda:
        model.cuda(device)


    print(model)

    model_save_path = 'model/' + folder_name + '/' + comment + '/'
    if (args.save_model):
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)
        # writer.add_text('exp', exp_comment)
        writer.add_text('optimizer', str(optimizer))
        writer.add_text('scheduler', str(scheduler))
        writer.add_text('model_save_path', model_save_path)
        writer.add_text('model', str(model))
        model_save_path = writer.log_dir
        writer.close()

    if (args.save_model):
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

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
    for epochidx in range(1, args.epochs):
        print(epochidx)
        train(args, model, device, train_loader,optimizer,scheduler,cuda, args.gpuidx)
        eval_loss, eval_score = eval(args, model, device, test_loader)

        eval_temp = np.array(eval_score)
        eval_temp = eval_temp.reshape(4, 18)  # 한폴드 9명 2세션 =18
        acc = eval_temp.mean(0) / args.batch_size
        acc_m = acc.mean(0)
        if acc_m > max:
            max = acc_m
        print("highest acc : ", max)

        acc_all = np.vstack([acc_all, acc])
        np.save('[DeepALL]acc_all_' + str(args.fold_idx), acc_all)

        if args.use_tensorboard:
            for subj in range(18):
                writer.add_scalar('eachsubj/'+str(subj), np.sum(eval_score[subj*2:subj*2+2]) / 200, epochidx)
            writer.add_scalar('Train/Loss', np.mean(train_loss) / args.batch_size, epochidx)
            writer.add_scalar('Train/Acc', np.mean(train_score) / args.batch_size, epochidx)
            writer.add_scalar('Eval/Loss', np.mean(eval_loss) / args.batch_size, epochidx)
            writer.add_scalar('Eval/Acc', np.mean(eval_score) / args.batch_size, epochidx)
            writer.close()
        if args.save_model:
            torch.save(model.state_dict(), model_save_path + 'danet_' + str(args.gamma) + '_' + str(epochidx) + '.pt')
    acc_all = np.delete(acc_all, [0, 0], axis=0)

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