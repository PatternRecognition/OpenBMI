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

    dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj,
                                trial=trial_train)
    dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj, trial=trial_val)

    triplet_dataset_train = TripletGiga4(x=x_data, y=y_subj, valtype=valtype, istrain=True, subj=train_subj,
                                         trial=trial_train)

    triplet_dataset_test = TripletGiga4(x=x_data, y=y_subj, valtype=valtype, istrain=False, subj=test_subj,
                                        trial=trial_val)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=args.batch_size, shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_test, batch_size=args.batch_size, shuffle=False)


    #create model
    import get_common as gc
    dgnet = gc.dgnet(gamma=args.gamma,margin=args.margin)
    model = dgnet.model
    if cuda:
        model.cuda(device)

    loss_fn = dgnet.loss_fn
    if cuda and (loss_fn is not None):
        loss_fn.cuda(device)

    optimizer = dgnet.optimizer
    milestones = dgnet.milestones
    scheduler = dgnet.scheduler
    exp_comment = dgnet.exp_comment

    print(model)

    model_save_path = 'model/' + folder_name + '/' + comment + '/'
    if (args.save_model):
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)
        writer.add_text('exp', exp_comment)
        writer.add_text('optimizer', str(optimizer))
        writer.add_text('scheduler', str(milestones))
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
        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, epochidx, args.epochs, cuda,
            args.gpuidx, log_interval=10)
        train_loss, train_score = eval(args, model.clf_net, device, train_loader)
        eval_loss, eval_score = eval(args, model.clf_net, device, test_loader)

        eval_temp = np.array(eval_score)
        eval_temp = eval_temp.reshape(4, 18) #한폴드 9명 2세션 =18
        acc = eval_temp.mean(0)/args.batch_size
        acc_m = acc.mean(0)
        if acc_m > max:
            max = acc_m
        print("highest acc : ", max)

        acc_all = np.vstack([acc_all, acc])
        np.save('[DG]acc_all_'+str(args.fold_idx), acc_all)

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
    args.device = torch.device("cuda:1" if args.cuda else "cpu")
    args.gpuidx = 1
    args.seed = 0
    args.use_tensorboard = False
    args.save_model = False

    # args.fold_idx = 0

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


#
