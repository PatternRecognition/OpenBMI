from __future__ import print_function
import argparse
import sys
import time
import copy

import numpy as np

from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle

paser = argparse.ArgumentParser()
args = paser.parse_args("")
args.seed = 123
args.val_size = 0.1
args.test_size = 0.1
args.shuffle = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def make_adj_matrix(map=None):
    if map == None:
        map = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 0, -1, 1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 56, 57, -1, 58, 59, -1, -1, -1, -1],
            [-1, -1, -1, 54, 2, 3, 4, 5, 6, 55, -1, -1, -1],
            [44, 45, -1, 7, 32, 8, -1, 9, 33, 10, -1, 50, 49],
            [-1, -1, 11, 34, 12, 35, 13, 36, 14, 37, 15, -1, -1],
            [16, 46, 47, 17, 38, 18, 39, 19, 40, 20, 51, 52, 21],
            [-1, -1, 48, 22, 23, 41, 24, 42, 25, 26, 53, -1, -1],
            [-1, -1, -1, -1, -1, 60, 43, 61, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 27, 28, 29, 30, 31, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    adj = np.zeros([62,62])
    for i in range(0,62):
        iidx = np.argwhere(map == i)
        for j in range(0,62):
            if i==j:
                adj[i, j] = 1
            else:
                jidx = np.argwhere(map == j)
                if (np.linalg.norm(iidx - jidx)) == 1.:
                    adj[i,j] = 1
    return adj

def convert_to_graph(x_data,y_data):
    #x_feature, x_adj 반환해야함
    x_adj = make_adj_matrix()
    x_feature = x_data #batch,ch,time(feature)
    y_data = y_data
    return x_data, x_adj, y_data

def load_smt(path='C:/Users/dk/PycharmProjects/data/giga'):
    with open(path+'/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)
    with open(path+'/smt1_scale.pkl', 'rb') as f:
        x_data1 = pickle.load(f)
    with open(path+'/smt2_scale.pkl', 'rb') as f:
        x_data2 = pickle.load(f)
    x_data = np.concatenate([x_data1, x_data2])
    # x_data = np.expand_dims(x_data, axis=1)
    return x_data,y_data


class GCNDataset(Dataset):
    def __init__(self, x,y, valtype, transform=None,istrain = True, sess=1,subj=None):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()

        x_data = x_data.reshape(108, -1, 62, 500)
        y_data = y_data.reshape(108, -1)

        if valtype == 'sess':
            if istrain:
                x_data = x_data[np.s_[0:54],  :, :, :]
                y_data = y_data[np.s_[0:54], :]
            else:
                x_data = x_data[np.s_[0 + 54:54 + 54], 100:200, :, :]  # tests sess2 online
                y_data = y_data[np.s_[0 + 54:54 + 54], 100:200]
        elif valtype == 'loso':
            if subj == None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data, np.s_[subj, subj + 54], 0)  # leave one subj
                y_data = np.delete(y_data, np.s_[subj, subj + 54], 0)
            else:
                x_data = x_data[np.s_[subj + 54], 100:200, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj + 54], 100:200]
        elif valtype == 'subj':
            if subj == None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, :, :, :]
                y_data = y_data[subj, :]
            else:
                x_data = x_data[subj, 100:200, :, :]  # tests sess2 online
                y_data = y_data[subj, 100:200]
        else:
            raise AssertionError()

        x_data = x_data.reshape(-1, 62, 500)
        # x_data = np.expand_dims(x_data, axis=1)
        y_data = y_data.reshape(-1)
        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()

        self.adj = make_adj_matrix()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.x_data[index,:,100:400]
        y = self.y_data[index]
        adj = self.adj
        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, adj, y



    #이걸 그냥 리스트로 받나?

    #예제코드 구현 해봐야할듯



def partition(list_feature, list_adj, list_logP, args):
    num_total = list_feature.shape[0]
    num_train = int(num_total * (1 - args.test_size - args.val_size))
    num_val = int(num_total * args.val_size)
    num_test = int(num_total * args.test_size)

    feature_train = list_feature[:num_train]
    adj_train = list_adj[:num_train]
    logP_train = list_logP[:num_train]
    feature_val = list_feature[num_train:num_train + num_val]
    adj_val = list_adj[num_train:num_train + num_val]
    logP_val = list_logP[num_train:num_train + num_val]
    feature_test = list_feature[num_total - num_test:]
    adj_test = list_adj[num_total - num_test:]
    logP_test = list_logP[num_total - num_test:]

    train_set = GCNDataset(feature_train, adj_train, logP_train)
    val_set = GCNDataset(feature_val, adj_val, logP_val)
    test_set = GCNDataset(feature_test, adj_test, logP_test)

    partition = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    return partition

# dict_partition = partition(list_feature, list_adj, list_logP, args)


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, n_atom, act=None, bn=False):
        super(GCNLayer, self).__init__()

        self.use_bn = bn
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(n_atom)
        self.activation = act

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        if self.use_bn:
            out = self.bn(out)
        if self.activation != None:
            out = self.activation(out)
        return out, adj


class SkipConnection(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out


class GatedSkipConnection(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0 - z, in_x)
        return out

    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1 + x2)


class GCNBlock(nn.Module):

    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_atom, bn=True, sc='gsc'):
        super(GCNBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i == 0 else hidden_dim,
                                        out_dim if i == n_layer - 1 else hidden_dim,
                                        n_atom,
                                        nn.ReLU() if i != n_layer - 1 else None,
                                        bn))
        self.relu = nn.ReLU()
        if sc == 'gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc == 'sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc == 'no':
            self.sc = None
        else:
            assert False, "Wrong sc type."

    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i == 0 else out), adj)
        if self.sc != None:
            out = self.sc(residual, out)
        out = self.relu(out)
        return out, adj


class ReadOut(nn.Module):

    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, 1)
        if self.activation != None:
            out = self.activation(out)
        return out


class Predictor(nn.Module):

    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        if self.activation != None:
            out = self.activation(out)
        return out


class GCNNet(nn.Module):

    def __init__(self, args):
        super(GCNNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1,10), stride=1)
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1,10), stride=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1,10), stride=1)
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1,10), stride=1)

        self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.blocks = nn.ModuleList()
        for i in range(args.n_block):
            self.blocks.append(GCNBlock(args.n_layer,
                                        args.in_dim if i == 0 else args.hidden_dim,
                                        args.hidden_dim,
                                        args.hidden_dim,
                                        args.n_atom,
                                        args.bn,
                                        args.sc))
        self.readout = ReadOut(args.hidden_dim,
                               args.pred_dim1,
                               act=nn.ReLU())
        self.pred1 = Predictor(args.pred_dim1,
                               args.pred_dim2,
                               act=nn.ReLU())
        self.pred2 = Predictor(args.pred_dim2,
                               args.pred_dim3,
                               act=nn.Tanh())
        self.pred3 = Predictor(args.pred_dim3,
                               args.out_dim)

    def forward(self, x, adj):
        out = x.unsqueeze(dim=1)
        out = self.conv1(out)
        #out = self.bn1(out)
        out = F.elu(out)
        out = F.max_pool2d(out, kernel_size=(1, 3), stride=(1, 2))

        out = self.conv2(out)
        #out = self.bn2(out)
        out = F.elu(out)
        out = F.max_pool2d(out, kernel_size=(1, 3), stride=(1, 2))

        out = self.conv3(out)
        #out = self.bn3(out)
        out = F.elu(out)
        out = F.max_pool2d(out, kernel_size=(1, 3), stride=(1, 2))

        out = self.conv4(out)
        #out = self.bn4(out)
        out = F.elu(out)
        out = F.max_pool2d(out, kernel_size=(1, 28), stride=(1, 28))

        out = out.squeeze()
        out = out.permute(0,2,1)
        for i, block in enumerate(self.blocks):
            out, adj = block((out if i == 0 else out), adj)
        out = self.readout(out)
        out = self.pred1(out)
        out = self.pred2(out)
        out = self.pred3(out)
        out = F.log_softmax(out, dim=1)
        return out




def train(model, device, optimizer, criterion, data_train, args,epoch):
    args.log_interval = 10
    epoch_train_loss = 0


    for batch_idx, batch in enumerate(data_train):
        x_feature = batch[0].to(device)
        x_adj = batch[1].to(device).float()
        target = batch[2].to(device)

        model.train()
        optimizer.zero_grad()
        output = model(x_feature, x_adj)
        #output.require_grad = False
        loss = F.nll_loss(output,target)

        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target), len(data_train.dataset),
                    100. * batch_idx / len(data_train), loss.item()))

        # bar.update(len(list_feature))

    epoch_train_loss /= len(data_train)

    return model, epoch_train_loss


def eval(model, device, data_test, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        logP_total = list()
        pred_logP_total = list()
        for i, batch in enumerate(data_test):
            x_feature = batch[0].to(device)
            x_adj = batch[1].to(device).float()
            target = batch[2].to(device)

            # logP_total += list_logP.tolist()
            # list_logP = list_logP.view(-1, 1)
            output = model(x_feature, x_adj)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_test.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(data_test.dataset),
        100. * correct / len(data_test.dataset)))

    return test_loss, correct


def experiment(device, args):
    time_start = time.time()

    model = GCNNet(args)
    model.to(device)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.l2_coef)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.l2_coef)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.l2_coef)
    else:
        assert False, 'Undefined Optimizer Type'

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)

    list_train_loss = list()
    # list_val_loss = list()


    x_data, y_data = load_smt()

    data_train = GCNDataset(x=x_data, y=y_data, valtype=args.valtype, istrain=True, sess=1)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True, num_workers=4)

    data_test = GCNDataset(x=x_data, y=y_data, valtype=args.valtype, istrain=False, sess=2, subj=-1)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, shuffle=False, num_workers=4)

    for epoch in range(args.epoch):
        #scheduler.step()
        model, train_loss = train(model, device, optimizer, criterion, train_loader, args, epoch)
        list_train_loss.append(train_loss)
        j_loss, j_score = eval(model, device, test_loader, args)




    # data_test = DataLoader(dict_partition['test'],
    #                        batch_size=args.batch_size,
    #                        shuffle=args.shuffle)

    # mae, std, logP_total, pred_logP_total = eval(model, device, data_test, args)
    #
    # time_end = time.time()
    # time_required = time_end - time_start
    #
    # args.list_train_loss = list_train_loss
    # # args.list_val_loss = list_val_loss
    # args.logP_total = logP_total
    # args.pred_logP_total = pred_logP_total
    # args.mae = mae
    # args.std = std
    # args.time_required = time_required

    return args

if __name__ == '__main__':
    args.batch_size = 200
    args.lr = 0.001
    args.l2_coef = 0
    args.optim = 'SGD'
    args.epoch = 100
    args.n_block = 2
    args.n_layer = 2
    args.n_atom =  62#노드수
    args.in_dim = 200#특징수
    args.hidden_dim = 62
    args.pred_dim1 = 10
    args.pred_dim2 = 20
    args.pred_dim3 = 40
    args.out_dim = 2
    args.bn = True
    args.sc = 'sc'
    args.atn = False
    args.step_size = 10
    args.gamma = 0.1
    args.valtype = 'sess'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    experiment(device, args)

