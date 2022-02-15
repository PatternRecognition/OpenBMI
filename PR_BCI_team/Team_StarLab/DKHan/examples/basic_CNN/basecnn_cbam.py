from __future__ import print_function
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
from cbam import *

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form. 

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, (1,10), 1)
        #self.conv1_2 = nn.Conv2d(20, 20, (1, 5), 1)
        self.conv2 = nn.Conv2d(25, 25, (62,1), 1)
        self.conv3 = nn.Conv2d(25, 50, (1,10), 1)
        self.conv4 = nn.Conv2d(50, 100, (1,10), 1)
        self.conv5 = nn.Conv2d(100, 200, (1,9), 1)

        # FC layers
        self.fc1 = nn.Linear(200 * 1 * 2, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.elu(self.conv1(x)) #temporal
        #x = F.relu(self.conv1_2(x))
        # x = F.max_pool2d(x, (1,2), (1,2))
        x = F.elu(self.conv2(x)) #spatial
        x = F.max_pool2d(x, (1, 3), 3)

        x = F.elu(self.conv3(x))
        x = F.max_pool2d(x, (1, 3), 3)

        x = F.elu(self.conv4(x))
        x = F.max_pool2d(x, (1, 2), 2)

        x = F.elu(self.conv5(x))
        #x = F.max_pool2d(x, (1, 3), 3)

        x = x.view(-1, 200 * 1 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1024

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,45), 1)
        self.fc1 = nn.Linear(self.num_filters * 1 * 149, self.num_hidden)
        self.bn = nn.BatchNorm1d(self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_fin = nn.Linear(self.num_hidden, 2)
        self.cbam = CBAM(self.num_filters,16)


    def forward(self, x):
        x = self.conv1(x[:, :, :, :])
        x = self.cbam(x)
        x = F.elu(x) #temporal
        x = F.max_pool2d(x, (1, 10), 3)


        x = x.view(-1, self.num_filters * 1 * 149)
        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))

        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc_fin(x)
        x = F.softmax(x, dim=1)
        return x

class Net_multi(nn.Module):
    def __init__(self):
        super(Net_multi, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1024

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,45), 1)
        self.fc1 = nn.Linear(self.num_filters * 1 * 149, self.num_hidden)
        self.bn = nn.BatchNorm1d(self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_lr = nn.Linear(self.num_hidden, 2)
        self.fc_subj = nn.Linear(self.num_hidden, 54)

    def forward(self, x):
        x = F.elu(self.conv1(x[:,:,:,:])) #temporal
        x = F.max_pool2d(x, (1, 10), 3)
        x = x.view(-1, self.num_filters * 1 * 149)
        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x2 = self.fc_subj(x)
        x1 = F.log_softmax(x1, dim=1)
        x2 = F.log_softmax(x2, dim=1)
        return x1,x2

class Net_cbam(nn.Module):
    def __init__(self):
        super(Net_cbam, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1024

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,45), 1)
        self.fc1 = nn.Linear(self.num_filters * 1 * 149, self.num_hidden)
        self.bn = nn.BatchNorm1d(self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_lr = nn.Linear(self.num_hidden, 2)
        self.fc_subj = nn.Linear(self.num_hidden, 54)

        self.cbam = CBAM(self.num_filters, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam(x)
        x = F.elu(x)  # temporal

        x = F.max_pool2d(x, (1, 10), 3)
        x = x.view(-1, self.num_filters * 1 * 149)
        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x2 = self.fc_subj(x)
        x1 = F.log_softmax(x1, dim=1)
        x2 = F.log_softmax(x2, dim=1)
        return x1,x2

class Net_cbam2(nn.Module):
    def __init__(self,use_cbam = True):
        super(Net_cbam2, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1000

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,10), stride=1) #템포럴
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, (1,20), stride=1)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, (1,20), stride=1)
        #self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, 9, stride=2)

        self.fc1 = nn.Linear(self.num_filters*1*94, self.num_hidden)
        #self.bn1 = nn.BatchNorm2d(self.num_filters)

        #self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_lr = nn.Linear(self.num_hidden, 2)
        self.fc_subj = nn.Linear(self.num_hidden, 54)
        if use_cbam:
            self.cbam1 = CBAM(self.num_filters, 16)
            self.cbam2 = CBAM(self.num_filters, 16)
        else:
            self.cbam1 = None
    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 2))
        x = self.conv2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 2))
        x = self.conv3(x)
        if not self.cbam1 == None:
            x = self.cbam1(x)
        x = F.elu(x)
        #x = F.avg_pool2d(x, (1, 2))
        #x = F.adaptive_avg_pool2d(x,(1,53))
        x = x.view(-1, self.num_filters*1*94)
        #x = x.view(-1, self.num_filters * 3 * 58)
        # x = F.elu(self.bn(self.fc1(x)))

        x = self.fc1(x)
        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x2 = self.fc_subj(x)
        x1 = F.log_softmax(x1, dim=1)
        x2 = F.log_softmax(x2, dim=1)
        return x1,x2

class Net_cbam3(nn.Module): #inception
    def __init__(self,use_cbam = True):
        super(Net_cbam3, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1000

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,45), stride=1) #템포럴
        self.conv2 = nn.Conv2d(1, self.num_filters, (62,1), stride=1)
        self.conv3 = nn.Conv2d(1, self.num_filters, (1,100), stride=(1,30))

        #self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, 9, stride=2)

        self.fc1 = nn.Linear(6720, self.num_hidden)
        #self.bn1 = nn.BatchNorm2d(self.num_filters)

        #self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_lr = nn.Linear(self.num_hidden, 2)
        self.fc_subj = nn.Linear(self.num_hidden, 54)
        if use_cbam:
            self.cbam1 = CBAM(self.num_filters, 16)
            self.cbam2 = CBAM(self.num_filters, 16)
        else:
            self.cbam1 = None
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = F.elu(out1)
        out1 = F.max_pool2d(out1, (1, 10),3) #1,149
        out1 = out1.view(-1, self.num_filters * 1 * 149)

        out2 = self.conv2(x)
        out2 = F.elu(out2)
        out2 = F.max_pool2d(out2, (1, 100))
        out2 = out2.view(-1, self.num_filters * 1 * 5)

        out3 = self.conv3(x)
        out3 = F.elu(out3)
        out3 = F.max_pool2d(out3, (62, 1))
        out3 = out3.view(-1, self.num_filters * 1 * 14)

        x = torch.cat([out1,out2,out3],1)
        if not self.cbam1 == None:
            x = self.cbam1(x)
        x = F.elu(x)
        #x = F.avg_pool2d(x, (1, 2))
        #x = F.adaptive_avg_pool2d(x,(1,53))
        x = x.view(-1, x.size(1))
        #x = x.view(-1, self.num_filters * 3 * 58)
        # x = F.elu(self.bn(self.fc1(x)))

        x = self.fc1(x)
        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x2 = self.fc_subj(x)
        x1 = F.log_softmax(x1, dim=1)
        x2 = F.log_softmax(x2, dim=1)
        return x1,x2

class Net_cbam4(nn.Module): #shallowconv
    def __init__(self,use_cbam = True):
        super(Net_cbam4, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1000

        self.conv1 = nn.Conv2d(1, self.num_filters, (1,25), stride=1) #템포럴
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, (62,1), stride=1) #채널

        self.fc1 = nn.Linear(self.num_filters * 1 * 27, self.num_hidden)
        #self.bn1 = nn.BatchNorm2d(self.num_filters)

        #self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_lr = nn.Linear(self.num_hidden, 2)
        self.fc_subj = nn.Linear(self.num_hidden, 54)
        if use_cbam:
            self.cbam1 = CBAM(self.num_filters, 16)
            self.cbam2 = CBAM(self.num_filters, 16)
        else:
            self.cbam1 = None
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x*x
        x = F.avg_pool2d(x, kernel_size = (1, 75), stride = (1,15)) #1,149


        x = x.view(-1, self.num_filters * 1 * 27)

        if not self.cbam1 == None:
            x = self.cbam1(x)

        x = torch.log(x)

        x = self.fc1(x)
        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x2 = self.fc_subj(x)
        x1 = F.log_softmax(x1, dim=1)
        x2 = F.log_softmax(x2, dim=1)
        return x1,x2

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.view(-1,1,62,301)
        target = target.view(-1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def train_mt(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.view(-1,1,62,301)
        #target = target.view(-1)
        optimizer.zero_grad()
        output1, output2= model(data)
        alpha = 1.0
        loss1 = F.nll_loss(output1, target[:,0])
        loss2 = F.nll_loss(output2, target[:, 1])
        loss = alpha*loss1+(1-alpha)*10/loss2
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss1.item(),loss2.item()))


def eval(args, model, device, test_loader):
    model.eval()
    loss1 = 0
    loss2 = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model(data)

            loss1 += F.nll_loss(output1, target[:,0], reduction='sum').item()
            loss2 += F.nll_loss(output2, target[:,1], reduction='sum').item() # sum up batch loss

            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct1 += pred1.eq(target[:,0].view_as(pred1)).sum().item()
            correct2 += pred2.eq(target[:,1].view_as(pred2)).sum().item()
    loss1 /= len(test_loader.dataset)

    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy1: {}/{} ({:.0f}%), Accuracy2: {}/{} ({:.0f}%)'.format(
        loss1,
        correct1,
        len(test_loader.dataset),
        100. * correct1 / len(test_loader.dataset),
        correct2,
        len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))

    return loss1, correct1


def windows(data, size, step):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += step


def segment_signal_without_transition(data, window_size, step):
	segments = []
	for (start, end) in windows(data, window_size, step):
		if(len(data[start:end]) == window_size):
			segments = segments + [data[start:end]]
	return np.array(segments)


def segment_dataset(X, window_size, step):
	win_x = []
	for i in range(X.shape[0]):
		win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
	win_x = np.array(win_x)
	return win_x


class GigaDataset_subject_wise(Dataset):
    """ Diabetes dataset."""
    #subject_wise 이며 test엔 online데이터만 사용
    # Initialize your data, download, etc.
    def __init__(self,transform=None,istrain = True,subj=0):
        self.transform = transform

        with open('data/epoch_data-1to4.pkl', 'rb') as f:
            x_data = pickle.load(f)
        x_data = np.expand_dims(x_data,axis=1)

        with open('data/epoch_labels.pkl', 'rb') as f:
            y_data = pickle.load(f)

        if istrain:
            x_data = np.delete(x_data,np.s_[200*subj:200*subj+200],0)
            y_data = np.delete(y_data,np.s_[200*subj:200*subj+200],0)
        else:
            # get online data
            x_data = x_data[200*subj+100:200*subj+200,:,:,:]
            y_data = y_data[200*subj+100:200*subj+200]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)

        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()



    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index]

        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len

class GigaDataset_LOSO(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, x, y, transform=None, istrain=True, fine=False, sess=1, subj=54):
        self.transform = transform

        x_data = x
        y_data = y

        if istrain:
            if fine == True:
                #get subj's offline data
                x_data = x_data[200 * (subj + 54):200 * (subj + 54) + 100, :, :, :]
                y_data = y_data[200 * (subj + 54):200 * (subj + 54) + 100,:]
            else:
                x_data = np.delete(x_data, np.s_[200 * subj:200 * subj + 200], 0)  # 1세션에서 지움
                y_data = np.delete(y_data, np.s_[200 * subj:200 * subj + 200], 0)

                x_data = np.delete(x_data, np.s_[10600+200*subj:10600+200*subj + 200], 0)  # 2세션에서 지움
                y_data = np.delete(y_data, np.s_[10600+200*subj:10600+200*subj + 200], 0)
        else:
            #2세션에 온라인만 테스트
            x_data = x_data[200*(subj+(sess-1)*54)+100:200*(subj+(sess-1)*54)+200,:,:,:]
            y_data = y_data[200*(subj+(sess-1)*54)+100:200*(subj+(sess-1)*54)+200,:]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index,:]

        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len

class GigaDataset_session_wise(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,x,y,transform=None,istrain = True, sess=1,subj=-1):
        self.transform = transform

        x_data = x
        y_data = y

        if istrain:
            x_data = x_data[10800*(sess-1):10800*(sess-1)+10800,:,:,:] #sess1 or 2학습용으로 선택
            y_data =  y_data[10800*(sess-1):10800*(sess-1)+10800]
            #x_data = np.delete(x_data, np.s_[200 * subj: 200 * subj + 200], 0)#LOO 적용
            #y_data = np.delete(y_data, np.s_[200 * subj: 200 * subj + 200], 0)
        else:
            if subj==-1:
                x_data = x_data[10800*(sess-1):10800*(sess-1)+10800,:,:,:] # sess2 online 전체 사용
                y_data = y_data[10800*(sess-1):10800*(sess-1)+10800]
            else:
                # get each subjects online data
                x_data = x_data[200*(subj+(sess-1)*54)+100:200*(subj+(sess-1)*54)+200,:,:,:]
                y_data = y_data[200*(subj+(sess-1)*54)+100:200*(subj+(sess-1)*54)+200,:]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index]

        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len

class GigaDataset_off2on(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,transform=None,istrain = True,subj=0):
        self.transform = transform

        with open('epoch_data.pkl', 'rb') as f:
            x_data = pickle.load(f)
        x_data = np.expand_dims(x_data,axis=1)

        with open('epoch_labels.pkl', 'rb') as f:
            y_data = pickle.load(f)

        if istrain:
            x_data = x_data[200*subj:200*subj+100,:,:,:]
            y_data = y_data[200*subj:200*subj+100]
        else:
            x_data = x_data[200*subj+100:200*subj+200,:,:,:]
            y_data = y_data[200*subj+100:200*subj+200]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index]

        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len

def load_smt(path='C:/Users/dk/PycharmProjects/data/giga'):
    with open(path+'/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    with open(path+'/smt1_2_scale.pkl', 'rb') as f:
        x_data1 = pickle.load(f)
    with open(path+'/smt2_2_scale.pkl', 'rb') as f:
        x_data2 = pickle.load(f)
    x_data = np.concatenate([x_data1, x_data2])
    x_data = np.expand_dims(x_data, axis=1)
    return x_data,y_data

def loso_sess2():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from datetime import datetime

    loging = True
    if loging:
        fname = "log/190321/base_cnn_LOSO_fine" + datetime.today().strftime("%m_%d_%H_%M") + ".txt"
        f = open(fname, 'w')

    x_data, y_data = load_smt()

    model_save_path = 'model/'
    for subj in range(0,54):
        model = Net2().to(device)
        # model.load_state_dict(torch.load('03_20_15_37basecnn.pt'))
        # model.eval()
        #
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer_fine = optim.SGD(model.parameters(), lr=0.005, momentum=args.momentum)
        dataset = GigaDataset_LOSO(x=x_data, y=y_data, istrain=True, sess=1,subj=subj)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        dataset_test = GigaDataset_LOSO(x=x_data, y=y_data, istrain=False, sess=2, subj=subj)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,**kwargs)


        dataset_fine = GigaDataset_LOSO(x=x_data, y=y_data, fine=True, istrain=True, sess=2, subj=subj)
        fine_loader = torch.utils.data.DataLoader(dataset_fine, batch_size=args.batch_size, shuffle=True, **kwargs)

        #LOSO joint training
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
        print("joint-train")
        j_loss, j_score = eval(args, model, device, test_loader)

        if (args.save_model):
            torch.save(model.state_dict(), model_save_path+ "J_" + str(subj) + 'basecnn.pt')

        #fine tuning
        for epoch in range(1, 20):
            train(args, model, device, fine_loader, optimizer_fine, epoch)
        print("fine-tuning")
        f_loss, f_score = eval(args, model, device, test_loader)

        if (args.save_model):
            torch.save(model.state_dict(), model_save_path+"F_" + str(subj) + 'basecnn.pt')

        if loging:
            f = open(fname, 'a')
            f.write(str(subj)+" "+"jl : "+ str(j_loss) + " " + str(j_score) + " " +
                    "fl : "+ str(f_loss) + " " + str(f_score) + '\n')
            f.close()



    #dataset_test = GigaDataset_session_wise(istrain=False, sess=2)
    #test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #loss, score = test(args, model, device, test_loader)

def loso_sess2_mt():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from datetime import datetime

    loging = True
    if loging:
        fname = "log/190329/basecnn_cbam_inception" + datetime.today().strftime("%m_%d_%H_%M") + ".txt"
        f = open(fname, 'w')

    x_data, y_data = load_smt()

    y_subj = np.zeros(21600)
    for subj in range(108):
        y_subj[subj*200:(subj+1)*200] = subj

    y_data = np.stack((y_data,y_subj),axis=1)

    import os
    model_save_path = 'model/shallowCNN'

    if not (os.path.isdir(model_save_path)):
        os.makedirs(os.path.join(model_save_path))



    loso = False

    if loso:
        for subj in range(0,54):
            model = Net_cbam4().to(device)
            #model.load_state_dict(torch.load(model_save_path+ "J_" + str(subj) + 'basecnn.pt'))

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            optimizer_fine = optim.SGD(model.parameters(), lr=0.005, momentum=args.momentum)

            dataset_train = GigaDataset_LOSO(x=x_data, y=y_data, istrain=True, sess=1, subj=subj)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)

            dataset_test = GigaDataset_LOSO(x=x_data, y=y_data, istrain=False, sess=2, subj=subj)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,**kwargs)

            dataset_fine = GigaDataset_LOSO(x=x_data, y=y_data, fine=True, istrain=True, sess=2, subj=subj)
            fine_loader = torch.utils.data.DataLoader(dataset_fine, batch_size=args.batch_size, shuffle=True, **kwargs)

            for epoch in range(1, args.epochs + 1):
                train_mt(args, model, device, train_loader, optimizer, epoch)
                print("joint-train")
                #LOSO joint training
                j_loss, j_score = eval(args, model, device, test_loader)

            # #fine tuning
            # for epoch in range(1, 10):
            #     train_mt(args, model, device, fine_loader, optimizer_fine, epoch)
            #
            # print("fine-tuning")
            # f_loss, f_score = eval(args, model, device, test_loader)

            if (args.save_model):
                torch.save(model.state_dict(), model_save_path+"F_" + str(subj) + 'basecnn.pt')

            if loging:
                f = open(fname, 'a')
                f.write(str(subj)+" "+"jl : "+ str(j_loss) + " " + str(j_score) + '\n')
                f.close()
    else:

        istrain = False
        model = Net_cbam4(use_cbam=False).to(device)
        if istrain:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            dataset_train = GigaDataset_session_wise(x=x_data, y=y_data, istrain=True, sess=1)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            model.load_state_dict(torch.load('model/shallowCNNshallow29.pt'))

        dataset_test = GigaDataset_session_wise(x=x_data, y=y_data, istrain=False, sess=2, subj=-1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)

        if istrain:
            for epoch in range(1, args.epochs + 1):
                train_mt(args, model, device, train_loader, optimizer, epoch)
                print("joint-train")
                # LOSO joint training
                j_loss, j_score = eval(args, model, device, test_loader)
                if (args.save_model):
                    torch.save(model.state_dict(), model_save_path + 'shallow'+str(epoch)+'.pt')

        for subj in range(0, 54):
            dataset_test = GigaDataset_session_wise(x=x_data, y=y_data, istrain=False, sess=2, subj=subj)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)

            j_loss, j_score = eval(args, model, device, test_loader)

            if loging:
                f = open(fname, 'a')
                f.write(str(subj) + " " + "jl : " + str(j_loss) + " " + str(j_score) + '\n')
                f.close()








    #dataset_test = GigaDataset_session_wise(istrain=False, sess=2)
    #test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #loss, score = test(args, model, device, test_loader)


def loso_sess1_eval():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from datetime import datetime

    loging = False
    loso = False

    if loging:
        fname = "log/190329/base_cnn_LOSO_sess1" + datetime.today().strftime("%m_%d_%H_%M") + ".txt"
        f = open(fname, 'w')

    x_data, y_data = load_smt()

    model_save_path = 'model/'

    if loso:
        for subj in range(0,54):
            model = Net2().to(device)
            model.load_state_dict(torch.load(model_save_path+ "J_" + str(subj) + 'basecnn.pt'))

            dataset_test = GigaDataset_LOSO(x=x_data, y=y_data, istrain=False, sess=1, subj=subj)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,**kwargs)


            j_loss, j_score = eval(args, model, device, test_loader)

            if loging:
                f = open(fname, 'a')
                f.write(str(subj)+" "+"jl : "+ str(j_loss) + " " + str(j_score) + '\n')
                f.close()
    else:

        for subj in range(0, 54):
            model = Net2().to(device)
            model.load_state_dict(torch.load(model_save_path + "J_" + str(subj) + 'basecnn.pt'))

            dataset_test = GigaDataset_LOSO(x=x_data, y=y_data, istrain=False, sess=1, subj=subj)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)

            j_loss, j_score = eval(args, model, device, test_loader)

            if loging:
                f = open(fname, 'a')
                f.write(str(subj) + " " + "jl : " + str(j_loss) + " " + str(j_score) + '\n')
                f.close()


    #dataset_test = GigaDataset_session_wise(istrain=False, sess=2)
    #test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #loss, score = test(args, model, device, test_loader)



if __name__ == '__main__':
    loso_sess2_mt()



