from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
from models.model_resnet import *
from models.model_openbmi import *
from models.model_3dcnn import *
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
from visualization import *

from torch.utils.tensorboard import SummaryWriter

def extract_embeddings(dataloader, model, num_ftrs=2):
    with torch.no_grad():
        model.eval()
        # num_ftrs = model.embedding_net.fc.out_features
        embeddings = np.zeros((len(dataloader.dataset), num_ftrs))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()

            embeddings[k:k+len(images)] = model(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.view(-1,1,62,301)
        target = target.view(-1)
        #data = nn.functional.interpolate(data,size=[300,300])
        optimizer.zero_grad()
        output = model(data)
        #output = nn.CrossEntropyLoss(output)
       # output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def eval(args, model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data = data.view(-1,1,62,data.shape[4])
            output = model(data)
            #output = nn.CrossEntropyLoss(output)
            #output = F.log_softmax(output, dim=1)

            test_loss.append(F.nll_loss(output, target, reduction='sum').item()) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss)/len(test_loader.dataset)
    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return test_loss, correct

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

class TripletGiga(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()

        x_data = x_data.reshape(108,-1,1,62,500)
        y_data = y_data.reshape(108,-1)

        if valtype =='sess':
            if istrain:
                x_data = x_data[np.s_[0:54],:,:,:,:]
                y_data = y_data[np.s_[0:54],:]
            else:
                x_data = x_data[np.s_[0+54:54+54],100:200,:,:,:] #tests sess2 online
                y_data = y_data[np.s_[0+54:54+54],100:200]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data,np.s_[subj,subj+54],0) #leave one subj
                y_data = np.delete(y_data,np.s_[subj,subj+54],0)
            else:
                x_data = x_data[np.s_[subj+54], 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54], 100:200]
        elif valtype == 'subj':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, :, :, :, :]
                y_data = y_data[subj, :]
            else:
                x_data = x_data[subj, 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[subj, 100:200]
        else:
            raise AssertionError()

        self.x_data = x_data.reshape(-1, 1, 62, 500)
        self.y_data = y_data.reshape(-1)
        self.len = self.y_data.shape[0]

        self.labels_set = set(self.y_data)
        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)

        if not istrain:
            self.labels_set = set(self.y_data)
            self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                     for label in self.labels_set}

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.y_data[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.y_data[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.x_data))]

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.istrain:
            img1 = self.x_data[index,:,:,100:500]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[y1])
            negative_label = np.random.choice(list(self.labels_set - set([y1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.x_data[positive_index,:,:,100:500]
            img3 = self.x_data[negative_index,:,:,100:500]
            y2 = self.y_data[positive_index]
            y3 = self.y_data[negative_index]

        else:
            img1 = self.x_data[self.test_triplets[index][0],:,:,100:500]
            img2 = self.x_data[self.test_triplets[index][1],:,:,100:500]
            img3 = self.x_data[self.test_triplets[index][2],:,:,100:500]
            y1 = self.y_data[self.test_triplets[index][0]]
            y2 = self.y_data[self.test_triplets[index][1]]
            y3 = self.y_data[self.test_triplets[index][2]]




        img1 = torch.from_numpy(img1).type(torch.FloatTensor)
        img2 = torch.from_numpy(img2).type(torch.FloatTensor)
        img3 = torch.from_numpy(img3).type(torch.FloatTensor)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), (y1,y2,y3)


    def __len__(self):
        return self.len
class TripletGiga2(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, setting='pn'):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()

        x_data = x_data.reshape(108,-1,1,62,500)
        y_data = y_data.reshape(108,-1)

        if valtype =='sess':
            if istrain:
                x_data = x_data[np.s_[0:54],:,:,:,:]
                y_data = y_data[np.s_[0:54],:]
            else:
                x_data = x_data[np.s_[0+54:54+54],100:200,:,:,:] #tests sess2 online
                y_data = y_data[np.s_[0+54:54+54],100:200]
        elif valtype == 'loso':
            if subj == None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data,np.s_[subj,subj+54],0) #leave one subj
                y_data = np.delete(y_data,np.s_[subj,subj+54],0)
            else:
                x_data = x_data[np.s_[subj+54], 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54], 100:200]
        elif valtype == 'subj':
            if subj == None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, :, :, :, :]
                y_data = y_data[subj, :]
            else:
                x_data = x_data[subj, 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[subj, 100:200]
        else:
            raise AssertionError()

        self.x_data = x_data.reshape(-1, 1, 62, 500)
        self.y_data = y_data.reshape(-1)
        self.len = self.y_data.shape[0]

        self.labels_set = set(self.y_data)
        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)
        self.setting = setting
        if not istrain:
            self.labels_set = set(self.y_data)
            self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                     for label in self.labels_set}


            if self.setting == 'pn':
                triplets = [[i,
                             random_state.choice(self.label_to_indices[self.y_data[i].item()]),#pp
                             random_state.choice(self.label_to_indices[
                                                     np.random.choice(list(set(
                                 self.y_data[np.where(
                                     (self.y_data % 2 != self.y_data[i] % 2))]
                             )- set([self.y_data[i].item()])))])
                             ]
                            for i in range(len(self.x_data))]
            elif self.setting == 'np':
                triplets = [[i,
                             random_state.choice(self.label_to_indices[self.y_data[i].item()]),  #pp
                             random_state.choice(self.label_to_indices[np.random.choice(list(set([self.y_data[
                                                                                                      i].item() + 1 if (
                                         self.y_data[i] % 2 == 0) else self.y_data[i].item() - 1])))])  # np
                             ]
                            for i in range(len(self.x_data))]

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.istrain:
            img1 = self.x_data[index,:,:,100:500]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[y1])
            if self.setting == 'pn':
                negative_label = np.random.choice(list(set(#pn
                                 self.y_data[np.where(
                                     (self.y_data % 2 == y1 % 2))]
                             )- set([y1])))
            elif self.setting == 'np':
                negative_label = np.random.choice(list(set([y1+1 if (
                        y1 % 2 == 0) else y1 - 1])))

            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.x_data[positive_index,:,:,100:500]
            img3 = self.x_data[negative_index,:,:,100:500]
            y2 = self.y_data[positive_index]
            y3 = self.y_data[negative_index]

        else:
            img1 = self.x_data[self.test_triplets[index][0],:,:,100:500]
            img2 = self.x_data[self.test_triplets[index][1],:,:,100:500]
            img3 = self.x_data[self.test_triplets[index][2],:,:,100:500]
            y1 = self.y_data[self.test_triplets[index][0]]
            y2 = self.y_data[self.test_triplets[index][1]]
            y3 = self.y_data[self.test_triplets[index][2]]




        img1 = torch.from_numpy(img1).type(torch.FloatTensor)
        img2 = torch.from_numpy(img2).type(torch.FloatTensor)
        img3 = torch.from_numpy(img3).type(torch.FloatTensor)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), (y1,y2,y3)


    def __len__(self):
        return self.len

class QuintupletGiga(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()

        x_data = x_data.reshape(108,-1,1,62,500)
        y_data = y_data.reshape(108,-1)

        if valtype =='sess':
            if istrain:
                x_data = x_data[np.s_[0:54],:,:,:,:]
                y_data = y_data[np.s_[0:54],:]
            else:
                x_data = x_data[np.s_[0+54:54+54],100:200,:,:,:] #tests sess2 online
                y_data = y_data[np.s_[0+54:54+54],100:200]
        elif valtype == 'loso':
            if subj == None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data,np.s_[subj,subj+54],0) #leave one subj
                y_data = np.delete(y_data,np.s_[subj,subj+54],0)
            else:
                x_data = x_data[np.s_[subj+54], 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54], 100:200]
        elif valtype == 'subj':
            if subj == None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, :, :, :, :]
                y_data = y_data[subj, :]
            else:
                x_data = x_data[subj, 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[subj, 100:200]
        else:
            raise AssertionError()

        self.x_data = x_data.reshape(-1, 1, 62, 500)
        self.y_data = y_data.reshape(-1)
        self.len = self.y_data.shape[0]

        self.labels_set = set(self.y_data)

        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)

        if not istrain:
            self.labels_set = set(self.y_data)
            self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                     for label in self.labels_set}

            quintuplets = [[i, #ref
                         random_state.choice(self.label_to_indices[self.y_data[i].item()]), #pp
                         random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set #pn
                                                                                         - set([self.y_data[i].item()])#자기자신제외
                                                                                         - set([np.where(self.y_data[i]%2==0,self.y_data[i]+1,self.y_data[i]-1).item()]) #동일 사용자 제외
                                                                                         - set(self.y_data[np.where(self.y_data%2!=(self.y_data[i]%2))]) #동일 클래스 제외
                                                                                         ))]), #pn
                         random_state.choice(self.label_to_indices[np.random.choice(list(set([self.y_data[i].item()+1 if (self.y_data[i]%2==0) else self.y_data[i].item()-1])))]), #np

                         random_state.choice(self.label_to_indices[np.random.choice(list(set(
                             self.y_data[np.where((self.y_data - self.y_data%2 != self.y_data[i] - self.y_data[i]%2)&(self.y_data%2!=self.y_data[i]%2))]
                           )))])
                         ]
                        for i in range(len(self.x_data))] #샘플수인덱스마다 트리플렛 생성



            #예를들러클래스가 109일때 해당하는 샘플들 인덱스 즉 포지티브 샘플을 하나 랜덤 초이스
            #
            self.test_triplets = quintuplets

    def __getitem__(self, index):
        if self.istrain:
            img1 = self.x_data[index,:,:,100:500]
            y1 = self.y_data[index]

            ppindex = y1

            ppindex = index
            while ppindex == index:
                ppindex = np.random.choice(self.label_to_indices[y1])

            pnindex = np.random.choice(self.label_to_indices[np.random.choice(list(self.labels_set  # pn
                                                        - set([self.y_data[index].item()])  # 자기자신제외
                                                        - set([np.where(self.y_data[index] % 2 == 0, self.y_data[index] + 1, self.y_data[index] - 1).item()])  # 동일 사용자 제외
                                                        - set(self.y_data[np.where(self.y_data % 2 != (self.y_data[index] % 2))])  # 동일 클래스 제외
                                                        ))])
            npindex = np.random.choice(self.label_to_indices[np.random.choice(list(set([self.y_data[index].item()+1 if (self.y_data[index]%2==0) else self.y_data[index].item()-1])))])
            nnindex = np.random.choice(self.label_to_indices[np.random.choice(list(set(
                             self.y_data[np.where((self.y_data - self.y_data%2 != self.y_data[index] - self.y_data[index]%2)&(self.y_data%2!=self.y_data[index]%2))]
                           )))])



            img2 = self.x_data[ppindex,:,:,100:500]
            img3 = self.x_data[pnindex,:,:,100:500]
            img4 = self.x_data[npindex,:,:,100:500]
            img5 = self.x_data[nnindex,:,:,100:500]


        else:
            img1 = self.x_data[self.test_triplets[index][0],:,:,100:500]
            img2 = self.x_data[self.test_triplets[index][1],:,:,100:500]
            img3 = self.x_data[self.test_triplets[index][2],:,:,100:500]
            img4 = self.x_data[self.test_triplets[index][3],:,:,100:500]
            img5 = self.x_data[self.test_triplets[index][4],:,:,100:500]
            y1 = self.y_data[self.test_triplets[index][0]]

        img1 = torch.from_numpy(img1).type(torch.FloatTensor)
        img2 = torch.from_numpy(img2).type(torch.FloatTensor)
        img3 = torch.from_numpy(img3).type(torch.FloatTensor)
        img4 = torch.from_numpy(img4).type(torch.FloatTensor)
        img5 = torch.from_numpy(img5).type(torch.FloatTensor)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)

        return (img1, img2, img3, img4, img5), (y1,)


    def __len__(self):
        return self.len

class GigaDataset(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()

        x_data = x_data.reshape(108,-1,1,62,500)
        y_data = y_data.reshape(108,-1)

        if valtype =='sess':
            if istrain:
                x_data = x_data[np.s_[0:54],:,:,:,:]
                y_data = y_data[np.s_[0:54],:]
            else:
                x_data = x_data[np.s_[0+54:54+54],100:200,:,:,:] #tests sess2 online
                y_data = y_data[np.s_[0+54:54+54],100:200]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data,np.s_[subj,subj+54],0) #leave one subj
                y_data = np.delete(y_data,np.s_[subj,subj+54],0)
            else:
                x_data = x_data[np.s_[subj+54], 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54], 100:200]
        elif valtype == 'subj':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, :, :, :, :]
                y_data = y_data[subj, :]
            else:
                x_data = x_data[subj, 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[subj, 100:200]
        else:
            raise AssertionError()

        x_data = x_data.reshape(-1, 1, 62, 500)
        y_data = y_data.reshape(-1)
        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)

        y_data = torch.from_numpy(y_data)
        self.x_data = x_data.type(torch.FloatTensor)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,100:500]
        y = self.y_data[index]


        # fs =100
        # N = 400
        # import librosa
        # import librosa.display
        #
        # xtemp = x.clone().view(-1)
        # f, t, Zxx = signal.spectrogram(xtemp,fs=fs,mode='psd')
        #
        # D = np.abs(librosa.stft(xtemp.numpy(),n_fft=30,center=False))
        #
        # librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
        # f, t, Zxx = signal.spectrogram(x[0,:,:],fs=fs,nperseg=60,noverlap=49,mode='psd')
        #
        # plt.pcolormesh(t, f,Zxx)
        # plt.title('STFT Magnitude')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        # x = torch.from_numpy(Zxx)
        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len

def load_smt(path='C:/Users/dk/PycharmProjects/data/giga'):
    with open(path+'/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    with open(path+'/smt1_scale.pkl', 'rb') as f:
        x_data1 = pickle.load(f)
    with open(path+'/smt2_scale.pkl', 'rb') as f:
        x_data2 = pickle.load(f)
    x_data = np.concatenate([x_data1, x_data2])
    x_data = np.expand_dims(x_data, axis=1)
    return x_data,y_data

def main():
    import torch
    from torch.optim import lr_scheduler
    import torch.optim as optim
    from torch.autograd import Variable
    from trainer import fit
    import numpy as np
    cuda = torch.cuda.is_available()
    # Training settings

    parser = argparse.ArgumentParser(description='cross subject domain adaptation')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
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

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    args = parser.parse_args()
    args.seed = 0
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    from datetime import datetime
    import os

    loging = False

    if (args.save_model):
        model_save_path = 'model/DANet(Deep4Net100)_test/'
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
    if loging:
        fname = model_save_path + datetime.today().strftime("%m_%d_%H_%M") + ".txt"
        f = open(fname, 'w')

    x_data, y_data = load_smt()

    #get subject number
    y_subj = np.zeros([108, 200])
    for i in range(108):
        y_subj[i, :] = i * 2
    y_subj = y_data.reshape(108, 200) + y_subj
    y_subj = y_subj.reshape(21600)

    # nonbciilli = np.s_[0,1,2,4,5,8,16,17,18,20,21,27,28,29,30,32,35,36,38,42,43,44,51]

    valtype = 'sess'
    if valtype == 'loso':
        pass
    elif valtype == 'sess':
        from networks import EmbeddingDeep4CNN, TripletNet, FineShallowCNN, EmbeddingDeepCNN, QuintupletNet, EmbeddingShallowCNN
        from losses import TripletLoss_dev2

        # make model for metric learning
        margin = 1.0
        gamma = 0.7
        embedding_net = EmbeddingDeep4CNN()

        print(embedding_net)

        model = TripletNet(embedding_net)
        #model.fc = nn.Linear(embedding_net.num_hidden,2)
        if cuda:
            model.cuda()
        loss_fn = TripletLoss_dev2(margin,gamma).cuda()


        n_epochs =1
        log_interval = 10
        #load_model_path = model_save_path+'dgnet1.027.pt' #'C:\\Users\dk\PycharmProjects\giga_cnn\구모델\\clf_83_8.pt'#'clf_29.pt' #'triplet_mg26.pt'#'clf_triplet2_5.pt' #'triplet_31.pt'
        load_model_path = None


        if load_model_path is not None:
            model.load_state_dict(torch.load(load_model_path))

        # For classification data
        dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, sess=1)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   )

        dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, sess=2, subj=-1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  )


        # For Domain adaptation
        triplet_dataset_train = TripletGiga(x=x_data, y=y_data,valtype=valtype, istrain=True, sess=1)
        triplet_train_loader  = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=args.batch_size, shuffle=True)

        triplet_dataset_test = TripletGiga(x=x_data, y=y_data,valtype=valtype, istrain=False, sess=2, subj=-1)
        triplet_test_loader  = torch.utils.data.DataLoader(triplet_dataset_test, batch_size=args.batch_size, shuffle=False)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)


        #model for validation
        evalmodel = nn.Sequential(model.embedding_net,
                                  model.fc,
                                  nn.LogSoftmax(dim=1)).to(device)
        print('____________DANet____________')
        print(model)


        #
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter()
        # images, labels = next(iter(train_loader))
        # import torchvision
        # #grid = torchvision.utils.make_grid(images)
        # writer.add_images('images',images)
        #
        # writer.add_embedding(metadata=train_embeddings_tl)
        # # writer.add_embedding(metadata = val_embeddings_tl)
        # writer.close()

        for temp in range(1, 50):
            fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
                log_interval)
            train_loss, j_score = eval(args, evalmodel, device, train_loader)
            eval_loss, j_score = eval(args, evalmodel, device, test_loader)
            writer.add_scalar('Train/Loss', np.mean(train_loss)/100, temp)
            writer.add_scalar('Eval/Loss', np.mean(eval_loss)/100, temp)
            writer.close()

            np.mean(train_loss)

            torch.save(model.state_dict(), model_save_path + 'dgnet_'+str(gamma)+'_'+ str(temp) + '.pt')


        #for visualization
        dataset_train_subj = GigaDataset(x=x_data, y=y_subj, valtype='subj', istrain=True, sess=1, subj=np.r_[0:10])
        train_loader_subj = torch.utils.data.DataLoader(dataset_train_subj, batch_size=args.batch_size, shuffle=True,
                                                   )

        dataset_test_subj = GigaDataset(x=x_data, y=y_subj, valtype='sess', istrain=False, sess=2, subj=-1)
        test_loader_subj = torch.utils.data.DataLoader(dataset_test_subj, batch_size=args.batch_size, shuffle=False,
                                                  )

        # train_embeddings_tl, train_labels_tl = extract_features(train_loader_subj.dataset, model.embedding_net.convnet,0,100)
        # val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader_subj, model.embedding_net, 1000)

        train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader_subj, model.embedding_net,model.embedding_net.num_hidden)
        val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader_subj, model.embedding_net,model.embedding_net.num_hidden)
        #
        # train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader_subj, model.embedding_net.convnet[0], 1000)
        # val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader_subj, model.embedding_net, 1000)

        # = train_labels_tl-train_labels_tl%2
        # from torchvision import datasets, models, transforms
        # temp = model.embedding_net.children()
        # newmodel = torch.nn.Sequential(*(list(model.embedding_net.children())[:]))


        # for param in model.embedding_net.parameters():
        #     param.requires_grad = True

        #newembedding_net = torch.nn.Sequential(*(list(model.embedding_net.children())[:]))
        #
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2,perplexity=30)
        #features = np.concatenate([train_embeddings_tl,val_embeddings_tl])
        #val_labels_tl = val_labels_tl+2
        #labels = np.concatenate([train_labels_tl,val_labels_tl])


        train_tsne = tsne.fit_transform(train_embeddings_tl)
        plot_features(train_tsne,train_labels_tl)
        plot_features(train_tsne, train_labels_tl%2)
        for i in range(0,10):
            plot_features(train_tsne[i*200:(i+1)*200],train_labels_tl[i*200:(i+1)*200])


        plot_features3d(train_tsne,train_labels_tl)


        val_tsne = tsne.fit_transform(val_embeddings_tl)
        plot_features(val_tsne,val_labels_tl-108)
        plot_features(val_tsne, val_labels_tl%2)
        plot_features3d(val_tsne,val_labels_tl%2)


        plot_embeddings(val_tsne, (val_labels_tl-108)-(val_labels_tl-108)%2)




        for param in model.embedding_net.parameters():
            param.requires_grad = True

        #embedding_net2 = EmbeddingDeep4CNN()
        newmodel = nn.Sequential(model.embedding_net,
                                nn.Linear(1000, 2),
                                nn.Dropout(p=0.5),
                                nn.LogSoftmax(dim=1),
                                ).to(device)

        print(newmodel)

        #newmodel.fc_lr = nn.Linear(1000,2)
        newmodel.to(device)
        optimizer = optim.SGD(newmodel.parameters(), lr=0.001, momentum=0.9)
        #optimizer = optim.Adam(newmodel.parameters())

        for epoch in range(1, 100):
            train(args, newmodel, device, train_loader, optimizer, epoch)
            j_loss, j_score = eval(args, newmodel, device, test_loader)

        if args.save_model:
            torch.save(newmodel.state_dict(),'clf_83_8.pt')
        newmodel.load_state_dict(torch.load(load_model_path))



if __name__ == '__main__':
    main()


