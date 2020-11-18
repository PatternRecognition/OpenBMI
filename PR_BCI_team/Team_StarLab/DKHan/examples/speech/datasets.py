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

class TripletGiga2(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, trial=None):
        self.transform = transform
        self.istrain = istrain
        if trial is None:
            if istrain:
                self.trial = 0,200
            else:
                self.trial = 100,200 #online
        else:
            self.trial = trial

        x_data = x.copy()
        y_data = y.copy()

        self.in_chans = x_data.shape[2]
        self.input_time_length = x_data.shape[3]

        x_data = x_data.reshape(108, -1, 1, self.in_chans, self.input_time_length)
        y_data = y_data.reshape(108,-1)

        if valtype == 'sess':
            if istrain:
                x_data = x_data[np.s_[0:54], self.trial[0]:self.trial[1], :, :, :]
                y_data = y_data[np.s_[0:54], self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[np.s_[0 + 54:54 + 54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[0 + 54:54 + 54], self.trial[0]:self.trial[1]]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data, np.s_[subj, subj + 54], 0)  # leave one subj
                y_data = np.delete(y_data, np.s_[subj, subj + 54], 0)
            else:
                x_data = x_data[np.s_[subj + 54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj + 54], self.trial[0]:self.trial[1]]
        elif valtype == 'subj':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, self.trial[0]:self.trial[1], :, :, :]
                y_data = y_data[subj, self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[subj, self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[subj, self.trial[0]:self.trial[1]]
        else:
            raise AssertionError()

        self.x_data = x_data.reshape(-1, 1, self.in_chans, self.input_time_length)
        self.y_data = y_data.reshape(-1)
        self.y_subj = (y_data.reshape(-1)-self.y_data%2)/2
        self.len = self.y_data.shape[0]

        self.labels_set = np.unique(self.y_data)
        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)

        if not istrain:
            self.labels_set = np.unique(self.y_data)
            self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                     for label in self.labels_set}



            triplets = [[i,
                         random_state.choice(self.label_to_indices[  # same task, diff subj
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2) & (
                                                                     (self.labels_set - self.labels_set % 2) / 2 !=
                                                                     self.y_subj[i]), True, False)]
                                                 )
                                             ]),
                         random_state.choice(self.label_to_indices[#diff task, diff subj
                                                 np.random.choice(
                                                     self.labels_set[np.where((self.labels_set%2!=self.y_data[i]%2) & ((self.labels_set-self.labels_set%2)/2 != self.y_subj[i]), True,False)]
                                                 )
                                             ])
                         ]
                        for i in range(self.len)]

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.istrain:
            img1 = self.x_data[index,:,:,:]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == y1 % 2) & (
                                                                     (self.labels_set - self.labels_set % 2) / 2 !=
                                                                     (y1-y1%2)/2), True, False)]
                                                 )])


            negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) & ((self.labels_set-self.labels_set%2)/2 != (y1-y1%2)/2), True,False)])
            # negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) & ((self.labels_set-self.labels_set%2)/2 != (y1-y1%2)/2), True,False)]) #diffsub
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.x_data[positive_index,:,:,:]
            img3 = self.x_data[negative_index,:,:,:]
            y2 = self.y_data[positive_index]
            y3 = self.y_data[negative_index]

        else:
            img1 = self.x_data[self.test_triplets[index][0],:,:,:]
            img2 = self.x_data[self.test_triplets[index][1],:,:,:]
            img3 = self.x_data[self.test_triplets[index][2],:,:,:]
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
        return (img1, img2, img3), (y1%2,y2%2,y3%2)


    def __len__(self):
        return self.len

class TripletGiga4(Dataset): #사용자가 구분이 가능하게
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, trial=None):
        self.transform = transform
        self.istrain = istrain
        if trial is None:
            if istrain:
                self.trial = 0,200
            else:
                self.trial = 100,200 #online
        else:
            self.trial = trial

        x_data = x.copy()
        y_data = y.copy()

        self.in_chans = x_data.shape[2]
        self.input_time_length = x_data.shape[3]

        x_data = x_data.reshape(108, -1, 1, self.in_chans, self.input_time_length)
        y_data = y_data.reshape(108,-1)

        if valtype == 'sess':
            if istrain:
                x_data = x_data[np.s_[0:54], self.trial[0]:self.trial[1], :, :, :]
                y_data = y_data[np.s_[0:54], self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[np.s_[0 + 54:54 + 54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[0 + 54:54 + 54], self.trial[0]:self.trial[1]]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data, np.s_[subj, subj + 54], 0)  # leave one subj
                y_data = np.delete(y_data, np.s_[subj, subj + 54], 0)
            else:
                x_data = x_data[np.s_[subj + 54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj + 54], self.trial[0]:self.trial[1]]
        elif valtype == 'subj':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, self.trial[0]:self.trial[1], :, :, :]
                y_data = y_data[subj, self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[subj, self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[subj, self.trial[0]:self.trial[1]]
        else:
            raise AssertionError()

        self.x_data = x_data.reshape(-1, 1, self.in_chans, self.input_time_length)
        self.y_data = y_data.reshape(-1)
        self.y_subj = (y_data.reshape(-1)-self.y_data%2)/2
        self.len = self.y_data.shape[0]

        self.labels_set = np.unique(self.y_data)
        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)

        if not istrain:
            self.labels_set = np.unique(self.y_data)
            self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                     for label in self.labels_set}



            triplets = [[i,
                         random_state.choice(self.label_to_indices[  # same task, same subj : positive
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2) & (
                                                                     (self.labels_set - self.labels_set % 2) / 2 ==
                                                                     self.y_subj[i]), True, False)]
                                                 )
                                             ]),
                         random_state.choice(self.label_to_indices[  # same task, diff subj : negative
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2) & (
                                                                 (self.labels_set - self.labels_set % 2) / 2 !=
                                                                 self.y_subj[i]), True, False)]
                                                 )
                                             ])
                         ]
                        for i in range(self.len)]

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.istrain:
            img1 = self.x_data[index,:,:,:]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == y1 % 2) & (
                                                                     (self.labels_set - self.labels_set % 2) / 2 ==
                                                                     (y1-y1%2)/2), True, False)]
                                                 )])


            # negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) , True,False)])
            negative_label = np.random.choice(self.labels_set[
                                                  np.where((self.labels_set%2==y1%2) &
                                                           ((self.labels_set-self.labels_set%2)/2 != (y1-y1%2)/2), True,False)]) #diffsub
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.x_data[positive_index,:,:,:]
            img3 = self.x_data[negative_index,:,:,:]
            y2 = self.y_data[positive_index]
            y3 = self.y_data[negative_index]

        else:
            img1 = self.x_data[self.test_triplets[index][0],:,:,:]
            img2 = self.x_data[self.test_triplets[index][1],:,:,:]
            img3 = self.x_data[self.test_triplets[index][2],:,:,:]
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
        return (img1, img2, img3), (y1%2,y2%2,y3%2)


    def __len__(self):
        return self.len

class TripletGiga5(Dataset): #transferacbility용
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, trial=None):
        self.transform = transform
        self.istrain = istrain
        if trial is None:
            if istrain:
                self.trial = 0,200
            else:
                self.trial = 100,200 #online
        else:
            self.trial = trial

        x_data = x.copy()
        y_data = y.copy()

        self.in_chans = x_data.shape[2]
        self.input_time_length = x_data.shape[3]

        x_data = x_data.reshape(108, -1, 1, self.in_chans, self.input_time_length)
        y_data = y_data.reshape(108,-1)

        if valtype == 'sess':
            if istrain:
                x_data = x_data[np.s_[0:54], self.trial[0]:self.trial[1], :, :, :]
                y_data = y_data[np.s_[0:54], self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[np.s_[0 + 54:54 + 54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[0 + 54:54 + 54], self.trial[0]:self.trial[1]]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data, np.s_[subj, subj + 54], 0)  # leave one subj
                y_data = np.delete(y_data, np.s_[subj, subj + 54], 0)
            else:
                x_data = x_data[np.s_[subj + 54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj + 54], self.trial[0]:self.trial[1]]
        elif valtype == 'subj':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj, self.trial[0]:self.trial[1], :, :, :]
                y_data = y_data[subj, self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[subj, self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[subj, self.trial[0]:self.trial[1]]
        else:
            raise AssertionError()

        self.x_data = x_data.reshape(-1, 1, self.in_chans, self.input_time_length)
        self.y_data = y_data.reshape(-1)
        self.y_subj = (y_data.reshape(-1)-self.y_data%2)/2
        self.len = self.y_data.shape[0]

        self.labels_set = np.unique(self.y_data)
        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)

        self.trsb = Transferability()
        if not istrain:
            self.labels_set = np.unique(self.y_data)
            self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                     for label in self.labels_set}


            #앵커와  같은테스크  두개를  뽑는다..
            triplets = [[i,
                         random_state.choice(self.label_to_indices[  # same task, diff subj : positive
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2) & (
                                                                     (self.labels_set - self.labels_set % 2) / 2 ==
                                                                     self.y_subj[i]), True, False)]
                                                 )
                                             ]),
                         random_state.choice(self.label_to_indices[  # same task, diff subj : positive
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2) & (
                                                                 (self.labels_set - self.labels_set % 2) / 2 !=
                                                                 self.y_subj[i]), True, False)]
                                                 )
                                             ])
                         ]
                        for i in range(self.len)]

            self.test_triplets = triplets




    def __getitem__(self, index):
        if self.istrain:
            img1 = self.x_data[index,:,:,:]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == y1 % 2) & (
                                                                     (self.labels_set - self.labels_set % 2) / 2 !=
                                                                     (y1-y1%2)/2), True, False)]
                                                 )])


            # negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) , True,False)])
            negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2==y1%2) & ((self.labels_set-self.labels_set%2)/2 != (y1-y1%2)/2), True,False)]) #samstask diffsub
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.x_data[positive_index,:,:,:]
            img3 = self.x_data[negative_index,:,:,:]

            anchor = int(self.y_subj[index])
            yi = int(self.y_subj[positive_index])
            yj = int(self.y_subj[negative_index])

            y2 = self.trsb.get_label(anchor, yi)
            y3 = self.trsb.get_label(anchor, yj)
        else:
            img1 = self.x_data[self.test_triplets[index][0],:,:,:]
            img2 = self.x_data[self.test_triplets[index][1],:,:,:]
            img3 = self.x_data[self.test_triplets[index][2],:,:,:]
            y1 = self.y_data[self.test_triplets[index][0]]

            anchor = int(self.y_subj[self.test_triplets[index][0]])
            yi = int(self.y_subj[self.test_triplets[index][1]])
            yj = int(self.y_subj[self.test_triplets[index][2]])
            y2 = self.trsb.get_label(anchor, yi)
            y3 = self.trsb.get_label(anchor, yj)


        img1 = torch.from_numpy(img1).type(torch.FloatTensor)
        img2 = torch.from_numpy(img2).type(torch.FloatTensor)
        img3 = torch.from_numpy(img3).type(torch.FloatTensor)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), (y1%2,y2%2,y3%2)


    def __len__(self):
        return self.len

class TripletSpeech(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset, subset=False):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        # self.transform = self.mnist_dataset.transform

        if self.train:
            if subset:
                self.train_data, self.train_labels = self.mnist_dataset.dataset.__getitem__(self.mnist_dataset.indices)
            else:
                self.train_labels = self.mnist_dataset.y_data
                self.train_data = self.mnist_dataset.x_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            if subset:
                self.test_data, self.test_labels = self.mnist_dataset.dataset.__getitem__(self.mnist_dataset.indices)
            else:
                self.test_labels = self.mnist_dataset.y_data
                self.test_data = self.mnist_dataset.x_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, y1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[y1])
            negative_label = np.random.choice(list(self.labels_set - set([y1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]

            y2 = self.train_labels[positive_index]
            y3 = self.train_labels[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
            y1 = self.test_labels[self.test_triplets[index][0]]
            y2 = self.test_labels[self.test_triplets[index][1]]
            y3 = self.test_labels[self.test_triplets[index][2]]



        # img1 = torch.from_numpy(img1).type(torch.FloatTensor)
        # img2 = torch.from_numpy(img2).type(torch.FloatTensor)
        # img3 = torch.from_numpy(img3).type(torch.FloatTensor)

        # if self.transform is not None:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)
        #     img3 = self.transform(img3)

        return (img1, img2, img3), (y1,y2,y3)

    def __len__(self):
        return len(self.mnist_dataset)

class SpeechDataset(Dataset):
    def __init__(self,x,y, idx=None,train=True):
        self.train  = train

        x_data = x.copy()
        y_data = y.copy()
        if idx is None:
            pass
        else:
            x_data = x_data[idx,:,:,:]
            y_data = y_data[idx]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        y_data = torch.from_numpy(y_data)
        self.x_data = x_data.type(torch.FloatTensor)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index]

        #
        # if self.transform:
        #     x = self.transform(x)

        return x, y

    def __len__(self):
        return self.len

def load_smt(path='data', fs=100):
    with open(path+'/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    if fs==100:
        with open(path + '/smt100_fix.pkl', 'rb') as f:
            x_data = pickle.load(f)
        x_data = np.expand_dims(x_data, axis=1)
        x_data = x_data[:, :, :, 100:]
    elif fs==250:
        with open(path + '/smt250_fix.pkl', 'rb') as f:
            x_data = pickle.load(f)
        x_data = np.expand_dims(x_data, axis=1)
        x_data = x_data[:, :, :, 250:]
    return x_data,y_data

def load_speech(path='data', data_type='i'):
    #'i':imagined speech
    #'o':overt
    if data_type == 'i':
        with open(path + 'epoch_labels.pkl', 'rb') as f:
            y_data = pickle.load(f)
        with open(path + 'epoch_scale.pkl', 'rb') as f:
            x_data = pickle.load(f)
        with open(path + 'epoch_sizes.pkl', 'rb') as f:
            subj_size = pickle.load(f)
    else:
        with open(path + 'epoch_sess3_labels.pkl', 'rb') as f:
            y_data = pickle.load(f)
        with open(path + 'epoch_sess3_scale.pkl', 'rb') as f:
            x_data = pickle.load(f)
        with open(path + 'epoch_sess3_sizes.pkl', 'rb') as f:
            subj_size = pickle.load(f)

    x_data = np.expand_dims(x_data, axis=1)
    x_data = x_data
    return x_data, y_data, subj_size

import pandas as pd

class Transferability():
    def __init__(self):
        self.data = self.__load_transferability__()

    def __load_transferability__(self,path=None):
        if path is None:
            path = 'data/108transfer.csv'
        data = pd.read_csv(path)
        data = 1-data.values[:,1:]
        eps = np.finfo(float).eps
        data2 = data+eps
        return 2*data2

    def get_label(self,x,y):
        return self.data[x,y] #앞이  train행  뒤가  test열
    #x,y -> train data, 앵커가  p,n을  얼마나  잘  분류  하게  하는  데이터인지를  나타내며, p,n의  성능이  출력되니까  실제  데이터  자체의  퀄리티도  고려??


