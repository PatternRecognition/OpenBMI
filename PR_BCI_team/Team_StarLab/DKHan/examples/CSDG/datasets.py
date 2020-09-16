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

class TripletGiga(Dataset):
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

        x_data = x_data.reshape(108,-1,1,self.in_chans,self.input_time_length)
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

        self.x_data = x_data.reshape(-1, 1, self.in_chans,self.input_time_length)
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
            img1 = self.x_data[index,:,:,:]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[y1])
            negative_label = np.random.choice(list(self.labels_set - set([y1])))
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
        return (img1, img2, img3), (y1,y2,y3)


    def __len__(self):
        return self.len
class TripletBCIC(Dataset):
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

        x_data = x_data.reshape(9,-1,1,self.in_chans,self.input_time_length)
        y_data = y_data.reshape(9,-1)

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

        self.x_data = x_data.reshape(-1, 1, self.in_chans,self.input_time_length)
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
            img1 = self.x_data[index,:,:,:]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[y1])
            negative_label = np.random.choice(list(self.labels_set - set([y1])))
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
        return (img1, img2, img3), (y1,y2,y3)


    def __len__(self):
        return self.len


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

class TripletGigaDA(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj_s=None, trial_s=None, subj_t=None, trial_t=None):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()




        self.in_chans = x_data.shape[2]
        self.input_time_length = x_data.shape[3]

        x_data = x_data.reshape(108, -1, 1, self.in_chans, self.input_time_length)
        y_data = y_data.reshape(108, -1)

        # source
        self.trial_s = trial_s

        if valtype == 'subj':
            if subj_s is None:
                raise AssertionError()
            x_data_s = x_data[subj_s, self.trial_s[0]:self.trial_s[1], :, :, :]
            y_data_s = y_data[subj_s, self.trial_s[0]:self.trial_s[1]]
        else:
            raise AssertionError()

        self.x_data_s = x_data_s.reshape(-1, 1, self.in_chans, self.input_time_length)
        self.y_data_s = y_data_s.reshape(-1)

        # self.y_subj = (y_data.reshape(-1)-self.y_data%2)/2

        #target
        self.trial_t = trial_t
        if valtype == 'subj':
            if subj_t is None:
                raise AssertionError()
            x_data_t = x_data[subj_t, self.trial_t[0]:self.trial_t[1], :, :, :]
            y_data_t = y_data[subj_t, self.trial_t[0]:self.trial_t[1]]
        else:
            raise AssertionError()

        self.x_data_t = x_data_t.reshape(-1, 1, self.in_chans, self.input_time_length)
        self.y_data_t = y_data_t.reshape(-1)
        self.len = self.y_data_t.shape[0]


        self.labels_set = np.unique(self.y_data_s)
        self.label_to_indices = {label: np.where(self.y_data_s == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)


    def __getitem__(self, index):
        img1 = self.x_data_t[index,:,:,100:500]
        y1 = self.y_data_t[index]

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
        img2 = self.x_data_s[positive_index,:,:,100:500]
        img3 = self.x_data_s[negative_index,:,:,100:500]
        y2 = self.y_data_s[positive_index]
        y3 = self.y_data_s[negative_index]



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
class TripletGiga3(Dataset):
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

        # self.labels_set = np.unique(self.y_data)
        # self.label_to_indices = {label: np.where(self.y_data == label)[0]
        #                          for label in self.labels_set}

        self.input_labels_set = np.unique(y_data)


        bci_excellent = np.r_[43,20,27,1,28,32,35,44,36,2]
        bci_excellent = np.concatenate([bci_excellent,bci_excellent+54])
        bci_excellent = np.concatenate([bci_excellent*2,bci_excellent*2+1])
        self.labels_set = np.unique(bci_excellent)

        self.labels_set = np.intersect1d(self.labels_set,self.input_labels_set)





        self.label_to_indices = {label: np.where(self.y_data == label)[0]
                                 for label in self.labels_set}

        random_state = np.random.RandomState(0)





        if not istrain:
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
            img1 = self.x_data[index,:,:,100:500]
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
        return (img1, img2, img3), (y1%2,y2%2,y3%2)


    def __len__(self):
        return self.len
class TripletGigaOrigin(Dataset):
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
                         random_state.choice(self.label_to_indices[  # samp task, diff subj
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2), True, False)]
                                                 )
                                             ]),
                         random_state.choice(self.label_to_indices[#diff task, same subj
                                                 np.random.choice(
                                                     self.labels_set[np.where((self.labels_set%2!=self.y_data[i]%2), True,False)]
                                                 )
                                             ])
                         ]
                        for i in range(self.len)]

            self.test_triplets = triplets

    def __getitem__(self, index):
        np.random.seed(0)

        if self.istrain:
            img1 = self.x_data[index,:,:,100:500]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == y1 % 2), True, False)]
                                                 )])


            negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) , True,False)])
            # negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) & ((self.labels_set-self.labels_set%2)/2 != (y1-y1%2)/2), True,False)]) #diffsub
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
        return (img1, img2, img3), (y1%2,y2%2,y3%2)


    def __len__(self):
        return self.len
class SiamesGiga(Dataset):
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
                         random_state.choice(self.label_to_indices[  # samp task, diff subj
                                                 np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == self.y_data[i] % 2), True, False)]
                                                 )
                                             ]),
                         random_state.choice(self.label_to_indices[#diff task, same subj
                                                 np.random.choice(
                                                     self.labels_set[np.where((self.labels_set%2!=self.y_data[i]%2), True,False)]
                                                 )
                                             ])
                         ]
                        for i in range(self.len)]

            self.test_triplets = triplets

    def __getitem__(self, index):
        np.random.seed(0)

        if self.istrain:
            img1 = self.x_data[index,:,:,100:500]
            y1 = self.y_data[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[np.random.choice(
                                                     self.labels_set[np.where(
                                                         (self.labels_set % 2 == y1 % 2), True, False)]
                                                 )])


            negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) , True,False)])
            # negative_label = np.random.choice(self.labels_set[np.where((self.labels_set%2!=y1%2) & ((self.labels_set-self.labels_set%2)/2 != (y1-y1%2)/2), True,False)]) #diffsub
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
        return (img1, img2, img3), (y1%2,y2%2,y3%2)


    def __len__(self):
        return self.len


class GigaDataset(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, trial=None):
        #trial : (tuple) trial = min, max
        self.transform = transform

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


        x_data = x_data.reshape(108,-1,1,self.in_chans,self.input_time_length)
        y_data = y_data.reshape(108,-1)

        if valtype =='sess':
            if istrain:
                x_data = x_data[np.s_[0:54],self.trial[0]:self.trial[1],:,:,:]
                y_data = y_data[np.s_[0:54],self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[np.s_[0+54:54+54],self.trial[0]:self.trial[1],:,:,:] #tests sess2 online
                y_data = y_data[np.s_[0+54:54+54],self.trial[0]:self.trial[1]]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data,np.s_[subj,subj+54],0) #leave one subj
                y_data = np.delete(y_data,np.s_[subj,subj+54],0)
            else:
                x_data = x_data[np.s_[subj+54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54], self.trial[0]:self.trial[1]]
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

        x_data = x_data.reshape(-1, 1, self.in_chans,self.input_time_length)
        y_data = y_data.reshape(-1)
        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)

        y_data = torch.from_numpy(y_data)
        self.x_data = x_data.type(torch.FloatTensor)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
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
class BCICDataset(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, trial=None):
        #trial : (tuple) trial = min, max
        self.transform = transform

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


        x_data = x_data.reshape(9,-1,1,self.in_chans,self.input_time_length)
        y_data = y_data.reshape(9,-1)

        if valtype =='sess':
            if istrain:
                x_data = x_data[np.s_[0:54],self.trial[0]:self.trial[1],:,:,:]
                y_data = y_data[np.s_[0:54],self.trial[0]:self.trial[1]]
            else:
                x_data = x_data[np.s_[0+54:54+54],self.trial[0]:self.trial[1],:,:,:] #tests sess2 online
                y_data = y_data[np.s_[0+54:54+54],self.trial[0]:self.trial[1]]
        elif valtype == 'loso':
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = np.delete(x_data,np.s_[subj,subj+54],0) #leave one subj
                y_data = np.delete(y_data,np.s_[subj,subj+54],0)
            else:
                x_data = x_data[np.s_[subj+54], self.trial[0]:self.trial[1], :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54], self.trial[0]:self.trial[1]]
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

        x_data = x_data.reshape(-1, 1, self.in_chans,self.input_time_length)
        y_data = y_data.reshape(-1)
        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)

        y_data = torch.from_numpy(y_data)
        self.x_data = x_data.type(torch.FloatTensor)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
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

def load_smt(path='C:/Users/dk/PycharmProjects/data/giga', fs=100):
    with open(path+'/epoch_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)

    # #기존 6폴드 성능은 이걸로 낸거였음
    # with open(path+'/smt250_1_new_norm.pkl', 'rb') as f:
    #     x_data1 = pickle.load(f)
    # with open(path+'/smt250_2_new_norm.pkl', 'rb') as f:
    #     x_data2 = pickle.load(f)
    #
    #
    # #
    # #
    # with open(path+'/smt1_new_norm.pkl', 'rb') as f:
    #     x_data1 = pickle.load(f)
    # with open(path+'/smt2_new_norm.pkl', 'rb') as f:
    #     x_data2 = pickle.load(f)
    #
    # x_data = np.concatenate([x_data1, x_data2])
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



def load_bcic(path='C:/Users/dk/PycharmProjects/test_braindecode/braindecode-master_new/examples', fs=100):
    with open(path+'/bcic4class_labels.pkl', 'rb') as f:
        y_data = pickle.load(f)


    with open(path + '/bcic4class.pkl', 'rb') as f:
        x_data = pickle.load(f)
    x_data = np.expand_dims(x_data, axis=1)
    x_data = x_data[:, :, :, 125:]

    from sklearn.preprocessing import Normalizer
    transformer = Normalizer()

    raw_data_train = np.zeros_like(x_data)

    for i in range(x_data.shape[0]):
        raw_fit = transformer.transform(x_data[i,0, :, :])
        # raw_fit = sc.fit_transform(x_data[i, :, :])
        # raw_fit = maxabs_scale(x_data[i, :, :])
        raw_data_train[i,0, :, :] = raw_fit[:, :]
        print(i)

    return x_data,y_data