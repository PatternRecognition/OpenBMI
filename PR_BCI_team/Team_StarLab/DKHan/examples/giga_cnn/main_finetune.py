from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pickle

from models.model_resnet import *
from models.model_openbmi import *
from models.model_3dcnn import *

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.view(-1,1,62,301)
        target = target.view(-1)
        #data = nn.functional.interpolate(data,size=[300,300])
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        train_loss+= loss.item()
    train_loss /= len(train_loader)
    return train_loss

def eval(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data = data.view(-1,1,62,data.shape[4])
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct

def train_mt(args, model, device, train_loader, optimizer, epoch, alpha = 1.0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.view(-1,1,62,301)
        #target = target.view(-1)
        optimizer.zero_grad()
        output1, output2= model(data)
        loss1 = F.nll_loss(output1, target[:,0])
        loss2 = F.nll_loss(output2, target[:, 1])
        loss = alpha*loss1+(1-alpha)*loss2
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss1.item(),loss2.item()))

def eval_mt(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model(data)

            test_loss += (F.nll_loss(output1, target[:,0], reduction='sum').item() \
                         + F.nll_loss(output2, target[:,1], reduction='sum').item())/2 # sum up batch loss

            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct1 += pred1.eq(target[:,0].view_as(pred1)).sum().item()
            correct2 += pred2.eq(target[:,1].view_as(pred2)).sum().item()
    test_loss /= len(test_loader.dataset)

    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy1: {}/{} ({:.0f}%), Accuracy2: {}/{} ({:.0f}%)'.format(
        test_loss,
        correct1,
        len(test_loader.dataset),
        100. * correct1 / len(test_loader.dataset),
        correct2,
        len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))

    return test_loss, correct1

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

class GigaDataset(Dataset):
    def __init__(self,x,y, valtype, transform=None,istrain = True, sess=1,subj=None, num_trial=100):
        self.transform = transform
        self.istrain = istrain
        x_data = x.copy()
        y_data = y.copy()

        x_data = x_data.reshape(108,-1,1,62,500)
        y_data = y_data.reshape(108,-1)

        if valtype == 'sess':
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
                x_data = x_data[np.s_[subj+54*(sess-1)], 100:200, :, :, :]  # tests sess2 online
                y_data = y_data[np.s_[subj+54*(sess-1)], 100:200]
        elif valtype == 'subj':
            #나중에 subj,sess,trial  idx다 받아서 하는식으로 만들자.
            if subj is None:
                raise AssertionError()
            if istrain:
                x_data = x_data[subj+54*(sess-1), 0:100, :, :, :]
                y_data = y_data[subj+54*(sess-1), 0:100]
            else:
                x_data = x_data[subj+54*(sess-1), 100:200, :, :, :]
                y_data = y_data[subj+54*(sess-1), 100:200]
        else:
            raise AssertionError()

        x_data = x_data.reshape(-1, 1, 62, 500)
        y_data = y_data.reshape(-1)
        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,100:500]
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

    with open(path+'/smt1_scale.pkl', 'rb') as f:
        x_data1 = pickle.load(f)
    with open(path+'/smt2_scale.pkl', 'rb') as f:
        x_data2 = pickle.load(f)
    x_data = np.concatenate([x_data1, x_data2])
    x_data = np.expand_dims(x_data, axis=1)
    return x_data,y_data

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from datetime import datetime
    import os
    loging = True

    ismultitask = False
    loso = False
    model_save_path = 'model/deep4cnn/'

    if (args.save_model):
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

    if loging:
        fname = model_save_path + datetime.today().strftime("%m_%d_%H_%M") + ".txt"
        f = open(fname, 'w')

    x_data, y_data = load_smt()

    if ismultitask:
        nonbciilli = np.s_[
            0, 1, 2, 4, 16, 17, 18, 20, 21, 27, 28, 29, 32, 35, 36, 38, 42, 43, 44, 54, 55, 56, 58, 59, 61, 71, 72, 73,
            74, 81, 82, 85, 86, 89, 90, 96, 97, 98, 99]
        y_subj = np.zeros([108,200])
        y_subj[nonbciilli,:]= 1
        y_subj = y_subj.reshape(21600)
        y_data = np.stack((y_data,y_subj),axis=1)

    valtype = 'sess'
    if valtype == 'loso':
        for subj in range(0,54):
            model = Deep4CNN(ismult=ismultitask).to(device)
            #model.load_state_dict(torch.load(model_save_path+ "J_" + str(subj) + 'basecnn.pt'))

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            optimizer_fine = optim.SGD(model.parameters(), lr=0.005, momentum=args.momentum)

            dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, sess=1, subj=subj)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)

            dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, sess=2, subj=subj)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,**kwargs)

            # dataset_fine = GigaDataset_LOSO(x=x_data, y=y_data, fine=True, istrain=True, sess=2, subj=subj)
            # fine_loader = torch.utils.data.DataLoader(dataset_fine, batch_size=args.batch_size, shuffle=True, **kwargs)

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                print("joint-train")
                #LOSO joint training
                j_loss, j_score = eval(args, model, device, test_loader)

                if epoch > 30:
                    if (args.save_model):
                        torch.save(model.state_dict(), model_save_path+ "model_" +str(subj) + "_" +str(epoch) + '.pt')


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
    elif valtype == 'sess':
        #model = ResNet_EEG(BasicBlock, [2, 2, 2, 2]).to(device)\
        from networks import EmbeddingDeep4CNN
        model = EmbeddingDeep4CNN()
        model.fc = nn.Linear(2000,2)
        model = nn.Sequential(
            model,
            nn.LogSoftmax(dim=1)
        )
        model.load_state_dict(torch.load(model_save_path+'model_65.pt'))
        model.to(device)

        args.lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        #optimizer = optim.Adam(model.parameters())

        for subj in range(0, 54):
            dataset_train = GigaDataset(x=x_data, y=y_data,valtype='subj', subj=subj, istrain=True, sess=2)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)

            dataset_test = GigaDataset(x=x_data, y=y_data,valtype='subj', subj=subj, istrain=False, sess=2)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)

            for epoch in range(1, 2):
                #train_loss = train(args, model, device, train_loader, optimizer, epoch)
                print("joint-train")
                eval_loss, eval_score = eval(args, model, device, test_loader)
            if loging:
                f = open(fname, 'a')
                f.write(str(subj) + " " + "Jl : " + str(eval_loss) + " " + str(eval_score) + '\n')
                f.close()

        for epoch in range(1, args.epochs + 1):
            train_loss = train(args, model, device, train_loader, optimizer, epoch)
            print("joint-train")
            # LOSO joint training
            eval_loss, eval_score = eval(args, model, device, test_loader)
            accuracy = eval_score/5400

            if (accuracy > 0.78) and (eval_loss-train_loss < 0.1):
                if (args.save_model):
                    torch.save(model.state_dict(), model_save_path + "model_" + str(epoch) + '.pt')


        for subj in range(54, 108): #sess2
            dataset_test = GigaDataset(x=x_data, y=y_data, valtype='subj', istrain=False, sess=2, subj=subj)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)

            j_loss, j_score = eval(args, model, device, test_loader)

            if loging:
                f = open(fname, 'a')
                f.write(str(subj) + " " + "jl : " + str(j_loss) + " " + str(j_score) + '\n')
                f.close()




if __name__ == '__main__':
    main()


