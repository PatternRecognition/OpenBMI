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
from datasets import *


def extract_embeddings(dataloader, model, num_ftrs=2):
    with torch.no_grad():
        model.eval()
        # num_ftrs = model.embedding_net.fc.out_features
        embeddings = np.zeros((len(dataloader.dataset), num_ftrs))
        labels = np.zeros(len(dataloader.dataset))
        preds = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()

            embeddings[k:k+len(images)] = model[0](images).data.cpu().numpy()
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)

            labels[k:k+len(images)] = target.numpy()
            preds[k:k+len(images)] = pred.squeeze().cpu().numpy()

            k += len(images)
    return embeddings, labels, preds

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

    fold_idx = 5
    gamma = 1
    DAsetting = False
    args = parser.parse_args()
    args.seed = 0
    args.use_tensorboard = True



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



    x_data, y_data = load_smt()
    #get subject number
    y_subj = np.zeros([108, 200])
    for i in range(108):
        y_subj[i, :] = i * 2
    y_subj = y_data.reshape(108, 200) + y_subj
    y_subj = y_subj.reshape(21600)

    # For classification data
    valtype='subj'
    # #subj - 0-27 train
    # train_subj1 = np.r_[0:27]
    # train_subj2 = np.r_[0:27]+54
    #
    # test_subj = np.r_[27:54,54+27:108]


    # For Domain adaptation setting
    if DAsetting:
        train_subj1 = np.r_[27:54]
        train_subj2 = np.r_[27:54] + 54

        test_subj = np.r_[0:27, 54 + 0:54 + 27]

        trial_s = (0, 200)
        trial_t = (0, 200)
        trial_val = (0, 200)

        dataset_train1 = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True,subj=train_subj1,trial=trial_s)
        dataset_train2 = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True,subj=train_subj2,trial=trial_t)
        dataset_train = dataset_train1.__add__(dataset_train2)
        dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False,subj=test_subj, trial=trial_val)

        triplet_dataset_train1 = TripletGiga(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj1,trial = trial_s )
        triplet_dataset_train2 = TripletGiga(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj2, trial=trial_t)
        triplet_dataset_train = triplet_dataset_train1.__add__(triplet_dataset_train2)
        triplet_dataset_test = TripletGiga(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj,trial = trial_val)
    else: #DG setting
        test_subj = np.r_[fold_idx*9:fold_idx*9+9,fold_idx+54:fold_idx+9+54]
        train_subj = np.setxor1d(np.r_[0:108],test_subj)

        trial_train = (0, 200)
        trial_val = (0, 200)


        dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj, trial=trial_train)
        dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj, trial=trial_val)

        triplet_dataset_train = TripletGiga(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj,
                                             trial=trial_train)
        triplet_dataset_test = TripletGiga(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj,
                                           trial=trial_val)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=args.batch_size, shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_test, batch_size=args.batch_size, shuffle=False)

    ###################################################################################################################
    # make model for metric learning
    from networks import Deep4Net, EmbeddingDeep4CNN,EmbeddingDeep4CNN_bn, TripletNet, FineShallowCNN, EmbeddingDeepCNN, QuintupletNet, EmbeddingShallowCNN
    from losses import TripletLoss_dev2

    margin = 1.0


    embedding_net = Deep4Net()
    print(embedding_net)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss_dev2(margin,gamma).cuda()

    n_epochs =1
    log_interval = 10




    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.5, last_epoch=-1)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=1, last_epoch=-1)



    #model for validation
    evalmodel = nn.Sequential(model.embedding_net,
                              model.fc,
                              nn.LogSoftmax(dim=1)).to(device)

    print('____________DANet____________')
    print(model)




    comment = 'fold_'+str(fold_idx)+'_g_' + str(gamma)
    #save someting
    if (args.save_model):
        model_save_path = 'model/Deep4Net_(test0604)/'+comment+'/'
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
    if loging:
        fname = model_save_path + datetime.today().strftime("%m_%d_%H_%M") + ".txt"
        f = open(fname, 'w')

    if args.use_tensorboard:
        writer = SummaryWriter(comment=comment)

    # load_model_path = model_save_path+'danet_0.7_49.pt' #'C:\\Users\dk\PycharmProjects\giga_cnn\구모델\\clf_83_8.pt'#'clf_29.pt' #'triplet_mg26.pt'#'clf_triplet2_5.pt' #'triplet_31.pt'
    load_model_path = None
    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))



    for temp in range(1, 50):
        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
            log_interval)
        print(temp)
        train_loss, train_score = eval(args, evalmodel, device, train_loader)
        eval_loss, eval_score = eval(args, evalmodel, device, test_loader)
        if args.use_tensorboard:
            writer.add_scalar('Train/Loss', np.mean(train_loss)/100, temp)
            writer.add_scalar('Train/Acc', np.mean(train_score)/100, temp)
            writer.add_scalar('Eval/Loss', np.mean(eval_loss)/100, temp)
            writer.add_scalar('Eval/Acc', np.mean(eval_score)/100, temp)
            writer.close()

        torch.save(model.state_dict(), model_save_path + 'danet_'+str(gamma)+'_'+ str(temp) + '.pt')



if __name__ == '__main__':
    print('hello')
    main()


