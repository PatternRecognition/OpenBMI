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

def train(args, model, device, train_loader, optimizer,scheduler, epoch=1):
    scheduler.step()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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

    fold_idx = 0
    gamma = 0.7
    margin = 1.0

    DAsetting = False
    args = parser.parse_args()
    args.seed = 0
    args.use_tensorboard = True
    args.save_model = True
    n_epochs = 200
    startepoch = 0

    folder_name = 'exp2'
    comment = '22ch_deep4' + str(fold_idx) + '_g_' + str(gamma) + '_m_' + str(margin)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if use_cuda else "cpu")
    gpuidx = 0
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    from datetime import datetime
    import os
    loging = False



    x_data, y_data = load_bcic(fs=250)
    y_subj = np.zeros([9, 576])
    for i in range(9):
        y_subj[i, :] = i * 2
    y_subj = y_data.reshape(9, 576) + y_subj
    y_subj = y_subj.reshape(9*576)


    valtype='subj'
    # if x_data.shape[2] != 60:



    test_subj = np.r_[2]
    # train_subj = np.setdiff1d(bci_excellent, test_subj)
    # bci_excellent.sort()

    print('test subj:'+ str(test_subj))
    train_subj = np.setdiff1d(np.r_[0:9],test_subj)

    trial_train = (0, 576)
    trial_val = (0, 576)



    dataset_train = BCICDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj, trial=trial_train)
    dataset_test = BCICDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj, trial=trial_val)




    triplet_dataset_train = TripletBCIC(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj,
                                         trial=trial_train)
    triplet_dataset_test = TripletBCIC(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj,
                                       trial=trial_val)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=args.batch_size, shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_test, batch_size=args.batch_size, shuffle=False)

    ###################################################################################################################
    # make model for metric learning
    # from networks import DWConvNet, basenet,Deep4Net_origin, Deep4Net, Deep4NetWs, EmbeddingDeep4CNN,EmbeddingDeep4CNN_bn, TripletNet, FineShallowCNN, EmbeddingDeepCNN, QuintupletNet, EmbeddingShallowCNN, TripletNet_conv_clf

    import get_common as gc
    from losses import TripletLoss_dev2, TripLoss, ContrastiveLoss_dk

    dgnet = gc.dgnet(gamma=gamma)
    model = dgnet.model
    if cuda:
        model.cuda(device)

    loss_fn = dgnet.loss_fn.cuda(device)

    log_interval = 10

    optimizer = dgnet.optimizer
    milestones = dgnet.milestones
    scheduler = dgnet.scheduler


    print('____________DANet____________')
    print(model)
    #
    # model_save_path = 'model/'+folder_name+'/'+comment+'/'
    # if (args.save_model):
    #     if not os.path.isdir(model_save_path):
    #         os.makedirs(model_save_path)
    #
    if args.use_tensorboard:
        writer = SummaryWriter(comment=comment)
        writer.add_text('optimizer', str(optimizer))
        writer.add_text('scheduler', str(milestones))
        writer.add_text('model_save_path', model_save_path)
        writer.add_text('model', str(model))
        writer.close()

    writer.log_dir
    load_model_path = 'C:\\Users\dk\PycharmProjects\csdg_exp2\model\exp3_22\danet_0.7_99.pt'
    # if startepoch > 0:
    #     load_model_path = model_save_path+'danet_'+str(gamma)+'_'+ str(startepoch) + '.pt'
    #     model_save_path = model_save_path +'(cont)'
    # else:
    #     load_model_path = None
    # if load_model_path is not None:
    # model.load_state_dict(torch.load(load_model_path,map_location='cuda:0'))
    #
    # for param in model.clf_net.parameters():
    #     param.requires_grad = False
    #
    #
    # model.clf_net.clf= nn.Sequential(nn.Linear(model.clf_net.embedding_net.num_hidden, 4),
    #                              nn.Dropout(),
    #                              nn.LogSoftmax(dim=1)).cuda()

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    # optimizer = optim.Adam(model.parameters(),lr=0.01)

    for epochidx in range(1,200):
        # fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, epochidx, n_epochs, cuda,gpuidx, log_interval)
        # print(epochidx)

        train(args, model.clf_net, device, train_loader, optimizer, scheduler)
        train_loss, train_score = eval(args, model.clf_net, device, train_loader)
        eval_loss, eval_score = eval(args, model.clf_net, device, test_loader)

        #
        # if args.use_tensorboard:
        #     writer.add_scalar('Train/Loss', np.mean(train_loss)/args.batch_size, epochidx)
        #     writer.add_scalar('Train/Acc', np.mean(train_score)/args.batch_size, epochidx)
        #     writer.add_scalar('Eval/Loss', np.mean(eval_loss)/args.batch_size, epochidx)
        #     writer.add_scalar('Eval/Acc', np.mean(eval_score)/args.batch_size, epochidx)
        #     writer.close()
        # if args.save_model:
        #     torch.save(model.state_dict(), model_save_path + 'danet_'+str(gamma)+'_'+ str(epochidx) + '.pt')



    #

if __name__ == '__main__':
    print('hello')
    main()

