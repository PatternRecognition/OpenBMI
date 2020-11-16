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
    gamma = 1.0
    margin = 0.3

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



    x_data, y_data = load_smt(fs=250)

    #get subject number
    y_subj = np.zeros([108, 200])
    for i in range(108):
        y_subj[i, :] = i * 2
    y_subj = y_data.reshape(108, 200) + y_subj
    y_subj = y_subj.reshape(21600)
    #y_subj = np.concatenate([y_data,y_subj],axis=1)


    # plt.imshow(x_data[100,0,:,:])
    # For classification data
    valtype='subj'
    # if x_data.shape[2] != 60:
    #     x_data = x_data[:,:,2:,:]
    # plt.imshow(x_data[1000,0,:,:])
    # #subj - 0-27 train
    # train_subj1 = np.r_[0:27]
    # train_subj2 = np.r_[0:27]+54
    #
    # test_subj = np.r_[27:54,54+27:108]

    chidx = np.r_[4,7,32,8,9,33,34,12,35,13,36,14,37,38,18,39,19,40,41,24,42,43] #오연조건
    # chidx = np.r_[2:56, 60:62]
    x_data = x_data[:,:,chidx,:]

    # For Domain adaptation setting
    if DAsetting:
        # test_subj = np.r_[fold_idx * 9:fold_idx * 9 + 9, fold_idx * 9 + 54:fold_idx * 9 + 9 + 54]
        test_subj_id= 39
        test_subj = np.r_[test_subj_id:test_subj_id+1]
        train_subj1 = np.setxor1d(np.r_[0:108], test_subj)
        train_subj2 = test_subj

        n_targets= 60
        trial_s = (0, 200)
        trial_t = (0, n_targets)

        trial_val = (n_targets, 200)

        # dataset_train1 = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True,subj=train_subj1,trial=trial_s)
        dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True,subj=train_subj2,trial=trial_t)
        # dataset_train = dataset_train1.__add__(dataset_train2)
        dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False,subj=test_subj, trial=trial_val)

        triplet_dataset_train = TripletGigaDA(x=x_data, y=y_subj, valtype=valtype, istrain=True, subj_s=train_subj1,trial_s = trial_s,
                                              subj_t=train_subj2,trial_t = trial_t)

        # triplet_dataset_train2 = TripletGiga2(x=x_data, y=y_subj, valtype=valtype, istrain=True, subj=train_subj2, trial=trial_t)
        # triplet_dataset_train = triplet_dataset_train1.__add__(triplet_dataset_train2)

        triplet_dataset_test = TripletGigaDA(x=x_data, y=y_subj, valtype=valtype, istrain=True, subj_s=train_subj1,
                                              trial_s=trial_s,subj_t=test_subj, trial_t=trial_val)

    else: #DG setting
        # test_subj = np.r_[fold_idx*9:fold_idx*9+9,fold_idx*9+54:fold_idx*9+9+54]
        # train_subj = test_subj
        # trial_train = (0, 30)
        # trial_val = (30, 200)
        #
        # bci_excellent = np.r_[43, 20, 27, 1, 28, 32, 35, 44, 36, 2]
        # bci_excellent = np.concatenate([bci_excellenth, bci_excellent + 54])




        test_subj = np.r_[fold_idx * 9:fold_idx * 9 + 9, fold_idx * 9 + 54:fold_idx * 9 + 9 + 54]
        # train_subj = np.setdiff1d(bci_excellent, test_subj)
        # bci_excellent.sort()

        print('test subj:'+ str(test_subj))
        train_subj = np.setdiff1d(np.r_[0:108],test_subj)

        trial_train = (0, 200)
        trial_val = (0, 200)


        dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, subj=train_subj, trial=trial_train)
        dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, subj=test_subj, trial=trial_val)

        triplet_dataset_train = TripletGiga2(x=x_data, y=y_subj, valtype=valtype, istrain=True, subj=train_subj,
                                             trial=trial_train)
        # triplet_dataset_train2 = TripletGiga2(x=x_data[:,:,:,10:], y=y_subj, valtype=valtype, istrain=True, subj=train_subj,
        #                                      trial=trial_train)
        # triplet_dataset_train = triplet_dataset_train1.__add__(triplet_dataset_train2)

        triplet_dataset_test = TripletGiga2(x=x_data, y=y_subj, valtype=valtype, istrain=False, subj=test_subj,
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

    model_save_path = 'model/'+folder_name+'/'+comment+'/'
    if (args.save_model):
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

    if args.use_tensorboard:
        writer = SummaryWriter(comment=comment)
        writer.add_text('optimizer', str(optimizer))
        writer.add_text('scheduler', str(milestones))
        writer.add_text('model_save_path', model_save_path)
        writer.add_text('model', str(model))
        writer.close()

    if startepoch > 0:
        load_model_path = model_save_path+'danet_'+str(gamma)+'_'+ str(startepoch) + '.pt'
        model_save_path = model_save_path +'(cont)'
    else:
        load_model_path = None
    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))








    for epochidx in range(1,200):
        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, epochidx, n_epochs, cuda,gpuidx, log_interval)
        print(epochidx)
        train_loss, train_score = eval(args, model.clf_net, device, train_loader)
        eval_loss, eval_score = eval(args, model.clf_net, device, test_loader)


        if args.use_tensorboard:
            writer.add_scalar('Train/Loss', np.mean(train_loss)/args.batch_size, epochidx)
            writer.add_scalar('Train/Acc', np.mean(train_score)/args.batch_size, epochidx)
            writer.add_scalar('Eval/Loss', np.mean(eval_loss)/args.batch_size, epochidx)
            writer.add_scalar('Eval/Acc', np.mean(eval_score)/args.batch_size, epochidx)
            writer.close()
        if args.save_model:
            torch.save(model.state_dict(), model_save_path + 'danet_'+str(gamma)+'_'+ str(epochidx) + '.pt')



    # # for visualization
    #
    # dataset_train_subj = GigaDataset(x=x_data, y=y_subj, valtype=valtype, istrain=True, subj=train_subj, trial=trial_train)
    # train_loader_subj = torch.utils.data.DataLoader(dataset_train_subj, batch_size=args.batch_size, shuffle=False)
    #
    # dataset_test_subj = GigaDataset(x=x_data, y=y_subj, valtype=valtype, istrain=False, subj=test_subj, trial=trial_val)
    # test_loader_subj = torch.utils.data.DataLoader(dataset_test_subj, batch_size=args.batch_size, shuffle=False)
    #
    # train_embeddings_tl, train_labels_tl, train_preds = extract_embeddings(train_loader_subj, evalmodel,model.embedding_net.num_hidden)
    # val_embeddings_tl, val_labels_tl, val_preds = extract_embeddings(test_loader_subj, evalmodel,model.embedding_net.num_hidden)
    #
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2,perplexity=30)
    #
    #
    # train_tsne = tsne.fit_transform(train_embeddings_tl)
    # plot_features(train_tsne,train_labels_tl-train_labels_tl%2)
    # plot_features(train_tsne, np.where(train_labels_tl>100,1,0))
    # plot_features(train_tsne, train_labels_tl%2)
    #
    # val_tsne = tsne.fit_transform(val_embeddings_tl)
    # plot_features(val_tsne, np.where(val_labels_tl > 100, 1, 0))
    # plot_features(val_tsne,val_labels_tl-108)
    # plot_features(val_tsne, val_labels_tl)
    #
    # np.savez('feature_fold'+str(fold_idx)+'_'+str(gamma),val_tsne=val_tsne,val_labels_tl=val_labels_tl)
    #
    # features_fold4 = np.load('feature_fold4_0.7.npz')
    # val_tsne = features_fold4['val_tsne']
    # val_labels_tl = features_fold4['val_labels_tl']
    #
    # #피쳐 트레인 테스트 합쳐서, generalization이 잘된건지 뿌려보기 위함
    # embeddings = np.concatenate([train_embeddings_tl,val_embeddings_tl])
    # labels = np.concatenate([train_labels_tl,val_labels_tl])
    # preds = np.concatenate([train_preds,val_preds])
    # tsne_result = tsne.fit_transform(embeddings)
    #
    # labels_task = np.concatenate([train_labels_tl%2, val_labels_tl%2+2])
    #
    # plot_features(tsne_result, labels_task%2)
    #
    #
    # plot_features(tsne_result[:18000],labels_task[:18000]%2)
    # plot_features(tsne_result[18000:], labels_task[18000:]%2)
    #
    # tsne_val = tsne_result[18000:]
    # labels_val = labels_task[18000:]%2
    # preds_val = preds[18000:]
    #
    # plot_features(tsne_val[preds_val!=labels_val],labels_val[preds_val!=labels_val])
    #
    # np.savez('feature_fold' + str(fold_idx) + '_' + str(gamma), tsne=tsne_result, labels=labels)
    # features_fold4 = np.load('feature_fold4_1.0.npz')
    # tsne_result = features_fold4['tsne']
    # labels_task = features_fold4['labels']
    #
    # fig, ax = plt.subplots()
    # import seaborn as sns
    # sns.pairplot(df)
    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(x='tsne-2d-one',y='tsne-2d-two',hue='y', data=df)
    #
    # df = pd.DataFrame()
    # df['tsne-2d-one'] = tsne_result[:, 0]
    # df['tsne-2d-two'] = tsne_result[:, 1]
    # df['y'] = labels_task
    #
    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", np.unique(labels).shape[0]),
    #     data=df,
    #     legend="full",
    #     alpha=0.5
    # )
    #
    #
    # plot_features(tsne_result[0:dataset_train_subj.len], np.where(labels < 100, np.where(labels%2==0,0,1), np.where(labels%2==0,20,21))[0:2200]) #plot train set
    # plot_features(tsne_result, np.where(labels < 100, np.where(labels%2==0,0,1), np.where(labels%2==0,20,21))) #plot source and target
    # plot_features(tsne_result[2200:], np.where(labels < 100,1,labels-labels%2)[2200:]) #each target
    # plot_features(tsne_result, np.where(labels%2==preds,1,0)) #each target
    # plot_features(tsne_result[np.where(labels%2==preds,False,True)], labels[np.where(labels%2==preds,False,True)]%2) #each target
    # plot_features(tsne_result[np.where(labels%2==preds,True,False)], labels[np.where(labels%2==preds,True,False)]%2) #each target
    #
    # acc = np.mean(np.where(train_labels_tl % 2 == train_preds, 1, 0))
    # plot_features(val_tsne, val_labels_tl - 108)
    #

if __name__ == '__main__':
    print('hello')
    main()

