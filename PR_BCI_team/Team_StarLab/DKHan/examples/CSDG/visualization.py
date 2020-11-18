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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

cnames = {
'red':                  '#FF0000',
'blue':                 '#0000FF',
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
giga_classes = list(range(108))

def plot_features(x_tsne, labels):
    df = pd.DataFrame()
    df['tsne-2d-one'] = x_tsne[:, 0]
    df['tsne-2d-two'] = x_tsne[:, 1]
    df['y'] = labels

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", np.unique(labels).shape[0]),
        data=df,
        legend="full",
        alpha=0.5
    )

def plot_features3d(features, labels):
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=features[:, 0],
        ys=features[:, 1],
        zs=features[:, 2],
        c=labels,
        cmap='rainbow'
    )
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    plt.show()

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(108):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=list(cnames.values())[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(giga_classes)


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        # self.features = torch.tensor(output, requires_grad=True).cuda()
        self.features = output.data.cpu().numpy()
    def close(self):
        self.hook.remove()



def getfeatures(model,dataloader,device):
    with torch.no_grad():
        model.eval()
        data, target = next(iter(dataloader))
        data = data.to(device)
        output = model(data).data.cpu().numpy()
        label = target.numpy()
        return output, label


#메모리 너무 먹는다
def extract_features(dataloader, model, layer, num_ftrs):
    with torch.no_grad():
        model.eval()
        # embeddings = np.zeros((len(dataloader.dataset), num_ftrs))
        embeddings = []
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings.append(model[layer](images).data.cpu().numpy())
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

import torch.nn as nn
def main():
    import main

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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

    x_data, y_data = main.load_smt()
    y_subj = np.zeros([108, 200])
    for i in range(108):
        y_subj[i, :] = i * 2
    y_subj = y_data.reshape(108, 200) + y_subj
    y_subj = y_subj.reshape(21600)

    load_model_path = 'C:\\Users\dk\PycharmProjects\giga_cnn\구모델\\clf_83_8.pt'
    from networks import EmbeddingDeep4CNN
    embedding_net = EmbeddingDeep4CNN()
    model = nn.Sequential(embedding_net,
                             nn.Linear(1000, 2),
                             nn.Dropout(p=0.5),
                             nn.LogSoftmax(dim=1),
                             ).to(device)
    model.load_state_dict(torch.load(load_model_path))

    from main import GigaDataset
    valtype = 'sess'
    dataset_train = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=True, sess=1)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                               **kwargs)
    dataset_test = GigaDataset(x=x_data, y=y_data, valtype=valtype, istrain=False, sess=2, subj=-1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                              **kwargs)




    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(1, 10):
        train(args, model, device, train_loader, optimizer, epoch)
        j_loss, j_score = eval(args, model, device, test_loader)



    feature,label = getfeatures(embedding_net.convnet[0:3],test_loader,device)
    right = x_data[np.where(y_data==0)]
    left = x_data[np.where(y_data==1)]

    grand_avg_r = np.mean(right,axis=0).squeeze()
    grand_avg_l = np.mean(left,axis=0).squeeze()



    plt.imshow(grand_avg_r[0])

    plt.imshow(grand_avg_l[0])

    feature_r = feature[np.where(label == 0)]
    feature_l = feature[np.where(label == 1)]



    grand_avg_r = np.mean(feature_r, axis=0).squeeze()
    grand_avg_l = np.mean(feature_l, axis=0).squeeze()
    plt.figure()
    plt.imshow(feature_l[1,:,0,:])

    plt.plot(feature_l[1,1,0,:])\

    import math
    from scipy.fftpack import fft
    from scipy.signal import bode
    N = 391
    T = 1/100



    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)


    yfabs = []
    for trial_idx in range(50):
        yf = fft(feature_l[trial_idx, :, 0, :])
        yfabs.append(2.0 / N * np.abs(yf[:,:N // 2]))
        yf_avg = np.mean(yfabs,0)
    fig, ax = plt.subplots()
    for i in range(100): #numof features
        ax.plot(xf[4:], (yf_avg[i,4:]))
    plt.show()

    boderesult = bode

    plt.show()


    fig, axarr = plt.subplots(10)
    for idx in range(10):
        axarr[idx].imshow(grand_avg_l[idx, :, :])


    j_loss, j_score = eval(args, model, device, test_loader)


if __name__ == '__main__':
    main()

