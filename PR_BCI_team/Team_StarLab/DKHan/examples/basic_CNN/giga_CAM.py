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
        self.num_filters = 40
        self.num_hidden = 100

        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,45), 1)
        self.fc1 = nn.Linear(11*self.num_filters * 1 * 49, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_fin = nn.Linear(self.num_hidden, 2)

    def forward(self, x):
        x = F.elu(self.conv1(x)) #temporal
        x = F.max_pool2d(x, (1, 10), 3)
        x = x.view(-1,self.num_filters * 1 * 49)
        #x = x.view(-1,11*1960)
        x = F.elu(self.fc1(x))

        x = torch.mean(x,1)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_fin(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1,1,62,200)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1, 62, 200)
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


def windows(data, size, step):
    start = 0
    while ((start+size) < data.shape[1]):
        yield int(start), int(start + size)
        start += step


def segment_signal_without_transition(data, window_size, step):
    #여긴 2차원으로 들어옴
    for (start, end) in windows(data, window_size, step):
        if(len(data[0,start:end]) == window_size):
            if start == 0:
                segments =data[:,start:end].unsqueeze(0).unsqueeze(0)
            else:
                segments = torch.cat([segments, data[:,start:end].unsqueeze(0).unsqueeze(0)],0)
    return segments


def segment_dataset(X, window_size, step):
	win_x = []
	for i in range(X.shape[0]): #각 트라이얼에 대해
		win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
	return win_x[0]


class GigaDataset_subject_wise(Dataset):
    """ Diabetes dataset."""
    #subject_wise 이며 test엔 online데이터만 사용
    # Initialize your data, download, etc.
    def __init__(self,transform=None,istrain = True,subj=0):
        self.transform = transform

        with open('data/epoch_data_scale2.pkl', 'rb') as f:
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
    def __init__(self,transform=None,istrain = True,subj=0):
        self.transform = transform

        with open('data/epoch_data_scale2.pkl', 'rb') as f:
            x_data = pickle.load(f)
        x_data = np.expand_dims(x_data,axis=1)

        with open('data/epoch_labels.pkl', 'rb') as f:
            y_data = pickle.load(f)

        if istrain:
            x_data = np.delete(x_data, np.s_[200 * subj:200 * subj + 200], 0)  # 1세션에서 지움
            y_data = np.delete(y_data, np.s_[200 * subj:200 * subj + 200], 0)

            x_data = np.delete(x_data, np.s_[10600+200*subj:10600+200*subj + 200], 0)  # 2세션에서 지움
            y_data = np.delete(y_data, np.s_[10600+200*subj:10600+200*subj + 200], 0)
        else:
            #2세션에 온라인만 테스트
            x_data = x_data[200*(subj+54)+100:200*(subj+54)+200,:,:,:]
            y_data = y_data[200*(subj+54)+100:200*(subj+54)+200]

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

class GigaDataset_session_wise(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self,transform=None,istrain = True, sess=1):
        self.transform = transform

        with open('data/epoch_data_scale2.pkl', 'rb') as f:
            x_data = pickle.load(f)
        x_data = np.expand_dims(x_data,axis=1)

        with open('data/epoch_labels.pkl', 'rb') as f:
            y_data = pickle.load(f)

        if istrain:
            x_data = x_data[10800*(sess-1):10800*(sess-1)+10800,:,:,:] #sess1 or 2학습용으로 선택
            y_data =  y_data[10800*(sess-1):10800*(sess-1)+10800]
            #x_data = np.delete(x_data, np.s_[200 * subj: 200 * subj + 200], 0)#LOO 적용
            #y_data = np.delete(y_data, np.s_[200 * subj: 200 * subj + 200], 0)
        else:
            x_data = x_data[10800*(sess-1):10800*(sess-1)+10800,:,:,:] #online만 사용
            y_data = y_data[10800*(sess-1):10800*(sess-1)+10800]

        self.len = y_data.shape[0]

        x_data = torch.from_numpy(x_data)
        self.x_data = x_data.type(torch.FloatTensor)

        y_data = torch.from_numpy(y_data)
        self.y_data = y_data.long()


    def __getitem__(self, index):
        x = self.x_data[index,:,:,:]
        y = self.y_data[index]


        x = segment_dataset(x,window_size=200,step=10)

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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1080, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
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

    # Creating data indices for training and validation splits:

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
    ])

    batch_size = 200

    #validation_split = .2
    #shuffle_dataset = True
    #random_seed = 42

    #dataset_size = len(dataset)
    #indices = list(range(dataset_size))
    #split = int(np.floor(validation_split * dataset_size))
    #if shuffle_dataset:
    #    np.random.seed(random_seed)
    #    np.random.shuffle(indices)

    #train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    #train_sampler = SubsetRandomSampler(train_indices)
    #valid_sampler = SubsetRandomSampler(val_indices)

    from datetime import datetime
    #f = open("log/base_cnn_subjwise(trainsess1_testsess2on)" + datetime.today().strftime("%m_%d_%H_%M") + ".txt", 'w')


    dataset = GigaDataset_session_wise(istrain = True, sess=1)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=True,num_workers=4)

    model = Net2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        dataset_test = GigaDataset_session_wise(istrain=False, sess=2)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
        loss, score = test(args, model, device, test_loader)


    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


    #dataset_test = GigaDataset_session_wise(istrain=False, sess=2)
    #test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #loss, score = test(args, model, device, test_loader)
    #f.write(str() + " " + str(loss) + " " + str(score) + '\n')
    #f.close()

if __name__ == '__main__':
    main()



