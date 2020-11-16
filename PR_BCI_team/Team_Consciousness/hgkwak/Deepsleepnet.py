import module.DataPreprocessing as DP
from module import Unet1D
import math
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torchsummary import summary
from matplotlib import pyplot as plt
import copy


seed = 2020
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
class DeepFeatureNet(nn.Module):
    def __init__(self, reuse_params=False):
        super(DeepFeatureNet, self).__init__()
        self.sampling_rate = 100
        self.input_size = 3000
        self.in_chan_size = 1
        self.n_classes = n_classes
        self.reuse_params = reuse_params
        self.input_dim = (24576 // batch_size)
        self.n_rnn_layers = 2
        self.hidden_size = 512
        self.relu=nn.ReLU()
        self.conv1d_s = nn.Sequential(nn.Conv1d(self.in_chan_size, 64, kernel_size=self.sampling_rate//2, stride=self.sampling_rate//16, padding=25, bias=False),
                                    nn.BatchNorm1d(64, momentum=0.001, eps=1e-5), nn.MaxPool1d(kernel_size=8, stride=8, padding=4), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4, bias=False),  nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=4, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=4, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.ReLU())
        self.conv1d_l = nn.Sequential(nn.Conv1d(self.in_chan_size, 64, kernel_size=self.sampling_rate*4, stride=self.sampling_rate//2, padding=200, bias=False),
                                    nn.BatchNorm1d(64, momentum=0.001, eps=1e-5), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1, padding=3, bias=False),  nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=3, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=3, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(512, self.hidden_size, self.n_rnn_layers)
        self.fc = nn.Linear(3456,2)

    def forward(self, data):
        self.batch_size = data.shape[0]
        ######## conv_layers 1 ########
        x1 = self.conv1d_s(data)
        data2 = data
        x2 = self.conv1d_l(data2)
        ######## conv_layers 2 ########
        x1 = x1.view(self.batch_size , -1)
        x2 = x2.view(self.batch_size , -1)
        ##### concat #####
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.dropout(x3)
        x3 = self.relu(x3)
        result = self.fc(x3)
        return result

##########################################################################
##########################################################################

class DeepSleepNet(nn.Module):
    def __init__(self, reuse_params=False):
        super(DeepSleepNet, self).__init__()
        self.sampling_rate = 100
        self.input_size = 3000
        self.in_chan_size = 1
        self.n_classes = n_classes
        self.reuse_params = reuse_params
        self.input_dim = (24576 // batch_size)
        self.n_rnn_layers = 2
        self.hidden_size = 512
        self.relu=nn.ReLU()
        self.fc = nn.Sequential(nn.Linear(self.input_dim, 512), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.seq_length = 10
        self.softmax = nn.Softmax()
        self.lstm = nn.LSTM(512, self.hidden_size, self.n_rnn_layers, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(3456,1024), nn.RelU())
        self.output = nn.Linear(512 ,self.n_classes)

    def forward(self, data):
        batch_size = data.shape[0]
        res = data
        res = self.fc(res)
        x = self.lstm(data)
        x = self.lstm(x)
        result = x + res
        result = self.dropout(result)
        result = self.output(result)
        return result

n_fold = 10
fold_idx = 0
n_classes = 2
interval_save_model = 100
interval_save_param = 1
interval_print_cm = 10
pretrain_epochs = 100
finetune_epochs = 100
batch_size = 8

resume = False
folder_name = "Track#2 Microsleep detection from single-channel EEG"
data_dir = './Dataset/'
output_dir = './Output/'

[dataset, labels] = DP.load_data(data_dir)
input_size = dataset[1][0].shape[0]
try:
    in_chan_size = dataset[1][0].shape[1]
except:
    in_chan_size = 1

conv_model = DeepFeatureNet().to(device=device)
summary(conv_model, (1,3000))
optimizer_conv = optim.Adam(conv_model.parameters(), lr=0.0001, weight_decay=1e-3)
#seq_model = DeepSleepNet()
#optimizer_seq = optim.Adam(seq_model.parameters(), lr=1e-4, weight_decay=0, betas=[1e-4,0.9])
#ummary(seq_model, (1,3000))

counter = 1
criterion = nn.CrossEntropyLoss().to(device=device)
kappa = 0
v_kappa = 0
while fold_idx < n_fold:
    print('##################### * Training for Cross Validation Fold{} started ... ####################'.format(fold_idx + 1))
    output_dir = os.path.join(output_dir, "fold{}".format(fold_idx))
    sample_list = dataset.keys()
    sample_list = sample_list - ['len']

    v_train_list = dataset[fold_idx]
    v_label_list = labels[fold_idx]
    valid_x = torch.FloatTensor(v_train_list).to(device=device)
    valid_y = torch.LongTensor(v_label_list.reshape(-1)).to(device=device)
    valid_data = TensorDataset(valid_x, valid_y).to(device=device)
    valid_data_ = DataLoader(dataset=valid_data, batch_size=1, shuffle=False).to(device=device)
    valid_idx = fold_idx
    sample_list = sample_list - {fold_idx}
    data_list = ''
    for sample in sample_list:
        data_name = 'Sample' + str(sample + 1) + ','
        data_list = data_list + data_name
    print('* Training dataset: ({}) // Validation dataset: Sample{}'.format(data_list, fold_idx + 1))
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []
    for epoch in range(pretrain_epochs):
        epo_loss = 0.0
        epo_acc = 0.0
        epo_vacc = 0.0
        epo_kappa = 0.0
        epo_v_kappa = 0.0
        loss_sum = 0.0
        train_list = []
        label_list = []
        for sample in sample_list:
            train_list += [dataset[sample]]
            label_list += [labels[sample]]
        train_list = np.concatenate(train_list)
        label_list = np.concatenate(label_list, 1)
        train_x = torch.FloatTensor(train_list).to(device=device)
        train_y = torch.LongTensor(label_list.reshape(-1)).to(device=device)
        train_data = TensorDataset(train_x, train_y).to(device=device)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False).to(device=device)

        all_pred_list = []
        all_v_pred_list = []
        for j, train in enumerate(train_data_):
            input, label = train
            input = input.view(-1,1,3000)
            optimizer_conv.zero_grad()
            pred_y = conv_model(input)
            loss = criterion(pred_y, label)
            loss.backward()
            optimizer_conv.step()
            loss_sum += loss
            label = label.numpy()
            pred_list = torch.max(pred_y,dim=1)
            pred_list = pred_list[1].detach().numpy()
            all_pred_list.append(pred_list)
        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]
        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)
        with torch.no_grad():
            for k, valid_ in enumerate(valid_data):
                v_input, v_label = valid_
                v_input = v_input.view(-1, 1, 3000)
                v_label = v_label.view(-1)
                v_pred_y = conv_model(v_input)
                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].detach().numpy()
                all_v_pred_list.append(v_pred_list)
            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)
            if math.isnan(v_kappa):
                v_kappa = 0.0
            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]
            print('* [[CV Fold{}, Epoch{}]] [TRAIN] loss:{:.3f}, acc:{:.3f}, kappa:{:.3f} // [Valid] acc:{:.3f}, kappa:{:.3f}'
                .format(fold_idx + 1, epoch + 1, loss_sum, acc, kappa, v_acc, v_kappa))
            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)
            print('Train CM\n', confusion_matrix(all_pred, train_y))
            print('Valid CM\n', confusion_matrix(all_v_pred, valid_y))
        epo_loss += loss_sum
        epo_acc += acc
        epo_kappa += kappa
        epo_vacc += v_acc
        epo_v_kappa += v_kappa
        if counter % interval_save_model == 0:
            print('Saved model weights')
        counter += 1
    x_scale = [i+1 for i in range(pretrain_epochs)]
    plt.xlim([0,pretrain_epochs])
    plt.plot(x_scale, loss_list, c='red')
    plt.plot(x_scale, t_acc_list, c='blue')
    plt.plot(x_scale, v_acc_list, c='green')
    plt.plot(x_scale, t_kappa_list)
    plt.plot(x_scale, v_kappa_list)
    plt.legend(['Train loss', 'Train acc', 'Valid acc', 'Train kappa', 'Valid kappa'])
    plt.show()
    print('* [[Mean Scores of CV Fold{}]] [TRAIN] loss:{:.3f}, acc:{:.3f} kappa:{:.3f} // [VALID] acc:{:.3f}, kappa:{:.3f}'
        .format(fold_idx+1, epo_loss/pretrain_epochs , epo_acc/pretrain_epochs, epo_kappa/pretrain_epochs, epo_vacc/pretrain_epochs, epo_v_kappa/pretrain_epochs))
    fold_idx += 1
print('#################### Training finished ####################')
