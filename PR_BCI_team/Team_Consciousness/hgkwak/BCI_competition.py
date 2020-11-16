import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as f
from torch import nn
from data_load import data_load
\
class Network(nn.Module):
    def __init__(self, make_hidden, batch_size, n_classes=2, output_channels=32, init_weights=False, normalize=True, **kwarg):
        super(Network, self).__init__()
        self.input_layer=nn.Sequential(
            nn.Conv1d(1, output_channels, kernel_size=3, padding=1, stride=1),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            )
        self.hidden, self.channels, self.structure=make_hidden
        self.output_layer = nn.Sequential(
            nn.Linear(self.channels, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(1024, n_classes)
            )

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        x = x.view(batch_size, -1)
        return x

class Auto(nn.Module):
    def __init__(self, init_weights=False, normalize=True, **kwarg):
        super(Auto, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(3000, 1024, bias=False),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3000)
            )

    def forward(self,x):
        x = self.layer(x)
        return x

def make_hidden(structure, normalize):
    input_channels=32
    layers=[]
    output_size = 3000
    padding=0
    fil_size=5
    stride=2
    output_size = int((output_size+2*padding-fil_size)/stride)+1
    for i in structure:
        if i == 'M':
            output_size = int((output_size+2*padding-fil_size)/stride)+1
            layers += [nn.MaxPool1d(kernel_size=5, stride=2)]
        else:
            layers += [nn.Conv1d(input_channels, i, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True)]
            if normalize:
                nn.BatchNorm1d(i, eps=1e-3)
            input_channels=i
    input_channels=input_channels*output_size
    return nn.Sequential(*layers), input_channels, structure

def net(structure,normalize):
    model = Network(make_hidden(structure, normalize),batch_size=1)
    print(model)
    return model

if torch.cuda.is_available():
    cuda = torch.device('cuda')
else:
    cuda= torch.device('cpu')

train_data, train_labels, val_data, val_labels = data_load()
time=3000
padding=0
fil_size=5
stride=2
pooling_n=4

structure=[64,'M',128,'M',256]
model=net(structure, normalize=True).to(cuda)
auto=Auto().to(cuda)
criterion=nn.CrossEntropyLoss().to(cuda)
criterion_auto=nn.KLDivLoss().to(cuda)

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer_auto = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)


#####################################################
########## Validation with traning dataset ##########
#####################################################
train_data=np.array(train_data)
train_labels=np.array(train_labels)
train_data_temp = train_data[0:40]
train_label_temp = train_labels[0:40]

validation_data_temp = train_data[40:]
validation_label_temp = train_labels[40:]

train_data[0][0].shape
train_labels[0]

auto, weight_auto = train.train_auto(auto, criterion_auto, optimizer_auto, train_data, train_labels, epoch=10)
model, weight = train.train(model, criterion, optimizer, train_data_temp, train_label_temp, epoch=1)
prediction = train.predict(model, validation_data_temp, validation_label_temp)
model_pre = net(structure, normalize=True).to(cuda)
model_pre.load_state_dict(weight)

#########################################################################
#########################################################################
#########################################################################
model, weight = train.train(model, criterion, optimizer, train_data, train_label, epoch=50)
prediction = train.predict(model, val_data, val_labels)

model_pre = net(structure, normalize=True).to(cuda)
model_pre.load_state_dict(weight)
