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
from .cbam import *

class Base_cnn(nn.Module):
    def __init__(self,use_attn = None):
        super(Base_cnn, self).__init__()
        self.num_filters = 112
        self.num_hidden = 512

        self.conv1 = nn.Conv2d(1, self.num_filters, (10,10), 1)
        self.fc1 = nn.Linear(self.num_filters * 1 * 1, self.num_hidden)
        #self.bn = nn.BatchNorm1d(self.num_hidden)
        #self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_fin = nn.Linear(self.num_hidden, 2)
        if not use_attn == None:
            self.cbam = CBAM(1,16)


    def forward(self, x):
        if not self.cbam == None:
            x = self.cbam(x)
        x = self.conv1(x[:, :, :, :])

        x = F.elu(x) #temporal
        #x = F.max_pool2d(x, (1, 10), 3)


        x = x.view(-1, self.num_filters * 1 * 1)
        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))

        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc_fin(x)
        x = F.log_softmax(x, dim=1)
        return x

class Base_cnn_dev(nn.Module):
    def __init__(self,use_attn = None):
        super(Base_cnn_dev, self).__init__()
        self.num_filters = 100
        self.num_hidden = 512

        self.conv1 = nn.Conv2d(1, self.num_filters, (1,10), 1) #temporal

        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, (62, 30), 1) #spatio-temporal
        self.fc1 = nn.Linear(self.num_filters * 1 * 1, self.num_hidden)
        #self.bn = nn.BatchNorm1d(self.num_hidden)
        #self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_fin = nn.Linear(self.num_hidden, 2)
        if not use_attn == None:
            self.cbam = CBAM(1,16)


    def forward(self, x):
        if not self.cbam == None:
            x = self.cbam(x)
        x = self.conv1(x[:, :, :, :])

        x = F.elu(x) #temporal
        #x = F.max_pool2d(x, (1, 10), 3)


        x = x.view(-1, self.num_filters * 1 * 1)
        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))

        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc_fin(x)
        x = F.log_softmax(x, dim=1)
        return x

class Base_cnn_mult(nn.Module):
    def __init__(self):
        super(Base_cnn_mult, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1024

        self.conv1 = nn.Conv2d(1, self.num_filters, (62,45), 1)
        self.fc1 = nn.Linear(self.num_filters * 1 * 83, self.num_hidden)
        self.bn = nn.BatchNorm1d(self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_lr = nn.Linear(self.num_hidden, 2)
        self.fc_subj = nn.Linear(self.num_hidden, 2)

    def forward(self, x):
        x = F.elu(self.conv1(x[:,:,:,:])) #temporal
        x = F.max_pool2d(x, (1, 10), 3)
        x = x.view(-1, self.num_filters * 1 * 83)
        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x2 = self.fc_subj(x)
        x1 = F.log_softmax(x1, dim=1)
        x2 = F.log_softmax(x2, dim=1)
        return x1,x2

class depthwise_separable_conv(nn.Module):
    def __init__(self):
        super(depthwise_separable_conv, self).__init__()
        self.num_filters = 100
        self.num_hidden = 1024
        self.depthwise1 = nn.Conv2d(1, 1, kernel_size=(62,45), padding=0, groups=1)
        torch.nn.init.xavier_uniform(self.depthwise1.weight)
        self.pointwise1 = nn.Conv2d(1, self.num_filters, kernel_size=1)
        torch.nn.init.xavier_uniform(self.pointwise1.weight)
        self.depthwise2 = nn.Conv2d(self.num_filters, self.num_filters, kernel_size=(1,10), padding=0, groups=self.num_filters)
        self.pointwise2 = nn.Conv2d(self.num_filters, self.num_filters, kernel_size=1)
        self.fc1 = nn.Linear(self.num_filters * 1 * 24, 2)
    def forward(self, x):
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = F.elu(x)

        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = F.elu(x)

        x = F.max_pool2d(x, (1, 10), 10)

        x = x.view(-1, self.num_filters * 1 * 24)
        # x = F.elu(self.bn(self.fc1(x)))
        x = self.fc1(x)

        x = F.dropout(x, training=self.training, p=0.5)
        # x = F.leaky_relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc_fin(x)
        x = F.log_softmax(x, dim=1)


        return x

class ResNet_EEG(nn.Module): #Resnet
    def __init__(self,block,layers, att_type=None, use_cbam = True):
        super(ResNet_EEG, self).__init__()
        self.num_filters = 40
        self.num_hidden = 960
        self.inplanes = 1
        self.layer1 = self._make_layer(block, 20, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 40, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 80, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 160, layers[3], stride=2, att_type=att_type)
        self.depthwise = nn.Conv2d(160, 160, kernel_size=(8, 8), padding=0,
                                    groups=160)
        self.pointwise = nn.Conv2d(160, 160, kernel_size=1)
        self.fc = nn.Linear(self.num_hidden, 2)
        #self.fc2 = nn.Linear(1024, 2)

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type == 'CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type == 'CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.max_pool2d(x,(1,5))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = F.dropout(x, training=self.training, p=0.5)
        #x = self.fc2(x)
        #x = F.dropout(x, training=self.training, p=0.5)
        x = F.log_softmax(x, dim=1)
        return x


class Base_dilated_cnn(nn.Module):
    def __init__(self):
        super(Base_dilated_cnn, self).__init__()
        self.num_filters = 128
        self.num_hidden = 1024

        self.conv1 = nn.Conv2d(1, 64, (62, 10), stride=1, dilation=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, (1,10), stride=10, dilation=(1, 2))
        self.fc1 = nn.Linear(self.num_filters * 1 * 7, self.num_hidden)
        self.bn = nn.BatchNorm1d(self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.fc_fin = nn.Linear(self.num_hidden, 2)
        self.cbam = CBAM(self.num_filters,16)


    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        #x = self.cbam(x)
        x = F.elu(x) #temporal
        x = F.max_pool2d(x, (1, 10), 1)


        x = x.view(-1, self.num_filters * 1 * 7)

        # x = F.elu(self.bn(self.fc1(x)))
        x = F.elu(self.fc1(x))

        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc_fin(x)
        x = F.log_softmax(x, dim=1)
        return x


class ShallowCNN(nn.Module): #shallowconv
    def __init__(self,use_cbam = False,ismult = False,use_bn = False):
        super(ShallowCNN, self).__init__()
        self.num_filters = 40
        self.num_hidden = 1000

        #self.SpatialGate = SpatialGate()
        # self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1)  # 템포럴
        # self.conv2 = nn.Conv2d(25, 25, kernel_size=(62, 1), stride=1)  # 채널

        self.conv1 = nn.Conv2d(1, 40, kernel_size= (1,25), stride=(1, 1)) #템포럴
        self.conv2 = nn.Conv2d(40,40, kernel_size = (62, 1), stride=(1, 1))  # 채널
        # self.cbam = CBAM(self.num_filters, 16)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(self.num_filters)
            self.bn2 = nn.BatchNorm2d(self.num_filters)
        else:
            self.bn1 = None
            self.bn2 = None

        if use_cbam:
            self.cbam1 = CBAM(self.num_filters,40)
            self.cbam2 = CBAM(self.num_filters, 40)
        else:
            self.cbam1 = None
            self.cbam2 = None


        #self.fc1 = nn.Linear(self.num_filters * 1 * 21, self.num_hidden)
        self.fc_lr = nn.Linear(self.num_filters * 1 * 21, 2)
        if ismult:
            self.fc_subj = nn.Linear(self.num_filters * 1 * 21, 2)
        else:
            self.fc_subj = None

    def forward(self, x):
        x = self.conv1(x)

        # x = self.SpatialGate(x)
        if not self.cbam1 ==None:
           x = self.cbam1(x)
        if not self.bn1 ==None:
            x = self.bn1(x)

        x = self.conv2(x)

        if not self.cbam2 ==None:
            x = self.cbam2(x)

        if not self.bn2 ==None:
            x = self.bn2(x)

        x = x*x

        x = F.avg_pool2d(x, kernel_size = (1, 75), stride = (1,15)) #1,149

        x = x.view(-1, self.num_filters * 1 * 21)

        x = torch.log(x)


        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x1 = F.dropout(x1, training=self.training, p=0.5)
        x1 = F.log_softmax(x1, dim=1)
        if not self.fc_subj == None:
            x2 = self.fc_subj(x)
            x2 = F.dropout(x2, training=self.training, p=0.5)
            x2 = F.log_softmax(x2, dim=1)
            return x1,x2
        else:
            return x1

class Deep4CNN(nn.Module): #shallowconv
    def __init__(self,use_cbam = False,ismult = False,use_bn = False):
        super(Deep4CNN, self).__init__()
        self.num_filters = 200
        self.num_hidden = 1000

        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1,10), stride=1) #템포럴
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(62, 1), stride=1)  # 채널

        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 10), stride=1)  # 채널

        self.conv4 = nn.Conv2d(50,100,kernel_size=(1,10),stride=1)

        self.conv5 = nn.Conv2d(100,200,kernel_size=(1,10),stride=1)

        #self.conv_classifier  = nn.Conv2d(200, 2, kernel_size=(9, 1), stride=(1, 1))

        if use_bn:
            self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.bn3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.bn4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        else:
            self.bn1 = None
            self.bn2 = None
            self.bn3 = None
            self.bn4 = None

        if use_cbam:
            self.cbam1 = CBAM(25,10)
            self.cbam2 = CBAM(50,10)
            self.cbam3 = None #CBAM(100,10)
            self.cbam4 = None # CBAM(200,10)

        else:
            self.cbam1 = None
            self.cbam2 = None
            self.cbam3 = None
            self.cbam4 = None


        self.fc1 = nn.Linear(self.num_filters * 1 * 10, self.num_hidden)
        self.fc_lr = nn.Linear(self.num_filters * 1 * 10, 2)
        if ismult:
            self.fc_subj = nn.Linear(self.num_filters * 1 * 14, 2)
        else:
            self.fc_subj = None

    def forward(self, x):
        #block1
        x = self.conv1(x)
        x = self.conv2(x)
        if not self.cbam1 ==None:
            x = self.cbam1(x)
        if not self.bn1 ==None:
            x = self.bn1(x)

        x = F.elu(x)
        x = F.max_pool2d(x, kernel_size = (1, 3), stride = (1, 2))


        #block2
        x = self.conv3(x)

        if not self.cbam2 ==None:
            x = self.cbam2(x)
        if not self.bn2 ==None:
            x = self.bn2(x)

        x = F.elu(x)
        x = F.max_pool2d(x, kernel_size=(1, 3), stride=(1, 2))


        #block3
        x = self.conv4(x)

        if not self.cbam3 == None:
            x = self.cbam3(x)
        if not self.bn3 == None:
            x = self.bn3(x)

        x = F.elu(x)
        x = F.max_pool2d(x, kernel_size=(1, 3), stride=(1, 2))

        #block4
        x = self.conv5(x)

        if not self.cbam4 ==None:
            x = self.cbam4(x)
        if not self.bn4 ==None:
            x = self.bn4(x)

        x = F.elu(x)
        x = F.max_pool2d(x, kernel_size=(1, 3), stride=(1, 3))


        x = x.view(-1, 200* 1 * 10)

        #x = torch.log(x)


        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x1 = self.fc_lr(x)
        x1 = F.dropout(x1, training=self.training, p=0.5)
        x1 = F.log_softmax(x1, dim=1)
        if not self.fc_subj == None:
            x2 = self.fc_subj(x)
            x2 = F.dropout(x2, training=self.training, p=0.5)
            x2 = F.log_softmax(x2, dim=1)
            return x1,x2
        else:
            return x1


class melCNN(nn.Module):
    def __init__(self):
        super(melCNN, self).__init__()

        self.conv1 = nn.Conv2d(62, 100, (6, 6), stride=1)  # 템포럴
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 100, (6, 6), stride=1)  # 템포럴
        self.bn2 = nn.BatchNorm2d(100)
        self.conv3 = nn.Conv2d(10, 20, (3, 3), stride=1)  # 템포럴

        self.fc1 = nn.Linear(1600, 2)
    def forward(self, x):
        x = x.squeeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        #
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=2)



        x = x.view(-1,1600)

        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)

        return x


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(24800, 1000)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(1000, 1000)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(1000, 2)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 24800)

        x = self.fc1(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.log_softmax(x, dim=1)

        return x