import torch.nn as nn
import torch.nn.functional as F
import torch
import layers as L
from models import cbam
from util import np_to_var
import numpy as np


class basenet(nn.Module):
    def __init__(self):
        super(basenet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1,100,(62, 40)),
                                     nn.ELU(),
                                     nn.AdaptiveAvgPool2d((1,10))
                                     )

        self.num_hidden = 1000

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingShallowCNN(nn.Module): #shallowconv
    def __init__(self):
        super(EmbeddingShallowCNN, self).__init__()
        self.num_filters = 50
        self.num_hidden = 1050
        self.convnet = nn.Sequential(nn.Conv2d(1, self.num_filters, (1,25), stride=1),
                                     nn.Conv2d(self.num_filters, self.num_filters, (62, 1), stride=1),
                                     nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def forward(self, x):
        x = self.convnet(x)
        x = x*x
        # x = F.adaptive_avg_pool2d(x, output_size = (1,20))
        x = F.avg_pool2d(x, kernel_size=(1, 75), stride=(1, 15))  # 1,149
        x = x.view(-1, self.num_filters * 1 * 21)
        x = torch.log(x)
        # x = self.l2normalize(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, clf_net):
        super(TripletNet, self).__init__()
        self.clf_net = clf_net
        self.embedding_net = self.clf_net.embedding_net
        self.num_hidden = self.clf_net.embedding_net.num_hidden


    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output1= self.l2normalize(output1)

        output2 = self.embedding_net(x2)
        output2 = self.l2normalize(output2)

        output3 = self.embedding_net(x3)
        output3 = self.l2normalize(output3)

        output4 = self.clf_net(x1)

        return output1, output2, output3, output4

    def get_embedding(self, x):
        return self.embedding_net(x)

    # def l2normalize(self,feature):
    #     epsilon = 1e-6
    #     norm = torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5).unsqueeze(1).expand_as(feature)
    #     return torch.div(feature, norm)
    #
    def l2normalize(self, feature):
        denominator= feature.pow(2).sum(1, keepdim=True).sqrt()
        return feature/denominator
class TripletNet_fusion(nn.Module):
    def __init__(self, clf_net):
        super(TripletNet_fusion, self).__init__()
        self.clf_net = clf_net
        self.embedding_net = self.clf_net.embedding_net
        self.num_hidden = self.clf_net.embedding_net.num_hidden


    def forward(self, x1, x2, x3):
        output1_g, output1_l = self.embedding_net(x1)
        output1_g= self.l2normalize(output1_g)

        output2_g, _ = self.embedding_net(x2)
        output2_g = self.l2normalize(output2_g)

        output3_g, _  = self.embedding_net(x3)
        output3_g = self.l2normalize(output3_g)



        output4 = self.clf_net(x1)

        return output1_g, output2_g, output3_g, output4

    def get_embedding(self, x):
        return self.embedding_net(x)

    # def l2normalize(self,feature):
    #     epsilon = 1e-6
    #     norm = torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5).unsqueeze(1).expand_as(feature)
    #     return torch.div(feature, norm)
    #
    def l2normalize(self, feature):
        denominator= feature.pow(2).sum(1, keepdim=True).sqrt()
        return feature/denominator

class ConvClfNet(nn.Module):
    def __init__(self, embedding_net):
        super(ConvClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Conv2d(embedding_net.n_ch4, embedding_net.n_classes, (1, embedding_net.final_conv_length), bias=True)

    def forward(self, x):
        output = self.embedding_net(x)
        output = output.view(output.size()[0],self.embedding_net.n_ch4,-1,self.embedding_net.final_conv_length)
        output = self.clf(output)
        output = output.view(output.size()[0], -1)
        output = F.log_softmax(output,dim=1)
        return output

class FcClfNet(nn.Module):
    def __init__(self, embedding_net, n_class=2):
        super(FcClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, n_class),
                                 nn.Dropout(),
                                 nn.LogSoftmax(dim=1))
    def forward(self, x):
        output = self.embedding_net(x)
        output = self.clf(output)
        return output

class FcClfNet_fusion(nn.Module):
    def __init__(self, embedding_net):
        super(FcClfNet_fusion, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, 2),
                                 nn.Dropout(),
                                 nn.LogSoftmax(dim=1))
    def forward(self, x):
        output1, output2 = self.embedding_net(x)
        output = torch.cat((output1,output2),dim=1)
        output = self.clf(output)
        return output

class TripletNet_deep4net(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_deep4net, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden


    def forward(self, x1, x2, x3):
        output = self.embedding_net(x1)
        output = F.log_softmax(output,dim=1)

        return output


class TripletNet_conv_clf(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_conv_clf, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        # self.fc = nn.Sequential(nn.Linear(self.num_hidden,500),
        #                         nn.ReLU( ,
        #                         nn.Dropout(),
        #                         nn.Linear(500,2),
        #
        # )
        self.clf = nn.Sequential(nn.Conv2d(embedding_net.n_ch4,embedding_net.n_classes,(1, embedding_net.final_conv_length), bias=True))
        # nn.init.xavier_normal_(self.fc.weight)
        # self.fc.bias.data.fill_(0.01)

    def forward(self, x1, x2, x3):

        anchor = self.embedding_net(x1)
        output4 = anchor.view(anchor.size()[0],self.embedding_net.n_ch4,-1,self.embedding_net.final_conv_length)
        output4 = self.clf(output4)

        output1= self.l2normalize(anchor)

        output2 = self.embedding_net(x2)
        output2 = self.l2normalize(output2)

        output3 = self.embedding_net(x3)
        output3 = self.l2normalize(output3)

        # output4 = output4.view(output4.size()[0], -1)
        output4 = output4.view(output4.size()[0], -1)
        output4 = F.log_softmax(output4,dim=1)

        return output1, output2, output3, output4

    def get_embedding(self, x):
        return self.embedding_net(x)
    def l2normalize(self, feature):
        denominator= feature.pow(2).sum(1, keepdim=True).sqrt()
        return feature/denominator

    def get_clfoutput(self,x):
        anchor = self.embedding_net(x)
        output4 = anchor.view(anchor.size()[0],self.embedding_net.n_ch4,-1,self.embedding_net.final_conv_length)
        output4 = self.clf(output4)
        output4 = output4.view(output4.size()[0], -1)
        output4 = F.log_softmax(output4,dim=1)
        return output4

class TripletClfNet(nn.Module):
    def __init__(self, embedding_net,classification_net,):
        super(TripletClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.fc = nn.Linear(1000, 2)

        self.classification_net = classification_net


    def forward(self, x1, x2, x3):
        anchor = self.embedding_net(x1)
        positive = self.embedding_net(x2)
        negative = self.embedding_net(x3)

        gather = self.fc(anchor)
        gather = F.log_softmax(gather,dim=1)

        clf = self.classification_net(x1)

        return anchor, positive, negative, gather, clf

    def get_embedding(self, x):
        return self.embedding_net(x)


class QuintupletNet(nn.Module):
    def __init__(self, embedding_net):
        super(QuintupletNet, self).__init__()
        self.embedding_net = embedding_net
        self.fc = nn.Linear(1000,2)

    def forward(self, x1, x2, x3, x4, x5):
        output1 = self.embedding_net(x1)

        output2 = self.embedding_net(x2)

        output3 = self.embedding_net(x3)

        output4 = self.embedding_net(x4)

        output5 = self.embedding_net(x5)

        output6 = self.fc(output1) #clf output
        output6 = F.log_softmax(output6, dim=1)

        return output1, output2, output3, output4, output5, output6

    def get_embedding(self, x):
        return self.embedding_net(x)


class Deep4Net(nn.Module):
    def __init__(self, n_ch, n_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4Net, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

        n_ch1 = 50
        n_ch2 = 75
        n_ch3 = 100
        n_ch4 = 100

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(62, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, n_ch4, kernel_size=(1, 10), stride=1),
                nn.BatchNorm2d(n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(62, 1), stride=1, bias=not self.batch_norm),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, n_ch4, kernel_size=(1, 10), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            )

        self.convnet.eval()


        out = self.convnet(np_to_var(np.ones(
            (1, 1, n_ch, n_time),
            dtype=np.float32)))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = self.final_conv_length*n_ch4
        # self.fc = nn.Sequential(nn.Linear(1000,1000),
        #                         nn.ReLU())
        # self.fc = nn.Linear(1000,1000)

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        # output = self.fc(output)
        # output = self.fc(output)
        # output = self.l2normalize(output)
        # x = F.elu(x)

        # x = self.l2normalize(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class Deep4NetWs(nn.Module):
    def __init__(self, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4NetWs, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

        self.convnet = nn.Sequential(
            L.Conv2d(1, 25, kernel_size=(1, 10), stride=1),

            L.Conv2d(25, 25, kernel_size=(62, 1), stride=1, bias=not self.batch_norm),
            L.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            L.Conv2d(25, 50, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
            L.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            L.Conv2d(50, 100, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
            L.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            L.Conv2d(100, 100, kernel_size=(1, 10), stride=1),
            L.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            )

        self.convnet.eval()


        out = self.convnet(np_to_var(np.ones(
            (1, 1, 62, 400),
            dtype=np.float32)))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = self.final_conv_length*100
        # self.fc = nn.Sequential(nn.Linear(1000,1000),
        #                         nn.ReLU())
        # self.fc = nn.Linear(1000,1000)

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        # output = self.fc(output)
        # output = self.l2normalize(output)
        # x = F.elu(x)

        # x = self.l2normalize(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class View(nn.Module):
	def __init__(self):
		super(View, self).__init__()
	def forward(self, x):
		return x.view(x.size()[0], -1)

class DWConvNet(nn.Module):
    def __init__(self, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(DWConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

        n_ch1 = 100
        n_ch2 = 100
        n_ch3 = 100

        self.convnet = nn.Sequential(
            # nn.Conv3d(1,50,(1, 1, 10),1),
            # nn.Conv3d(50, 50, (1, 1, 10), 1),

            nn.Conv2d(62, 62, kernel_size=(1, 10), stride=1, groups=62),
            cbam.ChannelGate(62),
            nn.Conv2d(62, n_ch1, 1, 1, 0, 1, 1, bias=False),
            # nn.Conv2d(n_ch1, n_ch1, kernel_size=(62, 1), stride=1, bias=not self.batch_norm),

            nn.BatchNorm2d(n_ch1,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
            nn.BatchNorm2d(n_ch2,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
            nn.BatchNorm2d(n_ch3,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(n_ch3,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            )

        self.convnet.eval()


        out = self.convnet(np_to_var(np.ones(
            (1, 62, 1, 400),
            dtype=np.float32)))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = self.final_conv_length*n_ch3
        # self.fc = nn.Sequential(nn.Linear(1000,1000),
        #                         nn.ReLU())
        # self.fc = nn.Linear(1000,1000)

    def forward(self, x):
        x.size()[0]
        x = x.view(x.size()[0],62,1,-1)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        # output = self.fc(output)
        # output = self.l2normalize(output)
        # x = F.elu(x)

        # x = self.l2normalize(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)



class DWConvNet2(nn.Module):
    def __init__(self, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(DWConvNet2, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

        n_ch1 = 100
        n_ch2 = 100
        n_ch3 = 100

        self.convnet = nn.Sequential(
            # nn.Conv3d(1,50,(1, 1, 10),1),
            # nn.Conv3d(50, 50, (1, 1, 10), 1),

            nn.Conv2d(62, 62, kernel_size=(1, 30), stride=1, groups=62),
            nn.Conv2d(62, n_ch1, 1, 1, 0, 1, 1, bias=False),
            # nn.ReLU(),


            # nn.Conv2d(n_ch1, n_ch1, kernel_size=(62, 1), stride=1, bias=not self.batch_norm),

            nn.BatchNorm2d(n_ch1,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
            nn.BatchNorm2d(n_ch2,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
            nn.BatchNorm2d(n_ch3,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(n_ch3,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            )

        self.convnet.eval()


        out = self.convnet(np_to_var(np.ones(
            (1, 62, 1, 400),
            dtype=np.float32)))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = self.final_conv_length*100
        # self.fc = nn.Sequential(nn.Linear(1000,1000),
        #                         nn.ReLU())
        # self.fc = nn.Linear(1000,1000)

    def forward(self, x):
        x.size()[0]
        x = x.view(x.size()[0],62,1,-1)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        # output = self.fc(output)
        # output = self.l2normalize(output)
        # x = F.elu(x)

        # x = self.l2normalize(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class Deep4Net_origin(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4Net_origin, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                # nn.ELU(),
                # nn.BatchNorm2d(n_ch1,
                #                momentum=self.batch_norm_alpha,
                #                affine=True,
                #                eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            )

        self.convnet.eval()


        out = self.convnet(np_to_var(np.ones(
            (1, 1, input_ch, input_time),
            dtype=np.float32)))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = self.final_conv_length*self.n_ch4
        # self.fc = nn.Sequential(nn.Linear(1000,1000),
        #                         nn.ReLU())
        # self.fc = nn.Linear(1000,1000)
        #
        # self.convnet.add_module(
        #     "conv_classifier",
        #     nn.Conv2d(
        #         self.n_ch4,
        #         self.n_classes,
        #         (1, self.final_conv_length),
        #         bias=True,
        #     ),
        # )


    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        # output = self.fc(output)
        # output = self.l2normalize(output)
        # x = F.elu(x)

        # x = self.l2normalize(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)



