import torch.nn as nn
import torch.nn.functional as F
import torch

from util import np_to_var
import numpy as np


class basenet(nn.Module):
    def __init__(self):
        super(basenet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1,1000,(62, 40)),
                                     nn.ELU(),
                                     nn.AdaptiveAvgPool2d((1,1))
                                     )

        self.num_hidden = 1000

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

    def get_embedding(self, x):
        return self.forward(x)

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
        output = F.log_softmax(output,dim=1)

        output = output.view(output.size()[0], -1)
        return output

class FcClfNet(nn.Module):
    def __init__(self, embedding_net, l2norm=False):
        super(FcClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, 2),
                                 nn.Dropout(),
                                 nn.LogSoftmax(dim=1))
        self.l2norm=l2norm
    def forward(self, x):
        output = self.embedding_net(x)
        if self.l2norm:
            output = self.l2normalize(output)
        output = self.clf(output)
        return output

    def l2normalize(self, feature):
        denominator= feature.pow(2).sum(1, keepdim=True).sqrt()
        return feature/denominator

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

from torch.nn import init

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


        if input_time==1000:
            if self.batch_norm:
                self.convnet = nn.Sequential(
                    nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
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
                    nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                )
            self.convnet.eval()
        else:
            if self.batch_norm:
                self.convnet = nn.Sequential(
                    nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                    nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
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

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)



class shallowNet_h(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,batch_norm=True,
                 batch_norm_alpha=0.1):
        super(shallowNet_h, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 100
        n_ch2 = 100
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 45), stride=(1,3)),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

                nn.Dropout(p=0.5),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
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






class EEGNet_v2(nn.Module):
    def __init__(self, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(EEGNet_v2, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

        n_ch1 = 100
        n_ch2 = 100
        n_ch3 = 100

        self.convnet = nn.Sequential(
            # nn.Conv3d(1,50,(1, 1, 10),1),
            # nn.Conv3d(50, 50, (1, 1, 10), 1),

            nn.Conv2d(1, 8, kernel_size=(1, 64), stride=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(62, 1), stride=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16, kernel_size=(1, 6), groups=16),
            nn.Conv2d(16, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25),
            )

        self.convnet.eval()


        out = self.convnet(np_to_var(np.ones(
            (1, 1, 62, 1000),
            dtype=np.float32)))
        n_out_time = out.cpu().data.numpy().shape[3]
        # self.final_conv_length = n_out_time

        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]
        # self.fc = nn.Sequential(nn.Linear(1000,1000),
        #                         nn.ReLU())
        # self.fc = nn.Linear(1000,1000)

    def forward(self, x):
        # x.size()[0]
        # x = x.view(x.size()[0],62,1,-1)
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



