import torch.nn as nn
import torch
import resnet_1d

class square(nn.Module):
    def __init__(self):
        super(square, self).__init__()

    def forward(self, x):
        out = x*x
        return out

class log(nn.Module):
    def __init__(self):
        super(log, self).__init__()

    def forward(self, x):
        out = self._log(x)
        return out

    def _log(self, x, eps=1e-6):
        return torch.log(torch.clamp(x, min=eps))

class ShallowNet_dense(nn.Module):
    def __init__(self,n_classes,input_ch,input_time):
        super(ShallowNet_dense, self).__init__()
        self.num_filters = 40
        self.n_classes = n_classes

        self.convnet = nn.Sequential(nn.Conv2d(1, self.num_filters, (1,25), stride=1),
                                     nn.Conv2d(self.num_filters, self.num_filters, (input_ch, 1), stride=1,bias=False),
                                     nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     square(),
                                     nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 1), padding=0),
                                     log(),
                                     nn.Dropout(p=0.5),
                                     nn.Conv2d(self.num_filters, n_classes, kernel_size=(1, 30),  stride=(1, 1), dilation=(1, 15)),
                                     nn.LogSoftmax(dim=1)
                                     )

        self.convnet.eval()
        out = self.convnet(torch.zeros((1, 1, input_ch, input_time)))
        self.out_size = out.size()[3]

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], self.n_classes,self.out_size)

        return output

class EEGNet_v2(nn.Module):
    def __init__(self,n_classes, input_ch,input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(EEGNet_v2, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), stride=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16 , kernel_size=(1,6),groups=16),
            nn.Conv2d(16, 16, kernel_size=(1,1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25),
            )
        self.convnet.eval()

        out = self.convnet(torch.zeros(1,1,input_ch,input_time))
        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class EEGNet_latest(nn.Module):
    def __init__(self,num_classes, input_ch,input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(EEGNet_latest, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        freq = 100
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, freq//2), stride=1, bias=False, padding=(1 , freq//4)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16 , kernel_size=(1,freq//4),padding=(0,freq//8), groups=16),
            nn.Conv2d(16, 16, kernel_size=(1,1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25)
            )
        self.convnet.eval()

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.num_hidden = out.size()[1] * out.size()[2] * out.size()[3]

        from pytorch_model_summary import summary

        print(summary(self.convnet, torch.zeros((1, 1, input_ch, input_time)), show_input=False))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

class SampleEEGNet(nn.Module):
    def __init__(self,num_classes, input_ch,input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(SampleEEGNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        freq = 100

        ch1 = 128
        ch2 = 256
        ch3 = 512
        ch4 = 128
        ch5 = 256


        self.convnet = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=(1,3), stride=(1,2), padding=(0 ,1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Conv2d(ch1, ch1, kernel_size=(1,3), stride=(1,2), padding=(0 ,1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Conv2d(ch1, ch1, kernel_size=(1,3), stride=(1,2), padding=(0 ,1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Conv2d(ch1, ch1, kernel_size=(1,3), stride=(1,2), padding=(0 ,1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Conv2d(ch1, ch1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Conv2d(ch1, ch1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Conv2d(ch1, ch2, kernel_size=(input_ch, 1), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(),

            nn.Conv2d(ch2, ch3, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(ch3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(),
            # nn.Conv2d(1, ch1, kernel_size=(1, 3), stride=(1, 3), padding=(1, 1), groups=ch1),
            # nn.BatchNorm2d(ch1),
            # nn.ReLU(),
            #
            # nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            # nn.BatchNorm2d(16),
            # nn.ELU(),
            # # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            # nn.AvgPool2d(kernel_size=(1,4)),
            # nn.Dropout(p=0.25),
            # nn.Conv2d(16, 16 , kernel_size=(1,freq//4),padding=(0,freq//8), groups=16),
            # nn.Conv2d(16, 16, kernel_size=(1,1)),
            # nn.BatchNorm2d(16),
            # nn.ELU(),
            # nn.AvgPool2d(kernel_size=(1, 8)),
            # nn.Dropout(p=0.25)
            )
        self.convnet.eval()

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.num_hidden = out.size()[1] * out.size()[2] * out.size()[3]

        from pytorch_model_summary import summary
        print(summary(self.convnet, torch.zeros((1, 1, input_ch, input_time)), show_input=False))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output




class Band_Spatial_Module(nn.Module):
    def __init__(self, in_ch , out_ch, kernel_band, kernel_spatial):
        super(Band_Spatial_Module, self).__init__()

        self.time_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_band, stride=1, bias=False, padding=(0 , kernel_band[1]//2), groups=in_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch))
        self.spatial_conv =  nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_spatial, stride=1, bias=False, ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        tmp = self.time_conv(x)
        feature = self.spatial_conv(tmp)
        feature = F.avg_pool2d(feature, (1,tmp.size()[3]))
        return  tmp, feature

import time
class MSNN(nn.Module):
    def __init__(self,num_classes, input_ch,input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(MSNN, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        freq = 100

        k1, F1 = (1, 100), 16
        k2, F2 = (1, 60), 32
        k3, F3 = (1, 20), 64

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, freq//2), stride=1, bias=False, padding=(0 , freq//4)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4))
        self.bs1 = Band_Spatial_Module(in_ch=4,out_ch=F1,kernel_band=k1,kernel_spatial= (input_ch,1))
        self.bs2 = Band_Spatial_Module(in_ch=F1, out_ch=F2, kernel_band=k2, kernel_spatial=(input_ch, 1))
        self.bs3 = Band_Spatial_Module(in_ch=F2, out_ch=F3, kernel_band=k3, kernel_spatial=(input_ch, 1))


        self.clf = nn.Sequential(nn.Linear(F1+F2+F3, self.n_classes))

        self.criterion = torch.nn.CrossEntropyLoss()



    def forward(self, x):
        output = self.temporal_conv(x)

        output, feature_low  = self.bs1(output)
        output, feature_mid = self.bs2(output)
        output, feature_high = self.bs3(output)

        features  = torch.cat([feature_low,feature_mid,feature_high], dim=1).squeeze()

        output = features.view(x.size()[0],-1)
        output = self.clf(output)
        output = output.view(x.size()[0], self.n_classes, -1)

        if output.size()[2]==1:
            output = output.squeeze(dim=2)


        return output

    def get_embedding(self, x):
        output = self.temporal_conv(x)

        output, feature_low  = self.bs1(output)
        output, feature_mid = self.bs2(output)
        output, feature_high = self.bs3(output)

        features  = torch.cat([feature_low,feature_mid,feature_high], dim=1).squeeze()

        return features



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
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            )
        self.convnet.eval()

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        # out = torch.zeros(1, 1, input_ch, input_time)
        #
        # for i, module in enumerate(self.convnet):
        #     print(module)
        #     out = module(out)
        #     print(out.size())
        #

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]




    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


import batchinstancenorm as bin
class Deep4Net_BIN(nn.Module):
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
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                bin.BatchInstanceNorm2d(),
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
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
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

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        # out = torch.zeros(1, 1, input_ch, input_time)
        #
        # for i, module in enumerate(self.convnet):
        #     print(module)
        #     out = module(out)
        #     print(out.size())
        #

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]




    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class Deep4Net_mtl(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4Net_mtl, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 64
        n_ch2 = 128
        n_ch3 = 256
        self.n_ch4 = 512


        self.convnet = nn.Sequential(
            nn.Conv2d(1, n_ch1, kernel_size=(1, 7), stride=1, padding=(0,3)),
            nn.BatchNorm2d(n_ch1,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 7), stride=1, padding=(0,3)),
            nn.BatchNorm2d(n_ch1,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 7), stride=1, padding=(0,3)),
            nn.BatchNorm2d(n_ch1,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),

            nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
            nn.BatchNorm2d(n_ch1,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(n_ch2,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Dropout(p=0.5),
            nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
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
            nn.AvgPool2d(kernel_size=(1, 14), stride=(1, 1)),
            )

        self.convnet.eval()

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.num_hidden = out.size()[1] * out.size()[2] * out.size()[3]

        from pytorch_model_summary import summary

        print(summary(self.convnet, torch.zeros((1, 1, input_ch, input_time)), show_input=False))




    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class Deep4Net_dense(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4Net_dense, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 200
        n_ch2 = 200
        n_ch3 = 200
        self.n_ch4 = 200


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
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
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

        out = self.convnet(torch.zeros(1, 1, input_ch, 1125))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class Deep4Net_scale(nn.Module):
    def __init__(self, n_classes,input_ch,input_time, scale=1, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4Net_scale, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = int(25*scale)
        n_ch2 = int(50*scale)
        n_ch3 = int(100*scale)
        self.n_ch4 = int(200*scale)


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
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
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

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]




    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)
class Deep4Net_wide(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Deep4Net_wide, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

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

        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.num_hidden = out.size()[1]*out.size()[2]*out.size()[3]


    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FcClfNet(nn.Module):
    def __init__(self, embedding_net, l2norm=False):
        super(FcClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, embedding_net.n_classes),
                                 nn.Dropout(),
                                 )
        # self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, 256),
        #                          nn.ReLU(),
        #                          nn.Dropout(),
        #                          nn.Linear(256, embedding_net.n_classes),
        #                          )
        self.l2norm=l2norm
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.clf(output)
        # output = F.log_softmax(output, dim=1)

        output = output.view(x.size()[0], self.embedding_net.n_classes, -1)
        if output.size()[2]==1:
            output = output.squeeze(dim=2)


        return output

    def get_embedding(self, x):
        return self.embedding_net(x)


class FcClfNet2(nn.Module):
    def __init__(self, embedding_net, l2norm=False):
        super(FcClfNet2, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        # self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, embedding_net.n_classes),
        #                          nn.Dropout(),
        #                          )
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, 256),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(256, embedding_net.n_classes),
                                 )
        self.l2norm=l2norm
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.clf(output)
        # output = F.log_softmax(output, dim=1)

        output = output.view(x.size()[0], self.embedding_net.n_classes, -1)
        if output.size()[2]==1:
            output = output.squeeze(dim=2)


        return output

    def get_embedding(self, x):
        return self.embedding_net(x)



class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
import torch.nn.functional as F

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)




class Generator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Generator, self).__init__()
        n_class = 2
        n_ch = 62
        n_time = 1000
        self.dim_neck = dim_neck
        self.freq = 32

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(62 + 108 if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x, subject_id):

        x = x.squeeze(1).transpose(2, 1)
        c_org = torch.zeros(1,108)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))



        x = torch.cat((x, c_org), dim=1)

        x = torch.zeros(20,62+108, 1024)
        for conv in self.convolutions:
            x = F.relu(conv(x))

        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        self.freq = 128


        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)

        x = torch.zeros(20,1024,64+108)

        self.lstm1 = nn.LSTM(32 * 2 + 108, 512, 1, batch_first=True)


        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(512, 1024, 2, batch_first=True)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)



        input = torch.zeros(64,7,200) #input

        self.lstm = nn.LSTM(200, 32, 2, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(input)
        out_forward = outputs[:, :, :32]
        out_backward = outputs[:, :, 32:]
        self.freq = 32
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))


        self.linear_projection = LinearNorm(200, 80)
        decoder_output = self.linear_projection(outputs)


def get_model(args):
    if args.model_name == 'MSNN':
        model = MSNN(args.n_class, args.n_ch, args.n_time)
    elif args.model_name == 'Deep4net':
        embedding_net = Deep4Net_origin(args.n_class, args.n_ch, args.n_time)
        model = FcClfNet(embedding_net)
    elif args.model_name == 'Deep4net_mtl':
        embedding_net = Deep4Net_mtl(args.n_class, args.n_ch, args.n_time)
        model = FcClfNet(embedding_net)
    elif args.model_name == 'EEGNet':
        embedding_net = EEGNet_latest(args.n_class, args.n_ch, args.n_time)
        model = FcClfNet(embedding_net)
    elif args.model_name == 'SampleEEGNet':
        embedding_net = SampleEEGNet(args.n_class, args.n_ch, args.n_time)
        model = FcClfNet(embedding_net)
    elif args.model_name == 'Resnet':
        embedding_net = resnet_1d.Resnet(args.n_class, args.n_ch, args.n_time)
        model = FcClfNet2(embedding_net)

    else:
        print('없는모델!')
        raise


    from pytorch_model_summary import summary
    print(summary(model, torch.zeros(1, 1, args.n_ch, args.n_time), show_input=False))

    return model


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    parser.add_argument('--data-root',
                        default='C:/Users/Starlab/Documents/onedrive/OneDrive - 고려대학교/untitled/convert/')
    parser.add_argument('--save-root', default='../data')
    parser.add_argument('--result-dir', default='/deep4net_origin_result_ch20_raw')
    parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current best Model')

    args = parser.parse_args()

    model = Deep4Net_origin(2,20,250)
    from pytorch_model_summary import summary

    print(summary(model, torch.zeros((1, 1, 20, 250)), show_input=False))
