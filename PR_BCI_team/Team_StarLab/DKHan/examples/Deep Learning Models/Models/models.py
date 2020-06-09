import torch.nn as nn
import torch

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


class FcClfNet(nn.Module):
    def __init__(self, embedding_net, l2norm=False):
        super(FcClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, embedding_net.n_classes),
                                 nn.Dropout())
        self.l2norm=l2norm
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.clf(output)
        # output = F.log_softmax(output, dim=1)

        output = output.view(output.size()[0], self.embedding_net.n_classes, -1)
        if output.size()[2]==1:
            output = output.squeeze()


        return output

