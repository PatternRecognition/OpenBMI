import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin, gamma):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, *args, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = F.relu(distance_positive - distance_negative +self.margin)

        triplet_loss =  losses.mean() if size_average else losses.sum()

        return triplet_loss, self.gamma

class TripletLoss_half(nn.Module):
    def __init__(self, margin, gamma):
        super(TripletLoss_half, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, *args, size_average=True):
        distance_positive = (anchor[0:500] - positive[0:500]).pow(2).sum(1)
        distance_negative = (anchor[0:500] - negative[0:500]).pow(2).sum(1)

        losses = F.relu(distance_positive - distance_negative +self.margin)

        triplet_loss =  losses.mean() if size_average else losses.sum()

        return triplet_loss, self.gamma

class ContrastiveLoss(nn.Module):
    def __init__(self, margin, gamma):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.eps = 1e-9

    def forward(self, anchor, positive, negative, *args, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = (distance_positive +
                        F.relu(self.margin - (distance_negative + self.eps).sqrt()).pow(2))

        loss =  losses.mean() if size_average else losses.sum()

        return loss, self.gamma

class TripLoss(nn.Module):
    def __init__(self, margin, gamma):
        super(TripLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, *args, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = 0.9 * distance_positive + 0.1 * F.relu(distance_positive - distance_negative + self.margin)

        triploss = losses.mean() if size_average else losses.sum()

        return triploss, self.gamma

class logratioLoss(nn.Module):
    def __init__(self, margin, gamma):
        super(logratioLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, *arg, size_average=True):
        distance_i = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_j = (anchor - negative).pow(2).sum(1)

        gt1 = arg[1].type(torch.cuda.FloatTensor)
        gt2 = arg[2].type(torch.cuda.FloatTensor)
        A = (torch.log(distance_i / distance_j))
        B = (torch.log(gt1 / gt2))
        # C = (torch.log(distance_i_j/gt3))
        loss_1 = (A - B).pow(2)
        # loss_2 = (B-C).pow(2).sum(0)
        loss=  loss_1.mean() if size_average else  loss_1.sum()
        return loss.sum(), self.gamma
