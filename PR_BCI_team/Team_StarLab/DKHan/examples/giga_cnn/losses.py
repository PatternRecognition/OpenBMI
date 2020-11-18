import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative,size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletLoss_dev(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss_dev, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, a_label, p_label, n_label, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)

        mask_diffclass = (a_label%2-n_label%2).pow(2).type(torch.float32)

        #같은피험자같은클래스(줄인다) , 다른클래스(늘린다), 다른피험자 같은 클래스(줄이다) 약하게?
        losses = F.relu(distance_positive - mask_diffclass*distance_negative + (1-mask_diffclass)*distance_negative*0.1 +self.margin)
        #print(losses)
        return losses.mean() if size_average else losses.sum()

class TripletLoss_dev2(nn.Module):
    """
    클래시피케이션 성능을 높이면서~~
    """

    def __init__(self, margin, gamma):
        super(TripletLoss_dev2, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, a_label, p_label, n_label, clf_loss, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)
        # clf_loss = self.fc(anchor)
        # clf_loss = F.log_softmax(clf_loss,dim=1)
        # target = (a_label%2).long()
        # clf_loss = F.nll_loss(clf_loss,target)

        #mask_diffclass = (a_label%2-n_label%2).pow(2).type(torch.float32)

        #같은피험자같은클래스(줄인다) , 다른클래스(늘린다), 다른피험자 같은 클래스(줄이다) 약하게?
        losses = F.relu(distance_positive - distance_negative +self.margin)
        #losses = 0.5*(distance_positive + F.relu(self.margin - (distance_negative + self.eps).sqrt()).pow(2))
        #print(losses)
        triplet_loss =  losses.mean() if size_average else losses.sum()

        # if self.gamma == 0.0:
        #     loss = (1 - self.gamma) * triplet_loss
        # else:
        #     loss = (1 - self.gamma) * triplet_loss + self.gamma * clf_loss
        # return loss

        return triplet_loss, self.gamma

class ContrastiveLoss_dk(nn.Module):
    def __init__(self, margin, gamma):
        super(ContrastiveLoss_dk, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.eps = 1e-9

    def forward(self, anchor, positive, negative, a_label, p_label, n_label, clf_loss, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = (distance_positive +
                        F.relu(self.margin - (distance_negative + self.eps).sqrt()).pow(2))

        loss =  losses.mean() if size_average else losses.sum()

        return loss, self.gamma



class TripLoss(nn.Module):
    """
    클래시피케이션 성능을 높이면서~~
    """

    def __init__(self, margin, gamma):
        super(TripLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, a_label, p_label, n_label, clf_loss, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)
        # clf_loss = self.fc(anchor)
        # clf_loss = F.log_softmax(clf_loss,dim=1)
        # target = (a_label%2).long()
        # clf_loss = F.nll_loss(clf_loss,target)

        mask_diffclass = (a_label%2-n_label%2).pow(2).type(torch.float32)

        #같은피험자같은클래스(줄인다) , 다른클래스(늘린다), 다른피험자 같은 클래스(줄이다) 약하게?
        losses = 0.1*distance_positive + 0.9*F.relu(distance_positive - distance_negative +self.margin)
        #losses = 0.5*(distance_positive + F.relu(self.margin - (distance_negative + self.eps).sqrt()).pow(2))
        #print(losses)
        triplet_loss =  losses.mean() if size_average else losses.sum()


        return triplet_loss, self.gamma

class TripletLoss_dev3(nn.Module):
    """
    클래시피케이션 성능을 높이면서~~
    """

    def __init__(self, margin):
        super(TripletLoss_dev3, self).__init__()
        self.margin = margin

    def forward(self, anchor, pp, pn, np, nn, clf_loss, size_average=True):
        dpp = self.dist(anchor,pp)
        dpn = self.dist(anchor,pn)
        dnp = self.dist(anchor,np)
        dnn = self.dist(anchor,nn)

        losses = F.relu(dpp-(dpn-0.5))+F.relu(dpp-(dnp-1))+F.relu(dpp-(dnn-1))



        triplet_loss =  losses.mean() if size_average else losses.sum()
        return 0.3*triplet_loss+0.7*clf_loss

    def dist(self,anchor, x):
        return (anchor - x).pow(2).sum(1)



        # return clf_loss

# class TripletLoss(nn.Module):
#     def __init__(self):
#         super(TripletLoss, self).__init__()
#
#     def forward(self, anchor, positive, negative, margin_pull, margin_push, size_average=True):
#         distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#         distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
#         loss_p1 = torch.max(torch.tensor([0, distance_positive - margin_pull], requires_grad=True).cuda())
#         loss_n1 = torch.max(torch.tensor([0, margin_push - distance_negative],requires_grad=True).cuda())
#         losses = loss_p1 + loss_n1
#         return losses.mean() if size_average else losses.sum()

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
