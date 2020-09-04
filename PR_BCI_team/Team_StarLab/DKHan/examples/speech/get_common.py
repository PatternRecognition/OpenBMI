from networks import Deep4Net_origin, ConvClfNet, TripletNet, FcClfNet
import networks as nets
from losses import *
from torch.optim import lr_scheduler
import torch.optim as optim

class dgnet():
   def __init__(self,gamma,margin):
        self.exp_comment = "ripletloss"
        self.gamma = gamma
        self.embedding_net = nets.Deep4Net_origin(13,64,501,batch_norm=True)
        self.clf_net = ConvClfNet(self.embedding_net)
        self.model = TripletNet(self.clf_net)
        self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.milestones = [20, 40, 60, 80]
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        self.loss_fn = TripletLoss(margin,gamma)