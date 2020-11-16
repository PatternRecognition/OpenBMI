from networks_new import Deep4Net_origin, ConvClfNet, TripletNet, FcClfNet
import networks_new as nets
from losses import TripletLoss_dev2, TripLoss, ContrastiveLoss_dk
from torch.optim import lr_scheduler
import torch.optim as optim


class dgnet():
    # def __init__(self,gamma):
    #     margin = 1.0
    #     self.gamma = gamma
    #     self.embedding_net = Deep4Net_origin(n_classes=2, input_ch=62, input_time=400)
    #     self.clf_net = ConvClfNet(self.embedding_net)
    #     self.model = TripletNet(self.clf_net)
    #     self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #     self.milestones = [30, 50, 70, 90]
    #     self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
    #     self.loss_fn = ContrastiveLoss_dk(margin,self.gamma)
    # def __init__(self,gamma):
    #     margin = 1.0
    #     self.gamma = gamma
    #     self.embedding_net = nets.Deep4Net()
    #     self.clf_net = FcClfNet(self.embedding_net)
    #     self.model = TripletNet(self.clf_net)
    #     self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    #     # self.optimizer = optim.Adam(self.model.parameters())
    #     self.milestones = [50, 100, 150, 200]
    #     self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
    #     self.loss_fn = TripletLoss_dev2(margin,self.gamma)
    # def __init__(self,gamma):
    #     #실험조건
    #     #brain decode랑 동일 모델 동일조건에서 스치듯 83프로 성능이 나오는지 확인코자함
    #     margin = 1.0
    #     self.gamma = gamma
    #     self.embedding_net = nets.Deep4Net_origin(n_classes=2, input_ch=62, input_time=400)
    #     self.clf_net = ConvClfNet(self.embedding_net)
    #     self.model = TripletNet(self.clf_net)
    #     self.optimizer = optim.Adam(self.model.parameters())
    #     self.milestones = [50, 100, 150, 200]
    #     self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
    #     self.loss_fn = TripletLoss_dev2(margin,self.gamma)
    # def __init__(self,gamma):
    #     #실험 내용: deep4net + fc clf 성능 위의 fix 100hz로 검증
    #     margin = 1.0
    #     self.gamma = gamma
    #     self.embedding_net = nets.Deep4Net()
    #     self.clf_net = FcClfNet(self.embedding_net)
    #     self.model = TripletNet(self.clf_net)
    #     # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    #     self.optimizer = optim.Adam(self.model.parameters())
    #     self.milestones = [50, 100, 150, 200]
    #     self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
    #     self.loss_fn = TripletLoss_dev2(margin,self.gamma)
    # def __init__(self,gamma):
    #     #실험 내용: deep4net + fc clf 성능 위의 fix 250hz로 검증 배치놈 안쓰고(좀더 유의한 특징뽑기 위해?>)
    #     margin = 1.0
    #     self.gamma = gamma
    #     self.embedding_net = nets.Deep4Net(n_ch=62,n_time=1000,batch_norm=False)
    #     self.clf_net = FcClfNet(self.embedding_net)
    #     self.model = TripletNet(self.clf_net)
    #     # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    #     self.optimizer = optim.Adam(self.model.parameters())
    #     self.milestones = [50, 100, 150, 200]
    #     self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
    #     self.loss_fn = TripletLoss_dev2(margin,self.gamma)


    def __init__(self, gamma):
        # 실험 내용: deep4net + fc clf 성능 위의 fix 250hz로 검증 배치놈 안쓰고(좀더 유의한 특징뽑기 위해?>)
        from models.model_3dcnn import Base_3dcnn
        margin = 1.0
        self.gamma = gamma
        self.embedding_net = nets.Deep4Net_origin(2,22,1000,batch_norm=True)
        self.clf_net = nets.FcClfNet(self.embedding_net, n_class=4)
        self.model = nets.TripletNet(self.clf_net)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.optimizer = optim.Adam(self.model.parameters())
        self.milestones = [50]
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        self.loss_fn = TripletLoss_dev2(margin, self.gamma)