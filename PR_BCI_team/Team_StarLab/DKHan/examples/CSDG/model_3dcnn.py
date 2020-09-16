from .model_openbmi import *
from .model_resnet import *

def makeadjmatrix(map=None):
    if map == None:
        map = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 0, -1, 1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 56, 57, -1, 58, 59, -1, -1, -1, -1],
            [-1, -1, -1, 54, 2, 3, 4, 5, 6, 55, -1, -1, -1],
            [44, 45, -1, 7, 32, 8, -1, 9, 33, 10, -1, 50, 49],
            [-1, -1, 11, 34, 12, 35, 13, 36, 14, 37, 15, -1, -1],
            [16, 46, 47, 17, 38, 18, 39, 19, 40, 20, 51, 52, 21],
            [-1, -1, 48, 22, 23, 41, 24, 42, 25, 26, 53, -1, -1],
            [-1, -1, -1, -1, -1, 60, 43, 61, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 27, 28, 29, 30, 31, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    adj = np.zeros([62,62])
    for i in range(0,62):
        iidx = np.argwhere(map == i)
        for j in range(0,62):
            jidx = np.argwhere(map == j)
            adj[i,j] = np.linalg.norm(iidx-jidx)
    return adj


from util import np_to_var
from networks_new import Deep4Net, Deep4Net_origin

class Base_3dcnn(nn.Module):
    def __init__(self):
        super(Base_3dcnn, self).__init__()
        self.SpatialTimeNet = Deep4Net(n_ch=62,n_time=1000,batch_norm=True)
        self.SpatialTimeNet = Deep4Net_origin(2,62,1000)

        self.map = np.array([
            [-1, -1, -1, -1, -1, 0, -1, 1, -1, -1, -1,-1,-1],
            [-1, -1, -1, -1, 56, 57, -1, 58, 59, -1, -1,-1,-1],
            [-1, -1, -1, 54, 2, 3, 4, 5, 6, 55, -1,-1,-1],
            [44, 45, -1, 7, 32, 8, -1, 9, 33, 10, -1,50,49],
            [-1, -1, 11, 34, 12, 35, 13, 36, 14, 37, 15,-1,-1],
            [16, 46, 47, 17, 38, 18, 39, 19, 40, 20, 51,52,21],
            [-1, -1, 48, 22, 23, 41, 24, 42, 25, 26, 53,-1,-1],
            [-1, -1, -1, -1, -1, 60, 43, 61, -1, -1, -1,-1, -1],
            [-1, -1, -1, -1, 27, 28, 29, 30, 31, -1, -1,-1, -1]])


        self.num_filters = 128
        self.num_hidden = 512

        self.convnet = nn.Sequential(nn.Conv3d(1,16,kernel_size=(10,3,3),padding=0),
                                     nn.BatchNorm3d(16),
                                     nn.ELU(),
                                     nn.MaxPool3d(kernel_size=(3,1,1), stride=(3,1,1)),
                                     nn.Conv3d(16, 32, kernel_size=(10, 3, 3), padding=0),
                                     nn.BatchNorm3d(32),
                                     nn.ELU(),
                                     nn.MaxPool3d(kernel_size=(3,1,1), stride=(3,1,1)),
                                     nn.Conv3d(32, 64, kernel_size=(10, 3, 3), padding=0),
                                     nn.BatchNorm3d(64),
                                     nn.ELU(),
                                     nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1)),
                                     nn.Conv3d(64, 128, kernel_size=(5, 3, 3), padding=0),
                                     nn.BatchNorm3d(128),
                                     nn.ELU(),
                                     nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1)),
                                     nn.Conv3d(128,  256, kernel_size=(5, 1, 5), padding=0),
                                     nn.BatchNorm3d(256),
                                     nn.ELU(),
                                     nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
                                     )

        out1 = self.convnet(np_to_var(np.ones(
            (1, 1, 1000,9,13),
            dtype=np.float32)))

        out2 = self.SpatialTimeNet(np_to_var(np.ones(
            (1, 1, 62,1000),
            dtype=np.float32)))

        out1 = out1.view(out1.size()[0], -1)
        out2 = out2.view(out2.size()[0], -1)
        out = torch.cat((out1, out2), dim=1)
        #self.bn1 = nn.BatchNorm3d()
        #self.conv2 = nn.Conv3d(16,32,kernel_size=(45,5,5),padding=0 )
        #self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0)
        #self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0)
        #self.conv5 = nn.Conv3d(500, 1000, kernel_size=(3, 3, 3), padding=0)
        # self.conv6 = nn.Conv3d(80, self.num_filters, kernel_size=(3, 3, 3), padding=0)

        # out1 = self._data_1Dto2D(np_to_var(np.ones(
        #     (1, 1, 62, 1000),
        #     dtype=np.float32)).cuda())

        self.num_hidden = out.size()[1]


    def forward(self, x):
        local_embedding = self._data_1Dto2D(x)
        local_embedding = self.convnet(local_embedding)
        local_embedding = local_embedding.view(local_embedding.size()[0], -1)

        global_embedding = self.SpatialTimeNet(x)


        return global_embedding, local_embedding

    def _data_1Dto2D(self, data, Y=9, X=13):
        data = data.permute([0,1,3,2])
        batch = data.size(0)
        ch = 1
        time = data.size(2)
        data_2D = torch.cuda.FloatTensor(batch,ch,time,Y,X).fill_(0)
        for y in range(Y):
            idx = self.map[y,self.map[y,:]>-1].tolist()
            if len(idx)>0:
                data_2D[:, :, :, y, np.where(self.map[y,:]>-1)] = data[:, :, :, idx].unsqueeze(dim=3)

        #
        # import matplotlib.pyplot as plt
        # from matplotlib import animation, rc
        # fig, ax = plt.subplots()
        # def animate(i):
        #     return data_2D[1,0,i,:,:].cpu()
        #
        # for i in  range(0,300):
        #     plt.imshow(data_2D[1,0,i,:,:].cpu())
        #     plt.show()

        return data_2D