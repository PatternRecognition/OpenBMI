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


class Base_3dcnn(nn.Module):
    def __init__(self):
        super(Base_3dcnn, self).__init__()
        self.map = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1],
            [-1, -1, -1, -1, -1, 0, -1, 1, -1, -1, -1,-1,-1],
            [-1, -1, -1, -1, 56, 57, -1, 58, 59, -1, -1,-1,-1],
            [-1, -1, -1, 54, 2, 3, 4, 5, 6, 55, -1,-1,-1],
            [44, 45, -1, 7, 32, 8, -1, 9, 33, 10, -1,50,49],
            [-1, -1, 11, 34, 12, 35, 13, 36, 14, 37, 15,-1,-1],
            [16, 46, 47, 17, 38, 18, 39, 19, 40, 20, 51,52,21],
            [-1, -1, 48, 22, 23, 41, 24, 42, 25, 26, 53,-1,-1],
            [-1, -1, -1, -1, -1, 60, 43, 61, -1, -1, -1,-1, -1],
            [-1, -1, -1, -1, 27, 28, 29, 30, 31, -1, -1,-1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1]])

        self.num_filters = 128
        self.num_hidden = 512

        self.conv1 = nn.Conv3d(1,40,kernel_size=(45,13,13),padding=0 )
        #self.bn1 = nn.BatchNorm3d()
        #self.conv2 = nn.Conv3d(16,32,kernel_size=(45,5,5),padding=0 )
        #self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0)
        #self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0)
        #self.conv5 = nn.Conv3d(500, 1000, kernel_size=(3, 3, 3), padding=0)
        # self.conv6 = nn.Conv3d(80, self.num_filters, kernel_size=(3, 3, 3), padding=0)

        self.fc1 = nn.Linear(40 * 1 * 116, 2)
        #self.bn = nn.BatchNorm1d(self.num_hidden)


    def forward(self, x):
        x = self._data_1Dto2D(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool3d(x,kernel_size = (10,1,1), stride = (3,1,1))

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool3d(x,kernel_size = (10,1,1), stride = (3,1,1))
        #
        # x = F.adaptive_max_pool3d(x, output_size=(5, 5, 5))
        # x = self.conv3(x)
        # x = F.relu(x)c
        #
        # x = self.conv4(x)
        # x = F.relu(x)

        #x = self.conv6(x)
        #x = self.cbam(x)
        #x = F.elu(x) #temporal
        #x = F.max_pool2d(x, (1, 10), 3)

        x = x.view(-1, 40 * 1 * 116)
        # x = F.elu(self.bn(self.fc1(x)))
        x = self.fc1(x)

        x = F.dropout(x, training=self.training,p=0.5)
        #x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)

        return x

    def _data_1Dto2D(self, data, Y=13, X=13):
        data = data.permute([0,1,3,2])
        batch = data.size(0)
        ch = 1
        time = data.size(2)
        data_2D = torch.cuda.FloatTensor(batch,ch,time,Y,X).fill_(0)
        for y in range(13):
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