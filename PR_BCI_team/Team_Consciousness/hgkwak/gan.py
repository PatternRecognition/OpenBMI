import numpy as np
import matplotlib.pylab as plt
from data_load import data_load
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch

batch_size=1
data_shape=[1,3000]
class Generator(nn.Module):
    def __init__(self, structure):
        super(Generator,self).__init__()
        def make_layers(chan_in, chan_out, normalize=False):
            layers=[]
            layers += [nn.Linear(chan_in, chan_out)]
            if normalize:
                layers += [nn.BatchNorm1d(chan_out,eps=1e-5)]
            layers += [nn.LeakyReLU(0.1, inplace=True)]
            return layers

        self.layers = []
        for i in range(1, len(structure)):
            if i == 1:
                self.layers += make_layers(structure[i-1], structure[i], normalize=False)
            else:
                self.layers += make_layers(structure[i-1], structure[i])
        self.generator = nn.Sequential(
            nn.Linear(data_shape[0]*data_shape[1],structure[0]),
            nn.LeakyReLU(0.1, inplace=True),
            *self.layers,
            nn.Linear(structure[-1], int(np.prod(data_shape))))
    def forward(self,z):
        data = z
        data = self.generator(data)
        data = data.view(data_shape[0], data_shape[1])
        return data

class Discriminator(nn.Module):
    def __init__(self, structure):
        super(Discriminator, self).__init__()
        self.rev_structure = list(reversed(structure))
        def make_layers(chan_in, chan_out, normalize=True):
            layers=[]
            layers += [nn.Linear(chan_in, chan_out)]
            layers += [nn.LeakyReLU(0.1, inplace=True)]
            return layers

        self.layers = []
        for i in range(1, len(self.rev_structure)):
            if i == 1:
                self.layers += make_layers(self.rev_structure[i-1], self.rev_structure[i], normalize=False)
            else:
                self.layers += make_layers(self.rev_structure[i-1], self.rev_structure[i])

        self.discriminator=nn.Sequential(
            nn.Linear(int(np.prod(data_shape)),self.rev_structure[0]),
            nn.LeakyReLU(0.1, inplace=True),
            *self.layers,
            nn.Linear(self.rev_structure[-1], 1),
            nn.Sigmoid()
        )

    def forward(self,data):
        data_flat = data.view(data_shape[0], -1)
        validity = self.discriminator(data_flat)
        return validity

train_data, train_labels, val_data, val_labels = data_load()
if torch.cuda.is_available():
    cuda = torch.device('cuda')
else:
    cuda= torch.device('cpu')

n_epochs=50
structure=[64,128,256,512,1024,2048]
adversarial_loss = torch.nn.BCELoss()
generator = Generator(structure)
discriminator = Discriminator(structure)

generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()

Generated_data_true = []
Generated_data_false = []

Tensor = torch.cuda.FloatTensor if cuda==torch.device('cuda') else torch.FloatTensor
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
batches_done=0
for epoch in range(n_epochs):
    for j in range(len(train_data)):
        generated_data_true = []
        generated_data_false = []
        trials=train_data[j].shape[0]
        batch_train=torch.tensor(train_data[j]).clone().detach().reshape(trials,-1)
        batch_target=torch.tensor(train_labels[j]).clone().detach()
        batch_dataset=torch.utils.data.TensorDataset(batch_train, batch_target)
        batch_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batch_size, num_workers=2)
        for i, data in enumerate(batch_loader):
            batch_inputs, batch_labels = data
            data_shape = batch_inputs.shape
            # Adversarial ground truths
            valid = Variable(Tensor(1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)
            # Configure input
            real_data = Variable(batch_inputs.type(Tensor))
            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (data_shape[0]*data_shape[1]))))
            # Generate a batch of images
            gen_data = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_data).view(1), valid)
            g_loss.backward()
            optimizer_G.step()
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples

            real_loss = adversarial_loss(discriminator(real_data).view(1), valid)
            fake_loss = adversarial_loss(discriminator(gen_data.detach()).view(1), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            batches_done = batches_done + 1

            if batch_labels == 1:
                generated_data_true.append(gen_data)
                generated_data_true_=torch.stack(generated_data_true,0)
            if batch_labels == 0:
                generated_data_false.append(gen_data)
                generated_data_false_=torch.stack(generated_data_false,0)

            fake = gen_data.to(device='cpu').detach().numpy()
            real = real_data.to(device='cpu').detach().numpy()
            if batch_labels == 1:
                if i % 100 == 0:
                    plt.figure(figsize=(30, 3))
                    plt.plot(real[0], c='r')
                    plt.plot(fake[0], c='b')
                    plt.savefig('./gan_generated/true/data_'+str(batches_done)+'.png')
                    print("[Epoch %d] [Sample: %d] [Trial: %d] [D loss: %f] [G loss: %f]"% (epoch,j+1, i+1, d_loss.item(), g_loss.item()))
                    print('[Real max: %4f, min: %4f] [Fake max: %4f, min: %4f]'%(real.min(),real.max(),fake.min(),fake.max()))
                    plt.show()
            if batch_labels == 0:
                    if i % 1000 == 0:
                        plt.figure(figsize=(30, 3))
                        plt.plot(real[0], c='r')
                        plt.plot(fake[0], c='b')
                        plt.savefig('./gan_generated/false/data_'+str(batches_done)+'.png')
                        print("[Epoch %d] [Sample: %d] [Trial: %d] [D loss: %f] [G loss: %f]"% (epoch,j+1, i+1, d_loss.item(), g_loss.item()))
                        print('[Real max: %4f, min: %4f] [Fake max: %4f, min: %4f]'%(real.min(),real.max(),fake.min(),fake.max()))
                        plt.show()

        Generated_data_true.append(torch.stack(generated_data_true_))
        Generated_data_false.append(torch.stack(generated_data_false_))
    PATH_data_true = './gan_ganerated_true_'+str(epoch+1)+'.pth'
    save=torch.save(generated_data_true_, PATH_data_true)
    PATH_data_false = './gan_ganerated_false_'+str(epoch+1)+'.pth'
    save=torch.save(Generated_data_false, PATH_data_false)

PATH1 = './gan_gen_weight.pth'
PATH2 = './gan_disc_weight.pth'
weight1=torch.save(generator.state_dict(), PATH1)
weight2=torch.save(discriminator.state_dict(), PATH2)

gen_pre = Generator(structure).cuda()
disc_pre = Discriminator(structure).cuda()
gen_pre.load_state_dict(torch.load(PATH1))
disc_pre.load_state_dict(torch.load(PATH2))
optimizer_G_pre = torch.optim.Adam(gen_pre.parameters(), lr=0.0001)
optimizer_D_pre = torch.optim.Adam(disc_pre.parameters(), lr=0.0001)
