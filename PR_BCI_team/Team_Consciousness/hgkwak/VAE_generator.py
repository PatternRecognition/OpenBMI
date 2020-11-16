from data_load import data_load
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

class WaveNet(nn.Module):
    def __init__(self, structure, window):
        super(WaveNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(window, structure[0], bias=False),
            nn.Tanh()
        )
        hidden_layers = []
        for i in range(1,len(structure)-1):
            hidden_layers += [nn.Linear(structure[i-1], structure[i]), nn.Tanh()]
        self.hidden_layer = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(structure[-2],structure[-1]),
            nn.Tanh(),
            nn.Linear(structure[-1],window)
        )
        self.window = window

    def forward(self, z):
        gen_data = self.input_layer(z)
        gen_data = self.hidden_layer(gen_data)
        gen_data = self.output_layer(gen_data)
        return gen_data

class Discriminator(nn.Module):
    def __init__(self, structure, window):
        super(Discriminator, self).__init__()
        self.rev_structure = list(reversed(structure))
        def make_layers(chan_in, chan_out, normalize=True):
            layers=[]
            layers += [nn.Linear(chan_in, chan_out)]
            layers += [nn.Tanh()]
            return layers

        self.layers = []
        for i in range(1, len(self.rev_structure)):
            if i == 1:
                self.layers += make_layers(self.rev_structure[i-1], self.rev_structure[i], normalize=False)
            else:
                self.layers += make_layers(self.rev_structure[i-1], self.rev_structure[i])

        self.discriminator=nn.Sequential(
            nn.Linear(window, self.rev_structure[0], bias=False),
            nn.Tanh(),
            *self.layers,
            nn.Linear(self.rev_structure[-1], 1),
            nn.Sigmoid()
        )

    def forward(self,data):
        validity = self.discriminator(data)
        return validity

def wavenet(structure, window):
    model = WaveNet(structure, window)
    print(model)
    print('Window size: %d'%window)
    return model, window

if torch.cuda.is_available():
    cuda = torch.device('cuda')
else:
    cuda= torch.device('cpu')

train_data, train_labels, val_data, val_labels = data_load()
structure = [64,128,256,512]
model, window = wavenet(structure, window=1000)
discriminator = Discriminator(structure, window)

criterion = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=1e-5)
n_epoch = 50

model.cuda()
discriminator.cuda()
criterion.cuda()

Tensor = torch.cuda.FloatTensor if cuda==torch.device('cuda') else torch.FloatTensor
batches_done=1
error=0
stop_counter=0
for epoch in range(n_epoch):
    if error == 1:
        break
    Generated_data_true=[]
    Generated_data_false=[]
    for i in range(len(train_data)):
        if error == 1:
            break
        print('Generating data for sample %d...'%(i+1))
        input_data = torch.tensor(train_data[i]).clone().detach()
        generated_data_true=[]
        generated_data_false=[]
        for j in range(len(input_data)):
            if error == 1:
                break
            input_data_trial = input_data[j].clone().detach()
            gen_trial = []
            for k in range(0,len(input_data_trial),window):
                if len(input_data_trial)%window != 0:
                    print('Window size should be the divisor of the length of input sequence')
                    error += 1
                    break
                valid = Variable(Tensor(1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)
                input_data_window = input_data_trial[k:window+k].clone().detach()
                real_data = Variable(input_data_window.type(Tensor))
                z = Variable(Tensor(np.random.normal(0, 1, (window))))
                gen_window = model(z)

                optimizer_G.zero_grad()
                g_loss = criterion(discriminator(gen_window), valid)
                g_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                real_loss = criterion(discriminator(real_data), valid)
                fake_loss = criterion(discriminator(gen_window.clone().detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                gen_trial.append(gen_window)
            batches_done=batches_done+1
            gen_trial_tensor = torch.cat(gen_trial)
            if train_labels[i][j] == 1:
                generated_data_true.append(gen_trial_tensor)
                generated_data_true_=torch.stack(generated_data_true,0)
            if train_labels[i][j] == 0:
                generated_data_false.append(gen_trial_tensor)
                generated_data_false_=torch.stack(generated_data_false,0)

            if train_labels[i][j] == 1:
                if (abs(d_loss - 1.386) <= 0.05) and (abs(g_loss - 0.693) <= 0.05):
                    fake_plot = gen_trial_tensor.to(device='cpu').detach().numpy()
                    real_plot = input_data_trial.to(device='cpu').detach().numpy()
                    print("'TRUE' label, Epoch: %d, Sample: %d, Trials: %d, [D loss: %f] [G loss: %f]"% (epoch+1,i+1,j+1,d_loss.item(), g_loss.item()))
                    print('[Real max: %4f, min: %4f] [Fake max: %4f, min: %4f]'%(real_plot.min(),real_plot.max(),fake_plot.min(),fake_plot.max()))
                    plt.figure(figsize=(90, 9))
                    for k in range(0,len(input_data_trial)+1,window):
                        plt.axvline(k, c='grey', alpha=0.3)
                    plt.plot(real_plot, c='r')
                    plt.plot(fake_plot, c='b')
                    plt.savefig('./gan_generated_new/true_new/data_'+str(batches_done)+'.png')
                    plt.show()

            if train_labels[i][j] == 0:
                if (abs(d_loss - 1.386) <= 0.05) and (abs(g_loss - 0.693) <= 0.05):
                    fake_plot = gen_trial_tensor.to(device='cpu').detach().numpy()
                    real_plot = input_data_trial.to(device='cpu').detach().numpy()
                    print("'FALSE' label, Epoch: %d, Sample: %d, Trials: %d, [D loss: %f] [G loss: %f]"% (epoch+1,i+1,j+1,d_loss.item(), g_loss.item()))
                    print('[Real max: %4f, min: %4f] [Fake max: %4f, min: %4f]'%(real_plot.min(),real_plot.max(),fake_plot.min(),fake_plot.max()))
                    figure=plt.figure(figsize=(90, 9))
                    for k in range(0,len(input_data_trial)+1,window):
                        plt.axvline(k, c='grey', alpha=0.3)
                    plt.plot(real_plot, c='r')
                    plt.plot(fake_plot, c='b')
                    plt.savefig('./gan_generated_new/false_new/data_'+str(batches_done)+'.png')
                    plt.show()
            if abs(d_loss - g_loss) >= 5:
                stop_counter += 1
                if batches_done % 100 == 0:
                    print(stop_counter)
                    print("Epoch: %d, Sample: %d, Trials: %d, [D loss: %f] [G loss: %f]"% (epoch+1,i+1,j+1,d_loss.item(), g_loss.item()))

            if abs(d_loss - g_loss) <= 1:
                stop_counter = stop_counter-1
                if batches_done % 100 == 0:
                    print(stop_counter)
                    print("Epoch: %d, Sample: %d, Trials: %d, [D loss: %f] [G loss: %f]"% (epoch+1,i+1,j+1,d_loss.item(), g_loss.item()))
            if stop_counter >= 100:
                print('Negatively stopped training')
                break
        if train_labels[i][j] != 0:
            Generated_data_true.append(generated_data_true_)
            #Generated_data_false.append(generated_data_false_)
    PATH_data_true = './gan_ganerated_new_true_new'+str(epoch+1)+'.pth'
    save=torch.save(generated_data_true_, PATH_data_true)
    PATH_data_false = './gan_ganerated_new_false_new'+str(epoch+1)+'.pth'
    save=torch.save(Generated_data_false, PATH_data_false)

PATH1 = './gan_gen_new_weight.pth'
PATH2 = './gan_disc_new_weight.pth'
weight1=torch.save(model.state_dict(), PATH1)
weight2=torch.save(discriminator.state_dict(), PATH2)
