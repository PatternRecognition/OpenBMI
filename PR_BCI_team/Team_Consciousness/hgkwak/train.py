import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pylab as plt
import torch.nn.functional as F

def train(model, criterion, optimizer, train_data, train_labels, epoch=10):
    label_false=0
    label_true=0
    trained_count=0
    print('---------- Training start ----------')
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
    else:
        cuda= torch.device('cpu')
    for epoch in range(epoch):
        epoch_avg_loss=0
        for i in range(len(train_data)):
            print('[Epoch %d] Training sample %d... '%(epoch+1,i+1))
            sample_avg_loss=0
            trials=train_data[i].shape[0]
            batch_train=torch.tensor(train_data[i]).clone().detach().reshape(trials,-1)
            batch_target=torch.tensor(train_labels[i]).clone().detach()
            for j in range(len(batch_train)):
                batch_input, batch_label = batch_train[j].to(device=cuda, dtype=torch.float), batch_target[j].to(device=cuda, dtype=torch.long)
                # zero the parameter gradients
                batch_input = F.normalize(batch_input.reshape(1,1,-1),dim=0)
                batch_label = batch_label.reshape(1)
                if batch_label == 0:
                    random_prop=np.random.randint(4)
                    if random_prop >= 1:
                        continue
                    else:
                        label_false += 1
                if batch_label == 1:
                    label_true += 1

                optimizer.zero_grad()
                outputs = model(batch_input).reshape(1,-1)
                act = nn.Softmax()
                loss = criterion(outputs, batch_label)
                sample_avg_loss = sample_avg_loss + loss
                loss.backward()
                optimizer.step()
                trained_count += 1
                # print statistics
            PATH = './model_weight_epochg'+str(epoch+1)+'.pth'
            weight=torch.save(model.state_dict(), PATH)

            sample_avg_loss = sample_avg_loss / trials
            epoch_avg_loss = epoch_avg_loss + sample_avg_loss
            print('Sample %d, Avg.Loss:%.4f'% (i+1,sample_avg_loss))
        epoch_avg_loss=epoch_avg_loss / len(train_data)
        print('------- Epoch %d, Avg.Loss:%.4f -------' % (epoch+1,epoch_avg_loss))
    print(label_true, label_false, trained_count)
    print('---------- Finished Training -----------')
    return model, weight

def predict(model, val_data, val_labels, batch_size=1):
    classes = [0., 1.]
    print('---------- Prediction start ----------')
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
    else:
        cuda= torch.device('cpu')
    with torch.no_grad():
        class_pred = [0.,0.]
        class_total = [0.,0.]
        predict_result=[]
        for i in range(len(val_data)):
            trials=val_data[i].shape[0]
            batch_val_data = torch.tensor(val_data[i]).clone().detach()
            batch_val_target = torch.tensor(val_labels[i]).clone().detach()
            for j in range(len(batch_val_data)):
                batch_val_input, batch_val_label = batch_val_data[j].to(device=cuda,dtype=torch.float), batch_val_target[j].to(device=cuda,dtype=torch.long)
                batch_val_input = batch_val_input.reshape(1,1,-1)
                batch_val_label = batch_val_label.reshape(1)
                model.eval()
                outputs = model(batch_val_input).reshape(1,-1)
                act = nn.Softmax()
                outputs = act(outputs.squeeze()).max()
                predict_result.append(outputs)
                c = (outputs == batch_val_label).squeeze()
                if batch_val_label == 0:
                    class_pred[0] += c
                    class_total[0] += 1
                else:
                    class_pred[1] += c
                    class_total[1] += 1
            print('Sample %d done'%(i+1))
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_pred[i] / class_total[i]))
        return predict_result

def train_auto(model, criterion, optimizer, train_data, train_labels, epoch=10):
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
    else:
        cuda= torch.device('cpu')
    Tensor = torch.cuda.FloatTensor if cuda==torch.device('cuda') else torch.FloatTensor
    label_false=0
    label_true=0
    trained_count=0
    print('---------- AE training start ----------')
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
    else:
        cuda= torch.device('cpu')
    for epoch in range(epoch):
        epoch_avg_loss=0
        for i in range(len(train_data)):
            print('[Epoch %d] Training sample %d... '%(epoch+1,i+1))
            sample_avg_loss=0
            trials=train_data[i].shape[0]
            batch_train=torch.tensor(train_data[i]).clone().detach().reshape(trials,-1)
            batch_target=torch.tensor(train_labels[i]).clone().detach()
            for j in range(len(batch_train)):
                if batch_target[j] == 0:
                    continue
                batch_input = F.normalize(batch_train[j].to(device=cuda, dtype=torch.float),dim=0)
                # zero the parameter gradients
                batch_label = Variable(batch_input.type(Tensor))
                optimizer.zero_grad()
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                sample_avg_loss = sample_avg_loss + loss
                loss.backward()
                optimizer.step()
                trained_count += 1
                # print statistics
            PATH = './model_weight_epoch_auto'+str(epoch+1)+'.pth'
            weight=torch.save(model.state_dict(), PATH)

            sample_avg_loss = sample_avg_loss / trials
            epoch_avg_loss = epoch_avg_loss + sample_avg_loss
            print('Sample %d, Avg.Loss:%.4f'% (i+1,sample_avg_loss))
        fake_plot = outputs.to(device='cpu').detach().numpy()
        real_plot = batch_label.to(device='cpu').detach().numpy()
        plt.figure(figsize=(90, 9))
        plt.plot(real_plot, c='r')
        plt.plot(fake_plot, c='b')
        plt.show()
        epoch_avg_loss=epoch_avg_loss / len(train_data)
        print('------- Epoch %d, Avg.Loss:%.4f -------' % (epoch+1,epoch_avg_loss))
    print('---------- Finished Training -----------')
    return model, weight
