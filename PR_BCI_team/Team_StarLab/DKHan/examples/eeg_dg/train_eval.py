import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class ConfidenceLabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(ConfidenceLabelSmoothingCrossEntropy, self).__init__()
        # self.confidence = [0.7425, 0.9325, 0.965, 0.5395, 0.86025, 0.754, 0.66475, 0.618, 0.7925, 0.6525, 0.5415,
        #                    0.5705, 0.6525, 0.59625, 0.6145, 0.62125, 0.7755, 0.866, 0.83425, 0.64125, 0.986, 0.82225,
        #                    0.70525, 0.5625, 0.5145, 0.5275, 0.57775, 0.918, 0.9175, 0.69575, 0.6555, 0.867, 0.945,
        #                    0.5155, 0.593, 0.976, 0.963, 0.591, 0.749, 0.5575, 0.52625, 0.6125, 0.83725, 0.97225,
        #                    0.93725, 0.6415, 0.61225, 0.584, 0.69175, 0.60825, 0.63575, 0.756, 0.61375, 0.53575]

        self.confidence = [0.713, 0.953, 0.947, 0.514, 0.933, 0.725, 0.6025, 0.5855, 0.821, 0.6175, 0.547, 0.5605, 0.7,
                           0.609, 0.5785, 0.638, 0.8005, 0.824, 0.834, 0.5155, 0.9775, 0.8615, 0.6305, 0.549, 0.517,
                           0.5915, 0.5285, 0.923, 0.855, 0.751, 0.675, 0.773, 0.9805, 0.53, 0.5255, 0.9685, 0.9535,
                           0.5515, 0.8795, 0.497, 0.529, 0.5335, 0.8645, 0.9595, 0.9245, 0.5265, 0.452, 0.6415, 0.696,
                           0.617, 0.683, 0.7255, 0.5995, 0.5815, 0.772, 0.912, 0.983, 0.565, 0.7875, 0.783, 0.727,
                           0.6505, 0.764, 0.6875, 0.536, 0.5805, 0.605, 0.5835, 0.6505, 0.6045, 0.7505, 0.908, 0.8345,
                           0.767, 0.9945, 0.783, 0.78, 0.576, 0.512, 0.4635, 0.627, 0.913, 0.98, 0.6405, 0.636, 0.961,
                           0.9095, 0.501, 0.6605, 0.9835, 0.9725, 0.6305, 0.6185, 0.618, 0.5235, 0.6915, 0.81, 0.985,
                           0.95, 0.7565, 0.7725, 0.5265, 0.6875, 0.5995, 0.5885, 0.7865, 0.628, 0.49, 0.985, 0.95,
                           0.7565, 0.7725, 0.5265, 0.6875, 0.5995, 0.5885, 0.7865, 0.628, 0.49
                           ]
    def forward(self, x, target, sid):
        confidencemat = torch.zeros_like(target,dtype=torch.float32)
        for i in range(len(target)):
            confidencemat[i] = self.confidence[sid[i]]

        smoothing = 1 - confidencemat
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)


        loss = torch.mul(confidencemat,nll_loss) + torch.mul(smoothing,smooth_loss)
        return loss.mean()


class CroppedLoss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def __call__(self, preds, targets):
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)


def train_crop(log_interval, model, device, train_loader, optimizer, scheduler, cuda, gpuidx, epoch=1):
    criterion = torch.nn.NLLLoss()
    lossfn = CroppedLoss(criterion)

    model.train()
    for batch_idx, datas in enumerate(train_loader):

        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

        optimizer.zero_grad()

        output = model(data)
        output = model.embedding_net(data)
        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()


def eval_crop(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
            outputs = []

            for i in range(2):
                outputs.append(model(data[:, :, :, i * 125:i * 125 + 1000]))

            result = torch.cat([outputs[0], outputs[1][:, :, model.out_size - 125:model.out_size]], dim=2)
            y_preds_per_trial = result.mean(dim=2)

            test_loss.append(F.nll_loss(y_preds_per_trial, target, reduction='sum').item())  # sum up batch loss
            pred = y_preds_per_trial.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss) / len(test_loader.dataset)
    # print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)


class MAE_loss(torch.nn.Module):
    def __init__(self, device):
        super(MAE_loss, self).__init__()
        self.device = device
        self.loss_function = torch.nn.L1Loss()

    def __call__(self, preds, targets):
        y_onehot = torch.FloatTensor(targets.size(0), 2).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)

        return self.loss_function(preds, y_onehot)


class MAE_loss(torch.nn.Module):
    def __init__(self, device):
        super(MAE_loss, self).__init__()
        self.device = device
        self.loss_function = torch.nn.L1Loss()

    def __call__(self, preds, targets):
        y_onehot = torch.FloatTensor(targets.size(0), 2).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, targets.unsqueeze(1), 1)

        return self.loss_function(preds, y_onehot)


import utils
import time


def train(log_interval, model, device, train_loader, optimizer, scheduler, cuda, gpuidx, epoch):
    losses = utils.AverageMeter('Loss', ':.4e')
    if isinstance(model, torch.nn.DataParallel):
        lossfn = model.module.criterion
    else:
        lossfn = model.criterion
        # lossfn = LabelSmoothingCrossEntropy()
        # lossfn = ConfidenceLabelSmoothingCrossEntropy()
    correct = []

    start = time.time()

    model.train()
    t_data = []
    t_model = []

    t3 = time.time()
    for batch_idx, datas in enumerate(train_loader):

        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
        t2 = time.time()
        t_data.append(t2 - t3)
        # print(t2)
        optimizer.zero_grad()

        output = model(data.unsqueeze(dim=1))

        pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct.append(pred.eq(target.view_as(pred)).sum().item())

        loss = lossfn(output, target)

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        t3 = time.time()
        t_model.append(t3 - t2)

    print("time :", time.time() - start)
    print(f"t_data : {sum(t_data)} , t_model : {sum(t_model)}")
    print(f'Train set: Accuracy: {sum(correct)}/{len(train_loader.dataset)} ({100. * sum(correct) / len(train_loader.dataset):.4f}%)')
def train_mtl(log_interval, model, device, train_loader, optimizer, scheduler, cuda, gpuidx, epoch):
    losses = utils.AverageMeter('Loss', ':.4e')
    if isinstance(model, torch.nn.DataParallel):
        lossfn = model.module.criterion
    else:
        lossfn = model.criterion
        # lossfn = LabelSmoothingCrossEntropy()
        # lossfn = ConfidenceLabelSmoothingCrossEntropy()
    correct = []

    start = time.time()

    model.train()
    t_data = []
    t_model = []

    t3 = time.time()
    for batch_idx, datas in enumerate(train_loader):

        data, target, subjid = datas[0].to(device), datas[1].to(device, dtype=torch.int64), datas[2].to(device, dtype=torch.int64)
        t2 = time.time()
        t_data.append(t2 - t3)
        # print(t2)
        optimizer.zero_grad()

        output = model(data.unsqueeze(dim=1))

        pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct.append(pred.eq(target.view_as(pred)).sum().item())

        loss = lossfn(output, 2*subjid+target)

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        t3 = time.time()
        t_model.append(t3 - t2)

    print("time :", time.time() - start)
    print(f"t_data : {sum(t_data)} , t_model : {sum(t_model)}")
    print(f'Train set: Accuracy: {sum(correct)}/{len(train_loader.dataset)} ({100. * sum(correct) / len(train_loader.dataset):.4f}%)')


def train_gpu(log_interval, model, device, train_loader, optimizer, scheduler, cuda, gpuidx, epoch=1):
    losses = utils.AverageMeter('Loss', ':.4e')
    if isinstance(model, torch.nn.DataParallel):
        lossfn = model.module.criterion
    else:
        lossfn = model.criterion
    correct = []
    import time
    start = time.time()
    model.train()

    t_data = []
    t_model = []

    t3 = time.time()
    for batch_idx, datas in enumerate(train_loader):

        data, target = datas[0], datas[1]
        t2 = time.time()
        t_data.append(t2 - t3)

        optimizer.zero_grad()

        output = model(data.unsqueeze(dim=1))
        pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct.append(pred.eq(target.view_as(pred)).sum().item())

        loss = lossfn(output, target)

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        t3 = time.time()
        t_model.append(t3 - t2)

    print("time :", time.time() - start)
    print(f"t_data : {sum(t_data)} , t_model : {sum(t_model)}")
    scheduler.step(losses.avg)
    print(f'Train set: Accuracy: {sum(correct)}/{len(train_loader.dataset)} ({100. * sum(correct) / len(train_loader.dataset):.4f}%)')


def eval(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

            output = model(data.unsqueeze(dim=1))

            test_loss.append(F.cross_entropy(output, target, reduction='sum').item())  # sum up batch loss

            pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss) / len(test_loader.dataset)
    # print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def eval_cali(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

            output = model(data.unsqueeze(dim=1))

            test_loss.append(F.cross_entropy(output, target, reduction='sum').item())  # sum up batch loss
            pred = F.softmax(output, dim=1)
            fpr, tpr, thresholds = roc_curve(target.cpu(), pred.cpu()[:,0])

            AUC = auc(fpr, tpr)

            correct.append(AUC)

    loss = sum(test_loss) / len(test_loader.dataset)
    # print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)


def vote(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    output = F.log_softmax(output, dim=1)
    _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    modevalue = torch.mode(pred%2)[0]

    return modevalue

def eval_mtl(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target, subjid = datas[0].to(device), datas[1].to(device, dtype=torch.int64), datas[2].to(device,
                                                                                                            dtype=torch.int64)

            output = model(data.unsqueeze(dim=1))

            pred = vote(output, subjid*2+target, (1,5))

            test_loss.append(F.cross_entropy(output, subjid*2+target, reduction='sum').item())  # sum up batch loss


            # pred_0 = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # pred = pred_0%2
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss) / len(test_loader.dataset)
    # print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)





def eval_ensemble(models, device, test_loader):
    for model in models:
        model.eval()

    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

            output = []
            for model in models:
                output.append(model(data.unsqueeze(dim=1)).unsqueeze(dim=2))

            temp = torch.cat(output, dim=2)
            temp2 = temp.mean(dim=2)
            test_loss.append(F.cross_entropy(temp2, target, reduction='sum').item())  # sum up batch loss

            pred = F.log_softmax(temp2, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss) / len(test_loader.dataset)
    # print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)
