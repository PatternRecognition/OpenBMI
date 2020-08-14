import torch
import torch.nn.functional as F

def train(log_interval ,model, device, train_loader, optimizer,scheduler, cuda, gpuidx, epoch=1):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, datas in enumerate(train_loader):

        data, target = datas[0].to(device,dtype=torch.float), datas[1].to(device, dtype=torch.int64)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()

def eval(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device,dtype=torch.float), datas[1].to(device, dtype=torch.int64)
            output = model(data[:,:,:])

            #output = nn.CrossEntropyLoss(output)
            #output = F.log_softmax(output, dim=1)

            test_loss.append(F.cross_entropy(output, target, reduction='sum').item()) # sum up batch loss

            pred = output.argmax(dim=1,keepdim=True)# get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss)/len(test_loader.dataset)
    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))



    return loss, 100. * sum(correct) / len(test_loader.dataset)

