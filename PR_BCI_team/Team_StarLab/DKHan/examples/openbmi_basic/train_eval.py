import torch
import torch.nn.functional as F
#단순  분류  모델 학습과  평가  부분


def train(args, model, device, train_loader, optimizer,scheduler, cuda, gpuidx, epoch=1):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)


        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()




def eval(args, model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    preds = []
    temps = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data = data.view(-1,1,62,data.shape[4])
            output = model(data)

            #output = nn.CrossEntropyLoss(output)
            #output = F.log_softmax(output, dim=1)

            test_loss.append(F.nll_loss(output, target, reduction='sum').item()) # sum up batch loss

            pred = output.argmax(dim=1,keepdim=True)# get the index of the max log-probability
            preds.append(pred.cpu().numpy())
            correct.append(pred.eq(target.view_as(pred)).sum().item())

            temp = pred.eq(target.view_as(pred)).cpu().numpy()
            temps.append(temp)


    loss = sum(test_loss)/len(test_loader.dataset)
    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))
    #
    # temp = np.array(preds)
    # import pandas as pd
    #
    # temp =temp.reshape(5100)
    # df = pd.DataFrame(temp)
    #
    # df.to_csv("5100result.csv", mode='w')

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset), temps