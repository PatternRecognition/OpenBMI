from main import *

def eval_subjwise_mt(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    accs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model(data)

            test_loss += (F.nll_loss(output1, target[:,0], reduction='sum').item() \
                         + F.nll_loss(output2, target[:,1], reduction='sum').item())/2 # sum up batch loss

            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct1 += pred1.eq(target[:,0].view_as(pred1)).sum().item()
            correct2 += pred2.eq(target[:,1].view_as(pred2)).sum().item()
            accs.append(pred1[100:200].eq(target[100:200,0].view_as(pred1[100:200])).sum().item()) #online ACC
    test_loss /= len(test_loader.dataset)

    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy1: {}/{} ({:.0f}%), Accuracy2: {}/{} ({:.0f}%)'.format(
        test_loss,
        correct1,
        len(test_loader.dataset),
        100. * correct1 / len(test_loader.dataset),
        correct2,
        len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))

    return test_loss, correct1, accs

def eval(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    accs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data = data.view(-1,1,62,data.shape[4])
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            accs.append(pred[:].eq(target[:].view_as(pred[:])).sum().item())  # online ACC
    test_loss /= len(test_loader.dataset)

    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct, accs

def subjwiseacc():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from datetime import datetime

    loging = True
    ismultitask = False
    loso = False


    x_data, y_data = load_smt()


    if ismultitask:
        nonbciilli = np.s_[
            0, 1, 2, 4, 16, 17, 18, 20, 21, 27, 28, 29, 32, 35, 36, 38, 42, 43, 44, 54, 55, 56, 58, 59, 61, 71, 72, 73,
            74, 81, 82, 85, 86, 89, 90, 96, 97, 98, 99]
        y_subj = np.zeros([108,200])
        y_subj[nonbciilli,:]= 1
        y_subj = y_subj.reshape(21600)
        y_data = np.stack((y_data,y_subj),axis=1)



    model_saved_path = 'model/deep4cnn/test1/'


    dataset_test = GigaDataset(x=x_data, y=y_data, istrain=False, sess=2, subj=-1 , valtype='sess')
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)
    fname = model_saved_path + datetime.today().strftime("%m_%d_%H_%M") + ".txt"

    f = open(fname, 'w')
    for epoch in range(51, 101):
        model = Deep4CNN(use_bn=True).to(device)
        model.load_state_dict(torch.load(model_saved_path + "model_" + str(epoch) + '.pt'))
        loss, score, accs = eval(args, model.embedding_net, device, test_loader)
        f = open(fname, 'a')
        f.write(str(accs) + '\n')
        f.close()


if __name__ == '__main__':
    subjwiseacc()
