import random
import numpy as np
import pickle

from mynetworks import Deep4Net_origin, ConvClfNet, FcClfNet, EEGNet_v2, EEGNet_gabor
from trte import *


def exp(subject_id):
    input_window_samples = 1125

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda:1' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed = 100  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    n_classes = 4

    test_train_split = 0.9
    dataset_size = 576
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_train_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:test_split], indices[test_split:]

    print(val_indices)

    PATH = '../datasets/'
    with open(PATH + 'bcic_datasets_prep.pkl', 'rb') as f:
        data = pickle.load(f)


    test_subj = np.r_[subject_id-1]

    print('test subj:' + str(test_subj))
    train_subj = np.setdiff1d(np.r_[0:9], test_subj)


    #make train test val
    tr = []
    val =[]
    #10%씩 떼어내서 val만듬
    for ids in train_subj:
        train_size = int(0.9 * len(data[ids]))
        test_size = len(data[ids]) - train_size

        tr_i = torch.utils.data.Subset(data[ids], indices=train_indices)
        val_i = torch.utils.data.Subset(data[ids], indices=val_indices)
        # tr_i, val_i = torch.utils.data.random_split(data[ids], [train_size, test_size])
        tr.append(tr_i)
        val.append(val_i)

    train_set = torch.utils.data.ConcatDataset(tr)
    valid_set = torch.utils.data.ConcatDataset(val)

    test_set = torch.utils.data.ConcatDataset([torch.utils.data.Subset(data[ids], indices=list(range(288,576))) for ids in test_subj])

    crop_size =1125
    embedding_net = EEGNet_v2(n_classes, 22, 1125)


    model = FcClfNet(embedding_net)

    print(model)

    batch_size =64
    epochs = 200

    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 200


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)



    # Send model to GPU
    if cuda:
        model.cuda(device=device)


    from torch.optim import lr_scheduler
    import torch.optim as optim

    import argparse
    parser = argparse.ArgumentParser(description='cross subject domain adaptation')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    args.gpuidx = 1
    args.seed = 0
    args.use_tensorboard = False
    args.save_model = False

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.5 * 0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 50)
    #
    # #test lr
    # lr = []
    # for i in range(200):
    #     scheduler.step()
    #     lr.append(scheduler.get_lr())
    #
    # import matplotlib.pyplot as plt
    # plt.plot(lr)

    curr_best_by_acc = 0
    curr_best_by_loss = 0
    curr_best_val_acc = 0
    curr_best_val_loss = 9999

    import pandas as pd
    results_columns = ['val_loss', 'test_loss', 'val_accuracy', 'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    for epochidx in range(1, epochs):
        print(epochidx)
        train(10, model, device, train_loader,optimizer,scheduler,cuda, args.gpuidx)
        tr_loss, tr_score = eval(model, device, train_loader)
        val_loss, val_score = eval(model, device, valid_loader)
        test_loss, test_score = eval(model, device, test_loader)

        if val_loss <=curr_best_val_loss:
            curr_best_val_loss = val_loss
            curr_best_by_loss = test_score
        if val_score >= curr_best_val_acc:
            curr_best_val_acc = val_score
            curr_best_by_acc = test_score



        results = {'val_loss': val_loss, 'test_loss': test_loss, 'val_accuracy' : val_score, 'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)
        print('curr_best_by_acc : {:.4f}, curr_best_by_loss : {:.4f}'.format(curr_best_by_acc,curr_best_by_loss))


    return df, curr_best_by_acc, curr_best_by_loss

if __name__ == '__main__':



    import pandas as pd
    df_all = pd.DataFrame()

    summary_columns = ['curr_best_by_acc', 'curr_best_by_loss']
    df_summary = pd.DataFrame(columns=summary_columns)

    for id in range(1,10):
        df, curr_best_by_acc, curr_best_by_loss= exp(id)
        df_all = pd.concat([df_all, df], axis=1)
        df_all.to_csv("bcic_ERM4.csv",mode='w')
        summary = {'curr_best_by_acc': curr_best_by_acc, 'curr_best_by_loss': curr_best_by_loss}
        df_summary = df_summary.append(summary, ignore_index=True)
        df_summary.to_csv("bcic_ERM_summary4.csv", mode='w')




