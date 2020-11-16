import random
import numpy as np
import pickle

from mynetworks import Deep4Net_origin, ConvClfNet, FcClfNet, EEGNet_v2, ShallowNet_dense
from trte import *
from torch.utils.data.sampler import SubsetRandomSampler




def exp(subject_id):
    import torch
    input_window_samples = 1000

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda:0' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed = 20190706  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    n_classes = 4

    PATH = '../datasets/'
    with open(PATH + 'bcic_datasets_[0,49].pkl', 'rb') as f:
        data = pickle.load(f)

    import torch

    print('subject:' + str(subject_id))


    #make train test
    tr = []
    val =[]
    test_train_split = 0.5

    dataset= data[subject_id]

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    test_split = int(np.floor(test_train_split * dataset_size))

    train_indices, test_indices = indices[:test_split], indices[test_split:]

    np.random.shuffle(train_indices)
    #분석
    sample_data = data[0].dataset
    sample_data.psd()
    from mne.viz import plot_epochs_image
    import mne
    plot_epochs_image(sample_data, picks=['C3','C4'])

    label = sample_data.read_label()

    sample_data.plot_projs_topomap()

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    from braindecode.models import ShallowFBCSPNet
    model = ShallowFBCSPNet(
        22,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=30,
    )

    from braindecode.models.util import to_dense_prediction_model, get_output_shape
    to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, 22, input_window_samples)[2]
    print("n_preds_per_input : ", n_preds_per_input)
    print(model)


    # crop_size =1000
    #
    #
    #
    #
    # model = ShallowNet_dense(n_classes, 22, crop_size)
    #
    # print(model)

    epochs = 100

    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 8

    train_set = torch.utils.data.Subset(dataset,indices= train_indices)
    test_set = torch.utils.data.Subset(dataset,indices= test_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
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
    args.gpuidx = 0
    args.seed = 0
    args.use_tensorboard = False
    args.save_model = False

    lr = 0.0625 * 0.01
    weight_decay = 0
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs-1)
    #
    # #test lr
    # lr = []
    # for i in range(200):
    #     scheduler.step()
    #     lr.append(scheduler.get_lr())
    #
    # import matplotlib.pyplot as plt
    # plt.plot(lr)

    import pandas as pd
    results_columns = ['test_loss', 'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    for epochidx in range(1, epochs):
        print(epochidx)
        train_crop(10, model, device, train_loader,optimizer,scheduler,cuda, args.gpuidx)
        test_loss, test_score = eval_crop(model, device, test_loader)
        results = {'test_loss': test_loss, 'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)

    return df


if __name__ == '__main__':
    import pandas as pd
    df_all = pd.DataFrame()
    for id in range(0,9):
        df = exp(id)
        df_all = pd.concat([df_all, df], axis=1)
        df_all.to_csv("center_shallow_with_bd.csv",mode='w')



