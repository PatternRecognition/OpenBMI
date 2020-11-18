import torch

from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from mynetworks import Deep4Net_origin, ConvClfNet, FcClfNet, EEGNet_v2_old
from trte import *
import torch

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=list(range(1,10)))

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    # keep only EEG sensors
    MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
    # convert from volt to microvolt, directly modifying the numpy array
    # NumpyPreproc(fn=lambda x: x * 1e3)
    # # bandpass filter
    # MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
    # exponential moving standardization
    # NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
    #     init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)

input_window_samples = 1125


n_classes=4
# Extract number of chans from dataset
n_chans = dataset[0][0].shape[0]

import numpy as np
from braindecode.datautil.windowers import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

from braindecode.datasets.base import BaseConcatDataset
splitted = windows_dataset.split('subject')

def exp(subject_id):
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda:1' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 10  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    test_subj = np.r_[subject_id]

    print('test subj:' + str(test_subj))
    train_subj = np.setdiff1d(np.r_[1:10], test_subj)

    tr = []
    val =[]

    #10%씩 떼어내서 val만듬
    for ids in train_subj:
        train_size = int(0.9 * len(splitted[ids]))
        test_size = len(splitted[ids]) - train_size
        tr_i, val_i = torch.utils.data.random_split(splitted[ids], [train_size, test_size])
        tr.append(tr_i)
        val.append(val_i)

    train_set = torch.utils.data.ConcatDataset(tr)
    valid_set = torch.utils.data.ConcatDataset(val)
    test_set = BaseConcatDataset([splitted[ids] for ids in test_subj])

    # model = Deep4Net(
    #     n_chans,
    #     n_classes,
    #     input_window_samples=input_window_samples,
    #     final_conv_length="auto",
    # )

    crop_size =1125
    embedding_net = EEGNet_v2_old(n_classes, n_chans, crop_size)
    model = FcClfNet(embedding_net)

    print(model)

    epochs = 100

    batch_size = 64

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

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.5 * 0.001)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 50)
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
    results_columns = ['val_loss', 'test_loss', 'val_accuracy', 'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    for epochidx in range(1, epochs):
        print(epochidx)
        train(10, model, device, train_loader,optimizer,scheduler,cuda, device)
        val_loss, val_score = eval(model, device, valid_loader)
        test_loss, test_score = eval(model, device, test_loader)
        results = {'val_loss': val_loss, 'test_loss': test_loss, 'val_accuracy' : val_score, 'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)

    return df

if __name__ == '__main__':
    import pandas as pd
    df_all = pd.DataFrame()
    for id in range(1,10):
        df = exp(id)
        df_all = pd.concat([df_all, df], axis=1)
        # df_all.to_csv("bcic_df_EEGNet_fc_trial_w/o_norm.csv",mode='w')



