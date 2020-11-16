import torch

from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from mynetworks import Deep4Net_origin, ConvClfNet, FcClfNet
from trte import *

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=list(range(1,3)))

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    # keep only EEG sensors
    MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
    # convert from volt to microvolt, directly modifying the numpy array
    NumpyPreproc(fn=lambda x: x * 1e6),
    # bandpass filter
    MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
    # exponential moving standardization
    NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
        init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)

input_window_samples = 1125




cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda:1' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 10  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

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
splitted = windows_dataset.split('session')

def exp(subject_id):
    import torch
    test_subj = np.r_[subject_id]

    print('test subj:' + str(test_subj))

    #20% validation
    train_size = int(0.9* len(splitted['session_T']))
    test_size = len(splitted['session_T']) - train_size



    # train_set, valid_set = torch.utils.data.random_split(splitted['session_T'], [train_size, test_size])
    train_set = splitted['session_T']
    test_set = splitted['session_E']



    # model = Deep4Net(
    #     n_chans,
    #     n_classes,
    #     input_window_samples=input_window_samples,
    #     final_conv_length="auto",
    # )

    from torch.utils.data import Dataset, ConcatDataset




    crop_size = 1000
    # embedding_net = Deep4Net_origin(n_classes, n_chans, crop_size)
    # model = FcClfNet(embedding_net)

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )

    from braindecode.models.util import to_dense_prediction_model, get_output_shape
    to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, 22, input_window_samples)[2]
    print("n_preds_per_input : ", n_preds_per_input)
    print(model)


    batch_size =8
    epochs = 200






    lr = 0.0625 * 0.01
    weight_decay = 0



    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)



    # Send model to GPU
    if cuda:
        model.cuda()

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

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.5 * 0.001)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-1)



    import pandas as pd
    results_columns = ['test_loss',  'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    for epochidx in range(1, epochs):
        print(epochidx)
        train_crop(10, model, device, train_loader,optimizer,scheduler,cuda, args.gpuidx)
        test_loss, test_score = eval_crop(model, device, test_loader)
        results = { 'test_loss': test_loss, 'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)

    return df

if __name__ == '__main__':
    import pandas as pd
    df_all = pd.DataFrame()
    for id in range(1,10):
        df = exp(id)
        df_all = pd.concat([df_all, df], axis=1)
        # df_all.to_csv("bcic_dk_ds_deep4net_fc_crop.csv",mode='w')



