import torch

from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from mynetworks import Deep4Net_origin, ConvClfNet, FcClfNet
from trte import *

def exp(subject_id):
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_id)

    from braindecode.datautil.preprocess import exponential_moving_standardize
    from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess

    low_cut_hz = 0.  # low cut frequency for filtering
    high_cut_hz = 49.  # high cut frequency for filtering
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
        # NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
        #     init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors)


    ######################################################################
    # Create model and compute windowing parameters
    # ---------------------------------------------
    #


    ######################################################################
    # In contrast to trialwise decoding, we first have to create the model
    # before we can cut the dataset into windows. This is because we need to
    # know the receptive field of the network to know how large the window
    # stride should be.
    #


    ######################################################################
    # We first choose the compute/input window size that will be fed to the
    # network during training This has to be larger than the networks
    # receptive field size and can otherwise be chosen for computational
    # efficiency (see explanations in the beginning of this tutorial). Here we
    # choose 1000 samples, which are 4 seconds for the 250 Hz sampling rate.
    #

    input_window_samples = 1000


    ######################################################################
    # Now we create the model. To enable it to be used in cropped decoding
    # efficiently, we manually set the length of the final convolution layer
    # to some length that makes the receptive field of the ConvNet smaller
    # than ``input_window_samples`` (see ``final_conv_length=30`` in the model
    # definition).
    #

    import torch
    from braindecode.util import set_random_seeds
    from braindecode.models import ShallowFBCSPNet, Deep4Net


    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda:1' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20190706  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes=4
    # Extract number of chans from dataset
    n_chans = dataset[0][0].shape[0]

    # model = Deep4Net(
    #     n_chans,
    #     n_classes,
    #     input_window_samples=input_window_samples,
    #     final_conv_length="auto",
    # )
    #
    #
    #
    # embedding_net = Deep4Net_origin(4, 22, input_window_samples)
    # model = FcClfNet(embedding_net)

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=30,
    )

    print(model)

    # Send model to GPU
    if cuda:
        model.cuda(device)





    ######################################################################
    # And now we transform model with strides to a model that outputs dense
    # prediction, so we can use it to obtain predictions for all
    # crops.
    #

    from braindecode.models.util import to_dense_prediction_model, get_output_shape
    to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
    print("n_preds_per_input : ", n_preds_per_input)
    print(model)



    ######################################################################
    # Cut the data into windows
    # -------------------------
    #


    ######################################################################
    # In contrast to trialwise decoding, we have to supply an explicit window size and window stride to the
    # ``create_windows_from_events`` function.
    #

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
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )


    ######################################################################
    # Split the dataset
    # -----------------
    #
    # This code is the same as in trialwise decoding.
    #







    from braindecode.datasets.base import BaseConcatDataset
    splitted = windows_dataset.split('session')

    train_set = splitted['session_T']
    valid_set = splitted['session_E']





    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 8
    n_epochs = 100


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)



    # Send model to GPU
    if cuda:
        model.cuda(device)

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

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-1)



    import pandas as pd
    results_columns = ['test_loss',  'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    for epochidx in range(1, n_epochs):
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
        df_all.to_csv("shallowFBCSP_abl.csv",mode='w')



