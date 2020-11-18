import sys
sys.path.append('../')

from torch.utils.data import Dataset
import os
import argparse
import models
from train_eval import *
import utils
import torch.cuda
import hyperparameter as hp

def exp(args,fold_idx, train_set,valid_set, test_set):
    path = args.save_root + args.result_dir
    seed =args.seed

    model = models.get_model(args)
    print(model)


    cuda = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model.load_state_dict(torch.load(os.path.join(
        path, 'models',
        f"model_fold{fold_idx}_best(loss).pt"), map_location=device))
    if cuda:
        model.cuda(device=device)

    all_test_score = []


    for subj in range(1):
        print(subj)
        for sess in range(2):
            for onoff in range(1):
                if sess == 0:
                    subj_id = subj
                else:
                    subj_id = subj + 1

                if onoff == 0:
                    data = torch.utils.data.Subset(test_set, range(subj_id*200,subj_id*200+200))
                else:
                    data = torch.utils.data.Subset(test_set, range(subj_id*200,subj_id*200+200))
                print(f'subject:{subj+1}, session:{sess}, onoff:{onoff}')
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
                # utils.blockPrint()
                test_loss, test_score = eval(model, device, test_loader)

                out, y, ids = get_feature(model, device, test_loader)

                feature_tr, y_tr, id = get_feature(model, device, train_loader)
                # feature_tr_gap = feature_tr.reshape(feature_tr.shape[0],200,6).mean(axis=2)
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, perplexity=30)
                selected_subj = np.r_[0:4]
                feature_tr_temp = feature_tr[np.isin(id,selected_subj)]

                feature = np.concatenate([feature_tr_temp, out])
                label = np.concatenate([y_tr[np.isin(id,selected_subj)],y])

                source_ids = np.char.mod('%d',id[np.isin(id,selected_subj)])
                source_ids_list = []
                for i in range(len(source_ids)):
                    source_ids_list.append(source_ids[i] +' (source)')

                for i in range(len(ids)):
                    source_ids_list.append(str(28) + ' (target)')

                #
                # subj = np.concatenate([id[np.isin(id,selected_subj)]+'source',ids])
                # source_target = np.zeros_like(subj)
                # source_target[0:len(feature_tr_temp)] = 1
                train_tsne = tsne.fit_transform(feature)

                np.char.mod('%d', subj)
                sns.set(style="darkgrid")
                plot_features(train_tsne, source_ids_list )
                plot_features(train_tsne, label )
                plot_pair(train_tsne, source_ids_list )
                np.unique(y_tr).size

                np.arange(0,19)



                out = np.concatenate(out)
                # all_test_score.append(test_score)
                # utils.enablePrint()
                # print(f"subject:{subj+1}, acc:{test_score}")


    df = pd.DataFrame(np.array(all_test_score).reshape(-1,4),columns=['sess1-off','sess1-on','sess2-off','sess2-on'])
    print(f"all acc: {np.mean(all_test_score):.4f}")







    print(df)
    return df



import seaborn as sns
import matplotlib.pyplot as plt

def plot_pair(x_tsne, labels):
    df = pd.DataFrame()
    df['tsne-2d-one'] = x_tsne[:, 0]
    df['tsne-2d-two'] = x_tsne[:, 1]
    df['subject'] = labels

    plt.figure(figsize=(10, 10))
    # plt.scatter(x_tsne[:, 0],x_tsne[:, 1],labels)

    sns.pairplot(
        df,
        hue="subject",
        palette=sns.color_palette("Set1",n_colors=np.unique(labels).size),
    )
    plt.show()

    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="subject",
    #     palette=sns.color_palette("Set1",n_colors=np.unique(labels).size),
    #     data=df,
    #     legend="full",
    # )
    # plt.show()


def plot_features(x_tsne, labels):
    df = pd.DataFrame()
    df['tsne-2d-one'] = x_tsne[:, 0]
    df['tsne-2d-two'] = x_tsne[:, 1]
    df['labels'] = labels

    plt.figure(figsize=(10, 10))

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="labels",
        palette=sns.color_palette("Set1",n_colors=np.unique(labels).size),
        data=df,
        legend="full",
    )
    plt.show()

def get_feature(model, device, test_loader):
    model.eval()
    outputs = []
    targets = []
    ids = []
    with torch.no_grad():
        for datas in test_loader:
            data, target, id = datas[0].to(device), datas[1], datas[2]

            output = model.get_embedding(data.unsqueeze(dim=1))
            outputs.append(output.cpu())
            targets.append(target)
            ids.append(id)

    return  np.concatenate(outputs), np.concatenate(targets), np.concatenate(ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    # parser.add_argument('--data-root',
    #                     default='C:/Users/Starlab/Documents/onedrive/OneDrive - 고려대학교/untitled/convert/')
    parser.add_argument('--data-root',
                        default='C:/Users/Starlab/Documents/MATLAB/')
    parser.add_argument('--save-root', default='../data')
    parser.add_argument('--result-dir', default=f'/{hp.model}_batch{hp.batch_size}_unshuffle')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.000625, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current best Model')

    args = parser.parse_args()
    args.gpuidx = 0
    import pandas as pd

    args.model_name = hp.model
    fold_idx = 0
    df_all = pd.DataFrame()

    args.seed=2020

    subject_list = np.r_[28]
    for fold_idx in subject_list:
        train_set, valid_set, test_set, args = utils.get_data_eeg_subject_subset_inference(args, fold_idx)
        df = exp(args, fold_idx , train_set, valid_set, test_set)
        df_all= pd.concat([df_all,df],axis=0)

        print(df_all)



