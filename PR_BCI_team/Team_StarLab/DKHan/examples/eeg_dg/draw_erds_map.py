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
import copy

def exp(args,fold_idx, train_set,valid_set, test_set):
    path = args.save_root + args.result_dir
    seed =args.seed

    df = pd.DataFrame(np.zeros([1,5]),columns=['subject','sess1-off','sess1-on','sess2-off','sess2-on'])
    if not os.path.isfile(os.path.join(
        path, 'models',
        f"model_fold{fold_idx}_best.pt")):
        return df




    model = models.get_model(args)
    print(model)


    cuda = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_list = []
    model.load_state_dict(torch.load(os.path.join(
        path, 'models',
        f"model_fold{fold_idx}_best(loss).pt"), map_location=device))
    if cuda:
        model.cuda(device=device)
    model_list.append(copy.deepcopy(model))

    model.load_state_dict(torch.load(os.path.join(
        path, 'models',
        f"model_fold{fold_idx}_best(loss).pt"), map_location=device))
    if cuda:
        model.cuda(device=device)
    model_list.append(copy.deepcopy(model))



    all_test_score = []


    for subj in range(1):
        print(subj)
        for sess in range(2):
            for onoff in range(2):
                if sess == 0:
                    subj_id = subj
                else:
                    subj_id = subj + 1

                if onoff == 0:
                    data = torch.utils.data.Subset(test_set, range(subj_id*200,subj_id*200+100))
                else:
                    data = torch.utils.data.Subset(test_set, range(subj_id*200+100,subj_id*200+200))
                print(f'subject:{subj+1}, session:{sess}, onoff:{onoff}')
                test_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
                # utils.blockPrint()

                # test_loss, test_score = eval(model, device, test_loader)

                model.eval()
                test_loss = []
                correct = []

                with torch.no_grad():
                    for datas in test_loader:
                        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

                        output = model(data.unsqueeze(dim=1))

                        test_loss.append(F.cross_entropy(output, target, reduction='sum').item())  # sum up batch loss

                        pred = F.log_softmax(output, dim=1).argmax(dim=1,
                                                                   keepdim=True)  # get the index of the max log-probability
                        correct.append(pred.eq(target.view_as(pred)).sum().item())

                loss = sum(test_loss) / len(test_loader.dataset)
                # print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

                print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    loss, sum(correct), len(test_loader.dataset),
                    100. * sum(correct) / len(test_loader.dataset)))

                test_score = 100. * sum(correct) / len(test_loader.dataset)

                all_test_score.append(test_score)
                # utils.enablePrint()
                print(f"subject:{subj+1}, acc:{test_score}")


    final_result = np.array(all_test_score).reshape(-1,4)
    df = pd.DataFrame(np.insert(final_result,0,fold_idx).reshape(-1,5),columns=['subject','sess1-off','sess1-on','sess2-off','sess2-on'])
    print(f"all acc: {np.mean(all_test_score):.4f}")


    print(df)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    # parser.add_argument('--data-root',
    #                     default='C:/Users/Starlab/Documents/onedrive/OneDrive - 고려대학교/untitled/convert/')
    parser.add_argument('--data-root',
                        default='C:/Users/Starlab/Documents/MATLAB/')
    parser.add_argument('--save-root', default='../data')
    parser.add_argument('--result-dir', default=f'/{hp.model}_batch{hp.batch_size}_ReLR2')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
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

    subject_list = np.r_[0:54]
    # subject_list = np.r_[20]
    for fold_idx in subject_list:
        train_set, valid_set, test_set, args = utils.get_data_eeg_subject_subset_inference(args, fold_idx)
        df = exp(args, fold_idx , train_set, valid_set, test_set)
        df_all= pd.concat([df_all,df],axis=0)

        print(df_all)

    print(df_all.mean())



