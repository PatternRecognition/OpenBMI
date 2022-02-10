import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
from torch.utils.data import Dataset
import random
import argparse
import models
from train_eval import *
import utils
import time
import hyperparameter as hp
from logger import eegdg_logger



def exp(args,fold_idx, train_set,valid_set, test_set):

    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')
        os.makedirs(path + '/logs')

    logger = eegdg_logger(path +f'/logs/{fold_idx}')

    with open(path + '/args.txt', 'w') as f:
        f.write(str(args))

    import torch.cuda
    cuda = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


    model = models.get_model(args)
    # model = FcClfNet(embedding_net)
    # model  = torch.nn.DataParallel(model)

    mb_params= utils.param_size(model)
    print(f"Model size = {mb_params:.4f} MB")
    if cuda:
        model.cuda(device=device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-1)

    results_columns = [f'valid_loss',f'test_loss',f'valid_accuracy',f'test_accuracy']
    df = pd.DataFrame(columns=results_columns)

    valid_acc = 0
    valid_min_loss = 100
    best_acc = 0
    best_acc_loss = 0
    max_acc = 0

    for epochidx in range(1, args.epochs):
        print(epochidx)
        start = time.time()
        train(10, model, device, train_loader,optimizer,scheduler,cuda, args.gpuidx)
        print(f'total time: {time.time()-start}')
        utils.blockPrint()
        train_loss, train_score = eval(model, device, train_loader)
        valid_loss, valid_score = eval(model, device, valid_loader)
        test_loss, test_score = eval(model, device, test_loader)
        utils.enablePrint()

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f'LR : {lr}')
        logger.log_training(train_loss, train_score, test_loss, test_score, lr, epochidx)

        results = {f'valid_loss':valid_loss,f'test_loss': test_loss, f'valid_accuracy':valid_score,f'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)

        if valid_score >= valid_acc:
            valid_acc = valid_score
            best_acc = test_score
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                f"model_fold{fold_idx}_best.pt"))
            best_epoch = epochidx

        if  valid_loss <= valid_min_loss:
            valid_min_loss = valid_loss
            best_acc_loss = test_score
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                f"model_fold{fold_idx}_best(loss).pt"))
            best_loss_epoch = epochidx

        if test_score >= max_acc:
            max_acc = test_score
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                f"model_fold{fold_idx}_max.pt"))
            max_epoch = epochidx

        print(f'current best acc : {best_acc:.4f} at epoch {best_epoch}')
        print(f'current best(loss) acc : {best_acc_loss:.4f} at epoch {best_loss_epoch}')
        print(f'current max acc : {max_acc:.4f} at epoch {max_epoch}')


    best_model = models.get_model(args)
    best_model.load_state_dict(torch.load(os.path.join(
        path, 'models',
        f"model_fold{fold_idx}_best.pt"), map_location=device))
    if cuda:
        best_model.cuda(device=device)

    print("best accuracy")
    _, _ = eval(best_model, device, test_loader)

    df = utils.get_testset_accuracy(best_model,device,test_set,args)

    return df


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    parser.add_argument('--data-root',
                        default='C:/Users/Starlab/Documents/onedrive/OneDrive - 고려대학교/untitled/convert/')
    parser.add_argument('--save-root', default='../data')
    parser.add_argument('--result-dir', default=f'/{hp.model}_norm')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=hp.lr, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current best Model')

    file_name =  os.path.basename(sys.argv[0])
    args = parser.parse_args()
    args.gpuidx = 0
    args.model_name = hp.model
    import pandas as pd
    df_all = pd.DataFrame()
    args.seed = 0
    fold_idx = 0
    # subject_list = np.r_[0, 1, 2, 4, 5, 8, 17, 18, 20, 27, 28, 32, 35, 36, 42, 43, 44, 51]
    subject_list = np.r_[13:27]
    for fold_idx in subject_list:
        print(f"subject : {fold_idx}")
        train_set,valid_set, test_set, args = utils.get_data_eeg_subject_subset(args, fold_idx)
        df = exp(args,fold_idx,train_set,valid_set, test_set)
        df_all= pd.concat([df_all,df],axis=0)
        print(df_all)


