import numpy as np

fold_idx= 0


for fold_idx in range(0,6):
    if fold_idx == 0:
        acc_DG = np.load('[DG]acc_all_' + str(fold_idx) + '.npy')
        acc_DeepAll = np.load('[DeepALL]acc_all_' + str(fold_idx) + '.npy')
        acc_DG = np.delete(acc_DG, [0, 0], axis=0)
        acc_DeepAll = np.delete(acc_DeepAll, [0, 0], axis=0)


    else:
        acc_DG_temp = np.load('[DG]acc_all_' + str(fold_idx) + '.npy')
        acc_DeepAll_temp = np.load('[DeepALL]acc_all_' + str(fold_idx) + '.npy')
        acc_DG_temp = np.delete(acc_DG_temp, [0, 0], axis=0)
        acc_DeepAll_temp = np.delete(acc_DeepAll_temp, [0, 0], axis=0)


        acc_DG = np.hstack([acc_DG,acc_DG_temp])
        acc_DeepAll = np.hstack([acc_DeepAll, acc_DeepAll_temp])



import pandas as pd


pd.DataFrame(acc_DG).to_csv("acc_DG.csv")
pd.DataFrame(acc_DeepAll).to_csv("acc_DeepAll.csv")

for fold_idx in range(0,6):
    print(np.r_[fold_idx * 9:fold_idx * 9 + 9, fold_idx * 9 + 54:fold_idx * 9 + 9 + 54])