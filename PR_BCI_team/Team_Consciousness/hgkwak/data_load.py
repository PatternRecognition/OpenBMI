from scipy import io
import os
import numpy as np

def data_load():
    train_path_dir='./dataset/Training/';
    train_file_list =  os.listdir(train_path_dir)
    val_path_dir='./dataset/Validation/'; val_file_list= os.listdir(val_path_dir)

    train_data=[]; val_data=[];
    train_epo=[]; train_label=[];
    val_epo=[]; val_label=[];

    ### Data load ###
    for file in train_file_list:
        train_data.append(io.loadmat(train_path_dir+file))
    for file in val_file_list:
        val_data.append(io.loadmat(val_path_dir+file))
    for i in range(len(train_data)):
        train_epo.append(train_data[i]['epo']['x'][0][0])
        train_label.append(train_data[i]['epo']['y'][0][0][0])
    for i in range(len(val_data)):
        val_epo.append(val_data[i]['epo']['x'][0][0])
        val_label.append(val_data[i]['epo']['y'][0][0][0])
    modified_label=[]
    modified_train=[]
    for i in range(len(train_label)):
        index=[]
        for j in range(len(train_label[i])):
            if train_label[i][j] in [2]:
                train_label[i][j] = 1
            if train_label[i][j] in [3,4,5,6]:
                index.append(j)
        re_label = np.delete(train_label[i], index)
        re_train = np.delete(train_epo[i], index, axis=0)
        modified_train.append(re_train)
        modified_label.append(re_label)
    return modified_train, modified_label, val_epo, val_label
