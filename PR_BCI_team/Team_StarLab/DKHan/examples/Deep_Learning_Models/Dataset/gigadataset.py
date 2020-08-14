import numpy as np
import pandas as pd
import bisect
from torch.utils.data import Dataset, ConcatDataset, IterableDataset

class GigaDataset(Dataset):
    def __init__(self, dataset, subj_id):
        self.dataset = dataset
        self.len = len(dataset[1])
        self.subj_id = subj_id
        # self.ch_idx = np.array(list(range(7, 11)) + list(range(12, 15)) + list(range(17, 21)) + list(range(32, 41)))


    def __len__(self):
        return self.len
    def __getitem__(self, idx):

        X = self.dataset[0][idx,:,:].astype('float32')*1e6
        y = self.dataset[1][idx]

        return X, y, self.subj_id

