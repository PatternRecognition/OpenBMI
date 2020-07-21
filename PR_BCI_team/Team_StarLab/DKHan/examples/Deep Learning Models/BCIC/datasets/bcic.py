import numpy as np
import pandas as pd
import bisect
from torch.utils.data import Dataset, ConcatDataset, IterableDataset

class BcicDataset(Dataset):
    def __init__(self, dataset, subj_id):
        self.dataset = dataset
        self.len = len(dataset)
        self.subj_id = subj_id
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.dataset.get_data(item =idx)[:,:,0:1125].astype('float32')*1e6
        y = self.dataset.events[:, -1][idx]-1

        return X, y, self.subj_id


class BcicConcatDataset(ConcatDataset):
    def __init__(self,datasets):
        self.datasets = datasets
        super(ConcatDataset, self).__init__(datasets)

        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __getitem__custom(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __getitem__(self, idx):
        x = self.__getitem__custom(idx)
        return