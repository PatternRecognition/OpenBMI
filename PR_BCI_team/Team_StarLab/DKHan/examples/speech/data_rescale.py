import scipy.io as sio
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from datetime import datetime

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

import pickle

with open('epoch_sess3.pkl', 'rb') as f:
    x_data = pickle.load(f)


sc = StandardScaler()
transformer = Normalizer()
i=0

# raw_data_train = np.zeros([x_data.shape[0],62,1125])
raw_data_train = np.zeros_like(x_data)

for i in range(x_data.shape[0]):

    raw_fit = transformer.transform(x_data[i, :, :])
    # raw_fit = sc.fit_transform(x_data[i, :, :])
    # raw_fit = maxabs_scale(x_data[i, :, :])
    raw_data_train[i,:,:] = raw_fit[:,:]
    print(i)


# plt.imshow(raw_fit)
with open('epoch_sess3_scale.pkl', 'wb') as f:
    pickle.dump(raw_data_train, f, protocol=4)

#시각화로 normalize가 제일 좋은거 확인