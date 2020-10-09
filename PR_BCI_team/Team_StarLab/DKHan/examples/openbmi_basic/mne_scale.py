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

x_data = np.load('C:/Users/Starlab/Documents/MATLAB/x_data_440_e.npy')

sc = StandardScaler()
transformer = Normalizer()
i=0

raw_data_train = np.zeros([x_data.shape[0],62,500])
# raw_data_train = np.zeros_like(x_data)

for i in range(x_data.shape[0]):
    raw_fit = transformer.transform(x_data[i, :, 0:500])
    raw_data_train[i,:,:] = raw_fit[:,:]
    print(i)

plt.imshow(x_data[35, :, 0:500])
plt.figure()
plt.imshow(raw_fit)

np.save('x_data_440_norm',raw_data_train)