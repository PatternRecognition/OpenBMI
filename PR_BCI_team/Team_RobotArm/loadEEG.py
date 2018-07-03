# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:54:34 2018

@author: wam
"""

import os
import scipy.io as scio

def loadEEG(path):
    os.chdir(path)
    clab = scio.loadmat('clab.mat')
    clab = clab['clab']
    fs = scio.loadmat('fs.mat')
    fs = fs['fs']
    x = scio.loadmat('x.mat')
    x = x['x']
    y = scio.loadmat('y.mat')
    y = y['y']
    t = scio.loadmat('t.mat')
    t = t['t']
    className = scio.loadmat('className.mat')
    className = className['className']
    return clab, x, y, t, className