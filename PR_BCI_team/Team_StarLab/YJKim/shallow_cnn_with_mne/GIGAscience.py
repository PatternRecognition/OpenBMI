import numpy as np
import mne
from scipy.io import loadmat
from gigadata2 import *

class GIGAscience(object):
    def __init__(self, filename, variable_names):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt, events = self.extract_data()
        cnt.info['events'] = events
        return cnt

    def extract_data(self):
        MI = loadmat(self.filename, struct_as_record=False, squeeze_me=True, variable_names = self.variable_names)
        raw_edf, events1 = load_gigadata(MI, self.variable_names)

        return raw_edf, events1