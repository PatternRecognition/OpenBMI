# LanguageModel.py -
# Copyright (C) 2009-2010  Sven Daehne
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import scipy.io
import os
import cPickle as pickle
from numpy import ones, outer, sum, isscalar, squeeze, array
import pylab as p
import Utils


class ConvertDataFormat():
        
    def __init__(self, file_name):
        self.file_name = file_name        
        
    def convert_mat_file(self):
        mat = scipy.io.loadmat(self.file_name)
        print mat
        f = open(self.file_name, 'wb')
        pickle.dump(mat, f)
        f.close()
        f = open(self.file_name,'rb')
        self.file_content = pickle.load(f)
        print self.file_content
        
if __name__ == "__main__":
    base_dir = 'D:\svn\pyff\src\Feedbacks\HexoSpeller\LanguageModels'
    file_name = 'german'
#    file_name = 'german.mat'
#    file_name = 'lm1to8.pckl'
    file_path = os.path.join(base_dir, file_name)
    lm = ConvertDataFormat(file_path)
#    print lm.symbol_list
#    print sum(lm.head_prob[0])
#    print n.sum(lm.head_prob[1],0)
        
    
 
