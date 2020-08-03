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

import os
import cPickle as pickle
from numpy import ones, outer, sum, isscalar, squeeze, array
import pylab as p
import Utils


class LanguageModel():
    
    delete_symbol = '<'
    
    def __init__(self, file_name, head_factors=[1.0, 0.9, 0.8, 0.6, 0.5], letter_factor=0.01, n_pred=2):
        self.file_name = file_name
        self.head_factors = head_factors
        self.letter_factor = letter_factor
        self.n_pred = n_pred
        self.load_mat_file()
        self.create_symbol_list()
        self.create_other_variables()
        
        
    def load_mat_file(self):
        f = open(self.file_name,'rb')
        self.file_content = pickle.load(f)
    
    def create_symbol_list(self):
        """ Get all the individual characters and store them in self.symbol_list, which is a list of lists. Each
        sublist contains five characters and there will be six sublists in symbol_list. """
        all_chars = self.file_content['charset'][0]
        self.nr_chars = len(all_chars)
        self.char_set = all_chars
        self.char_set_list = []
        for s in self.char_set:
            self.char_set_list.append(s)

        symbol_list = []
        sub_list = []
        symbol_list.append(sub_list)
        for i, c in enumerate(all_chars):
            sub_list.append(c)
            if (i+1) % 5 == 0:
                sub_list = []
                symbol_list.append(sub_list)
                
        self.symbol_list = symbol_list
    
    def create_other_variables(self):
        # create the word look-up tables
        self.head_table = self._create_word_table(self.file_content['head_table'])
        self.pred_table = self._create_word_table(self.file_content['pred_table'])
        #  create and normalize the probability tables
        self.head_prob = self._normalize_probability_tables(self.file_content['head_prob'])
        self.pred_prob = self._normalize_probability_tables(self.file_content['pred_prob'])
        
    def _create_word_table(self, ascii_table):
        word_table = []
        ascii_table = squeeze(ascii_table)
        for matrix in ascii_table:
            table = []
            if matrix.size > 0:
                for row in matrix:
                    if isscalar(row):
                        table.append(chr(row))
                    else:
                        s = ''
                        for elem in row:
                            s = s + chr(elem)
                        table.append(s)
            word_table.append(table)
        return word_table

    def _normalize_probability_tables(self, tables):
        n_tables = []
        tables = squeeze(tables)
        vec = squeeze(array(tables[0]))
        n_tables.append(vec / float(vec.sum()))
        for i in range(1,len(tables)):
            M = tables[i]
            M = M / outer(ones(self.nr_chars), sum(M,0))
            n_tables.append(M)
        return n_tables
        
    
    def get_probabilities(self, spelled_text):
        """ Returns a probability distribution over the character set, based on the spelled_text. """
        # find the beginning of the last written word and store it in word
        words = spelled_text.split('_')
        word = words[-1] # word after the last '_'
        word_length = len(word)
        # hp - probability based on "head", i.e. the half complete word from the beginning on
        if word_length < len(self.head_table):
            # find the index of the word in the look-up table
            word_list = self.head_table[word_length]
            index = self._find_word_index(word, word_list)
        else:
            index = None
        if index == None:
            # if the word was not found in the look-up table of if it is already too long, 
            # then we can't make a prediction based on it, hence we use the priors only
            hp = self.head_prob[0]
        else:
            hp = self.head_prob[word_length][:,index]
        self.hp = hp
        # partial predictive match based on the k previous letters, with k being either self.n_pred or less if the word is too short
        k = min(word_length, self.n_pred)
        # try to find the combination of the k previous letters in the lookup table
        index = None
        while index==None and k > 0:
            word_list = self.pred_table[k]
            k_prev_letters = word[-k:]
            index = self._find_word_index(k_prev_letters, word_list)
            k = k - 1
        if index == None or k==0:
            # the k_prev_letter combination could not be found in the table
            pp = self.pred_prob[0]
        else:
            pp = self.pred_prob[k+1][:,index]
        self.pp = pp
        hf = self.head_factors[min(word_length, len(self.head_factors)-1)]
        prob = hp*hf + pp*(1-hf)
        if k > 0:
            prob = self.pred_prob[0]*self.letter_factor + prob*(1-self.letter_factor)
        return prob
    
    def update_symbol_list_sorting(self, spelled_text):
        """ 
        Update the sorting of symbols in the symbol list according to the current probability distribution which depends 
        the on currently spelled text. 
        """
        probs = self.get_probabilities(spelled_text)
#        probs = ones(29)/29.0
        self.probs_list = []
        symbol_list = Utils.copy_list(self.symbol_list)
        start = 0
        for i, sub_list in enumerate(symbol_list):
            stop = min(start + len(sub_list), len(probs))
            values = Utils.array_to_list(probs[start:stop])
            self.probs_list.append(Utils.copy_list(values))
            symbol_list[i] = Utils.sort_list_according_to_values(sub_list, values)
            start = stop
        # add the back symbol
        symbol_list[-1].append(self.delete_symbol)
        return symbol_list
    
    def get_symbol_list(self):
        return self.update_symbol_list_sorting('')
    
    def get_most_probable_symbol_sublist_index(self):
        """ Goes through the probability list that corresponds to the sorted symbol list and returns the index
        of the sublist that has the highest max probability value, i.e. the sublist containing the most likely
        next letter. """
        idx = 0
        max_prob_value = -1
        for i,prob_sub_list in enumerate(self.probs_list):
            sub_list_max = Utils.max_with_idx(prob_sub_list)[0]
            if sub_list_max > max_prob_value:
                max_prob_value = sub_list_max
                idx = i
        return idx
    
    def get_symbol_index(self, symbol):
        """ If the given symbol is in the original symbol list from the language model file, then its index in that list 
        is returned. Otherwise None will be returned. """
        try:
            idx = self.char_set_list.index(symbol)
        except:
            idx = None
        return idx
    
    def _find_word_index(self, word, word_list):
        """ A little helper method that returns the index of word in word_list. If word is not contained in word_list, None will
        be returned. """
        try:
            idx = word_list.index(word)
        except ValueError:
            idx = None
        return idx
    
if __name__ == "__main__":
    base_dir = 'D:\svn\pyff\src\Feedbacks\HexoSpeller\LanguageModels'
#    file_name = 'german.pckl'
#    file_name = 'german.mat'
#    file_name = 'lm1to8.pckl'
    file_name = 'english_all.pckl'
    file_path = os.path.join(base_dir, file_name)
    lm = LanguageModel(file_path)
#    print lm.symbol_list
#    print sum(lm.head_prob[0])
#    print n.sum(lm.head_prob[1],0)
    
    p.figure()
    to_be_spelled = 'ICH_BIN'
    to_be_spelled = 'LANGUAGE'
    word = ''
    for i,letter in enumerate(to_be_spelled):
        probs = lm.get_probabilities(word)
        print lm.update_symbol_list_sorting(word)
        p.subplot(len(to_be_spelled),1, i+1)
        p.plot(probs)
        p.plot(lm.hp)
        p.plot(lm.pp)
        p.legend(('probs', 'hp', 'pp'))
        p.ylabel(word)
        p.xticks(range(29), lm.char_set_list)
        word = word + letter
        
    p.show()
    
    
 
