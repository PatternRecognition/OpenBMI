import pygame 
import numpy
import time
import sys,os
import random
#import Object
from conversion import ncr_to_python
from conversion import ucn_to_python

class MakeMix():
    
    def init(self):
        pass

    def make_dictionary(self, lessons):
        """ makes a dictionary from a given file"""
        dictionary = []
        for index in lessons:
    
            file = ''.join(['lektion_', str(index), '.txt'])
            datei = open(file,'r')
            for zeile in datei.readlines():
                # get rid off \n at end of line
                zeile = " ".join(zeile.split("\n"))
                zeile = " ".join(zeile.split(";"))
                # append the difficulty level = nr. of strokes
                zeile = "\t".join([zeile, str(index)])
            #zeile = zeile.split("\t")
            #zeile[0] = ncr_to_python(zeile[0])
            #zeile[1] = "".join(zeile[1].split(" "))
                dictionary.append(zeile)
                #print(zeile)
            datei.close()
        #print(len(dictionary))
            # extra symbols 
        file = 'extra_symbols.txt'
        datei = open(file,'r')
        for zeile in datei.readlines():
                # get rid off \n at end of line
            zeile = "".join(zeile.split("\n"))
                #zeile = " ".join(zeile.split(";"))
                # append the difficulty level = nr. of strokes
                #zeile = "\t".join([zeile, str(index)])
            #zeile = zeile.split("\t")
            #zeile[0] = ncr_to_python(zeile[0])
            #zeile[1] = "".join(zeile[1].split(" "))
                #print(zeile)
            dictionary.append(zeile)
        datei.close()
        #print(len(dictionary))
        #print(dictionary)
        numpy.random.shuffle(dictionary)
        print(len(dictionary))
        print(pygame.size(dictionary))
        #for i in range(8):
       #     datei = open(''.join(['lektion_mixed_', str(i+1), '.txt']), 'w')
       #     for pair in dictionary[((i)*30):((i+1)*30)]:
                #if (i==0):
                    #print pair
                #print type(pair)
       #         datei.write(pair)
       #         datei.write('\n')
       #     datei.close()

if __name__ == '__main__':
    #vc = VocabularyDeveloperFeedback(None)
    #vc.on_init()
    #vc.on_play()
    m = MakeMix()
    m.make_dictionary(range(1,18))
    print 'Done'       
         