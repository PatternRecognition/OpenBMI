'''
This is a Feedback class which shows a pair of vocabulary which the
 subject should learn.
'''
#DEBUG: 1. problem with present_word [11.1.11,18.82] solved [12.1.11,]
#       2. while training/presenting/learning, two words are concatenated at
#          space. probably due to ''.join statement. [13.1.11, 00.30] 
#                                                    fixed [13.1.11, 17.42]
#       3. shuffle the lectures into different txt files [13.1.11, 00.30]
#            fixed [13.1.11, 17.42]
#       4. in distractor presentation may be also check for distractor gap!!
#            [13.1.11, 00.30]
#       5. matlab does'nt detect errors from python while writing files (is this
#            general?)
#       6. convert all to lowercase and then compare
#       7. show distractors before asking them
#       8. minimum bag size equal to distractor size?equal to distractor size+1?
#       9. store all initializations and final result? probably give some 
#          immediate statistics too
import pygame 
import numpy
import time
import sys,os
import random
from pygame.locals import *
from conversion import ncr_to_python
from conversion import ucn_to_python


from FeedbackBase.MainloopFeedback import MainloopFeedback


class VocabularyDeveloperFeedback(MainloopFeedback):
    """Feedback for showing pairs of vocabulary"""
    
    # Markers written to parallel port

    INIT_FEEDBACK = 200
    GAME_STATUS_PLAY = 20
    GAME_STATUS_PAUSE = 21
    GAME_OVER = 254
    WELCOME = 50
    FIXATION_PERIOD = 101
    PRESENTATION_PERIOD = 102
    INTER_PERIOD = 103
    FIXATION_PERIOD_TEST = 201
    PRESENTATION_PERIOD_TEST = 202
    INTER_PERIOD_TEST = 203
    TESTING = 150
    TESTING_CORRECT = 151
    TESTING_INCORRECT = 152
    TRAINING = 160
    TRAINING_CORRECT = 161
    TRAINING_INCORRECT = 162
    ANSWER_CORRECT = 171
    ANSWER_INCORRECT = 172
    
    DISTRACTOR = 180 # newly added.. so as to not analyze distractor trials
    
    MATH_TEST = 190
    NEW_PAIR = 30
    KNOWN_PAIR = 31 
    TEST = 32 
    POSSIBLE_LEARNING = 111
    UPDATE = 112
    
    def init(self):
        """Called at the beginning of the Feedback's lifecycle.
        
        More specifically: in Feedback.on_init().
        """
        self.logger.debug("on_init")
        self.send_parallel(self.INIT_FEEDBACK)
        
        # time before showing
        self.fixation_time = 1.25
        # time after key pressing
        self.inter_time = 0.25
        self.time_b4_enter_press = self.inter_time
        self.fixation_time_test = 1.25
        self.inter_time_test = 0.25
        self.user_feedback_time = 1
        self.fullscreen =  False
        #self.fullscreen =  True
        self.screenWidth =  1000
        self.screenHeight =  700
        self.backgroundColor = (255,255,255)
        self.cursorColor = (0,0,0)
        self.fontColor = self.cursorColor
        
        self.part = 1
        #print 'self.part'+str(self.part)
        self.VP = 'test'
        # create a directory of VP if it does'nt exist
        # is(dir),if not then mkdir() etc etc
        self.lection_path = os.path.dirname(sys.modules[__name__].__file__)+'/'
        self.learnt_words = 0
        #self.st_path = 'D:/data/bbciRaw/'
        self.st_path = 'C:/bbci_data/bbciRaw/'
        #self.store_path = ''
        self.store_path = ''.join([self.st_path, self.VP, '/'])
        #self.store_path = '/home/sophie/Dokumente/HiWiJob/Data/'
  
        self.f = 0
        
        self.finnish = False
        self.multiple_choice = False
        
        # here is stored how often they have been remembered correctly
        self.nr_of_max_reps = 3
        self.nr_of_correct_reps = 2
        self.max_dict_length = 60 # probably need to be removed for the new paradigm!!!
        self.nr_of_words_2_b_shown = 4
        self.distractor_size = 2 # is this enough?? or would 5-6 gap be good
        self.tot_words = 260 # total number of words
        
        self.bag = []
        self.distractor_bag =[]
        self.distrctDict = []
        self.dict_indices = [] # to be read from the dictionary, 3rd column
        self.distractor_file = 'lektion_distractor.txt'
               
        self.minBagSize = self.distractor_size # else 5??
        
        
        self.initial_bag_size = 4
        self.bag_size = self.initial_bag_size
        self.bag_filling_index = 0
        self.showed_sequence = []
        self.asked_sequence = []

        self.maths_filename = 'maths.txt'
        self.maths_questions = self.make_maths_questions(''.join\
                                    ([self.lection_path, self.maths_filename]))
        self.Filenames = ['lektion_simple_1.txt'] 
        self.distrctDict = []
        if self.finnish:
            self.Filenames = ['Lektion_Finnisch_Kommunikation.txt']
            
        
        
    def pre_mainloop(self): #check with Sophie (parts have changed so be careful)
        print 'store_path'+str(self.store_path)
        # create logfile with run time for future reference
        localtime=time.localtime()
        self.logfilename = 'logfile_'+str(localtime[0])+'_' \
                                     +str(localtime[1])+'_' \
                                     +str(localtime[2])+'_' \
                                     +str(localtime[3])+'_' \
                                     +str(localtime[4])+'_' \
                                     +str(localtime[5]) \
                                     +'.txt'
        #self.store_training_file=
        #self.store_test_file=
        self.store_logfile = ''.join([self.store_path, self.logfilename])
        [self.distrctDict,self.distractor_bag] = self.make_dictionary(''.join\
                                    ([self.lection_path, self.distractor_file]))
        # chnge the make_dictionary to include indices as well
        # to obtain the necessary indices for the distractor bag for each part
        if not(self.part==3) and not(self.part==6) and not(self.part==7):
            
            if self.part>3 and self.part<6:
                self.part = self.part - 2
            self.distr_show = []
            self.distractor_bag = range((self.part-1) * 5 + 241, (self.part-1) \
                                        * 5 + 246)

        """Called before entering the mainloop, e.g. after on_play."""
        self.init_pygame()
        self.init_graphics()
        if ((self.part == 2) and not(self.finnish)):
            self.Filenames = ['lektion_simple_2.txt'] 
        if ((self.part == 4) and not(self.finnish)):
            self.Filenames = ['lektion_simple_3.txt'] 
        if ((self.part == 5) and not(self.finnish)):
            self.Filenames = ['lektion_simple_4.txt']
            
        if ((self.part == 3) and not(self.finnish)):
            self.part_final = self.part
            self.Filenames = ['lektion_simple_2.txt', 'lektion_simple_2.txt']             
       
        if ((self.part == 6) and not(self.finnish)):
            self.part_final = self.part
            self.part = 3
            #self.Filenames = ['lektion_mixed_old_5.txt', 'lektion_mixed_old_6.txt', 
            #                  'lektion_mixed_old_7.txt', 'lektion_mixed_old_8.txt']
            self.Filenames = ['lektion_simple_3.txt','lektion_simple_4.txt']
        if ((self.part == 7) and not(self.finnish)):
            self.part_final = self.part
            self.part = 3
            self.Filenames = ['lektion_simple_1.txt','lektion_simple_2.txt',
                              'lektion_simple_3.txt','lektion_simple_4.txt']
        self.send_parallel_and_write(self.GAME_STATUS_PLAY)
    
    
    def post_mainloop(self):
        """Called after leaving the mainloop, e.g. after stop or quit."""
        self.logger.debug("on_quit")
        self.send_parallel_and_write(self.GAME_OVER)
        pygame.quit()
            
    def tick(self):
        """
        Called repeatedly in the mainloop no matter if the Feedback is paused
        or not.
        """
        self.process_pygame_events()
        pygame.time.wait(10)
        #self.elapsed = self.clock.tick(self.FPS)
        pass
    
    def pause_tick(self):
        """
        Called repeatedly in the mainloop if the Feedback is paused.
        """
        self.do_print("Pause", self.fontColor, self.size / 4)
        print 'in pause'
    
    def play_tick(self):
        """
        Called repeatedly in the mainloop if the Feedback is not paused.
        """
        
        if(self.part != 3):
        
            if (self.part == 1):
                self.welcome()
            if (self.part > 4):
                numpy.random.shuffle(self.maths_questions)
            
            tmp_index = (self.part -1) * 4 * len(self.Filenames)
            tmp_index = tmp_index % len(self.maths_questions)
            
            for filename in self.Filenames:

                self.init_lection2(''.join([self.lection_path, filename]))
                # self.dictionary also changes in this function (ask Sophie)
                self.training_oneShw()
                self.store_training(filename)
            
                answers = self.ask_maths_questions(self.maths_questions\
                                                   [tmp_index:(tmp_index + 4)])
                self.store_maths_testing(filename, answers)
                tmp_index += 4
                tmp_index = tmp_index % len(self.maths_questions)
            
                answer_array, test_dict = self.testing()
                self.store_testing(filename, answer_array, test_dict)
                

        # here all words are asked at the very end
        else:
            final_dict = []
            for filename in self.Filenames:
                self.init_lection2(''.join([self.lection_path, filename]))
                final_dict.extend(self.dictionary)
            self.dictionary = final_dict
            answer_array, test_dict = self.testing()
            self.store_testing('final'.__add__(str(self.part_final)), \
                               answer_array, test_dict)
        
        self._running = False
    
 
        
    def init_lection2(self, filename):
        """sets up the parameters which are used for each lection"""
        # here the word pairs are stored
        # finnish
        if self.finnish:
            self.dictionary = self.make_finnish_dictionary(filename)
        # chinese
        else: # needs to read dictionary indices too in the third column of the text files
            [self.dictionary, self.dict_indices] = self.make_dictionary(filename)
        # create separate dictionary for distractors
        # reading absolute indices required
        #self.distrctDict = self.dictionary[self.distrInd[0]:self.distrInd[1]]
#        if (len(self.dictionary) > self.max_dict_length): # for new paradigm this probably does'nt matter
#            self.dictionary = self.dictionary[0:self.max_dict_length]   
        
            
        # change the length of dictionary size
        self.array_correct = numpy.zeros((self.tot_words, self.nr_of_max_reps))
        # this stores the trial number
        self.array_trial = numpy.zeros((self.tot_words, self.nr_of_max_reps))
        
        # this is the array which stores the reaction times
        self.array_RT = numpy.zeros((self.tot_words, self.nr_of_max_reps))
        self.array_RT_1st_press = numpy.zeros((self.tot_words, self.nr_of_max_reps))

        self.list_correct = numpy.zeros(len(self.dictionary))        
        
        # here is stored which pairs should be called
        self.bag_size = self.initial_bag_size
        if (self.bag_size > len(self.dictionary)):
            self.bag_size = len(self.dictionary)
            
        self.bag = self.dict_indices[0:self.bag_size] # need to be replaced with the index of the word
        # probably extract into the dictionary itself
        
        # here is stored which is the next index of the dictionary 
        #the bag can be filled with
        self.bag_filling_index = self.bag_size
        #self.sequence = list(numpy.arange(len(self.dictionary)))
        #self.ask_sequence = make_initial_ask_sequence()
        self.showed_sequence = []
        self.asked_sequence = []
        #self.index = 0
        # this is the index where the pair which has not been remembered should be inserted in the sequence
        #self.incorrect_index = len(self.dictionary)
        #self.answers = []
        
    def update_bag(self, pair_to_remove):
        """removes the pair out of the bag and puts a new one in"""
        # NOTE: to remove the same element from both bad and bad_index
        #        1. reassign bag_index to bag, it'll be shuffled anyway
        #        2. search for the common element and remove based on host index
        
        # Using method 1: Test if this works. But still distractors are not bein
        # accessed
        self.bag_index = []
        self.bag_index = self.bag[:]
        
        self.bag.remove(pair_to_remove)
        self.bag_index.remove(pair_to_remove)
        if (self.bag_filling_index < len(self.dictionary)):
            self.bag.append(self.dict_indices[self.bag_filling_index]) # need to be replaced by self.dict_indices
            self.bag_index.append(self.dict_indices[self.bag_filling_index])
            
            tmp=self.dictionary[self.bag_filling_index]
            self.show_pair(tmp[0:2])
            # store which words have been shown
            self.showed_sequence.append(self.dict_indices[self.bag_filling_index]) #list to store shown words
            self.send_parallel_and_write(self.UPDATE, self.dict_indices[self.bag_filling_index],'Update')
            # no words have been asked here
            self.asked_sequence.append(-1)
            self.trial = self.trial+1;
            self.bag_filling_index += 1
            
        self.learnt_words += 1
        print '+++++++++++++++++++++++++++++++++++++++++++++++++'
        print 'Learnt Words: '+str(self.learnt_words)+'/'+str(60)
        print '*************************************************'
    def training_oneShw(self):
        """ here all words are shown once and then only asked. upon wrong response, the
            correct word is then shown: thereby learning"""
        # Is this right? or should a limited number of words only be showed every time??
        # Each lecture has 60 words in total and per day 120 words are learnt.
        # All 60 need to be learnt in one go??
        # Assuming above case, in the code below, once the bag shrinks below the minimum size of
        #   bag, the new words are added from the dictionary
        # Also note that the showed and asked sequence is the same
        # NOTE: In this paradigm the exchanged word is not shown explicitely. It is asked after
        #        the exchange directly. Is this alright???
        #training_done = False
        #exchange = False
        self.send_parallel_and_write(self.TRAINING, task='Training')
        self.do_print('Training')
        time.sleep(self.inter_time)
        self.trial = 0;
        
        # show all words in the bag
        self.bag_index = self.bag[:]
        for pair_index in self.bag_index:# note indices of words are not yet included
            # NOTE: pair index is the absolute index of words
            self.logger.debug("new pair is shown")
            self.send_parallel_and_write(self.NEW_PAIR, pair_index, 'present')
            #self.store_training_parallel(pair_index, answer=-1,rxn=0,rxn_1press=0)
            #temp_ind = self.bag.index(pair_index)
            # NOTE: since dictionary is not changing with bag, the temp_ind
            #       refers to the first five elements only. This needs to be
            #       updated as well. Sol: use (n-1)*60+1 (remove the 1 for py)
            #       then abs word number from bag can be used.
            #tmp = self.dictionary[temp_ind]
            tmp = self.dictionary[self.dict_indices.index(pair_index)]
            self.show_pair(tmp[0:2])
            # store which words have been shown
            self.showed_sequence.append(pair_index) #list to store shown words
            # no words have been asked here
            self.asked_sequence.append(-1)
            self.trial = self.trial+1;
        # training without distractors i.e., until bag size is big enough without distractors
        print '+++++++++++++++++++++++++++++++++++++++++++++++++'
        print 'Learnt Words: '+str(self.learnt_words)+'/'+str(60)
        print '*************************************************'
        
        while (len(self.bag_index) > self.minBagSize):# this maintains a constant bag size else #self.distractor_size): 
            # self.distractor_size defines the minimum distance between word repetitions
            # once the bag falls below this levels a suitable no of distractors need to be added
            # QUESTION: show there be new distractor once the bag size goes down??
            
            # randomize the bag and obtain the first word index
            # QUESTION: Would randomization be more useful than ordered presentation??
            #            Although here randomization is used!!
#            print 'self.array_trial= '+str(self.array_trial)
#            print 'self.array_correct= '+str(self.array_correct)
#            print 'self.bag_index= '+str(self.bag_index)
            
            numpy.random.shuffle(self.bag_index)
            rand_index = self.bag_index[0]
#            print 'self.bag_index= '+str(self.bag_index)
#            print 'self.bag= '+str(self.bag)
            print 'show_seq= '+str(self.showed_sequence)
            # check if the word is not repeated for at least a gap of distractor_size
            presentation = self.verify_gap(rand_index)
#            print presentation
            # if N-x'th recall condition verified then finally ask the word
            if presentation:                
                temp_index = self.bag.index(rand_index)
                self.present_word(temp_index)
            #print 'BAG-SIZE is '+str(len(self.bag_index))
            #print '+++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'Learnt Words: '+str(self.learnt_words)+'/'+str(60)
            #print '*************************************************'
        # training with distractors i.e., when bag size has shrunk below minBagSize
        while (len(self.bag_index) > 0) and (len(self.bag_index) <= self.minBagSize):
            # read the distractors into a separate dictionary
#            print '************************************************************'
#            print 'DISTRACTOR SEQUENCE ENTERED :)'
#            print '************************************************************'
            # then proceed like above
            # go through the bag number of times == length of the bag
            # or just put a if else clause
            for _ in range(len(self.bag_index)):
                numpy.random.shuffle(self.bag_index)
                rand_index = self.bag_index[0] #call this pair or abs_index
                # check if the word is not repeated for at least a gap of distractor_size
                presentation = self.verify_gap(rand_index)
                if presentation:
                    #distractor = False
                    temp_index = self.bag.index(rand_index)
                    self.present_word(temp_index)
                    print 'BAG-SIZE is '+str(len(self.bag_index))
                    break
#            print 'self.bag_index= '+str(self.bag_index)
#            print 'self.bag= '+str(self.bag)
#            print 'show_seq= '+str(self.showed_sequence)
#            print 'presentation is '+str(presentation)
                
            if not(presentation):
#                print 'Ahoy Matie, Captain Distractor Sir'
                # if none of the cycles above gave a positive presentation, then
                # use a distractor
                
                # show all distractor first: for every distractor loop only one
                if len(self.distr_show)< len(self.distractor_bag):
                    rand_index = len(self.distr_show)
                    self.logger.debug("distractor is asked")
                    self.send_parallel_and_write(self.DISTRACTOR, \
                                                 self.distractor_bag[rand_index], \
                                                 'distractor')
                    tmp = self.distrctDict[rand_index]
                    self.show_pair(tmp[0:2])
                    self.distr_show.append(self.distractor_bag[rand_index])
                    self.trial=self.trial+1
                    
                    # store in arrays
                    self.showed_sequence.append(self.distractor_bag[rand_index])
                    self.asked_sequence.append(self.distractor_bag[rand_index])
                    
                else:
                    rand_index = numpy.random.randint(0,len(self.distractor_bag))
                    #numpy.random.shuffle(self.distractor_bag)
                    #rand_index = self.distractor_bag[0]
                    presentation = True
                    #distractor = True
                    
                    # present the distractor (NOTE: distractors were not presented earlier)
                    #    or should the distractors be presented along with the training bag
                    self.logger.debug("distractor is asked")
    #                temp_ind = 
                    self.send_parallel_and_write(self.DISTRACTOR, \
                                                 self.distractor_bag[rand_index], \
                                                 'distractor')
                    _=self.ask_answer(self.distrctDict[rand_index], \
                                             self.distractor_bag[rand_index])
                    # '_' is a temporary variable
                    self.trial=self.trial+1
                    
                    # store in arrays
                    self.showed_sequence.append(self.distractor_bag[rand_index])
                    self.asked_sequence.append(self.distractor_bag[rand_index])
                    # for distractors do we need to store the results???
    #                self.array_correct[self.distractor_bag[rand_index]][ind] = -1.0
    #                self.array_trial[self.bag[rand_index]][ind] =  self.trial
                    print 'BAG-SIZE is '+str(len(self.bag_index))
            #print '+++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'Learnt Words: '+str(self.learnt_words)+'/'+str(60)
            #print '*************************************************'
                
    def present_word(self,rand_index):
        """ tests a given pair and stores the results in respective arrays"""
        # rand_index is temp_index from calling function: refers to the index of words in bag 
        # Problem in this section [11.1.11,18.82]
        # NOTE: Also the naming of variable rand_index is confusing. Need to 
        #       change.
        exchange = False
        word_index = self.bag[rand_index]-1 # array_correct, trial, RT have 
                                            # tot_words x max_rep size, hence
                                            # word_index refers to absolute word
                                            # index. Note: the inidices of the
                                            # above arrays are abs indices 
                                            # already
        pair_index = self.bag[rand_index]
        
        self.logger.debug("test is asked")
        self.send_parallel_and_write(self.TEST, self.bag[rand_index], task='test')
        
        answer = self.ask_answer(self.dictionary[self.dict_indices.index(pair_index)], self.bag[rand_index])
        # send markers 
        if answer[0]:
            self.send_parallel_and_write(self.TRAINING_CORRECT, self.bag[rand_index])
        else:
            self.send_parallel_and_write(self.TRAINING_INCORRECT, self.bag[rand_index])
        
        self.trial = self.trial+1; #update the trial number
        # stores which pair has been asked NOTE: showed and asked sequence is the same
        # except for initial all show presentation
        self.showed_sequence.append(self.bag[rand_index])
        self.asked_sequence.append(self.bag[rand_index])
        # stores the reaction time
        # NOTE: 1.rand_index refers to 
        ind = list(self.array_correct[word_index]).index(0.0)
        self.array_RT[word_index][ind] = answer[2]
        self.array_RT_1st_press[word_index][ind] = answer[3]
        
        # storing answer in array_correct and array_trial
        if answer[0]:
            # if the answer has been correct 1 is stored
            # NOTE:1.problem with passing rand_index since it is temporary
            #      2.also bag index cannot be used either since the bag contents
            #        can change
            #      3.hence best use an index that refers to word index in 
            #        array_correct and array_trial
            self.array_correct[word_index][ind] = 1.0
            self.array_trial[word_index][ind] =  self.trial
            # if the answer was often enough correct
            # the word is thrown out of the bag
            if (ind >= (self.nr_of_correct_reps - 1)):
                exchange = True
                for i in range(self.nr_of_correct_reps):
                    if (self.array_correct[word_index][ind - i] == -1.0):
                        exchange = False
        
                if exchange: # need to look into how this update works as well
                    self.update_bag(self.bag[rand_index])
        
        else:
            # if the answer has not been correct -1 is stored
            self.array_correct[word_index][ind] = - 1.0
            self.array_trial[word_index][ind] =  self.trial
        
            
        # if the word was asked for often enough it gets also thrown out of the bag
        # or if there's only one word left, the lesson is finished as well
        #
        if (not(exchange) and (ind == (self.nr_of_max_reps - 1))):
            self.update_bag(self.bag[rand_index])
        
    def verify_gap(self,rand_index):
        """ returns 'True' if the word is not repeated for at least a gap of 
            distractor_size"""
        value = list(cmp(rand_index,varble) for varble in self.showed_sequence\
                     [len(self.showed_sequence)-self.distractor_size:])
#        print 'value= '+str(value) 
#        decision_var = sum(value)
#        temp = 1 in value
        if 0 in value:#decision_var == -1 * self.distractor_size:
            return False
        else:
            return True    
   
        
    def show_pair(self, pair, first_seen=False):
        """
        shows one pair
        """
        # Fixation Part
        self.logger.debug("Fixation period started")# what is this fixation period??
        self.send_parallel_and_write(self.FIXATION_PERIOD, task='show')
        time.sleep(self.fixation_time)
        
        # Presentation Part
        self.logger.debug("Presentation period started")
        self.send_parallel(self.PRESENTATION_PERIOD )
        # shows the sign and the word
        if first_seen:
            self.do_print([pair[0], '', "Press enter to continue"], 
                      size_list=[None, None, 20], center=0)
        
            # time.sleep(self.presentation_time)
            self.wait_until_enter()
        
            self.do_print([pair[1], '', "Press enter to continue"], 
                      size_list=[None, None, 20], center=0)
            self.wait_until_enter2()
            #time.sleep(self.presentation_time)
            # show both words together in relation
        else:
            self.do_print([" - ".join(pair), '', "Press enter to continue"], 
                      size_list=[None, None, 20], center=0)
            self.wait_until_enter2()
 
        #time.sleep(self.presentation_time)
        
        # Inter Part
        self.logger.debug("Inter period started")
        self.send_parallel(self.INTER_PERIOD)
        self.do_print('')
        time.sleep(self.inter_time)
       
    def ask_test(self, pair, index):
        """
        shows one pair
        """
        # Fixation Part
        self.logger.debug("Fixation period of test started")
        self.send_parallel(self.FIXATION_PERIOD_TEST)
        time.sleep(self.fixation_time_test)
        
        # Presentation Part
        self.logger.debug("Presentation period of test started")
        self.send_parallel(self.PRESENTATION_PERIOD_TEST)
        test_list = ['TEST:', 'What is', pair[0]]
        answer_list = [pair[1]]
        #answer_list = [pair[2]]
        next_4_elems = []
        for _ in range(4):
            elem_index = random.randint(0, numpy.shape(self.dictionary)[0]-1)
            while ((elem_index in next_4_elems) or (elem_index == index)):
                elem_index = random.randint(0, numpy.shape(self.dictionary)[0]-1)
            next_4_elems.append(elem_index)
            answer_list.append(self.dictionary[elem_index][1])
            #answer_list.append(self.dictionary[elem_index][2])
        random.shuffle(answer_list)
               
        showed_list = []
        showed_list.extend(answer_list)
        showed_list.insert(0, '')
        showed_list.insert(0, ' '.join([pair[0], ':']))
        showed_list.append('no idea')
        self.do_print(test_list, size=40)
        time.sleep(self.fixation_time_test)
        self.do_print(showed_list, size=30)
        #time.sleep(self.fixation_presentation_test_max)
        done = False
        time1 = time.time()
        time2 = 0
        answer = 0
        while not done:
            for event in pygame.event.get():
                if ((event.type == pygame.KEYDOWN) or (event.type == pygame.KEYUP)):
                    _ = pygame.key.name(event.key)
                    if (_ in ['1', '2', '3', '4', '5', '6']):
                        answer = int(_)
#                   if ((answer == 6) or (answer_list[answer - 1] != pair[2])):
                    if ((answer == 6) or (answer_list[answer - 1] != pair[1])):
                        correct = False
                    else: correct = True
                    done = True
                    time2 = time.time()
        
        self.do_print('')           
        time.sleep(self.inter_time_test)
        
        # This is the direct feedback for the user
        if correct:
            self.do_print('The answer has been correct', size=30)
            time.sleep(self.user_feedback_time)
        else:
#            self.do_print(['The answer has not been correct:', '', pair[0], 'means', pair[2]], size=30)
            self.do_print(['The answer has not been correct:', '', \
                           " - ".join(pair), "Press enter to continue"], 
                          size_list=[30, 30, None, 20], center=2)
            self.wait_until_enter()
        #time.sleep(self.user_feedback_time)
        
        
        # Inter Part
        self.logger.debug("Inter period of test started")
        self.send_parallel(self.INTER_PERIOD_TEST)
        self.do_print('')
        time.sleep(self.inter_time_test)
        
        # returns whether the answer has been correct and at which position the 
        # correct word has been written
        position = 1 + answer_list.index(pair[1])
        reaction_time = time2 - time1       
        return [correct, position, reaction_time]
       
    def ask_answer(self, pair, index, give_feedback = True):
        """asks for the answer"""
        
        # Fixation Part
        self.logger.debug("Fixation period of test started")
        self.send_parallel(self.FIXATION_PERIOD_TEST)
        time.sleep(self.fixation_time_test)
        
        # Presentation Part
        self.logger.debug("Presentation period of test started")
        self.send_parallel(self.PRESENTATION_PERIOD_TEST)
        showed_item = " ".join(['What is ', pair[0], '?'])
        self.do_print(showed_item)
        time1 = time.time()
        time2 = 0
        time3 = 0
        done = False
        wait_for_1st_press = True
        answer = []
        # make sure that question is really seen
        time.sleep(self.time_b4_enter_press)
        pygame.event.clear()
        while not done:
            for event in pygame.event.get():
                if (event.type == pygame.KEYUP):
                    if wait_for_1st_press:
                        time3 = time.time()
                        wait_for_1st_press = False
                    _ = pygame.key.name(event.key)
                    # the word is finished
                    if (_ == 'return'):
                        done = True
                        time2 = time.time()
                        if ("".join(answer) == pair[1]):
                            correct = True
                            self.send_parallel(self.ANSWER_CORRECT)
                        else:
                            correct = False
                            self.send_parallel(self.ANSWER_INCORRECT)
                    # if the last entered letter shall be deleted
                    elif((_ == 'backspace') and (len(answer) > 0)):
                        answer.pop()
                    # if the entered key is alphabetic it is part of the answer
                    elif(_ == 'space'):
                        answer.append(' ')
                    elif(_.isalpha()):
                        answer.append(_) 
                    elif((_ == '-') or (_ == ',')):
                        answer.append(_)   
                    self.do_print([showed_item, "".join(answer)])
                    
        # This is the direct feedback for the user
        if give_feedback:
            if correct:
                self.do_print('The answer has been correct', color=(50,155,50),\
                                                                        size=30)
                time.sleep(self.user_feedback_time)
            else:
                self.do_print(['The answer has not been correct:', \
                            '', " - ".join(pair), "Press enter to continue"], \
                            color=(215,50,50), \
                            size_list=[30, 30, None, 20], center=2)
                self.wait_until_enter2()
                self.logger.debug("Inter period of test started")
                self.send_parallel(self.INTER_PERIOD_TEST)
                self.do_print('')           
                time.sleep(self.inter_time_test)
        
        self.do_print('')
        
        position = 1
        reaction_time = time2 - time1
        reaction_time_1st_press = time3 - time1
        return [correct, position, reaction_time, reaction_time_1st_press]
    
    def ask_maths_questions(self, questions):
        """asks some mathematical questions"""
        # Fixation Part
        self.logger.debug("Period of math test started")
        self.send_parallel_and_write(self.MATH_TEST, task='maths')
        self.do_print('Maths Testing')
        time.sleep(self.inter_time_test)
        
        m_answers = []
        
        for q in questions:
            
            pair = q.split('\t')
            # Fixation Part
            self.logger.debug("Fixation period of test started")
            self.send_parallel(self.FIXATION_PERIOD_TEST)
            time.sleep(self.fixation_time_test)
        
            # Presentation Part
            self.logger.debug("Presentation period of test started")
            self.send_parallel(self.PRESENTATION_PERIOD_TEST)
            showed_item = " ".join(['What is ', pair[0], '?'])
            self.do_print(showed_item)
            done = False
            answer = []
            correct = False
            # make sure that question is really seen
            time.sleep(self.time_b4_enter_press)
            pygame.event.clear()
            while not done:
                for event in pygame.event.get():
                    if (event.type == pygame.KEYUP):
                        _ = pygame.key.name(event.key)
#                       # the word is finished
                        if (_ == 'return'):
                            done = True
                            if ("".join(answer) == pair[1].split(' ')[0]):
                                correct = True
                                self.send_parallel(self.ANSWER_CORRECT)
                            else: 
                                correct = False
                                self.send_parallel(self.ANSWER_INCORRECT)
#                       # if the last entered letter shall be deleted
                        elif ((_ == 'backspace') and (len(answer) > 0)):
                            answer.pop()
#                       # if the entered key is alphabetic it is part of the answer
#                       #elif(_.isalpha()):
                        else:
                            answer.append(_)    
                        self.do_print([showed_item, "".join(answer)])
            self.do_print('')
            m_answers.append(correct)
        return m_answers
             
        
    
    def testing(self):
        """tests the whole dictionary"""
        # makes a deep copy of dictionary and shuffles it   
        test_dictionary = []
        temp_indices = range(len(self.dict_indices))
        test_dictionary.extend(self.dictionary) 
#        test_dictionary=[]
        random.shuffle(temp_indices)
        #check that the first item which is asked is not the last shown one
#        if (self.part != 3):
#            while (test_dictionary[0] == self.dictionary[self.asked_sequence[len(self.asked_sequence) - 1]]):
#                random.shuffle(test_dictionary)
        #print 'shuffled'
        
        # here the answers are stored: correct? reaction time? reaction_time_1st_press
        answer_array = numpy.zeros((len(test_dictionary), 3))
        test_order = []
        index = 0
        # every pair is tested
        self.send_parallel_and_write(self.TESTING, task='testing')
        self.do_print('Testing')
        time.sleep(self.inter_time_test)
        for i in temp_indices:
            pair = test_dictionary[i]
            answer = self.ask_answer(pair, 0, False)
            if answer[0]:
                self.send_parallel_and_write(self.TESTING_CORRECT, \
                                             self.dict_indices[i])
            else:
                self.send_parallel_and_write(self.TESTING_INCORRECT, \
                                             self.dict_indices[i])
            test_order.append(self.dict_indices[i])
            answer_array[index][0] = answer[0]
            answer_array[index][1] = answer[2]
            answer_array[index][2] = answer[3]
            index += 1
        # here the percentage of the correct answers is calculated and
        # presented to the user as some feedback
        pc = list(answer_array.T[0]).count(True) * 100 / len(test_dictionary)
        self.do_print(' '.join(['Your answers were', str(pc), '% correct']), \
                                                                    size=30)
        time.sleep(self.user_feedback_time)
        return answer_array, test_order
             
       
    def wait_until_enter(self):
        """ waits until the enter button has been pressed """
        done = False
        # make sure that word is really seen
        time.sleep(self.time_b4_enter_press)
        pygame.event.clear()
        while not done:
            for event in pygame.event.get():
                if (event.type == pygame.KEYUP):
                    _ = pygame.key.name(event.key)
                    if (_ == 'return'):
                        done = True 

    def wait_until_enter2(self):
        """ waits until the enter button has been pressed """
        done = False
        # make sure that word is really seen
        time.sleep(self.time_b4_enter_press)
        pygame.event.clear()
        while not done:
            for event in pygame.event.get(): 
                if (event.type == pygame.KEYUP):
                    _ = pygame.key.name(event.key)
                    if (_ == 'return'):
                        self.send_parallel_and_write(self.POSSIBLE_LEARNING, \
                                                task='possible learnin period')
                        done = True 
       
    def on_control_event(self, data):
        self.logger.debug("on_control_event: %s" % str(data))
        self.f = data["data"][ - 1] 
#        print data
        
    def on_interaction_event(self, data):
        # this one is equivalent to:
        # self.myVariable = self._someVariable
        self.myVariable = data.get("someVariable")
#        print self.myVariable  
#        print data 
       
    def init_pygame(self):
        """
        Sets up pygame and the screen and the clock.
        """
        pygame.init()
        pygame.display.set_caption('Vocabulary Developer Feedback')
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.screenWidth, \
                                                   self.screenHeight), \
                                                   pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screenWidth, \
                                                   self.screenHeight),\
                                                    pygame.RESIZABLE)
        self.w = self.screen.get_width()
        self.h = self.screen.get_height()
        self.clock = pygame.time.Clock()
        
    def init_graphics(self):
        """
        Initialize the surfaces and fonts depending on the screen size.
        """
        self.screen = pygame.display.get_surface()
        (self.screenWidth, self.screenHeight) = (self.screen.get_width(), \
                                                 self.screen.get_height())
        self.size = min(self.screen.get_height(), self.screen.get_width())
        #self.borderWidth = int(self.size*self.borderWidthRatio/2)
        
        # background 
        self.background = pygame.Surface((self.screen.get_width(), \
                                          self.screen.get_height()))
        self.backgroundRect = self.background.get_rect\
                                (center=self.screen.get_rect().center)
        self.background.fill(self.backgroundColor)

        
    def do_print(self, text, color=None, size=None, superimpose=False, 
                 size_list=None, center=-1):
        """
        Print the given text in the given color and size on the screen.
        If text is a list, multiple items will be used, one for each list entry.
        """

        u_type = type(u'\u4e36')

        if not color:
            color = self.fontColor
        if not size:
            size = self.size/10
        font = pygame.font.Font(''.join([self.lection_path, 'Cyberbit.ttf']),\
                                 size)
        
        if not superimpose:
            self.draw_init()
        
        if type(text) is list:
            height = pygame.font.Font.get_linesize(font)
            top = -(2*len(text)-1)*height/2
            for t in range(len(text)):
                if ((size_list != None) and (size_list[t] != None)): 
                    size = size_list[t]
                else: size = self.size/10
                if (t != 0):
                    color = self.fontColor
                font = pygame.font.Font(''.join([self.lection_path,\
                                                  'Cyberbit.ttf']), size)
                tt = text[t]
                if (type(tt) != u_type):
                    tt.decode("utf-8")
                surface = font.render(tt, 1, color)
                if(t == center): 
                    self.screen.blit(surface, surface.get_rect\
                                     (center=self.screen.get_rect().center))
                else: 
                    self.screen.blit(surface, \
                                     surface.get_rect(midtop=(self.screenWidth/2,\
                                     self.screenHeight/2+top+t*2*height)))
        else:
            if (type(text) != u_type):
                    text.decode("utf-8")
            surface = font.render(text, 1, color)
            self.screen.blit(surface, surface.get_rect\
                             (center=self.screen.get_rect().center))
        pygame.display.update()
    
    def draw_init(self):
        """
        Draws the initial screen.
        """
        self.screen.blit(self.background, self.backgroundRect)
        #self.screen.blit(self.border, self.borderRect)
        #self.screen.blit(self.inner, self.innerRect)
        
        
    def process_pygame_events(self):
        """
        Process the the pygame event queue and react on VIDEORESIZE.
        """
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.resized = True
                self.size_old = self.size
                h = min(event.w, event.h)
                self.screen = pygame.display.set_mode((event.w, h), \
                                                      pygame.RESIZABLE) 
                self.init_graphics()
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                step = 0
                if event.unicode == u"a": step = -0.1
                elif event.unicode == u"d" : step = 0.1
                self.f += step
                if self.f < -1: self.f = -1
                if self.f > 1: self.f = 1
    
    def welcome(self):
        """shows welcome screen in beginning """
        
        self.logger.debug("Welcome started")
        self.send_parallel_and_write(self.WELCOME)
        
        self.do_print(["Welcome to the", "Chinese Vocabulary Developer", '', 
                       "Press enter to continue"], 
                      size_list=[None, None, None, 20], center=1)
        self.wait_until_enter()
        #time.sleep(self.presentation_time)
        
        # Inter Part - 250ms after finger tap
        self.logger.debug("Inter period started")
        self.send_parallel(self.INTER_PERIOD)
        self.do_print('')
        time.sleep(self.inter_time)
    
    def make_maths_questions(self, file):
        """ makes maths_questions from a given file"""
        questions = []
        datei = open(file,'r')
        for zeile in datei.readlines():
            # get rid off \n at end of line
            zeile = " ".join(zeile.split("\n"))
#            zeile = " ".join(zeile.split(";"))
#            zeile = zeile.split("\t")
#            zeile[0] = ncr_to_python(zeile[0])
#            zeile[1] = "".join(zeile[1].split(" "))
            questions.append(zeile)
        datei.close()
        return questions 
    
    def make_dictionary(self, file):
        """ makes a dictionary from a given file and also reads the indices"""
        dictionary = []
        indices = []
        datei = open(file,'r')
        for zeile in datei.readlines():
            # get rid off \n at end of line
            zeile = " ".join(zeile.split("\n"))
            zeile = " ".join(zeile.split(";"))
            zeile = zeile.split("\t")
            #print zeile[0]
            zeile[0] = ncr_to_python(zeile[0])
            #zeile[1] = "".join(zeile[1].split(" "))
            zeile[1] = zeile[1].rstrip()
            dictionary.append(zeile[0:2])
            indices.append(int(zeile[2]))
#            zeile[2] = "".join(zeile[2].split(" "))
            
        datei.close()
        result = [dictionary, indices]
        return result
    
    def make_initial_ask_sequence(self):
        """makes the initial asking sequence"""
        seq = [-1, 0, -1]
        for i in range(len(self.dictionary)):
            seq[i+1] = i
            i = i+1
        return seq

    # trigger, abs_index are integers, task is a string
    def send_parallel_and_write(self, trigger, abs_index=None, task=None):
    
        self.send_parallel(trigger)
        logfile = open(self.store_logfile,'a')
    
        t = time.localtime()
        logfile.write(''.join([str(list(t)), '\t']))
        logfile.write(''.join([str(trigger), '\t']))
    
        if(abs_index != None):
            logfile.write(''.join(['index:', str(abs_index), '\t']))
        if(task != None):
            logfile.write(''.join(['task:', task, '\t']))
    
        logfile.write('\n')
        logfile.close()    
        
    def store_training_parallel(self, answer=-1,rxn=0,rxn_1press=0):
        """stores the training data(sequence, array_correct, array_RT)""" 
        file = '_'.join([self.VP, 'training', self.store_training_file])
        datei = open(''.join([self.store_path, file]), "a") 
        
        _ = self.showed_sequence[-1]
        temp = [self.trial, self.showed_sequence[-1], self.asked_sequence[-1], answer, rxn, rxn_1press]
        
        datei.write(str(temp))
        datei.write('\n')
        #datei.write('\n')
             
        datei.close()
        
    def store_testing_parallel(self, filename, answer_array, test_dict):
        """stores the testing data""" 
       
        file = '_'.join([self.VP, 'testing', filename])
        datei = open(''.join([self.store_path, file]), "w") 
        
        for _ in test_dict:
            datei.write('\t' + str(_))
        datei.write('\n')
        datei.write('\n')
        
        for i in range(numpy.shape(answer_array)[0]):
            datei.writelines(str(list(answer_array[i])))
            datei.write('\n') 
            
        datei.close()
    def store_training(self, filename):
        """stores the training data(sequence, array_correct, array_RT)""" 
        file = '_'.join([self.VP, 'training', filename])
        datei = open(''.join([self.store_path, file]), "w") 
        
        for _ in self.showed_sequence:
            datei.write('\t' + str(_))
        datei.write('\n')
        datei.write('\n')
        
        for _ in self.asked_sequence:
            datei.write('\t' + str(_))
        datei.write('\n')
        datei.write('\n')
        
        for i in range(numpy.shape(self.array_correct)[0]):
            datei.writelines(str(list(self.array_correct[i])))
            datei.write('\n')
        datei.write('\n')
        
        
        for i in range(numpy.shape(self.array_RT)[0]):
            datei.writelines(str(list(self.array_RT[i])))
            datei.write('\n') 
        datei.write('\n')
            
        for i in range(numpy.shape(self.array_RT_1st_press)[0]):
            datei.writelines(str(list(self.array_RT_1st_press[i])))
            datei.write('\n') 
        datei.write('\n')
            
        for i in range(numpy.shape(self.array_trial)[0]):
            datei.writelines(str(list(self.array_trial[i])))
            datei.write('\n')

             
        datei.close()
        
    def store_testing(self, filename, answer_array, test_dict):
        """stores the testing data""" 
        print test_dict
        print answer_array
        print self.dictionary
        file = '_'.join([self.VP, 'testing', filename])
        datei = open(''.join([self.store_path, file]), "w") 
        
        for _ in test_dict:
            datei.write('\t' + str(_))
        datei.write('\n')
        datei.write('\n')
        
        for i in range(numpy.shape(answer_array)[0]):
            datei.writelines(str(list(answer_array[i])))
            datei.write('\n') 
            
        datei.close()
        
    def store_maths_testing(self, filename, answers):
        """stores the maths testing""" 
        file = '_'.join([self.VP, 'maths', filename])
        datei = open(''.join([self.store_path, file]), "w") 
        
        for _ in answers:
            datei.write(str(_) + '\n')
                
        datei.close()
        
if __name__ == '__main__':
    vc = VocabularyDeveloperFeedback(None)
    vc.on_init()
    vc.on_play()
