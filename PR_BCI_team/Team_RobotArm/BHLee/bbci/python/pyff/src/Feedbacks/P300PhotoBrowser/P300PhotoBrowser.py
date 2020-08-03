import math, sys, random, time
import pygame
import os
from pygame.locals import *
from numpy import *
import glutils

from p300browser import *

from FeedbackBase.MainloopFeedback import MainloopFeedback

import load_highlights

P300_START_EXP = 251
P300_END_EXP = 254
P300_COUNTDOWN_START = 70
P300_START_BLOCK = 0
P300_END_BLOCK = 81
P300_START_TRIAL = 1
P300_END_TRIAL = 2
File_Path = os.path.dirname(globals()["__file__"]) + "/highlights.csv"

 
[
                STATE_INITIAL,
                STATE_STARTING_BLOCK,
                STATE_STARTING_TRIAL,
                STATE_SUBTRIAL,
                STATE_BETWEEN_TRIALS,
                STATE_BETWEEN_BLOCKS,
                STATE_FINISHED,
                STATE_DISPLAY_IMAGE,
] = range(8)

class P300PhotoBrowser(MainloopFeedback):
        def init(self):
                """Create the class and initialise variables that must be in place before
                pre_mainloop is called"""

                self.screen_w = 1100
                self.screen_h = 700
                self.screenPos = [100, 100]
                self._data = {}
                self.online_mode = False

                self.block_count = 3
                self.trial_count = 1
                self.trial_highlight_duration = 3500
                self.trial_pre_highlight_duration = 2000
                self.trial_pause_duration = 2000
                self.subtrial_count = 60 #2160 #6
                self.stimulation_duration = 100
                self.inter_stimulus_duration = 200
                self.inter_trial_duration = 3000
                self.inter_block_duration = 15000
                self.inter_block_countdown_duration = 3000
                self.highlight_count = 6
                self.subtrials_per_frame = 6 # nr of subtrials for displaying all images

                # switch for effect
                self.rotation = True
                self.brightness = True
                self.enlarge = True
                self.mask = True

                # gobals vars
                self.row = 6
                self.col = 6
                self.max_photos = self.row * self.col
                self.rowColEval = False
                self.target_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

                self.total_subtrial_counter = 0
                self.init_scoring_matrix()

                self.viewer = None

                p = os.path.join(os.getcwd(), File_Path)
                if os.path.exists(p):
                        preloaded_highlight_indexes = load_highlights.load2(p, self.subtrial_count, self.highlight_count)
                        print "Loading highlights from highlights.csv!"
                        self.highlight_indexes = preloaded_highlight_indexes
                else:
                        self.highlight_indexes = []
                        for i in range(self.trial_count):
                                self.highlight_indexes.append([])
                                #index_list = []
                                #for x in range(self.subtrial_count/self.highlight_count):
                                #       index_list += [k for k in range(30)]
                                #random.shuffle(index_list)
                                indexes = {}
                                for j in range(self.subtrial_count):
                                        if len(indexes) == 0:
                                                for x in range(self.max_photos):
                                                        indexes[x] = x
                                        new_indexes = random.sample(indexes, self.highlight_count)
                                        for ni in new_indexes:
                                                del indexes[ni]
                                        self.highlight_indexes[i].append(new_indexes)

#               for i in range(self.trial_count):
#                       indexes = {}
#                       for j in range(self.subtrial_count):
#                               for k in self.highlight_indexes[i][j]:
#                                       if indexes.has_key(k):
#                                               indexes[k] += 1
#                                       else:
#                                               indexes[k] = 1
#
#                       for k in indexes.keys():
#                               if indexes[k] != self.highlight_count:
#                                       print "Error, value %d is shown %d times (trial %d, subtrial %d)!" % (k, indexes[k], i, j)

                self._subtrial_pause_elapsed = 0
                self._subtrial_stimulation_elapsed = 0
                self._inter_stimulus_elapsed = 0
                self._trial_elapsed = 0
                self._block_elapsed = 0

                self.MARKER_START = 20
                self.HIGHLIGHT_START = 120
                self.RESET_STATES_TIME = 100

                self._state = STATE_INITIAL

                self._current_block = 0
                self._current_trial = -1
                self._current_subtrial = 0

                self._highlighted_image = -1
                self._finished = False

                self._stimulus_active = False
                self._markers_sent = False
                self._markers_elapsed = 0
                self._last_marker_sent = 0
                self._current_highlights =  None
                self._current_highlights_index = 0
                self._last_highlight = -1
                self._display_elapsed = 0

                self.image_display_time = 5000
                self.image_display = True

                self.copy_task = True
                self._in_pre_highlight_pause = False

                self.highlight_all_selected = True
                self._subtrial_scores_received = False
                self.startup_sleep = 1

                self.udp_markers_enable = True #_udp_markers_socket error
                self._markers_reset = False # _markers_reset error
                self._skip_cycle = False
                
        def init_scoring_matrix(self):
            self.scoring_matrix = []
            if self.rowColEval:
                for i in range(self.row + self.col):
                    self.scoring_matrix.append([0.0 for x in range((self.subtrial_count / self.subtrials_per_frame)/2)])
            else:
                for i in range(self.max_photos):        # IMAGE COUNT
                    self.scoring_matrix.append([0.0 for x in range(self.subtrial_count / self.subtrials_per_frame)])             

        def pre_mainloop(self):
                self.p300_setup()
                print "Initialised!"
                #if not self.online_mode:
                #       self.send_udp(P300_START_EXP)
                time.sleep(self.startup_sleep)
                self.send_parallel(P300_START_EXP) # exp start marker
                # manually update state
                self._state = STATE_STARTING_BLOCK

        def post_mainloop(self):

                print "End of experiment!"

        #initialize opengl with a simple ortho projection
        def init_opengl(self):
                glClearColor(0,0,0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(0, self.w, 0, self.h, -1, 500)
                glMatrixMode(GL_MODELVIEW)

                #enable texturing and alpha
                glEnable(GL_TEXTURE_2D)
                glEnable(GL_BLEND)
                glEnable(GL_LINE_SMOOTH)

                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Initialise pygame, and load the fonts
        def init_pygame(self,w,h):
                pygame.init()
                os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screenPos[0], self.screenPos[1])
                default_font_name = pygame.font.match_font('bitstreamverasansmono', 'verdana', 'sans')
                if not default_font_name:
                        self.default_font_name = pygame.font.get_default_font()
                self.default_font = pygame.font.Font(default_font_name, 64)
                self.small_font = pygame.font.Font(default_font_name, 12)

                self.screen = pygame.display.set_mode((w,h) , pygame.OPENGL|pygame.DOUBLEBUF)
                #store screen size
                self.w = self.screen.get_width()
                self.h = self.screen.get_height()

                self.init_opengl()
                self.glfont = GLFont(self.default_font, (255,255,255))

        #initialise any surfaces that are required
        def init_surfaces(self):
                pass

        # init routine, sets up the engine, then enters the main loop
        def p300_setup(self):
                self.init_pygame(self.screen_w,self.screen_h)
                self.clock = pygame.time.Clock()
                self.start_time = time.clock()
                self.init_surfaces()
                self.photos = PhotoSet(self.max_photos)
                tick = False
                if self.online_mode or self.highlight_all_selected:
                        tick = True
                self.viewer = P300PhotoViewer(self.photos, self.highlight_indexes, self.w, self.h, self.row, self.col, tick)
                self.fps = 60
                self.phase = 0

                #self.main_loop()

        # handles shutdown
        def quit(self):
                pygame.quit()
                sys.exit(0)

        def draw_paused_text(self):
                size = self.glfont.get_size("Paused")
                #size = [1450,800]
                position = ((self.w-size[0])/2, (self.h-size[1])/2)
                glPushMatrix()
                glTranslatef(position[0], position[1], 0)
                self.glfont.render("Paused")
                glPopMatrix()

        def draw_finished_text(self):
                size = self.glfont.get_size("Ende")
                #size = [1550,800]
                position = ((self.w-size[0])/2, (self.h-size[1])/2)
                glPushMatrix()
                glTranslatef(position[0], position[1], 0)
                self.glfont.render("Ende")
                glPopMatrix()

        def draw_countdown_text(self, t):
                text = "%d seconds..." % (1+(t/1000))
                size = self.glfont.get_size(text)
                #size = [1450, 600]
                position = ((self.w-size[0])/2, (self.h-size[1])/2+(self.h/4.0))
                glPushMatrix()
                glTranslatef(position[0], position[1], 0)
                self.glfont.render(text)
                glPopMatrix()

        def flip(self):
                # clear the transformation matrix, and clear the screen too
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
                if self._state != STATE_BETWEEN_TRIALS and self._state != STATE_BETWEEN_BLOCKS and self._state != STATE_FINISHED:
                        self.viewer.render(self.w, self.h, self.rotation, self.brightness, self.enlarge, self.mask)
                else:
                        glEnable(GL_TEXTURE_2D)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glColor3f(1.0, 1.0, 1.0)
                        if self._state == STATE_BETWEEN_BLOCKS:
                                self.draw_paused_text()
                                if self._block_elapsed >= self.inter_block_duration - self.inter_block_countdown_duration:
                                        self.draw_countdown_text(self.inter_block_duration - self._block_elapsed)
                        elif self._state == STATE_FINISHED:
                                self.draw_finished_text()
                pygame.display.flip()

        def tick(self):
                self.elapsed = self.clock.tick(self.fps)
                self.handle_events()
                self.update_state()
                if self._stimulus_active:
                        self.viewer.update(self._current_trial, self._current_subtrial)
                self.flip()

        def play_tick(self):
                pass

        def post_tick(self):
                pass

        def handle_start_block(self):
                # send block start marker
                # reset current trial to 0
                # change state to STATE_STARTING_TRIAL
                #if not self.online_mode:
                #       self.send_udp(P300_START_BLOCK)
                self.send_parallel(P300_START_BLOCK)
                self._current_trial = 0
                self._state = STATE_STARTING_TRIAL
                self._block_elapsed = 0
                self._highlighted_image = -1
                self._trial_elapsed = 0
                print "> Starting block %d" % self._current_block

        def handle_start_trial(self):
                # send trial start marker
                # highlight random image for 1 second
                # pause for 2000ms
                # reset current subtrial to 0
                # change state to STATE_SUBTRIAL

                # select a random image for highlighting
                if self._trial_elapsed >= self.trial_pre_highlight_duration and not self._in_pre_highlight_pause:
                        self._in_pre_highlight_pause = True
                        if not self.online_mode:
                                self.send_udp(P300_START_TRIAL)
                        self.send_parallel(P300_START_TRIAL)

                        if self.copy_task:
                                self._current_target = self.target_list[0]
                                del self.target_list[0]
                                #self._current_target = random.randint(0, len(self.viewer.photo_set.photos())-1)
                                self.viewer.set_highlight(self._current_target)

                        print "> Starting trial %d" % self._current_trial
                        # send marker for selected image
                        time.sleep(0.02)
                        if not self.online_mode:
                                self.send_udp(self.HIGHLIGHT_START)
                        if self.copy_task:
                                self.send_parallel(self.HIGHLIGHT_START)#+1+self.viewer.highlight)
                                self._last_highlight = self.viewer.highlight

                self._trial_elapsed += self.elapsed

                if self._trial_elapsed > (self.trial_highlight_duration + self.trial_pre_highlight_duration):
                        self.viewer.clear_highlight()

                if self._trial_elapsed > (self.trial_pause_duration + self.trial_highlight_duration + self.trial_pre_highlight_duration):
                        self._state = STATE_SUBTRIAL
                        self._trial_elapsed = 0
                        self._current_subtrial = 0
                        self._subtrial_stimulation_elapsed = 0
                        self._subtrial_pause_elapsed = 0
                        print "> Starting subtrials!"

        def get_indexes(self):
                self.viewer.update(self._current_trial, self._current_subtrial)
                state = self.viewer.stimulation_state.transpose()
                self._current_highlights = []
                for x in range(self.row):
                        for y in range(self.col):
                                if state[x][y]:
                                        self._current_highlights.append((x*6)+y)
                print state
                self._current_highlights_index = 0

        def handle_subtrial(self):
                self._subtrial_stimulation_elapsed += self.elapsed
                self._subtrial_pause_elapsed += self.elapsed

                first_marker = False
                if self._subtrial_stimulation_elapsed > self.stimulation_duration:
                        self._stimulus_active = False
                else:
                        self._stimulus_active = True
                        if not self._markers_sent:
                                self._markers_sent = True
                                first_marker = True

                                # send first special marker
                                self.send_parallel(self.MARKER_START)
                                self._markers_reset = False

                if not first_marker and not self._current_highlights:
                        print "Getting indexes"
                        self.get_indexes()
                        if self.online_mode:
                                #self.send_udp('%d\n' % self.MARKER_START)
                                self.send_udp(self.MARKER_START)
                        else:
                                print self._current_target
                                print self._current_highlights
                                if self._current_target in self._current_highlights:
                                        #self.send_udp('S101\n')
                                        self.send_udp(101)
                                else:
                                        #self.send_udp('S  1\n')
                                        self.send_udp(1)

                elif not first_marker and self._current_highlights and self._current_highlights_index < len(self._current_highlights):
                        if not self._skip_cycle:
                                print "Sending image marker %d" % (self.MARKER_START+1+self._current_highlights[self._current_highlights_index])
                                # if this was the highlighted image
                                if self._current_highlights[self._current_highlights_index] == self._last_highlight and not self.online_mode:
                                        self.send_parallel(self.HIGHLIGHT_START+1+int(self._last_highlight))
                                else:
                                        self.send_parallel(self.MARKER_START+1+self._current_highlights[self._current_highlights_index])
                                self._current_highlights_index+=1
                                self.markers_elapsed = 0
                                self._skip_cycle = True
                        else:
                                self._skip_cycle = False

                if self._subtrial_pause_elapsed >= self.RESET_STATES_TIME and not self._markers_reset and not self.online_mode:
                        #self.send_udp('S  0\n')
                        self.send_udp(0)
                        self._markers_reset = True

                if self._subtrial_pause_elapsed >= self.stimulation_duration + self.inter_stimulus_duration:
                        # move on to next subtrial
                        print "> Finished subtrial %d (%d)" % (self._current_subtrial, self._subtrial_pause_elapsed)
                        self._current_subtrial += 1
                        self._subtrial_stimulation_elapsed = 0
                        self._subtrial_pause_elapsed = 0
                        self._stimulus_active = False
                        self._markers_elapsed = 0
                        self._markers_sent = False
                        self._skip_cycle = False
                        self._current_highlights = None
                        if not self.online_mode:
                                self.update_scores([random.random()], True) # XXX

                        if self._current_subtrial >= self.subtrial_count:
                                self._display_elapsed = 0
                                if self.image_display:
                                        self._state = STATE_DISPLAY_IMAGE #STATE_BETWEEN_TRIALS
                                else:
                                        self._state = STATE_BETWEEN_TRIALS
                                self._trial_elapsed = 0


        def handle_trial_wait(self):
                # send trial end marker
                # inter trial pause of 5000ms
                if self._trial_elapsed == 0:
                        self._in_pre_highlight_pause = False
                        self.send_parallel(P300_END_TRIAL)
                        self.send_parallel(P300_END_TRIAL)
                        self._current_trial += 1
                        self._trial_elapsed = 0
                        print "> Waiting between trials"

                self._trial_elapsed += self.elapsed
                #print "> Wait time: %05d\r" % self._trial_elapsed
                if self._trial_elapsed >= self.inter_trial_duration and (not self.online_mode or (self.online_mode and self._subtrial_scores_received)):
                        self._block_elapsed = 0
                        self._trial_elapsed = 0
                        self._subtrial_scores_received = False

                        if self._current_trial >= self.trial_count:
                                self._state = STATE_BETWEEN_BLOCKS
                        else:
                                self._state = STATE_STARTING_TRIAL
                                
                                self.init_scoring_matrix()

        def handle_block_wait(self):
                # send block_end marker
                # inter-block pause of 15seconds
                # show "pause" text
                # if possible countdown displayed for last 3 seconds
                if self._block_elapsed == 0:
                        self.send_udp(P300_END_BLOCK)
                        self.send_parallel(P300_END_BLOCK)
                        print "> Finished block %d" % self._current_block
                        self._current_block += 1
                        self._current_trial = 0
                        # if on last block, stop
                        if self._current_block >= self.block_count:
                                self._state = STATE_FINISHED
                                return

                self._block_elapsed += self.elapsed
                #print "> Block wait elapsed: %d" % self._block_elapsed

                if self._block_elapsed >= (self.inter_block_duration - self.inter_block_countdown_duration):
                        # display countdown
                        pass

                if self._block_elapsed >= self.inter_block_duration:
                        self._state = STATE_STARTING_TRIAL

        def handle_finished(self):
                if not self._finished:
                        self.send_udp(P300_END_EXP)
                        self.send_parallel(P300_END_EXP)
                        self._finished = True

                #print "FINISHED"
                # Display "Ende" message

        def handle_display_image(self):
                self._display_elapsed += self.elapsed
                if self._display_elapsed >= self.image_display_time:
                        self.viewer.fullscreen = False
                        #self.viewer.clear_selected_images()
                        self._state = STATE_BETWEEN_TRIALS
                else:
                        self.viewer.fullscreen = True

        def update_state(self):
                if self._state == STATE_INITIAL:
                        pass
                        print "> Initial state"
                elif self._state == STATE_STARTING_BLOCK:
                        self.handle_start_block()
                elif self._state == STATE_STARTING_TRIAL:
                        self.handle_start_trial()
                elif self._state == STATE_SUBTRIAL:
                        self.handle_subtrial()
                elif self._state == STATE_BETWEEN_TRIALS:
                        self.handle_trial_wait()
                elif self._state == STATE_BETWEEN_BLOCKS:
                        self.handle_block_wait()
                elif self._state == STATE_FINISHED:
                        self.handle_finished()
                elif self._state == STATE_DISPLAY_IMAGE:
                        if self._subtrial_scores_received:
                                self.handle_display_image()
                else:
                        print "*** Unknown state ***"

        def keyup(self,event):
                if event.key == K_ESCAPE:
                        self.quit()
                if event.key == K_s:
                        self.viewer.set_highlight(random.randint(0,24))
                if event.key== K_c:
                        self.viewer.clear_highlight()
                if event.key == K_e:
                        self.viewer.add_selected_image(random.randint(0, self.max_photos), True)

        def handle_events(self):
                for event in pygame.event.get():
                        if event.type==KEYDOWN:
                                if event.key==K_ESCAPE:
                                        self.quit()
                        if event.type==KEYUP:
                                self.keyup(event)
                        if event.type == QUIT:
                                self.quit()

        def update_scores(self, data, fake=False):
                trial_number = int(self._current_trial)
                self.total_subtrial_counter += 1
                subtrial_number = int(self.total_subtrial_counter)

                print "Scores received for trial %d, subtrial %d" % (trial_number, subtrial_number)
                highlights_for_these_scores = self.highlight_indexes[trial_number][subtrial_number-1]

                # update scores for highlighted images
                score = data[0]

                print "Adding", score, " to images ", highlights_for_these_scores
                print "Frame", int((subtrial_number-1)/6)
                
                if self.rowColEval:
                        # find out if row or column
                        # find out which row or column
                        # self.scoring_matrix[row 0 - 5, col 6 - 11]
                        # add score to self.scoring_matrix[INDEX_OF_ROW_OR_COL][int(math.floor((subtrial_number-1)/self.subtrials_per_frame))] = score
                        # use self.row and self.col for generic solution
                        value = []
                                                                        
                        for i in range(2):
                                col = highlights_for_these_scores[i]
                                row = 0
                                
                                while col >= (self.col):
                                        col = col - self.col
                                        row = row + 1
                                        
                                value.append([row, col])     
                        
                        if (value[0][1] + 1) == value[1][1]:
                            # is row
                            print "scoring_matrix[%d][%d] = %f" % (value[0][0], int(math.floor(((subtrial_number-1)/2)/self.subtrials_per_frame)), score)
                            self.scoring_matrix[value[0][0]][int(math.floor(((subtrial_number-1)/2)/self.subtrials_per_frame))] = score
                        else:
                            # is col
                            print "scoring_matrix[%d][%d] = %f" % ((value[0][1] + self.col), int(math.floor(((subtrial_number-1)/2)/self.subtrials_per_frame)), score)
                            self.scoring_matrix[int(value[0][1] + self.col)][int(math.floor(((subtrial_number-1)/2)/self.subtrials_per_frame))] = score                             
                else:
                        for i in highlights_for_these_scores:
                                print "scoring_matrix[%d][%d] = %f" % (int(i), int(math.floor((subtrial_number-1)/self.subtrials_per_frame)), score)
                                #print self.subtrials_per_frame
                                self.scoring_matrix[int(i)][int(math.floor((subtrial_number-1)/self.subtrials_per_frame))] = score

                print "Total subtrial count:", self.total_subtrial_counter

                if subtrial_number == self.subtrial_count:
                        print "Last subtrial completed, calculating selected image"

                        if self.rowColEval:
                                # find winning row and winning column
                                # find the intersecting image -> winner
                                # find score of intersecting image -> winValue
                                scoresMat = array(self.scoring_matrix)
                                print scoresMat
                                scoresMat = median(scoresMat,1)
                                row = scoresMat[:self.row].argmin()
                                col = scoresMat[self.row:].argmin()
                                winner = int((self.col * row) + col)
                                winValue = scoresMat[row]*scoresMat[self.col+col]
                        else:
                                scoresMat = array(self.scoring_matrix)
                                print scoresMat
                                scoresMat = median(scoresMat,1)
                                winner = scoresMat.argmin()
                                winValue = scoresMat[winner]
                        #min = 999
                        #minimg = 0
                        #for i in range(30): # IMAGE COUNT
                        #       s = self.scoring_matrix[i][subtrial_number-1]
                        #               if s != 0.0 and s < min:
                        #               min = s
                        #               minimg = i

                        print "Image %d was selected, score %f" % (winner, winValue)
                        self.viewer.set_winner(winner)
                        if self.online_mode:
                                self.viewer.add_selected_image(winner, not self.highlight_all_selected)
                        elif fake:
                                print "Faking selection of image", self._current_target
                                self.viewer.add_selected_image(self._current_target, not self.highlight_all_selected)

                        self._subtrial_scores_received = True
                        self.total_subtrial_counter = 0

        def on_control_event(self,data):
                if data.has_key(u'cl_output'):
                        score_data = data[u'cl_output']
                        print score_data
                        self.update_scores(score_data)

        def on_interaction_event(self, data):
                #self.scoring_matrix = []
                #for i in range(30):    # IMAGE COUNT
                #       self.scoring_matrix.append([0.0 for x in range(self.subtrial_count/self.subtrials_per_frame)])
                #self.total_subtrial_counter = 0

                if data.has_key(u'highlight_indexes'):
                        self.highlight_indexes
                        if self.viewer != None:
                                self.viewer.update_indexes(self.highlight_indexes)
                elif data.has_key(u'subtrial_count') or data.has_key(u'highlight_count'):
                        p = os.path.join(os.getcwd(), File_Path)
                        if os.path.exists(p):
                                preloaded_highlight_indexes = load_highlights.load2(p, self.subtrial_count, self.highlight_count)
                                print "Loading highlights from highlights.csv!"
                                self.highlight_indexes = preloaded_highlight_indexes
                        self.init_scoring_matrix()
                elif data.has_key(u'rowColEval'):
                        self.init_scoring_matrix()

if __name__ == "__main__":
        # simulate pyff for rapid testing
        os.chdir("../..")
        p = P300PhotoBrowser()
        p.on_init()
        p._on_play()
