import math, sys, random, time
import pygame
import os
from pygame.locals import *
import glutils

#sys.path.append("e:\\work\\pyff\\src\\Feedbacks\MultiFeedback")
from FeedbackBase.MainloopFeedback import MainloopFeedback

# markers 1 63 area reserved for decimal representation of binary true/false labels.
# marker 99 is the special case where ALL classifiers were wrong.

MSF_START_EXP = 251
MSF_END_EXP = 254
MSF_COUNTDOWN_START = 70
MSF_START_TRIAL = 0
MSF_END_TRIAL = 0
MSF_TRIAL_LEFT = 1
MSF_TRIAL_RIGHT = 2

MAX_CLASSIFIERS = 6

HIT, MISS, DRAW = range(3)

class MultiScaleFeedback(MainloopFeedback):

	def init(self):
		"""Create the class and initialise variables that must be in place before
		pre_mainloop is called"""

		self.send_parallel(MSF_START_EXP) # exp start marker
		self.initialised=False
		self.screen_w = 1920
		self.screen_h = 1200
		self.screenPos = [-1920, 400]
		self.fps = 25		
		self.trail_persist = 0.9
		self.bubble_mode = False
		self.bubble_height = 40
		self.bubble_min_width = 40
		self.x_spread = 4
		self.y_spread = 4
		self.bias = [.0, .0, .0, .0, .0]
		self.gain = .2		
		self._data = {}

		self.number_of_trials = 40			# number of trials to perform
		self.countdown_duration = 5 		# initial countdown duration (seconds)
		self.countdown_remaining = self.countdown_duration
		self.countdown_elapsed = 0
		self.trial_duration = 4			# trial duration (seconds)
		self._trial_elapsed = 0 			# elapsed time in current trial
		self._arrow_time_elapsed = 0 		# elapsed time when displaying fixation cross and arrow
		self._cross_time_elapsed =0  		# elapsed time when displaying fixation cross
		self.current_trial = 1 				# current trial number (starts at 1)
		self.show_arrow_duration = 1 		# time in seconds to show fixation cross AND arrow
		self.show_cross_duration = 1 		# time in seconds to show fixation cross only
		self.blank_screen_min_duration = 0.8 	# minimum duration in seconds for blank screen between trials
		self.blank_screen_max_duration = 1.5 	# maximum duration in seconds for blank screen between trials
		self._pause_duration = 5 			# time to show blank screen between trials; randomly recalculated from the above pair of variables for each trial
		self._paused_elapsed = 0
		self.classifier_limit = 0 			# controls how many classifier inputs are actually used in the feedback, default is 0 (all)
		self.v_offset = self.screen_h/2 				# controls the vertical offset of the clouds

		# state variables
		self._countdown_active = True 
		self._countdown_starting = True
		self._trial_active = False
		self._trial_starting = False
		self._showing_arrow = False
		self._showing_cross = False
		self._between_feedbacks = False
		self._showing_scores = False

		# holds a direction (left/right) for each trial
		self.trial_directions = []
		self.generate_trial_directions()

		self.voting_scores = [0, 0, 0]
		self.weighted_scores = [0, 0, 0]
		self.weight_factor = [1.0 for i in range(MAX_CLASSIFIERS)] 
		self.class_labels = []
		self.score_display_period = 20 # display scores every x trials
		self.score_display_duration = 5 # time to display the scores for
		self._scores_elapsed = 0

		self._last_samples = None

		self.special_marker = 99
		self.trial_marker_delay = 0.01
		self.show_draws = 0;
	
		# colours for the clouds in Pygame HSLA format (if you have more than MAX_CLASSIFIER classifier inputs, add more colours here!)
		self.colours = 	[
				(10.0, 50, 50, 100),
				(60.0, 50, 50, 100),
				(120.0, 50, 50, 100),
				(180.0, 50, 50, 100),
				(240.0, 50, 50, 100),
				(300.0, 50, 50, 100)
						]				

		# colour of the fixation cross
		self.cross_colour = (150, 150, 150)

	def init_opengl(self):
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, self.screen_w, 0, self.screen_h, -1, 500)
		glMatrixMode(GL_MODELVIEW)
		
		#enable texturing and alpha
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_LINE_SMOOTH)
		
	def load_gl_textures(self):
		self._large_bubble = glutils.GLSprite("bubble_large.png")
		self._small_bubble = glutils.GLSprite("bubble_small.png")
		
	def init_surfaces(self):
		"""Initialise the surfaces for drawing onto"""
		self._back_buf = pygame.Surface(self._screen.get_size())
		self._draw_buf = pygame.Surface(self._screen.get_size())
		self._other_buf = pygame.Surface(self._screen.get_size())
		self._back_buf = self._back_buf.convert()
		self._draw_buf = self._draw_buf.convert()
		self._other_buf = self._other_buf.convert()
		self._back_buf.set_alpha(200)
		self._other_buf.set_alpha(255)
		self._back_buf.fill((0,0,0))
		self._draw_buf.fill((0,0,0))
		self._other_buf.fill((0,0,0))
		self._draw_buf.set_alpha(self.trail_persist*255.0)
		
	def transparent_clear(self):
		  """clear the buffers, using a semi transparent blit"""
		  self._other_buf.blit(self._draw_buf,(0,0))
		  self._draw_buf.blit(self._back_buf, (0,0))
		  #smoke trail blur effect		  
		  self._draw_buf.blit(self._other_buf, (0,-2))
		
	def on_interaction_event(self, data):
		"""Called every time the data updates"""
		if self.initialised:
			self._draw_buf.set_alpha(self.trail_persist*255.0)

		if data.has_key(u'number_of_trials'):
			print "Number of trials updated: ", data[u'number_of_trials']
			self.generate_trial_directions()

	def generate_trial_list(self, length):
		tmp = [MSF_TRIAL_LEFT for x in range(length)]
		replaced = []
		while len(replaced) != length/2:
			r = random.random()
			if r > 0.5:
				while True:
					p = random.randint(0, length-1)
					if p not in replaced:
						break

				tmp[p] = MSF_TRIAL_RIGHT
				replaced.append(p)
		return tmp

	def generate_trial_directions(self):
		# if length is 0, being called from init
		if len(self.trial_directions) == 0:
			self.trial_directions = self.generate_trial_list(self.number_of_trials)
			return
		
		# number of trials has been changed, need to resize the list

		# if the number of trials has been decreased, just truncate the list
		if self.number_of_trials <= len(self.trial_directions):
			self.trial_directions = self.trial_directions[:self.number_of_trials]
			# and reset current trial if required
			if self.current_trial > self.number_of_trials:
				self.current_trial = self.number_of_trials
			return


		# if the number of trials has been increased, preserve the existing list and current_trial value,
		# and append the new trials onto the existing list
		self.trial_directions += self.generate_trial_list(self.number_of_trials - len(self.trial_directions))
		
	def text(self, text, position, color, center=False):		
		textimage = self._default_font.render(text, True, color)
		if center:
			position = (position[0]-textimage.get_width()/2, position[1]-textimage.get_height()/2)
			
		self._screen.blit(textimage, position)
   
	def pre_mainloop(self):
		# init pygame, load fonts
		os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screenPos[0], self.screenPos[1])
		pygame.init()
		default_font_name = pygame.font.match_font('bitstreamverasans', 'verdana', 'sans')
		if not default_font_name:		   
			default_font_name = pygame.font.get_default_font()				
		self._default_font = pygame.font.Font(default_font_name, 36)
		self._small_font = pygame.font.Font(default_font_name, 18)
		self._screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)				
		self._clock = pygame.time.Clock()
		self.init_surfaces()
		self._w = self._screen.get_width()
		self._h = self._screen.get_height()
		self.initialised = True

	def post_mainloop(self):
		pygame.quit()
		self.send_parallel(MSF_END_TRIAL)
		
	# always called every tick
	def tick(self):
		self.elapsed = self._clock.tick(self.fps)
		self.handle_events()
		
	def on_control_event(self,data):
		pass
		
	def post_tick(self):
		pygame.display.flip()
	
	# called only when paused
	def pause_tick(self):
		self._screen.fill((128,128,128))
		self.text("Paused", (255,128,128), (self._w/2, self._h/2), center=True)
		self.post_tick()
		
	def get_bubble_color(self, row):
		dcolor = (255,100,100)
		col = row[0] - row[-1]
		if col<-1:
			col = -1
		if col>1:
			col = 1
		col = (col + 1)/2.0		
		dcolor = ((1-col)*255,col*255,100)
		return dcolor
		
	def draw_particles(self, row, x_scale):
		ncolors = len(row)
		
		# draw samples
		for v in range(40):
			for k,i in enumerate(row):	   
				spread = random.gauss(0, self.x_spread)				
				v_spread = random.gauss(0,self.y_spread)
				color = pygame.Color(0,0,0,255)
				color.hsla = self.colours[k] # (360.0*(k/float(ncolors)),50,50,100)
				pygame.draw.circle(self._draw_buf, color, (i*x_scale+self._w/2+spread, self._h-self.v_offset+v_spread), 2)
				
		
	def draw_bubble(self, row, x_scale):
		# compute bubble size
		minval = min(row)
		maxval = max(row)
		width = max([self.bubble_min_width, ((maxval-minval)*x_scale)])
		rect = Rect(minval*x_scale+self._w/2,self._h-100, width, self.bubble_height)				
		# color code the bubble
		dcolor = self.get_bubble_color(row)
		
		pygame.draw.ellipse(self._draw_buf, dcolor, rect)
		pygame.draw.ellipse(self._draw_buf, (255,255,255), rect, 1)
	
	def draw_targets(self):
		pygame.draw.rect(self._screen, (255,255,255), (self._w/6-50, self._h-100, 50, 100))
		pygame.draw.rect(self._screen, (255,255,255), (self._w-(self._h/6), self._h-100, 50, 100))

	def handle_countdown(self):
		self._screen.fill((33,33,33))
		if self._countdown_starting:
			self.countdown_remaining = self.countdown_duration
			self._countdown_starting = False
			self.countdown_elapsed = 0
			self.send_parallel(MSF_COUNTDOWN_START)

		self.countdown_elapsed += self.elapsed
		self.countdown_remaining = int(self.countdown_duration - (self.countdown_elapsed/1000.0))
		self.text('Starting in %d seconds...' % self.countdown_remaining, [self._w/2, self._h/2], (255,255,255), True)
		if self.countdown_remaining <= 0.0:
			self._countdown_active = False
			self._trial_active = True
			self._trial_starting = True
			self._showing_cross = True

	def draw_cross(self):
		pygame.draw.line(self._screen, self.cross_colour, (self._w/2-100, self._h/2), (self._w/2+100, self._h/2), 2)
		pygame.draw.line(self._screen, self.cross_colour, (self._w/2, self._h/2-100), (self._w/2, self._h/2+100), 2)
	
	def draw_arrow(self, direction):
		if direction == MSF_TRIAL_LEFT:
			pygame.draw.rect(self._screen, (170, 170, 170, 100), Rect(self._w/2-80, self._h/2-10, 80, 20), 0)
			pygame.draw.polygon(self._screen, (170, 170, 170, 100), [(self._w/2-110, self._h/2), (self._w/2-80, self._h/2-20), (self._w/2-80, self._h/2+20)], 0)
		else:
			pygame.draw.rect(self._screen, (170, 170, 170), Rect(self._w/2, self._h/2-10, 80, 20), 0)
			pygame.draw.polygon(self._screen, (170, 170, 170), [(self._w/2+110, self._h/2), (self._w/2+80, self._h/2-20), (self._w/2+80, self._h/2+20)], 0)

	def update_scores(self):
		class_result = []
		for i in self._last_samples:
			if i >= 0:
				class_result.append(1)
			else:
				class_result.append(-1)

		print class_result

		# calculate marker to send 
		marker = 0
		for i in range(len(class_result)):
			if class_result[i] == 1:
				marker += 2 ** ((len(class_result) - 1) - i)
                marker += 100

		print "Marker = %d" % marker
		#if marker == 0:
		#	marker = self.special_marker 
		self.send_parallel(marker)

		# append class labels to the list of all results
		self.class_labels.append(class_result)

		print "Calculating voting score"
		# calculate voting score
		classifier_count = self.classifier_limit
		if classifier_count == 0:
			classifier_count = MAX_CLASSIFIERS
		
		voting_score = sum(map(lambda x: x * (1.0/classifier_count), class_result))
		if voting_score > 0 and self.trial_directions[self.current_trial-1] == MSF_TRIAL_RIGHT:
			self.voting_scores[HIT] += 1
		elif voting_score < 0 and self.trial_directions[self.current_trial-1] == MSF_TRIAL_LEFT:
			self.voting_scores[HIT] += 1
		elif voting_score == 0.0:
			self.voting_scores[DRAW] += 1
		else:
			self.voting_scores[MISS] += 1

		# calculate weighted score
		print "Calculating weighted score"
		weighted_score_tmp = []
		for x in range(len(class_result)):
			weighted_score_tmp.append(class_result[x] * self.weight_factor[x])

		weighted_score = sum(weighted_score_tmp)

		if weighted_score > 0 and self.trial_directions[self.current_trial-1] == MSF_TRIAL_RIGHT:
			self.weighted_scores[HIT] += 1
		elif weighted_score < 0 and self.trial_directions[self.current_trial-1] == MSF_TRIAL_LEFT:
			self.weighted_scores[HIT] += 1
		elif weighted_score == 0.0:
			self.weighted_scores[DRAW] += 1
		else:
			self.weighted_scores[MISS] += 1

	def draw_scores(self):
                if self.show_draws:
        		self.text('hit : miss : draw   %d:%d:%d / %d:%d:%d' % tuple(self.voting_scores + self.weighted_scores), (self._w/2, self._h/2), (200, 200, 200), True)
        	else:
        		self.text('hit : miss   %d:%d / %d:%d' % tuple(self.voting_scores[:2] + self.weighted_scores[:2]), (self._w/2, self._h/2), (200, 200, 200), True)

	def handle_trial(self):
		if self._trial_starting:
			self._trial_starting = False
			self._trial_elapsed = 0
			self._arrow_time_elapsed = 0
			self._cross_time_elapsed = 0
			self.send_parallel(MSF_START_TRIAL)
			time.sleep(self.trial_marker_delay)
			if self.current_trial > self.number_of_trials:
				return
			self.send_parallel(self.trial_directions[self.current_trial-1])

		if self._showing_cross:
			# draw fixation cross
			self._screen.fill((0,0,0))
			self.draw_cross()

			self._cross_time_elapsed += self.elapsed

			if (self._cross_time_elapsed / 1000.0) >= self.show_cross_duration:
				self._showing_arrow = True

			if self._showing_arrow:
				self._arrow_time_elapsed += self.elapsed

				self.draw_arrow(self.trial_directions[self.current_trial-1])
				if (self._arrow_time_elapsed/1000.0) >= self.show_arrow_duration:
					self._showing_arrow = False
					self._showing_cross = False
			return
		else:
			self._trial_elapsed += self.elapsed

		trial_remaining = self.trial_duration - (self._trial_elapsed / 1000.0)
		#print "Trial time remaining: %.2f seconds" % trial_remaining

		if trial_remaining <= 0.0:
			self._trial_active = False
			self._trial_starting = False
			self.send_parallel(MSF_END_TRIAL)
			time.sleep(self.trial_marker_delay)


			self.update_scores()

			self.current_trial += 1
	
			self._between_feedbacks = True
			# randomly pause for a time between min/max values
			# randint(a, b) returns a value x s.t. a <= x <= b
			#self._pause_duration = random.randint(self.blank_screen_min_duration, self.blank_screen_max_duration)
			self._pause_duration = random.uniform(self.blank_screen_min_duration, self.blank_screen_max_duration)
			self._paused_elapsed = 0
			self._scores_elapsed = 0
			if (self.current_trial - 1) % self.score_display_period == 0:
				self._showing_scores = True
			return

		self.transparent_clear()

		# NB depends on classifier being -1.0--1.0!
		# will scale the range to fit in 2/3 of screen width		
		x_scale = self._w*self.gain	   
		
		if not self._data.has_key(u"cl_output"):
			#return
			class_outputs = [-0.5, 0.5, 1.0]
		else:
			class_outputs = self._data[u"cl_output"]
			
			# limit number of outputs here if required
			if self.classifier_limit > 0:
				class_outputs = class_outputs[:self.classifier_limit]

                class_outputs = [class_outputs[i] + self.bias[i] for i in range(len(class_outputs))]
			

		# special case of 1 input
		if isinstance(class_outputs, float):
			class_outputs = [class_outputs]

		self._last_samples = class_outputs

		if self.bubble_mode:
			self.draw_bubble(class_outputs, x_scale)
		else:
			self.draw_particles(class_outputs, x_scale)
		# blit and finish
		self._screen.blit(self._draw_buf, (0,0))		

		# draw left/right divider line
		pygame.draw.line(self._screen, (55, 55, 55), (self._w/2, 0), (self._w/2, self._h))

	def handle_paused(self):
		self._screen.fill((0,0,0))

		if self._showing_scores:
			self.draw_scores()
			self._scores_elapsed += self.elapsed
			if (self._scores_elapsed / 1000.0) >= self.score_display_duration:
				self._showing_scores = False
				self._scores_elapsed = 0
			return

		self._paused_elapsed += self.elapsed
		if (self._paused_elapsed / 1000.0) >= self._pause_duration:
			self._between_feedbacks = False
			self._draw_buf.fill((0,0,0))
			self._back_buf.fill((0,0,0))
			self._other_buf.fill((0,0,0))

			if self.current_trial > self.number_of_trials:
				return
			self._trial_active = True
			self._trial_starting = True
			self._showing_cross = True
			self._showing_arrow = False

	def handle_finished(self):
		self._screen.fill((77,77,77))
		self.text('%d trials completed' % self.number_of_trials, [self._w/2, self._h/2], (255,255,255), True)

	# called only when not paused
	def play_tick(self):
		if self._countdown_active:
			self.handle_countdown()
		elif self._trial_active:
			self.handle_trial()
		elif self._between_feedbacks:
			self.handle_paused()
		else:
			self.handle_finished()
		
		self.post_tick()
		
	def handle_events(self):
		  """Make sure the event loop is emptied"""
		  for event in pygame.event.get():
			pass
	
