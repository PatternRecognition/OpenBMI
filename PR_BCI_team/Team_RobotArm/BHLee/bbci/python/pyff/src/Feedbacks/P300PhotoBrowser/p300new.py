import sys,time,os,random,cPickle, math
import traceback

import pygame, thread
from pygame.locals import *

import thread
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from glutils import *
from filter import *
import numpy

MAX_PHOTOS = 30
			
#  Encapsualtes a cache of images
class Cache:
		def __init__(self, max_size=32):
			self.cache = {}
			self.max_size = max_size
			
		def add_element(self, key, value):			
			if len(self.cache)>=self.max_size:
				# remove oldest accessed element
				min_time = time.clock()
				min_key = None
				for k in self.cache:
					v,t = self.cache[k]
					if t<min_time:
						min_time = t
						min_key = k
				self.cache.min[min_key].delete()
				del self.cache[min_key]
				
			self.cache[key] = (value, time.clock())
			
			
		def get_element(self, key):
			v = self.cache.get(key, None)
			if v:
				value, t = v
				self.cache[key] = (value, time.clock())
				return value
			else:
				return None
			
# A set of images, in a particular directory, with caching and thumbnailing
class PhotoSet:
	def __init__(self, directory, cache="cache"):
		if not os.path.exists(cache):
			os.mkdir(cache)
		self.cache = cache
			
		photo_list = []
		photos = os.listdir(directory)
		for photo in photos:
			path,fname = os.path.split(photo)
			base,ext = os.path.splitext(photo)
			if ext==".jpg" or ext==".jpeg":
				photo_list.append(photo)
		self.directory = directory
		
		# limit photo list to max_photos
		self.photo_list = photo_list
		if len(self.photo_list)>MAX_PHOTOS:
			self.photo_list = self.photo_list[:MAX_PHOTOS]
		else:
			self.photo_list = self.photo_list
		self.cache_images()
		
	def cache_images(self):
		self.thumbs = {}
		self.img_cache = Cache()
		for photo in self.photo_list:
			fname = os.path.join(self.directory, photo)
			
			thumb_name = os.path.join(self.cache, photo+".thumb.png")
			small_name = os.path.join(self.cache, photo+".small.png")
										 
			if not os.path.exists(thumb_name) or not os.path.exists(small_name):
				print "Creating thumbnail %s" % photo													   
				try:
					img = pygame.image.load(fname)
				except:			
					print "Could not open %s" % fname
					continue
				
				aspect = float(img.get_width())/img.get_height()
				# scale the images and store them
				thumb = pygame.transform.smoothscale(img, (128*aspect, 128))
				small = pygame.transform.smoothscale(img, (512*aspect, 512))
				
				pygame.image.save(thumb,thumb_name)
				pygame.image.save(small,small_name)
				
				del img, small
			else:
				thumb = GLSprite(thumb_name,real_size=False, constant_height=True)
				
			self.thumbs[photo] = thumb
	
	# return a photo name given an index
	def get_photo_name(self, index):
		while index<0:
			index = index  + len(self.photo_list)
			
		index = index % len(self.photo_list)
		return self.photo_list[index]
	
	# return a thumb nail
	def get_thumb(self, img):
		return self.thumbs[img]
		
	def photos(self):
		return self.photo_list
		
	# get a full-sized image, from the cache if we can
	def get_image(self, img):
		c = self.img_cache.get_element(img)
		if c:
			return c
		else:
			c = GLSprite(os.path.join(self.cache, img+".small.png"), real_size = False)
			self.img_cache.add_element(img, c)
			return c
	
# Simulated annealing to optimize the layout of images which are stimulated
class Annealer:
	def __init__(self, w, h, initial_density=0.2, iterations=500):
		self.w = w
		self.h = h
		self.initial_density = initial_density
		self.iterations = iterations
		self.history = numpy.ones((w,h))*100
		
	def anneal(self):
		t = 100
		t_decay = 0.98
		state = self.sample_initial_state(self.initial_density)
		score = self.score_state(state)
		
		# annealing loop
		for i in range(self.iterations):
			new_state = self.neighbour(state,t)
			new_score = self.score_state(new_state)
			
			# transitions
			if new_score<score:
				state = new_state
				score = new_score
			else:
				#print "math.exp parameters:", new_score, score, t
				#if numpy.isnan(score):
				#	score = 0.0001
				p = math.exp(-(new_score - score)/t)
				if random.random()<p:
					state = new_state
					score = new_score
				
			# temperature schedule
			t = t * t_decay
			#print t,score
			
		self.state = state
		self.score = score
		self.update_history()
		
	def update_history(self):
		self.history = self.history - (self.state * self.history)
		self.history = self.history + 1
			
	def score_state(self, state):
		n = numpy.sum(numpy.sum(state))+0.1
		max_on = (self.w*self.h)
		s = state
		hpscore = s[1:,:] * s[:-1,:]
		vpscore = s[:,1:] * s[:,:-1]
		d1pscore = s[1:,1:] * s[:-1,:-1]
		d2pscore = s[1:,:-1] * s[:-1,1:]
		
		pscore = (numpy.sum(numpy.sum(hpscore)) + numpy.sum(numpy.sum(vpscore)
		+numpy.sum(numpy.sum(d1pscore))*0.5 + numpy.sum(numpy.sum(d2pscore))*0.5
		)) / n
		
		hscore = (1.0/(self.history)) * s
		hscore = numpy.sum(numpy.sum(hscore)) / n
		nscore = abs((n-float(max_on*self.initial_density))/n)
		#score = pscore*10+nscore*4+hscore*10
		score = pscore*10+nscore*100+hscore*10
		
		return score
		
	def sample_initial_state(self, initial_density):
		state = numpy.random.random((self.w,self.h)) + initial_density
		state = numpy.clip(numpy.floor(state), 0, 1)
		return state
			
	def neighbour(self, state, t):
		n = 1+t
		state = numpy.array(state)
		for i in range(n):
			x = random.randrange(0,int(self.w))
			y = random.randrange(0,int(self.h))
			state[x,y] = 1-state[x,y]
		return state

class SimpleImageSelect:
	def __init__(self, indexes, _w, _h):
		self.image_indexes = indexes
		self.w = _w
		self.h = _h

	def get_state(self, trial, subtrial):
		state = numpy.zeros((self.w, self.h))
		indexes = self.image_indexes[trial][subtrial]
		#print indexes
		for i in indexes:
			state[i%self.w][i/(self.h+1)] = 1

		return state

# The main photoviewer class
class P300PhotoViewer:
	def __init__(self, photo_set, image_indexes):
		self.photo_set = photo_set
		self.mask = GLSprite(os.path.join(os.getcwd(),"mask.png"), real_size=False)
		self.shadow = GLSprite(os.path.join(os.getcwd(),"shadow.png"), real_size=False)
		self.frame = GLSprite(os.path.join(os.getcwd(),"frame.png"), real_size=False)
		self.index = 0
		self.visible_images = []
		self.image_space = {}
		self.stimulation_cycle_time = 0.1 #0.8
		self.stimulation_time = 0.2
		self.last_stimulation_time = time.clock()

		self.indexer = SimpleImageSelect(image_indexes, 6, 5)
		
		self.image_density =  0.16666 #  0.1
		self.photo_size = 150
		self.flash_level = 0.4
		self.rotate_level = 5
		self.scale_level = 0.1
		self.image_spacing = 1.2
		self.highlight = None
		self.selected_images = []
		
		self.layout()
		
	# highlight an image
	def set_highlight(self, image_number):
		self.highlight = image_number
		
	# clear the highlight
	def clear_highlight(self):
		self.highlight = None
		
	# Sample from the stimulation set
	def sample_stimulation(self, trial, subtrial):
		dt = time.clock()-self.last_stimulation_time
		if dt>self.stimulation_cycle_time:
			#self.annealer.anneal()			
			#self.stimulation_state = self.annealer.state
			self.stimulation_state = self.indexer.get_state(trial, subtrial)

			self.last_stimulation_time = time.clock()

			#print self.stimulation_state.transpose()
			#print self.stimulation_state.shape
			#count = 0
			#for i in simulation_v:
			#	for j in i:
			#		if j > 0.0:
			#			count += 1
			#print "Highlight", count
			
		# WRITE OUT SIMULATION STATE HERE
		
	# Just layout the photographs, the first time only
	def layout(self):		
		photos = self.photo_set.photos()
		l = len(photos)
		xlen = 6 # math.sqrt(l)+1
		ylen = 5 # math.sqrt(l)
		x = 0
		y = 0		
		#self.annealer = Annealer(xlen, ylen, initial_density = self.image_density) #Annealer(xlen+1, ylen+1, initial_density=self.image_density)		
		#self.annealer.anneal()
		#self.stimulation_state = self.annealer.state
		self.stimulation_state = self.indexer.get_state(0, 0)
		self.size = (xlen,ylen)
		for photo in photos:		
			self.image_space[photo] = (x,y)
			x = x + 1
			if x>=xlen:
				x = 0
				y = y + 1
		
	def render(self, w, h):
		
		step_scale = self.image_spacing
		size = self.photo_size
		xsize = self.photo_size*step_scale
		ysize = self.photo_size*step_scale
		
		border_size = 1.0
		photos = self.photo_set.photos()
		glTranslatef(xsize/2,h-ysize/2,0)
		
		dt = 1.0-((time.clock()-self.last_stimulation_time) / self.stimulation_time)
		if dt<0.0:
			dt = 0.0
		simulation_v = self.stimulation_state * dt
		
		# set scaling paramters
		flash = self.flash_level
		scale = self.scale_level
		jump = 0.0
		rotate_rate = 0
		rotate_amt = self.rotate_level
		rotate = rotate_amt*math.cos(time.clock()*rotate_rate)
		
		for index,photo in enumerate(photos):
			(xc, yc) = self.image_space[photo]
			s = simulation_v[xc][yc]
			x,y = xsize*xc, -ysize*yc
			glPushMatrix()
			glTranslatef(x,y,0)
			thumb = self.photo_set.get_thumb(photo)
			glScalef(size*border_size*(1+s*scale), size*border_size*(1+s*scale),1)
			aspect = thumb.h/float(thumb.w)
			# set up the transformation
			glScalef(1,aspect,1)
			glRotatef(s*rotate,0,0,1)		   
			glTranslatef(0,jump*s,0)
			
			# draw the thumbnail
			glColor4f(1,1,1,1)
			glPushMatrix()			
			glCallList(thumb.sprite)
			glPopMatrix()			
			
			glDisable(GL_TEXTURE_2D)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE)
			glColor4f(1,1,1,s*flash)
			
			# Draw the flash as a textured quad
			glBegin(GL_QUADS)
			glVertex3f(-0.5,-0.5,0)
			glVertex3f(0.5,-0.5,0)
			glVertex3f(0.5,0.5,0)
			glVertex3f(-0.5,0.5,0)
			glEnd()
			
			if index==self.highlight:
				glColor4f(0.2,0.5,0.2,0.4)			
				# Draw the flash as a textured quad
				glBegin(GL_QUADS)
				glVertex3f(-0.7,-0.7,0)
				glVertex3f(0.7,-0.7,0)
				glVertex3f(0.7,0.7,0)
				glVertex3f(-0.7,0.7,0)
				glEnd()

			if index in self.selected_images:
				glColor4f(0.5, 0.2, 0.2, 0.4)
				glBegin(GL_QUADS)
				glVertex3f(-0.7,-0.7,0)
				glVertex3f(0.7,-0.7,0)
				glVertex3f(0.7,0.7,0)
				glVertex3f(-0.7,0.7,0)
				glEnd()

			# draw the mask
			glEnable(GL_TEXTURE_2D)
			glCallList(self.mask.sprite)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			
			# draw the frame
			glColor4f(1,1,1,1)
			glCallList(self.frame.sprite)
			glPopMatrix()
		
	def update(self, trial, subtrial):
		self.sample_stimulation(trial, subtrial)
		
	def flick(self):		
		pass	
		
	def select(self, img):
		pass

	def add_selected_image(self, index, discard_previous):
		if discard_previous:
			self.selected_images = [index]
		else:
			if index in self.selected_images:
				self.selected_images.remove(index)
			else:
				self.selected_images.append(index)

	def update_indexes(self, indexes):
		self.indexer = SimpleImageSelect(indexes, 6, 5)

class P300BrowserTest():

	def __init__(self):
		"""Create the class and initialise variables that must be in place before
		pre_mainloop is called"""

		self.screen_w = 1150
		self.screen_h = 900
		self.screenPos = [0, 0]
		self._data = {}
		self.online_mode = True	# offline/online toggle

		self.block_count = 10
		self.trial_count = 10
		self.trial_highlight_duration = 1000
		self.trial_pause_duration = 2000
		self.subtrial_count = 6
		self.stimulation_duration = 100
		self.inter_stimulus_duration = 300
		self.inter_trial_duration = 5000
		self.inter_block_duration = 15000
		self.inter_block_countdown_duration = 3000
		self.highlight_count = 5
		self.subtrials_per_frame = 6 # nr of subtrials for displaying all images

		self.scoring_matrix = []
		for i in range(30): 	# IMAGE COUNT
			self.scoring_matrix.append([0.0 for x in range(self.subtrial_count/self.subtrials_per_frame)])
		self.total_subtrial_counter = 0

		self.viewer = None

		if preloaded_highlight_indexes:
			print "Loading highlights from highlights.csv!"
			self.highlight_indexes = preloaded_highlight_indexes
		else:
			self.highlight_indexes = []
			for i in range(self.trial_count):
				self.highlight_indexes.append([])
				for j in range(self.subtrial_count):
					self.highlight_indexes[i].append([k + (j*5) % 30 for k in range(self.highlight_count)])

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

		self.highlight_all_selected = False
		self._subtrial_scores_received = False

	def pre_mainloop(self):
		self.p300_setup()
		print "Initialised!"
		#if not self.online_mode:
		#	self.send_udp(P300_START_EXP)
		self.send_parallel(P300_START_EXP) # exp start marker
		time.sleep(1) # 1s pause after start exp marker
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
		self.photos = PhotoSet(os.path.join(os.getcwd(), "Feedbacks/P300PhotoBrowser/photos"))
		self.viewer = P300PhotoViewer(self.photos, self.highlight_indexes)
		self.fps = 60
		self.phase = 0
		
		#self.main_loop()
		
	# handles shutdown
	def quit(self):
		pygame.quit()		
		sys.exit(0)

	def draw_paused_text(self):
		glColor3f(1.0, 1.0, 1.0)
		#size = self.glfont.get_size(text)
		size = [1450,800]
		position = (self.w-size[0]/2, self.h-size[1]/2)
		glPushMatrix()
		glTranslatef(position[0], position[1], 0)
		self.glfont.render("Paused")
		glPopMatrix()

	def draw_finished_text(self):
		glColor3f(1.0, 1.0, 1.0)
		#size = self.glfont.get_size(text)
		size = [1550,800]
		position = (self.w-size[0]/2, self.h-size[1]/2)
		glPushMatrix()
		glTranslatef(position[0], position[1], 0)
		self.glfont.render("Ende")
		glPopMatrix()

	def draw_countdown_text(self, t):
		glColor3f(1.0, 1.0, 1.0)
		size = [1450, 600]
		position = (self.w-size[0]/2, self.h-size[1]/2)
		glPushMatrix()
		glTranslatef(position[0], position[1], 0)
		self.glfont.render("%d seconds..." % (1+(t/1000)))
		glPopMatrix() 

	def flip(self):
		# clear the transformation matrix, and clear the screen too
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		if self._state != STATE_BETWEEN_TRIALS and self._state != STATE_BETWEEN_BLOCKS and self._state != STATE_FINISHED:
			self.viewer.render(self.w, self.h)
		else:
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
		#	self.send_udp(P300_START_BLOCK)
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
		if self._trial_elapsed == 0:
			#if not self.online_mode:
			#	self.send_udp(P300_START_TRIAL)
			self.send_parallel(P300_START_TRIAL)
			self._current_target = random.randint(0, len(self.viewer.photo_set.photos())-1)
			self.viewer.set_highlight(self._current_target)
			print "> Starting trial %d" % self._current_trial
			# send marker for selected image
			time.sleep(0.02)
			#if not self.online_mode:
			#	self.send_udp(self.HIGHLIGHT_START)
			self.send_parallel(self.HIGHLIGHT_START)#+1+self.viewer.highlight)
			self._last_highlight = self.viewer.highlight
		
		self._trial_elapsed += self.elapsed

		if self._trial_elapsed > self.trial_highlight_duration:
			self.viewer.clear_highlight()

		if self._trial_elapsed > self.trial_pause_duration + self.trial_highlight_duration:
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
		for x in range(5):
			for y in range(6):
				if state[x][y]:
					self._current_highlights.append((x*6)+y)
		#print self._current_highlights
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
				self.send_udp('%d\n' % self.MARKER_START)
			else:
				print self._current_target
				print self._current_highlights
				if self._current_target in self._current_highlights:
					self.send_udp('S101\n')
				else: 
					self.send_udp('S  1\n')			

		elif not first_marker and self._current_highlights and self._current_highlights_index < len(self._current_highlights):
			print "Sending image marker %d" % (self.MARKER_START+1+self._current_highlights[self._current_highlights_index])
			# if this was the highlighted image
			if self._current_highlights[self._current_highlights_index] == self._last_highlight and not self.online_mode:
				self.send_parallel(self.HIGHLIGHT_START+1+self._last_highlight)
			else:			
				self.send_parallel(self.MARKER_START+1+self._current_highlights[self._current_highlights_index])
			self._current_highlights_index+=1
			self.markers_elapsed = 0
		
		if self._subtrial_pause_elapsed >= self.RESET_STATES_TIME and not self._markers_reset and not self.online_mode:
			self.send_udp('S  0\n')
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
			self._current_highlights = None
			self.update_scores([random.random()]) # XXX

			if self._current_subtrial >= self.subtrial_count:
				self._state = STATE_BETWEEN_TRIALS
				self._trial_elapsed = 0
				

	def handle_trial_wait(self):
		# send trial end marker
		# inter trial pause of 5000ms
		if self._trial_elapsed == 0:
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

				self.scoring_matrix = []
				for i in range(30): 	# IMAGE COUNT
					self.scoring_matrix.append([0.0 for x in range(self.subtrial_count/self.subtrials_per_frame)])

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

		self._block_elapsed += self.elapsed
		#print "> Block wait elapsed: %d" % self._block_elapsed

		if self._block_elapsed >= (self.inter_block_duration - self.inter_block_countdown_duration):
			# display countdown
			pass

		if self._block_elapsed >= self.inter_block_duration:
			# if on last block, stop
			if self._current_block >= self.block_count:
				self._state = STATE_FINISHED
			else:
				self._state = STATE_STARTING_TRIAL

	def handle_finished(self):
		if not self._finished:
			self.send_udp(P300_END_EXP)			
			self.send_parallel(P300_END_EXP)
			self._finished = True

		print "FINISHED"
		# Display "Ende" message

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
		else:
			print "*** Unknown state ***"

	def keyup(self,event):
		if event.key == K_ESCAPE:
			self.quit()
		if event.key == K_s:
			self.viewer.set_highlight(random.randint(0,24))
		if event.key== K_c:
			self.viewer.clear_highlight()

	def handle_events(self):
		for event in pygame.event.get():  
			if event.type==KEYDOWN:				
				if event.key==K_ESCAPE:
					self.quit()
			if event.type==KEYUP:
				self.keyup(event)
			if event.type == QUIT:
				self.quit()

	def update_scores(self, data):
		trial_number = int(self._current_trial)
		self.total_subtrial_counter += 1
		subtrial_number = int(self.total_subtrial_counter)
		
		print "Scores received for trial %d, subtrial %d" % (trial_number, subtrial_number)
		highlights_for_these_scores = self.highlight_indexes[trial_number][subtrial_number-1]

		# update scores for highlighted images
		score = data[0]

		print "Adding", score, " to images ", highlights_for_these_scores
		print "Frame", int((subtrial_number-1)/6)
		
		for i in highlights_for_these_scores:
			print "scoring_matrix[%d][%d] = %f" % (int(i), int(math.floor((subtrial_number-1)/self.subtrials_per_frame)), score)
			self.scoring_matrix[int(i)][int(math.floor((subtrial_number-1)/self.subtrials_per_frame))] = score

		print "Total subtrial count:", self.total_subtrial_counter

		if subtrial_number == self.subtrial_count:
			print "Last subtrial completed, calculating selected image"

			scoresMat = array(self.scoring_matrix)
			scoresMat = median(scoresMat,1)
			winner = scoresMat.argmin()
			winValue = scoresMat[winner]
							  
			#min = 999
			#minimg = 0
			#for i in range(30): # IMAGE COUNT
			#	s = self.scoring_matrix[i][subtrial_number-1]
			#		if s != 0.0 and s < min:
			#		min = s
			#		minimg = i

			print "Image %d was selected, score %f" % (winner, winValue)
			self.viewer.add_selected_image(winner, not self.highlight_all_selected)

			self._subtrial_scores_received = True
			self.total_subtrial_counter = 0

	def on_control_event(self,data):
		if data.has_key(u'cl_output'):
			score_data = data[u'cl_output']
			print score_data
			#XXX self.update_scores(score_data)
	
	def on_interaction_event(self, data):
		#self.scoring_matrix = []
		#for i in range(30): 	# IMAGE COUNT
		#	self.scoring_matrix.append([0.0 for x in range(self.subtrial_count/self.subtrials_per_frame)])
		#self.total_subtrial_counter = 0
		
		if data.has_key(u'highlight_indexes'):
			self.highlight_indexes
			if self.viewer != None:
				self.viewer.update_indexes(self.highlight_indexes
	

if __name__ == "__main__":
	p = P300BrowserTest()
	p.p300_setup()
	p.run()
