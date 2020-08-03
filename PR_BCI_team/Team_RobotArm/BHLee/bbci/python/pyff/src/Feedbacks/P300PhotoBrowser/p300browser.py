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
	def __init__(self, max_photos):

		cache = os.path.dirname(globals()["__file__"]) + "/cache"
		dphotos = os.path.dirname(globals()["__file__"]) + "/photos"


		if not os.path.exists(cache):
			os.mkdir(cache)
		self.cache = cache
		self.max_photos = max_photos

		photo_list = []
		photos = os.listdir(dphotos)
		for photo in photos:
			path,fname = os.path.split(photo)
			base,ext = os.path.splitext(photo)
			if ext==".jpg" or ext==".jpeg":
				photo_list.append(photo)
		self.dphotos = dphotos

		# limit photo list to max_photos
		self.photo_list = photo_list
		if len(self.photo_list)>self.max_photos:
			self.photo_list = self.photo_list[:self.max_photos]
		else:
			self.photo_list = self.photo_list
		self.cache_images()

	def cache_images(self):
		self.thumbs = {}
		self.img_cache = Cache()
		for photo in self.photo_list:
			fname = os.path.join(self.dphotos, photo)

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

		for i in indexes:
			w = i
			h = 0
			while w >= (self.w):
				w = w - self.w
				h = h + 1

			state[w][h] = 1
#		for i in indexes:
#			state[i%self.w][i/(self.h+1)] = 1

		return state

# The main photoviewer class
class P300PhotoViewer:
	def __init__(self, photo_set, image_indexes, w, h, row, col, online_mode):
		self.photo_set = photo_set

		self.mask = GLSprite((os.path.dirname(globals()["__file__"]) + "/mask.png"), real_size=False)
		self.shadow = GLSprite((os.path.dirname(globals()["__file__"]) + "/shadow.png"), real_size=False)
		self.frame = GLSprite((os.path.dirname(globals()["__file__"]) + "/frame.png"), real_size=False)
		self.tick = GLSprite((os.path.dirname(globals()["__file__"]) + "/tick-icon.png"), real_size=False)
		self.index = 0
		self.visible_images = []
		self.image_space = {}
		self.stimulation_cycle_time = 0.1 #0.8
		self.stimulation_time = 0.2
		self.last_stimulation_time = time.clock()

		self.row = row
		self.col = col
		self.online_mode = online_mode

		self.indexer = SimpleImageSelect(image_indexes, self.col, self.row)

		self.image_density =  0.16666 #  0.1

		self.photo_size = 150 # default photo size in pixels

		self.flash_level = 0.4
		self.rotate_level = 10
		self.scale_level = 0.1
		self.image_spacing = 1.2
		self.highlight = None
		self.selected_images = []
		self.offset = 20

		if ((w / (self.col*self.image_spacing))*self.row) < (h - self.photo_size*self.image_spacing/2):
			self.photo_size = w / (self.col*self.image_spacing) - self.offset
		else:
			self.photo_size = h / (self.row*self.image_spacing) - self.offset

		self.fullscreen = False
		self._winner = -1

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
		xlen = self.col # math.sqrt(l)+1
		ylen = self.row # math.sqrt(l)
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

	def set_winner(self, winner):
		self._winner = winner
    	
	def render(self, w, h, rotation, brightness, enlarge, mask):
		if self.fullscreen:
			self.render_fullscreen(w, h)
		else:
			self.render_normal(w, h, rotation, brightness, enlarge, mask)

	def render_fullscreen(self, w, h):
		glClearColor(0.15,0.15,0.15,0)
		glClear(GL_COLOR_BUFFER_BIT)

		glTranslatef(180/2,h-180/2,0)
		glEnable(GL_TEXTURE_2D)
		glColor4f(1,1,1,1)
		glPushMatrix()

		if len(self.selected_images) > 0 or self._winner > -1:
			img = self.photo_set.get_image(self.photo_set.get_photo_name(self._winner))		
			aspect = img.h/float(img.w)
			iw = img.w / 5.0
			ih = img.h / (5.0*aspect)
			glTranslatef((w-iw)/2, -(h-ih)/2, 0)
			glScalef(self.photo_size*(1+self.scale_level), self.photo_size*(1+self.scale_level),1)

			glScalef(5,5*aspect,1)
			glCallList(img.sprite)	

		glPopMatrix()

		#glDisable(GL_TEXTURE_2D)


	def render_normal(self, w, h, rotation, brightness, enlarge, mask):
		glClearColor(0,0,0,0)
		glClear(GL_COLOR_BUFFER_BIT)

		step_scale = self.image_spacing
		size = self.photo_size
		xsize = self.photo_size*step_scale
		ysize = self.photo_size*step_scale

		border_size = 1.0
		photos = self.photo_set.photos()
		glTranslatef((xsize/2)+((w-xsize*self.col)/2),h-(ysize/2)-((h-ysize*self.row)/2),0)
		#glTranslatef(xsize/2,h-ysize/2,0)

		dt = 1.0-((time.clock()-self.last_stimulation_time) / self.stimulation_time)
		if dt<0.0:
			dt = 0.0
		simulation_v = self.stimulation_state * dt

		# set scaling paramters
		if rotation and enlarge and mask:
                    flash = self.flash_level
                else:
                    flash = self.flash_level*2
                    
		scale = self.scale_level
		jump = 0.0
		# todo: Set rotate_rate to 100 for cool jitter effect
		# rotate_rate = 100
		rotate_rate = 0
		rotate_amt = self.rotate_level
		rotate = rotate_amt*math.cos(time.clock()*rotate_rate)

		for index,photo in enumerate(photos):
			
			# switch
			if enlarge:
				scale = self.scale_level
			else:
				scale = 0

			(xc, yc) = self.image_space[photo]
			s = simulation_v[xc][yc]
			x,y = xsize*xc, -ysize*yc

			glPushMatrix()
			glTranslatef(x,y,0)
			#glTranslatef(x,y,0)
			thumb = self.photo_set.get_thumb(photo)
			glScalef(size*border_size*(1+s*scale), size*border_size*(1+s*scale),1)
			
			aspect = thumb.h/float(thumb.w)
			# set up the transformation
			glScalef(1,aspect,1)

			glTranslatef(0,jump*s,0)

			# draw the thumbnail
			glColor4f(1,1,1,1)
			glPushMatrix()
			glCallList(thumb.sprite)
			glPopMatrix()

			glDisable(GL_TEXTURE_2D)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE)

			# switch
			if brightness:
				glColor4f(1,1,1,s*flash)
			else:
				glColor4f(1,1,1,0)

			# Draw the flash as a textured quad
			glBegin(GL_QUADS)
			glVertex3f(-0.5,-0.5,0)
			glVertex3f(0.5,-0.5,0)
			glVertex3f(0.5,0.5,0)
			glVertex3f(-0.5,0.5,0)
			glEnd()

			# switch
			if rotation:
				glRotatef(s*rotate,0,0,1)

			if index==self.highlight:
				glColor4f(0.2,0.5,0.8,1.0)
				# Draw the flash as a textured quad
				glBegin(GL_QUADS)
				glVertex3f(-0.7,-0.7,0)
				glVertex3f(0.7,-0.7,0)
				glVertex3f(0.7,0.7,0)
				glVertex3f(-0.7,0.7,0)
				glEnd()

			# draw the mask
			# switch
			if mask:
				glEnable(GL_TEXTURE_2D)
				glCallList(self.mask.sprite)
				glBlendFunc(GL_ONE,GL_ZERO)
				#glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
				#glColor4f(1,1,1,s*flash)

			# sets the hook
			if (index in self.selected_images) and self.online_mode:
				glEnable(GL_TEXTURE_2D)
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
				glColor4f(1,1,1,1)
				glPushMatrix()
				glTranslatef(0.35, -0.4, 0)
				glScalef(0.5, 0.5, 1)
				glCallList(self.tick.sprite)
				glPopMatrix()
				#glColor4f(0.5, 0.2, 0.2, 1.0)
				#glBegin(GL_QUADS)
				#glVertex3f(-0.7,-0.7,0)
				#glVertex3f(0.7,-0.7,0)
				#glVertex3f(0.7,0.7,0)
				#glVertex3f(-0.7,0.7,0)
				#glEnd()
				pass

			# makes around corners
			glEnable(GL_TEXTURE_2D)
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

	def clear_selected_images(self):
		self.selected_images = []

	def add_selected_image(self, index, discard_previous):
		print "Adding selected image", index
		if discard_previous:
			self.selected_images = [index]
		else:
			if index in self.selected_images:
				self.selected_images.remove(index)
			else:
				self.selected_images.append(index)

	def update_indexes(self, indexes):
		self.indexer = SimpleImageSelect(indexes, self.w, self.h)
