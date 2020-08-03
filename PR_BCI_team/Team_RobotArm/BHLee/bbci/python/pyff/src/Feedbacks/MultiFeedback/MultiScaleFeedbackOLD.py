import math, sys, random
import pygame
from pygame.locals import *
import glutils

sys.path.append("e:\\work\\pyff\\src\\Feedbacks\MultiFeedback")
from FeedbackBase.MainloopFeedback import MainloopFeedback


class MultiScaleFeedback(MainloopFeedback):

    def init(self):
        """Create the class and initialise variables that must be in place before
        pre_mainloop is called"""
        self.initialised=False
        self.screen_w = 800
        self.screen_h = 600
        self.fps = 25        
        self.trail_persist = 0.9
        self.bubble_mode = False
        self.bubble_height = 40
        self.bubble_min_width = 40
        self.x_spread = 5
        self.y_spread = 5
	self.stepsize = 0.05
	self.relative = True
	self.bias = 0.3
        self._data = {}
	self._prevData = {}
        
        
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
        
        
    def text(self, text, position, color, center=False):        
        textimage = self._default_font.render(text, True, color)
        if center:
            self.position = (self.position[0]-textimage.get_width()/2, self.position[1]-textimage.get_height()/2)
        else:
            self.screen.blit(text, position)
    
   
        
    
    
    
    def pre_mainloop(self):
        
        # init pygame, load fonts
        pygame.init()
        default_font_name = pygame.font.match_font('bitstreamverasans', 'verdana', 'sans')
        if not default_font_name:           
            default_font_name = pygame.font.get_default_font()                
        self._default_font = pygame.font.Font(default_font_name, 36)
        self._small_font = pygame.font.Font(default_font_name, 18)
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))                
        self._clock = pygame.time.Clock()
        self.init_surfaces()
        self._w = self._screen.get_width()
        self._h = self._screen.get_height()
        self.initialised = True

    def post_mainloop(self):
        pygame.quit()
        
                
        
    # always called every tick
    def tick(self):
        self._clock.tick(self.fps)
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
                color.hsla = (360.0*(k/float(ncolors)),50,50,100)
                pygame.draw.circle(self._draw_buf, color, (i*x_scale+self._w/2+spread, self._h-80+v_spread), 2)
                
        
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
    
    # called only when not paused
    def play_tick(self):
        self.transparent_clear()
        
        # NB depends on classifier being -1.0--1.0!
        # will scale the range to fit in 2/3 of screen width        
        x_scale = self._w/3       
        
        if not self._data.has_key(u"cl_output"):
            return
            
        class_outputs = self._data[u"cl_output"]
	
	if len(self._prevData) == 0:
		self._prevData = [0]*10
	
	for i,j in enumerate(class_outputs):
		class_outputs[i] += self.bias

		if self.relative:
			class_outputs[i] = self._prevData[i]+cmp(class_outputs[i],0)*self.stepsize
			if class_outputs[i] < -1:
				class_outputs[i] = -1
			if class_outputs[i] > 1:
				class_outputs[i] = 1

			self._prevData[i] = class_outputs[i]
        

        if self.bubble_mode:
            self.draw_bubble(class_outputs, x_scale)
        else:
            self.draw_particles(class_outputs, x_scale)

        # blit and finish
        self._screen.blit(self._draw_buf, (0,0))        
      
        
        
        self.post_tick()
        
    def handle_events(self):
          """Make sure the event loop is emptied"""
          for event in pygame.event.get():
            pass
    
