import math, sys, random, time
import pygame
from pygame.locals import *

sys.path.append("e:\\work\\pyff\\src\\Feedbacks\MultiFeedback")
from FeedbackBase.MainloopFeedback import MainloopFeedback
from glutils import *


# parallel port constants
# TARGET_LEFT = 1
# TARGET_RIGHT = 2
# HIT_CURSOR_LEFT = 11
# HIT_CURSOR_RIGHT = 12
# MISS_CURSOR_LEFT = 21                                   
# MISS_CURSOR_RIGHT = 22
# TRIAL_END_REJECT = 24
# COUNTDOWN_START = 30
# START_POSITION_CONTROL = 33
# HIT_FT_CURSOR_LEFT = 41
# HIT_FT_CURSOR_RIGHT = 42
# MISS_FT_CURSOR_LEFT = 51
# MISS_FT_CURSOR_RIGHT = 52
# INIT_FEEDBACK = 200
# GAME_STATUS_PLAY = 210
# GAME_STATUS_PAUSE = 211
# GAME_OVER = 254
   

class MultiScaleFeedbackGL(MainloopFeedback):

    def init(self):
        """Create the class and initialise variables that must be in place before
        pre_mainloop is called"""
        self.initialised=False
        self.screen_w = 800
        self.screen_h = 600
        self.fps = 50    
        self.trail_persist = 0.99
        self.bubble_mode = False
        self.bubble_height = 40
        self.bubble_min_width = 40
        self.perspective_mode = False
        self.glow_mode = False
        self.integration_mode = False
        self.integration_rate = 1.0
        self.scale = 1.0
        self.n_particles = 20
        self.x_spread = 2
        self.y_spread = 10
        self._states = []
        self.control_stream = 3
        
        self._data = {}
        
        
        
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
        self._back_buffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._back_buffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGB,0,0,1024,1024,0)
        
    
        
        
    def load_gl_textures(self):
        path = os.path.dirname(globals()["__file__"]) 
        self._large_bubble = GLSprite(os.path.join(path, "bubble_large.png"))
        self._small_bubble = GLSprite(os.path.join(path, "bubble_small.png"))
    
        
        
        
    def transparent_clear(self):
          """clear the buffers, using a semi transparent blit"""
          pass
    
        
        
    def on_interaction_event(self, data):
        """Called every time the data updates"""
        
        if self.initialised:
            self._target_width = self._w/6-10
        
        
    def text(self, text, position, color, center=False):        
        glColor3f(color[0], color[1], color[2])
        if center:
            size = self._font.get_size(text)
            self.position = (self.position[0]-size[0]/2, self.position[1]-size[1]/2)
        glPushMatrix()
        glTranslatef(position[0], position[1], 0)
        self._font.render(text)
        glPopMatrix()
        
        
    def load_circle(self):
        self.circle = glGenLists(1)
        glNewList(self.circle)       
        for i in range(64):        
            x = math.sin(i/64.0*(2*math.pi))/2.0
            y = math.cos(i/64.0*(2*math.pi))/2.0
            glVertex3f(x,y,0)
               
        glEndList()
    
    def pre_mainloop(self):
        
        # init pygame, load fonts
        pygame.init()
        default_font_name = pygame.font.match_font('bitstreamverasans', 'verdana', 'sans')
        if not default_font_name:           
            default_font_name = pygame.font.get_default_font()                
        self._default_font = pygame.font.Font(default_font_name, 36)
                
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.OPENGL|pygame.DOUBLEBUF)                
        self.init_opengl()
        self._font = GLFont(self._default_font, (255,255,255))
        self._clock = pygame.time.Clock()
        self.load_gl_textures()
        
        self._w = self._screen.get_width()
        self._h = self._screen.get_height()
        
        self.last_time = time.clock()
        self.initialised = True
        self._target_width = self._w/6-10

    def post_mainloop(self):
        pygame.quit()
        
                
        
    def integrate(self):
    
        # don't integrate if no data
        if not self._data.has_key(u"cl_output"):
            return
                    
        row = self._data[u"cl_output"]        
    
        # compute exact frame length
        new_time = time.clock()
        dt = self.last_time - new_time
        self.last_time = new_time
        
        #extend or contract the state vector as needed if the values change
        if len(self._states)!=len(row):
            self._states = [0]*len(row)
    
        # integrate
        if self.integration_mode:            
            for i, v in enumerate(row):
                self._states[i] = self._states[i]+v*dt*self.integration_rate
        else:
            for i, v in enumerate(row):
                self._states[i] = v*self.scale
                
        # clip
        for i in range(len(self._states)):
            if self._states[i]<-1.0:
                self._states[i] = -1
            if self._states[i]>1.0:
                self._states[i] = 1
            
        
    
    # always called every tick
    def tick(self):    
        self.integrate()
        self._clock.tick(self.fps)
        self.handle_events()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        
          
    def draw_target(self, left_side=True, color=(1,1,1)):
        w = self._target_width
        h = self._h/3
        
        if left_side:
            x = 0
        else:
            x = self._w - w
            
        glPushMatrix()
        glTranslatef(x,0, 0)
        glScalef(w,h,1)
        glDisable(GL_TEXTURE_2D)
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_QUADS)
        glVertex3f(0,0,0)
        glVertex3f(1,0,0)
        glVertex3f(1,1,0)
        glVertex3f(0,1,0)
        glEnd()
        glEnable(GL_TEXTURE_2D)
        glPopMatrix()
        
        
    def on_control_event(self,data):
        self._fade_level = 1-self.trail_persist
        
    def post_tick(self):

        pygame.display.flip()
    
    # called only when paused
    def pause_tick(self):
        glClearColor(0.5, 0.5, 0.5, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        self.text("Paused", (1,0.5,0.5), (self._w/2, self._h/2), center=True)
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
        for v in range(self.n_particles):
            
            for k,i in enumerate(row):       
                spread = random.gauss(0, self.x_spread)                
                v_spread = random.gauss(0,self.y_spread)
                color = pygame.Color(0,0,0,255)
                color.hsva = (360.0*(k/float(ncolors)),100,100,100)
                
                
                x =i*x_scale+self._w/2+spread
                y = self._h-80+v_spread
                
                
                glPushMatrix()
                
                glTranslatef(x, self.screen_h-y, 0)
                # highlight the controlled target
                if k==self.control_stream:
                    glScalef(20,20,20)
                else:
                    glScalef(8,8,8)
                glEnable(GL_TEXTURE_2D)
                
                if self.glow_mode:
                    glBlendFunc(GL_SRC_ALPHA,GL_ONE)
                
                glColor4f(color.r/256.0, color.g/256.0, color.b/256.0, 0.25)
                glCallList(self._small_bubble.sprite)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDisable(GL_TEXTURE_2D)
                glPopMatrix()
        
        
    def draw_bubble(self, row, x_scale):
        # compute bubble size
        minval = min(row)
        maxval = max(row)
        width = max([self.bubble_min_width, ((maxval-minval)*x_scale)])
        rect = Rect(minval*x_scale+self._w/2,self._h-100, width, self.bubble_height)                
        # color code the bubble
        dcolor = self.get_bubble_color(row)
            
        glColor4f(dcolor[0]/256.0, dcolor[1]/256.0, dcolor[2]/256.0, 0.9)
        glEnable(GL_TEXTURE_2D)
        glPushMatrix()
        glTranslatef(minval*x_scale+self._w/2+width/2,100, 0)
        glScalef(width, self.bubble_height, 1)
        glCallList(self._large_bubble.sprite)
        glPopMatrix()
        
    
    
    
    # copy a version
    def copy_shift_fade(self):
        
        # fade
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_TEXTURE_2D)
        glColor4f(0,0,0,self._fade_level)
        glBegin(GL_QUADS)
        glVertex3f(0,0,0)
        glVertex3f(self._w,0,0)
        glVertex3f(self._w,self._h,0)
        glVertex3f(0,self._h,0)
        glEnd()
        
        # copy
        glViewport(0,0,1024,1024)
        glBindTexture(GL_TEXTURE_2D, self._back_buffer)
        glCopyTexSubImage2D(GL_TEXTURE_2D,0,self._target_width,0,self._target_width,0,self._w-(self._target_width*2),1024)
        glViewport(0,0,self._w, self._h)
        
        #shift draw
        glTranslatef(0,2,0)
        
        glTranslatef(self._w/2, self._h/2, 0)        
        
        if self.perspective_mode:
            glScalef(0.97,0.97,1)
        glTranslatef(-self._w/2, -self._h/2, 0)
       
        glColor4f(1,1,1,1)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._back_buffer)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(0,0,0)
        glTexCoord2f(1, 0)
        glVertex3f(1024,0,0)
        glTexCoord2f(1, 1)
        glVertex3f(1024,1024,0)
        glTexCoord2f(0, 1)
        glVertex3f(0,1024,0)
        
        glEnd()
        
        
        
    
    
    # called only when not paused
    def play_tick(self):
        #glClearColor(0,0,0, 1)
        #glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        self.copy_shift_fade()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        
        # NB depends on classifier being -1.0--1.0!
        # will scale the range to fit in 2/3 of screen width        
        x_scale = self._w/3       
        
        class_outputs = self._states
        if self.bubble_mode:
            self.draw_bubble(class_outputs, x_scale)
        else:
            self.draw_particles(class_outputs, x_scale)

        self.draw_target(left_side=True)
        self.draw_target(left_side=False)
        self.post_tick()
        
    def handle_events(self):
          """Make sure the event loop is emptied"""
          for event in pygame.event.get():
            pass
    