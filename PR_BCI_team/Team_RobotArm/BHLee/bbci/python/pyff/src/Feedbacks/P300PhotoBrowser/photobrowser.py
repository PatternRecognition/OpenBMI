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
        self.photo_list = photo_list
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
                thumb = pygame.transform.smoothscale(img, (64*aspect, 64))
                small = pygame.transform.smoothscale(img, (512*aspect, 512))
                
                pygame.image.save(thumb,thumb_name)
                pygame.image.save(small,small_name)
                
                del img, small
            else:
                thumb = GLSprite(thumb_name,real_size=True)
                
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
            
            c = GLSprite(os.path.join(self.cache, img+".small.png"), real_size = True)
            self.img_cache.add_element(img, c)
            return c
    
            
        
        
class PhotoViewer:
    def __init__(self, photo_set):
        self.photo_set = photo_set
        self.index = 0
        self.prev_image = None
        self.fade = 0
        self.gain = 0.5
        self.move_bar = 0.0
        self.move_base = 0.0
        #self.integrator = LeakyIntegrator(leakage=0.99, saturation=(-1,1))
        #self.move_chain = ProcessChain([self.integrator,FirstOrderLag(0.9),OperatorFilter(lambda x:math.pow(x,0.3))])
        
    def render(self, w, h):
    
        rotate = self.move_bar * 60
        x = 32 
        y = h/2
        i = self.index
        min_size = 1.0
        mid_index = self.index
        
        
        
        glPushMatrix()
        
        glColor4f(1,1,1,1)        
        # draw the necklace
        while x<w-10:
            d = (abs((x-w/2)/(float(w/2))))            
            if d<min_size:
                min_size = d
                mid_index = i
            scale = 1.0/(d*2+0.5)
            photo = self.photo_set.get_photo_name(i)
            thumb_sprite = self.photo_set.get_thumb(photo)
            glPushMatrix()
            y = h/8 + d*d*200*d
            glTranslatef(x+(thumb_sprite.w/2)*scale,y,0)
            glScalef(scale,scale,1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)                        
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glCallList(thumb_sprite.sprite)
            
            glPopMatrix()
            x = x + thumb_sprite.w*scale+5*scale
            i = i + 1
        glPopMatrix()
        
        # draw the mainimage        
        photo = self.photo_set.get_photo_name(mid_index)        
        img_sprite = self.photo_set.get_image(photo)   
        self.cur_image = img_sprite
        
        
        
        glColor4f(1,1,1,1-self.fade)        
        glPushMatrix()
        
        glTranslatef(w/2,h-40-img_sprite.h/2,0)        
        
        glCallList(img_sprite.sprite)        
        glPopMatrix()
        
        # draw fading old image
        if self.fade!=0:
            glColor4f(1,1,1,self.fade)        
            glPushMatrix()            
            glTranslatef(w/2,h-40-self.prev_image.h/2,0)                        
            glCallList(self.prev_image.sprite)
            glPopMatrix()
        
        
        
        # draw the pointer
        
        
        glDisable(GL_TEXTURE_2D)
        
        glTranslatef(w/2,h/4,0)
        glRotatef(rotate,0,0,1)
        
        glColor4f(1,0.5,0.5,0.3)        
        glBegin(GL_TRIANGLES)
        glVertex3f(0,-60,0)
        glVertex3f(-20,0,0)
        glVertex3f(20,0,0)
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor4f(1,1,1,0.5)        
        glBegin(GL_TRIANGLES)
        glVertex3f(0,-60,0)
        glVertex3f(-20,0,0)
        glVertex3f(20,0,0)
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_TEXTURE_2D)
        
        
        self.update()
        
        
    def update(self):
        # update fade constants
        if self.fade>0.02:
            self.fade = self.fade * 0.95
        else:
            self.fade = 0
            
        self.move_base = self.move_base* 0.96
        
        if self.move_base>1.0:
            self.move_base = 1.0
        if self.move_base<-1.0:
            self.move_base = -1.0
            
        if self.move_base!=0:
            sign = self.move_base/abs(self.move_base)
            self.move_bar = math.sqrt(abs(self.move_base))*sign
        else:
            self.move_bar = 0.0
        
        
                        
    def move(self, dir):        
        self.move_base = self.move_base + dir * self.gain                        
        if abs(self.move_bar)>=1.0:
            self.change_index(int(self.move_bar))
            self.move_base *= 0.2
            
            
        
    
    def change_index(self, dir=1):
        self.prev_image = self.cur_image
        self.fade = 1
        self.index = self.index + dir
        
            


class P300PhotoViewer:
    def __init__(self, photo_set):
        self.photo_set = photo_set
        self.index = 0
        self.visible_images = []
        self.image_space = {}
        self.layout()
        
        
        
    def layout(self):
        photos = self.photo_set.photos()
        xlen = len(math.sqrt(photos))
        ylen = len(math.sqrt(photos))
        x = 0
        y = 0
        for i in range(len(photos)):        
            self.image_space[photo] = (x,y)
            x = x + 1
            if x>xlen:
                x = 0
                y = y + 1
            
            
        
    def render(self, w, h):   
        self.update()
    
        
        
    def update(self):
        
        
                        
    def flick(self):        
        r = random.choice(self.visible_images)
        return r
            
        
    def select(self, img):
        pass
    
        
            
            
        
 
# Skeleton class                                          
class Skeleton:


    
    #initialize opengl with a simple ortho projection
    def init_opengl(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
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
        self.default_font = pygame.font.Font(default_font_name, 36)
        self.small_font = pygame.font.Font(default_font_name, 12)
        self.screen = pygame.display.set_mode((w,h) , pygame.OPENGL|pygame.DOUBLEBUF)                  
        #store screen size
        self.w = self.screen.get_width()
        self.h = self.screen.get_height()
        self.init_opengl()
       
      
        
    #initialise any surfaces that are required
    def init_surfaces(self):        
        pass
        
            

               
    # init routine, sets up the engine, then enters the main loop
    def __init__(self):    
        self.init_pygame(800,600)
        self.init_surfaces()
        self.photos = PhotoSet("photos")
        self.viewer = PhotoViewer(self.photos)
        self.fps = 60
        self.phase = 0
        self.clock = pygame.time.Clock()
        self.start_time = time.clock()
        self.main_loop()
        
    # handles shutdown
    def quit(self):
        pygame.quit()        
        sys.exit(0)
        
    # this is the redraw code. Add drawing code between the "LOCK" and "END LOCK" sections
    def flip(self):
          # clear the transformation matrix, and clear the screen too
          glMatrixMode(GL_MODELVIEW)
          glLoadIdentity()
          glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
          
          self.viewer.render(self.w, self.h)
          pygame.display.flip()
     
       
    # Get the array containing the up or down states for each key. For example, key_state[K_UP] is true if the up arrow key is pressed, [K_q] if Q is pressed, etc.
    def check_key_state(self):
          self.key_state = pygame.key.get_pressed()
          
            
           
    #frame loop. Called on every frame. all calculation shpuld be carried out here     
    def tick(self):          
          self.clock.tick(self.fps)                             
          self.handle_events()                   
          delta_t = time.clock() - self.start_time  
          self.start_time = time.clock()                  
          self.check_key_state()   
          if self.key_state[K_LEFT]:
            self.viewer.move(0.1) 
          if self.key_state[K_RIGHT]:
            self.viewer.move(-0.1) 
          self.flip()

          
          

   
    #returns last mouse position
    def get_mouse(self):
          return self.mouse_pos
     
    #Event handlers. These are called as events arrive
    def keydown(self,event):
          return
     
    def keyup(self,event):
          if event.key == K_ESCAPE:
               self.quit()
          
          return
          
    def mouseup(self,event):
          return
     
    def mousedown(self,event):
          return
          
    def mousemove(self,event):
          (self.mouse_pos) = event.pos
          return
    
   
    
    #event handling code. Calls the relevant handlers
    def handle_events(self):
          for event in pygame.event.get():  
               if event.type==KEYDOWN:                
                    if event.key==K_ESCAPE:
                        self.quit()
               if event.type==KEYUP:
                    self.keyup(event)
                    
               if event.type == QUIT:
                    self.quit()
               elif event.type == MOUSEBUTTONUP:
                    self.mouseup(event)
               elif event.type == MOUSEBUTTONDOWN:
                    self.mousedown(event)
               elif event.type == MOUSEMOTION:     
                    self.mousemove(event)

        
                    

         
    #main loop. Just runs tick until the program exits     
    def main_loop(self):
        while 1:
            self.tick()
         
     
#Create and run the engine     
s = Skeleton()


          



 