import sys, time, os, random, math
import pygame
import liquid_sim
import numpy
import soundmodule
from FeedbackBase.MainloopFeedback import MainloopFeedback


STATE_NOTHING = - 1
STATE_HOLDING = 0
STATE_MOVING = 1
STATE_RESPONSE = 2
STATE_STARTING = 3
STATE_FINISHED = 4

TRIGGER_LEFT_ANNOUNCED = 1      #- left target announced
TRIGGER_RIGHT_ANNOUNCED = 2     #- right target announced
TRIGGER_BOTTOM_ANNOUNCED = 3    #- bottom target announced

TRIGGER_LEFT_CORRECT = 11       #- left target (correctly) touched
TRIGGER_RIGHT_CORRECT = 12      #- right target (correctly) touched
TRIGGER_BOTTOM_CORRECT = 13     #- bottom target (correctly) touched

TRIGGER_LEFT_ERROR = 21         #- left target erroneously touched
TRIGGER_RIGHT_ERROR = 22        #- right target erroneously touched
TRIGGER_BOTTOM_ERROR = 23       #- bottom target erroneously touched
TRIGGER_NOTHING_ERROR = 24

TRIGGER_MOVE_START = 60         #- Cursor starts to move 
TRIGGER_START = 210             #- Recording starts
TRIGGER_STOP = 212              #- Recording stops

    
class Attractor:
    def __init__(self, x, y, strength):
        self.x = x
        self.y = y
        self.strength = strength
        self.value = 0
        
        
class Liquid_Feedback(MainloopFeedback):
    """Liquid Feedback."""

    def init(self):
        self.init_trial()        
        self.init_parameters()
        
        self.sim = None
        self.data = [0, 0, 0, 0, 0, 0, 0, 0, 0]


    def pre_mainloop(self):
        self.init_engine()
        self.logger.debug("Playing.")
        self.start_trial()


    def post_mainloop(self):
        self.finish()
        self.quit()
        self.logger.debug("Quitting.")


    def tick(self):
        self.handle_events()
        self.clock.tick(self.fps)


    def pause_tick(self):
        pass


    def play_tick(self):
        # update trial state
        if self.trial_type > 0:
            if self.state == STATE_FINISHED:
                return            
            if time.clock() > self.next_change_time:
                self.next_action()
            self.last_i_time = time.clock()
                    
        delta_t = time.clock() - self.start_time  
        self.start_time = time.clock()                  

        #update liquid dynamics
        if self.trial_type == 0:
            self.sim.updateDynamics()          
        if self.trial_type > 0:
            #in trials, only update in the moving state; stay unchanged in other states
            if self.state == STATE_MOVING:
                self.sim.updateDynamics()          
            else:
                pass
            #draw
            self.trial_flip()
        else:
            self.test_flip()

        #sonify
        self.sonify()
                
        #map values to attractors
        if len(self.data) >= len(self.attractors):
            i = 0
            for attractor in self.attractors:                
                attractor.value = self.clip(self.data[i])
                i = i + 1
        else:
            self.logger.error("Too short data vector!")
                        

    def on_interaction_event(self, data):
        if self.sim:
            self.update_parms()

        
    def on_control_event(self, data):
        self.data = data["cl_output"]
        h = 0
        for d in self.data:
            h = h - self.clip(d) * math.log(self.clip(d) + 1e-6)
        self.entropy = h

################################################################################
# /Feedback methods   
################################################################################
    
    def doCountdown(self):
        self.countdown = self.countdown_start - time.clock() + self.counter_time + 1
        if self.countdown <= 1:
            self.setAction(self.showTarget, time.clock())
            self.in_countdown = 0
        else:
            self.setAction(self.doCountdown, time.clock())
        
        
    def runFeedback(self):
        self.setAction(self.runFeedback, time.clock())               
        self.checkTargets()

        
    def start_trial(self):
        self.logger.debug("Starting Trial.")
            
        self.send_parallel(TRIGGER_START)            
        self.resetLiquid()
        for i in range(len(self.colors)):
            self.colors [i] = self.grey                        
        self.trial_no = 0        
    
        self.state = STATE_STARTING        
        self.countdown = 0
        self.in_countdown = 0        
        self.setAction(self.startCountdown, time.clock() + 0.1)
            
            
    def setAction(self, action, when):
        self.next_action = action
        self.next_change_time = when

        
    def startCountdown(self):        
        self.in_countdown = 1
        self.countdown_start = time.clock()        
        self.countdown = self.counter_time
        self.setAction(self.doCountdown, time.clock())
        
        
    def finish(self):
        self.state = STATE_FINISHED
        
        
    def startFeedback(self):
        self.resetLiquid()
        self.targetColors()
        self.send_parallel(TRIGGER_MOVE_START)
        self.last_i_time = time.clock()
        self.target_start_time = time.clock()
        self.state = STATE_MOVING        
        self.runFeedback()       
        
        
    def showResponse(self):   
        self.state = STATE_HOLDING
        self.last_target = self.target       
        self.resetLiquid()
        self.trial_no = self.trial_no + 1
        if self.trial_no > self.n_trials:            
            self.setAction(self.finish, time.clock())
            self.send_parallel(TRIGGER_STOP)
        else:
            self.responseColors()        
            self.setAction(self.showTarget, time.clock() + self.response_time)
            
    
    def showTarget(self):
        self.target = self.rng.choice([0, 1, 2])
        
        if self.target == 0:
            self.send_parallel(TRIGGER_LEFT_ANNOUNCED)
        elif self.target == 1:
            self.send_parallel(TRIGGER_RIGHT_ANNOUNCED)            
        elif self.target == 2:
            self.send_parallel(TRIGGER_BOTTOM_ANNOUNCED)
        
        self.targetColors()
        self.setAction(self.startFeedback, time.clock() + self.hold_time)
        
        
    def responseColors(self):
        #put the right colors on
        for i in range(len(self.colors)):
            self.colors[i] = self.grey
        
        # hit_target is the target number we struck
        if self.correct:        
            self.colors[self.hit_target] = self.hit_color
        else:
            self.colors[self.hit_target] = self.miss_color
        
        
    def targetColors(self):
        for i in range(len(self.attractors)):
            self.colors [i] = self.grey            
        self.colors[self.target] = self.target_color
                
                
    def distance(self, a, b):
        d = math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y))
        return d

        
    def checkTargets(self):
        """Check if we hit the targets.
        
        If so, send the trigger, log the hit, and move to show the response and
        update the target.
        """        
        hit = False                     
        # compute where the mass lies (i.e the particle centers)
        counts = [0] * len(self.attractors)
        
        for drop in self.sim.drops:
            i = 0
            for attractor in self.attractors:
                if self.distance(attractor, drop) < self.trial_target_radius:
                    #increase count by one drops' worth
                    counts[i] = counts[i] + 1.0 / self.liquid_droplets
                i = i + 1
            
        # find the target with most mass
        max_count = 0
        max_index = - 1
        i = 0
        for count in counts:
            if count > max_count:
                max_count = count
                max_index = i
            i = i + 1
            
        if max_index >= 0 and counts[max_index] > self.trial_mass_ratio:          
            self.hit_target = max_index
        else:
            self.hit_target = 3                                                                                  
               
        # if time trial, we check as soon as time is up
        if self.trial_type == 1:
            if time.clock() - self.target_start_time > self.trial_time:
                hit = True
               
        #otherwise, we need to check whether the mass is above the threshold
        if self.trial_type == 2:
            if counts[max_index] > self.trial_mass_ratio:
                hit = True
                
        if hit:
            if self.hit_target == self.target:
                self.correct = True
                if self.target == 0:
                    self.send_parallel(TRIGGER_LEFT_CORRECT)                
                if self.target == 1:
                    self.send_parallel(TRIGGER_RIGHT_CORRECT)                
                if self.target == 2:
                    self.send_parallel(TRIGGER_BOTTOM_CORRECT)                
            else:
                self.correct = False
                if self.hit_target == 0:
                    self.send_parallel(TRIGGER_LEFT_ERROR)                
                if self.hit_target == 1:
                    self.send_parallel(TRIGGER_RIGHT_ERROR)                
                if self.hit_target == 2:
                    self.send_parallel(TRIGGER_BOTTOM_ERROR)                
                if self.hit_target == 3:
                    self.send_parallel(TRIGGER_NOTHING_ERROR)                
            self.setAction(self.showResponse, time.clock())                
            
       
    def init_parameters(self):
        # FIXME: ist diese Methode wirklich noetig?

        # Determines size of field of inter-particle repulsion
        self.liquid_repulsion_scale = - 10
        # Determines size of field of inter-particle attraction
        self.liquid_attraction_scale = - 920
        # Determines strength of inter-particle repulsion
        self.liquid_repulsion_strength = 0.018
        # Determines strength of inter-particle attraction
        self.liquid_attraction_strength = 1.9
        # Spread of liquid at start (in pixels)
        self.start_spread = 200
        # Do not adjust! 
        self.liquid_min_distance = 5
        self.temperature = 0
        # Number of droplets in liquid
        self.liquid_droplets = 80
        # Damping factor 0--1
        # 0 = no movement
        # 1 = no damping
        self.friction = 0.95
        # Strength of force moving the liquid. 
        # Higher = faster movement
        self.attractor_scale = 1.8
        # Size of droplet lookup table. Do not adjust
        self.liquid_droplet_size = 100
        # Strength of a droplet (i.e. scaling of droplet function)               
        self.liquid_droplet_strength = 22
        # Screen size
        self.screen_width = 800
        self.screen_height = math.sqrt(self.screen_width**2 - (self.screen_width/2.0)**2)
        # If true, will run in fullscreen mode
        self.fullscreen = False
        # Relative strength of corner attractors. Do not adjust
        self.attract1_strength = 1
        self.attract2_strength = 1
        self.attract3_strength = 1
        # Time for countdown at start of trial (seconds)
        self.counter_time =  5
        # Time cursor is held before movement (while target is visible)
        self.hold_time =  0.5
        # Time cursor is held after selection complete
        self.response_time =  0.5
        # Shape of the space. -1 -- 1. 
        # 0 = triangle
        # 0.5 = star
        # -1.0 = hexagon
        self.triangle_shape = 0.0
        # Relative speed of the system
        self.speed = 1.0
        # If true, will produce splashing sounds
        self.sound = False
        # Amount of damping applied as a function of entropy. 0 -- 1. 
        # 0, no damping is applied. 
        # 1, high entropy greatly damps movement.
        self.entropy_damping = 0.0
        # trial_type can be 
        # 0 (just testing) 
        # 1 (timed trials) 
        # 2 (mass inside target radius)
        self.trial_type = 1
        # number of trials in a session
        self.n_trials = 30
        # only used if trial_type is 1, seconds until one selection is complete
        self.trial_time =  4
        # ratio of mass that must be inside the target radius
        self.trial_mass_ratio = 0.5
        # size of the corner target regions in pixels
        self.trial_target_radius = 100      
    
    
    def init_trial(self):
        self.rng = random.Random()
        
        #target colors
        self.grey = (128, 128, 128)
        self.target_color = (0, 0, 255)
        self.hit_color = (0, 255, 0)
        self.miss_color = (255, 0, 0)
                         
        self.colors = [self.grey, self.grey, self.grey, self.grey]
   
        self.correct = False # was the target hit correctly
        self.target = 0 # which target are we aiming for        

        
    def compute_restrictor_triangles(self, p1, p2, p3, delta):
        """Compute the inside triangles of a larger triangle. 
        
        Delta specifies the size of the internal triangles. 
        Must be 0---1. 1 = completely filled, 0 = no internal triangles
        """        
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p3[0]
        y3 = p3[1]

        crx = (x1 + x2 + x3) / 3
        cry = (y1 + y2 + y3) / 3
                
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        
        cx2 = (x2 + x3) / 2
        cy2 = (y2 + y3) / 2
        
        cx3 = (x1 + x3) / 2
        cy3 = (y1 + y3) / 2
        
        v1x1 = (delta) * crx + (1 - delta) * cx1
        v1y1 = (delta) * cry + (1 - delta) * cy1
        
        v2x1 = (delta) * crx + (1 - delta) * cx2
        v2y1 = (delta) * cry + (1 - delta) * cy2
        
        v3x1 = (delta) * crx + (1 - delta) * cx3
        v3y1 = (delta) * cry + (1 - delta) * cy3
        
        v1 = ((v1x1, v1y1), (x1, y1), (x2, y2))
        v2 = ((v2x1, v2y1), (x2, y2), (x3, y3))
        v3 = ((v3x1, v3y1), (x1, y1), (x3, y3))        
        return (v1, v2, v3)
        
        
    def rescale(self, t, scl):
        """Rescale a triangle by the given factor."""
        k = []
        for vertex in t:
            v = list(vertex)
            v[0] = v[0] - self.screen_width / 2
            v[1] = v[1] - self.screen_height / 2
            v[0] *= scl
            v[1] *= scl
            v[0] = v[0] + self.screen_width / 2
            v[1] = v[1] + self.screen_height / 2
            k.append(v)
        return tuple(k)
            
        
    def gen_triangle_list(self, delta):
        """Generate the list of obstacle triangles."""
        t1 = ((self.attractors[0].x, self.attractors[0].y),
                    (self.attractors[1].x, self.attractors[1].y),
                    (self.attractors[2].x, self.attractors[2].y))
        (t2, t3, t4) = self.compute_restrictor_triangles(t1[0], t1[1], t1[2], delta)
        
        if self.triangle_shape < 0:
            t1 = self.rescale(t1, 1.01)
        self. triangles = [t1, t2, t3, t4]
        
        
    def isInside(self, x, y):
        inside = self.inside_triangle(self.triangles[0], x, y)
        
        for t in self.triangles[1:]:
            if self.inside_triangle(t, x, y):
                if self.triangle_shape > 0:
                    inside = False
                else:
                    inside = True
        return inside

        
    def getField(self, x, y, dx, dy):        
        """Defines the forces acting upon the molecules."""                   
        fx = 0
        fy = 0
        
        w = self.sim.getWidth()
        h = self.sim.getHeight()                        
        
        if not self.isInside(x, y):                                                           
            fx = (x - w / 2) 
            fy = (y - h / 2) 
            flen = math.sqrt(fx * fx + fy * fy)
            fx /= flen
            fy /= flen
            fx *= - 1
            fy *= - 1                                              
        ax = 0
        ay = 0
        for attractor in self.attractors:
            d = math.sqrt((x - attractor.x) * (x - attractor.x) + (y - attractor.y) * (y - attractor.y))
            dx = ((x - attractor.x) / d) * attractor.strength * attractor.value 
            dy = ((y - attractor.y) / d) * attractor.strength * attractor.value
            ax = ax - dx
            ay = ay - dy
                              
        return (fx + ax * self.attractor_scale, fy + ay * self.attractor_scale)
        
        
    def getFriction(self, x, y, dx, dy):
        delta = self.entropy_damping * 1 / (self.entropy * 2 + 1) + (1 - self.entropy_damping) 
        if not self.isInside(x + dx, y + dy):
            dx = x - self.screen_width / 2
            dy = y - self.screen_height / 2
            mag = math.sqrt(dx * dx + dy * dy)
            dx /= mag *- 5
            dy /= mag *- 5
            return (dx, dy)
        else:
            return (self.friction * dx * delta, self.friction * dy * delta)
        
    
    def init_pygame(self):
        """Initialise pygame and load the fonts."""
        pygame.init()
        default_font_name = pygame.font.match_font('bitstreamverasansmono', 'verdana', 'sans')
        if not default_font_name:           
            self.default_font_name = pygame.font.get_default_font()  
        self.default_font = pygame.font.Font(default_font_name, 72)
        self.small_font = pygame.font.Font(default_font_name, 12)
        
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)                  
        else:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))                  
        #store screen size
        self.w = self.screen.get_width()
        self.h = self.screen.get_height()

      
    def init_surfaces(self):
        """Initialise any surfaces that are required."""        
        self.draw_buf = pygame.Surface(self.screen.get_size())                    
        self.draw_buf.fill((0, 0, 0))
        self.back_buf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self.back_buf = self.back_buf.convert_alpha()

 
    def inside_triangle(self, t, x0, y0):
        ((x1, y1), (x2, y2), (x3, y3)) = t
        #barycentric co-ordinates test
        b0 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        
        if abs(b0) < 0.0001:
            b0 = 0.0001 
        b1 = ((x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0)) / float(b0)
        b2 = ((x3 - x0) * (y1 - y0) - (x1 - x0) * (y3 - y0)) / float(b0)
        b3 = ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / float(b0)
        
        return b3 > 0 and b1 > 0 and b2 > 0            


    def init_sim(self):                
        self.sim = liquid_sim.LiquidSimulator(width=int(self.screen_width), height=int(self.screen_height), field_fn=self, friction_fn=self)        
        boundary = numpy.zeros((self.sim.getWidth(), self.sim.getHeight()))        

        self.gen_triangle_list(self.triangle_shape)
        #should be triangular, with vertices at the attractor centres        
        #expensive!
        for i in range(self.sim.getWidth()):
            for j in range(self.sim.getHeight()):
                if not self.isInside(i, j):
                    boundary[i, j] = 1
                    pygame.draw.line(self.draw_buf, (200, 200, 200), (i, j), (i, j), 1)

        self.sim.setBoundary(boundary)
                        
        self.update_parms()
        for i in range(self.liquid_droplets):        
            self.sim.addDroplet(x=int(random.random()*self.start_spread) + self.screen_width / 2, y=int(random.random()*self.start_spread) + self.screen_height / 2, width=self.liquid_droplet_size, height=self.liquid_droplet_size, strength=self.liquid_droplet_strength)
        #for i in range(self.liquid_droplets):
        #    self.sim.drops[i].fmult = random.random()/2 + 0.5

            
    def update_parms(self):
        self.sim.setProperty('repulsion_scale', self.liquid_repulsion_scale)
        self.sim.setProperty('attraction_scale', self.liquid_attraction_scale)
        self.sim.setProperty('attraction_strength', self.liquid_attraction_strength)
        self.sim.setProperty('repulsion_strength', self.liquid_repulsion_strength)
        self.sim.setProperty('min_distance', self.liquid_min_distance)
        self.sim.setProperty('temperature', self.temperature)
        
        self.sim.setProperty('dt', self.speed)
        self.screen_width = int(self.screen_width)
        self.screen_height = int(self.screen_height)
        
        
    def resetLiquid(self):
        sx = 0
        sy = 0
        for attractor in self.attractors:
            sx = sx + attractor.x / float(len(self.attractors))
            sy = sy + attractor.y / float(len(self.attractors))
        
        for i in range(self.liquid_droplets):                    
            self.sim.setDroplet(i, x=int(random.random()*self.start_spread - self.start_spread / 2) + sx, y=int(random.random()*self.start_spread - self.start_spread / 2) + sy)
            
        self.sim.updateDynamics()

        
    def init_engine(self):    
        """Set up the engine."""
        self.init_pygame()
        self.init_surfaces()
        
        # set the attractor positions (these depend on the screen size!)
        margin = 50
        self._attract1_x = margin
        self._attract1_y = margin
        
        self._attract2_x = self.screen_width - margin
        self._attract2_y = margin
        
        #compute third point so that the outer triangle is equilateral
        baseline = self.screen_width - 2 * margin
        
        ax = math.cos(2 * math.pi / 6) * baseline + margin
        ay = math.sin(2 * math.pi / 6) * baseline + margin
        
        self._attract3_x = ax
        self._attract3_y = ay
        
        self.fps = 60
        self.entropy = 0
        self.clock = pygame.time.Clock()
        
        self.start_time = time.clock()
        
        self.attractors = []

        #create the attractors
        self.attractors.append(Attractor(self._attract1_x, self._attract1_y, self.attract1_strength))
        self.attractors.append(Attractor(self._attract2_x, self._attract2_y, self.attract2_strength))
        self.attractors.append(Attractor(self._attract3_x, self._attract3_y, self.attract3_strength))
                
        #create the sonification 
        dir = os.path.dirname(globals()["__file__"])        
        self.sound_module = soundmodule.SoundModule(os.path.join(dir, "sounds"))
        self.sound_module.load_bank('water')
        self.init_sim()
        
        #get ready!
        self.state = STATE_NOTHING

        
    def quit(self):
        """Handle shutdown."""
        self.logger.debug("quit")
        pygame.quit()        
                     

    def trial_flip(self):
        """Draw the screen in trial mode."""
        self.screen.blit(self.draw_buf, (0, 0))
        self.screen.lock()       
        #LOCK          
        radius = self.trial_target_radius          
        #draw the corner items
        i = 0
        for attractor in self.attractors:                    
            #show correct colors in trial mode
            col = self.colors[i]
            i = i + 1
            pygame.draw.circle(self.screen, (255, 0, 0), (attractor.x, attractor.y), radius, 1)
            pygame.draw.circle(self.screen, col, (attractor.x, attractor.y), 45)

        if self.state == STATE_MOVING:
            liquid_color = (70, 90, 180)
        else:
            liquid_color = (100, 100, 100)                    
        #draw the liquid image
        polys = self.sim.getPolygons(y=(self.h - self.sim.getHeight()) / 2, step=10, height=7e6)
                              
        for poly in polys:                        
            if len(poly) > 2:                                
                pygame.draw.polygon(self.screen, liquid_color, poly)
                pygame.draw.aalines(self.screen, (255, 255, 255), 1, poly)
                                                             
        #END LOCK
        self.screen.unlock()         
          
        #draw countdown (must be done outside of lock)
        if self.state == STATE_STARTING:
            text = self.default_font.render(str(int(self.countdown)), 1, (255, 10, 10))            
            textpos = text.get_rect(centerx=self.screen_width / 2, centery=self.screen_height / 2)            
            self.screen.blit(text, textpos)
        if self.state == STATE_FINISHED:
            text = self.default_font.render("Finished", 1, (255, 10, 10))            
            textpos = text.get_rect(centerx=self.screen_width / 2, centery=self.screen_height / 2)            
            self.screen.blit(text, textpos)
        pygame.display.update()     

     
    def test_flip(self):
        """Draw the screen in non-trial mode."""
        self.screen.blit(self.draw_buf, (0, 0))
        self.screen.lock()       
        #LOCK
        radius = 45
        #draw the corner items
        for attractor in self.attractors:                    
            col = (attractor.value * 200, attractor.value * 200, attractor.value * 200)
            pygame.draw.circle(self.screen, col, (attractor.x, attractor.y), radius)
                                           
        #draw the liquid image
        polys = self.sim.getPolygons(y=(self.h - self.sim.getHeight()) / 2, step=10, height=7e6)
        for poly in polys:                        
            if len(poly) > 2:                                
                pygame.draw.polygon(self.screen, (100, 150, 180), poly)
                pygame.draw.aalines(self.screen, (255, 255, 255), 1, poly)

        #END LOCK
        self.screen.unlock()
        pygame.display.update()
     
     
    def sonify(self):
        """Sonify the droplets."""
        if self.state == STATE_MOVING and self.sound:
            for mol in self.sim.drops:
                if mol.energy > 30:
                    self.sound_module.play_from_bank('water', mol.energy)

          
    def clip(self, var):
        """Return a value between 0..1.

        if var < 0 return 0.0
        if var > 1 return 1.0
        else return var
        """
        if var > 1.0: 
            return 1.0
        if var < 0.0: 
            return 0.0
        return var

        
    def handle_events(self):
        """Handle events and call relevant handlers."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            else:
                pass


        
if __name__ == "__main__":
    lfb = Liquid_Feedback(None)
    lfb.on_init()
    lfb.on_play()
                                                                                                                                                                                  
