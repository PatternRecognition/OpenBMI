#!/usr/bin/env python

# SMR_NeuroFeedback.py -
# Copyright (C) 2010  Marton Danoczy
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import math

import pygame

from FeedbackBase.PygameFeedback import PygameFeedback


STATE_RUNNING = 1
STATE_TESTMODE = 2
STATE_COUNTDOWN = 3
STATE_TRIAL = 4
STATE_PAUSE = 5
STATE_END = 6

TRIGGER_START_EXP = 251
TRIGGER_END_EXP = 254
TRIGGER_START_SMR = 252
TRIGGER_PAUSE_SMR = 101

BUFFER_SIZE = 10000

class SMR_NeuroFeedback(PygameFeedback):
    
    def init(self):
        PygameFeedback.init(self)
        # Raw values as sent by the BCI
        self.values = [0.0, 0.0]
	self._buffer = range(0, BUFFER_SIZE)
	self.buffer_ptr = 0
	self.buffer_maxindex = 0
	self.percentiles = [0.15, 0.85]
        #
	#self.PauseColor = (178,34,34) # reddish
	self.PauseColor = (205,92,92) #sand red
        self.backgroundColor = [0x40, 0x40, 0x40] #grey
	self.borderColor = (190,190,190)
        #self.rectangleColor  = [0x00, 0x80, 0x00] #dark green
        #self.triangleColor   = [0x00, 0xc0, 0x00] # lighter green
	self.rectangleColor  = (173,216,230) #light blue
        self.triangleColor   = (193,205,193) # light violet
	#self.scoreBarColorRest = [0x00, 0x70, 0x70] #blue
	self.scoreBarColorRest = (222,184,135) #sand
	self.scoreBarColorHit = (205,92,92) #sand red
	#self.scoreBarColorHit = (255,215,0) #gold

        self.bar_width_fraction = 0.3 #width of the rectangle/triangle base
        self.fontsize = 130
        self.caption = "SMR Neurofeedback training" # caption on the top of the window
        # Normal trialmode vs endless trial
        self.testmode = False
        self.state = STATE_TESTMODE

	self.trialTime = 4*60
	self.countdown_duration=8
	self.current_state_duration = 0
        self.current_trial = 0
	self.pauseTime = 10
	self.startScoreBarPos = 0.1
	self.NewPos = 0
	
    
    def init_graphics(self):
        PygameFeedback.init_graphics(self)
	self.initScoreBar(self.startScoreBarPos)
	self.initScore = self.calc_initScoreBarval(self.startScoreBarPos)
	self.NewPos= self.initScore
    
    def pre_mainloop(self):
        PygameFeedback.pre_mainloop(self)
        self.send_parallel(TRIGGER_START_EXP)
        if not self.testmode:
	    self.state = STATE_COUNTDOWN
        else:
            self.state = STATE_TESTMODE
        self.clock.tick()

        
    def post_mainloop(self):
        PygameFeedback.post_mainloop(self)
        
    
    def tick(self):
        PygameFeedback.tick(self)
	self.current_state_duration += self.elapsed / 1000. # so the time gets in seconds 
       
    def play_tick(self):
	PygameFeedback.play_tick(self)

	if self.state == STATE_COUNTDOWN and self.current_state_duration >= self.countdown_duration:
            self.state = STATE_TRIAL
            self.current_state_duration = 0
	    self.send_parallel(TRIGGER_START_SMR)
            # Prepare next trial
            self.current_trial += 1
	elif self.state == STATE_TRIAL and self.current_state_duration >= self.trialTime:
		self.state= STATE_END
	   # self.state = STATE_PAUSE
	  #  self.current_state_duration = 0
	  #  self.send_parallel(TRIGGER_PAUSE_SMR)
#	elif self.state == STATE_PAUSE and self.current_state_duration >= self.pauseTime:
	 #   self.state = STATE_COUNTDOWN
	  #  self.current_state_duration = 0
    
	self.screen.fill(self.backgroundColor) # Blank the screen
      
        if self.state == STATE_COUNTDOWN:
            self.countdown_tick()
        elif self.state == STATE_TRIAL:
		self.NewPos = self.trial_tick(self.NewPos)
		self.calculate_smr_range()
	    
	elif self.state == STATE_PAUSE:
            self.pause_tick()
	    self.NewPos= self.initScore
   	elif self.state == STATE_END:
		self.write_smr_range()
		self.end_tick()
	
	
        pygame.display.flip() # Update the screen


    def trial_tick(self, BarHeight):
   
	SMRHeight = self.updateTrianglePos(self.values)

	if  SMRHeight <= BarHeight:

	    #if SMRHeight == BarHeight:
		#self.send_parallel(TRIGGER_REACHED_BAR)

	    newBarHeight = self.updateScoreBar(self.scoreBar,SMRHeight,self.scoreBarColorHit, 11)

	elif SMRHeight > BarHeight:
	    newBarHeight = self.updateScoreBar(self.scoreBar,BarHeight,self.scoreBarColorRest, 8)

	return newBarHeight


    def updateTrianglePos(self,val):

	w, h = self.screen.get_width(), self.screen.get_height()
        htri, hrec = self.values
	hrec = max(0,min(1,hrec))
	htri = max(0,min(1,htri))
        
        mid = 0.5*w
        l = mid*(1.0 - self.bar_width_fraction)
        r = mid*(1.0 + self.bar_width_fraction)
        yrec = h*(1.0-hrec)
        rectangle = pygame.Rect(l, yrec, w*self.bar_width_fraction, h-yrec-((h-yrec)/2)) #schwebende version
	#rectangle = pygame.Rect(l, yrec, w*self.bar_width_fraction, h-yrec) #normal version
	ytri = min(yrec,h*(1.0-htri))
	#old traingle version
	triangle = [ [l,yrec], [r,yrec], [mid,ytri] ]
	pygame.draw.rect(self.screen, self.rectangleColor, rectangle)
	pygame.draw.polygon(self.screen, self.triangleColor, triangle)

	## new version to put the spitze in a box and only update it
#	self.triangleSurface=pygame.Surface((w*self.bar_width_fraction,max(0,yrec-ytri)))
#	self.triangleRect=self.triangleSurface.get_rect(centerx=mid,centery=(yrec+ytri)/2)
#	self.triangleSurface.fill(self.backgroundColor)
#	h_surftri=self.triangleSurface.get_height()
#	w_surftri=self.triangleSurface.get_width()

#	triangle = [ [0,h_surftri], [0.5*w_surftri,0] , [w_surftri,h_surftri], ]


	## komische skalierung
        #if hrec>0.5:
        #    ytri = yrec - yrec*htri
        #else:
        #    ytri = yrec - (h*hrec)*htri

	#YData_slow= y2*2-1;
	# YData_fast= max(y1*2-1, YData_slow); 

        
#	pygame.draw.rect(self.screen, self.rectangleColor, rectangle)

	#pygame.draw.polygon(self.screen, self.triangleColor, triangle)
#	pygame.draw.polygon(self.triangleSurface, self.triangleColor, triangle)
	#self.triangleSurface.fill(self.triangleColor)
#	self.screen.blit(self.triangleSurface, self.triangleRect)
#	pygame.display.update(self.triangleRect)

	######## end of test###
	return yrec

    def initScoreBar(self,startPos):

	w, h = self.screen.get_width(), self.screen.get_height()
	startHeight=(1-startPos)*h

	self.scoreBar=pygame.Surface((w,h))
	self.scoreBar.set_colorkey((0,0,0))
	self.scoreBarRect=self.scoreBar.get_rect(center=self.screen.get_rect().center)

	pygame.draw.line(self.screen, self.scoreBarColorRest, (0,startHeight), (w,startHeight), 12)
	return startHeight

    def calc_initScoreBarval(self,initScore):
	w, h = self.screen.get_width(), self.screen.get_height()
	initBarHeight=(1-initScore)*h
	return initBarHeight

    def updateScoreBar(self, scoreBar, score, color, barWidth):
	w, h = self.screen.get_width(), self.screen.get_height()
	pygame.draw.line(self.screen, color, (0,score), (w,score), barWidth)	

	return score

    def pause_tick(self):
        PygameFeedback.pause_tick(self)
        self.print_center(self.screen, "Pause", self.PauseColor, self.fontsize)

    def end_tick(self):
        self.send_parallel(TRIGGER_END_EXP)
        self.print_center(self.screen, "End", self.PauseColor, self.fontsize)
		
        
    def countdown_tick(self):
        t = int(math.ceil(self.countdown_duration - self.current_state_duration))
        self.print_center(self.screen, str(t), self.PauseColor, self.fontsize)
        
    
    def on_control_event(self, data):
        self.values = data.get("cl_output", [0, 0])
	print data
    

    def print_center(self, srf, text, color, size):
        """Print the string on the surface."""
        font = pygame.font.Font(None, size)
        surface = font.render(text, 1, color)
        srf.blit(surface, surface.get_rect(center=srf.get_rect().center))

    def calculate_smr_range(self):
        self._buffer[self.buffer_ptr] = self.values[0]
        self.buffer_ptr = (self.buffer_ptr + 1) % BUFFER_SIZE
        if self.buffer_ptr > self.buffer_maxindex:
            self.buffer_maxindex = self.buffer_ptr
##	if self.values[0] < self.smr_range[0]:
##	      self.smr_range[0] = self.values[0]
##
##	if self.values[0] > self.smr_range[1]:
##	      self.smr_range[1] = self.values[0]

	
    def write_smr_range(self):
        del self._buffer[self.buffer_maxindex:BUFFER_SIZE]
        self._buffer.sort()
        pos = map(lambda x: int(x*self.buffer_maxindex), self.percentiles )
	text_file = open("d:/svn/bbci/acquisition/setups/smr_neurofeedback_season2/smr_range.txt", "w")
	text_file.write(str(self._buffer[pos[0]]) + "," + str(self._buffer[pos[1]]) + "\n")
	text_file.close()

	
        

if __name__ == "__main__":
    fb = SMR_NeuroFeedback()
    fb.on_init()
    fb.on_play()
