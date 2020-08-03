#!/usr/bin/env python
# -*- coding: utf-8 -*-

# HexoNavi/stuff.py -
# Copyright (C) 2009  Márton Danóczy
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

import pygame
import math
import random
import os.path

col_trans = (1,2,3)

class HexagonalGrid:
    """A hexagonal crid class
    
    Hexagonal coordinates are: 2 axes, a == 0°, b == -60°
    """

    axes_angle = [0, -math.pi/3.0]  # a == 0° and b == -60°
    axes = [(math.cos(phi), math.sin(phi)) for phi in axes_angle]

    sides_angle = [2*math.pi*i/6.0 for i in range(6)]
    sides       = [(math.cos(a)/2.0, math.sin(a)/2.0) for a in sides_angle]

    edges_angle = [2*math.pi*(i+0.5)/6.0 for i in range(6)]
    edges       = [(math.cos(a)/2.0, math.sin(a)/2.0) for a in edges_angle]
    xsize       = max(x for (x,y) in edges) - min(x for (x,y) in edges)
    ysize       = max(y for (x,y) in edges) - min(y for (x,y) in edges)
    
    def __init__(self, level_filename):
        self.read_level(level_filename)
        self.color_background = (64, 64, 64)
        self.fontColor = (192, 0, 0)
        self.hex_colors_inside = ((192, 192, 192), (32,32,32))
        self.hex_color_border = (0, 0, 0)
        self.hex_color_stimulus = (192, 255, 0)
        self.hex_width_border = 8
        
    def read_level(self, filename):
        path = os.path.dirname( globals()["__file__"] )
        infile = open(os.path.join(path,filename), "rb")
        infile.next()  #skip first line
            
        self.level = {}
        for line in infile:
            try:
                line = line.partition('#')[0].strip()   #remove comment
                if not line:
                    continue
                a,b,kind = line.split()
                pos = (int(a),int(b))
                if kind.isdigit():
                    self.level[pos] = int(kind)
                else:
                    self.__dict__[kind] = pos
            except:
                raise RuntimeError("Can't parse line " + line)
                
        
    def scale(self, screen, box, show_coordinates=False):
        
        #set hexagon size to unit hexagon
        self.radius = 1.0                    #distance edge to edge
        self.short  = math.sqrt(3.0/4.0)     #distance side to side
        self.origin = 0.0, 0.0
        
        #find dimensions of level in units of hexagon radius
        level_xy = [self.hexagonal2carthesian(a,b) for a,b in self.level.iterkeys()]
        level_x, level_y = zip(*level_xy)
        x_rng =  max(level_x) - min(level_x)
        y_rng =  max(level_y) - min(level_y)
        
        #calculate the level's optimal position on screen
        self.radius = min(box.width/(self.short + x_rng), box.height/(self.radius + y_rng))
        self.short  = self.radius * math.sqrt(3.0/4.0)
        self.origin = box.centerx-self.radius*x_rng/2.0, box.centery+self.radius*y_rng/2.0
        
        #create a surface for each kind of hexagon
        self.hex_surfaces = [self.create_surface(self.hex_width_border, self.hex_color_border, col)
                             for col in self.hex_colors_inside]

        self.stim_sprites = [Stimulus(self.create_stimulus_surface(stim, self.hex_width_border+1, self.hex_color_stimulus))
                              for stim in range(6)]
        
        self.surface = pygame.Surface(screen.get_rect().size)
        self.surface.fill(self.color_background)
        
        font = pygame.font.Font(None, 20)
            
        #and blit them all to the background surface in their respective position
        for (a,b),kind in self.level.iteritems():
            x,y = self.hexagonal2carthesian(a,b)
            self.surface.blit(self.hex_surfaces[kind], self.hex_surfaces[kind].get_rect(center = (x,y)))
            if show_coordinates:
                txt = font.render("a:%d, b:%d" % (a,b), 1, self.fontColor)
                self.surface.blit(txt, txt.get_rect(center=(x,y)))
        
        
    def create_surface(self, line_wid, line_col, fill_col):
        surf = pygame.Surface((self.xsize*self.radius + line_wid,
                               self.ysize*self.radius + line_wid))
        surf.fill(col_trans)
        center = surf.get_rect().center
        if line_wid>0:
            pygame.draw.polygon(surf, line_col, self.get_polygon_points(center, +line_wid/2.0))
            pygame.draw.polygon(surf, fill_col, self.get_polygon_points(center, -line_wid/2.0))
        else:
            pygame.draw.polygon(surf, fill_col, self.get_polygon_points(center))
        surf.set_colorkey(col_trans,pygame.RLEACCEL)
        return surf.convert()
    
    def create_stimulus_surface(self, stim, line_wid, line_col):
        r = self.radius/1.5
        s,h,t,w1,w2 = .25, .6, .5, .1, .25 #start, head, hook, wid1, wid2
        a = self.sides_angle[stim] 
        b = a + math.pi/2
        sa,ca,sb,cb = r*math.sin(a),r*math.cos(a),r*math.sin(b),r*math.cos(b)
        
        poly = [(ca, sa),
                (ca*t+cb*w2, sa*t+sb*w2), (ca*h+cb*w1, sa*h+sb*w1),
                (cb*w1, sb*w1), (-cb*w1, -sb*w1),
                (ca*h-cb*w1, sa*h-sb*w1), (ca*t-cb*w2, sa*t-sb*w2)]


        surf = pygame.Surface((r*2,r*2))
        surf.fill(col_trans)
        center = surf.get_rect().center
        poly = [(x+center[0], y+center[1]) for (x,y) in poly]
        pygame.draw.polygon(surf, line_col, poly)            
        surf.set_colorkey(col_trans,pygame.RLEACCEL)
        return surf.convert()

    def shiftpoly(self,poly,center):
        return 
    
    def get_polygon_points(self, center, displacement=0):
        w = self.radius + displacement
        return [(w*p[0]+center[0], w*p[1]+center[1]) for p in self.edges]
    
    def is_walkable(self, pos):
        try:
            return self.level[pos] == 0
        except KeyError:
            return False
    
    def get_accessible_hexagons(self, a, b):
        return [(a+1,b),(a+1,b-1),(a,b-1),(a-1,b),(a-1,b+1),(a,b+1)]
    
    def get_walkable_hexagons(self, a, b):
        all = self.get_accessible_hexagons(a, b)
        return set(p for p in all if self.is_walkable(p)) 
    
    def hexagonal2carthesian(self, a, b):
        a, b = a*self.short, b*self.short
        return (self.axes[0][0]*a + self.axes[1][0]*b + self.origin[0],
                self.axes[0][1]*a + self.axes[1][1]*b + self.origin[1])

    def carthesian2hexagonal(self, x, y):
        x,y = x-self.origin[0], y-self.origin[1]
        a = (x/self.axes[1][0] - y/self.axes[1][1]) / (self.axes[0][0] - self.axes[0][1]) 
        b = (x - a*self.axes[0][0])/self.axes[1][1]
        return a/self.short, b/self.short


class Stimulus(pygame.sprite.Sprite):
    
    def __init__(self, surf):
        pygame.sprite.Sprite.__init__(self)
        self.image = surf.convert()
        self.rect = self.image.get_rect()
        
    def update(self, pos):
        self.rect.center = pos
    

class Avatar(pygame.sprite.Sprite):
    
    def __init__(self, color):

        num_segments = 12
        size = 31.0
        center = (size+1.0)/2.0
        radius = (size-1.0)/2.0

        pygame.sprite.Sprite.__init__(self)
      
        surf = pygame.Surface((size, size))
        surf.fill(col_trans)
        surf.set_colorkey(col_trans, pygame.RLEACCEL)
        angles = [2*math.pi*i/num_segments for i in range(num_segments)]
        pygame.draw.polygon(surf, color, [(math.cos(a)*radius+center,math.sin(a)*radius+center) for a in angles])

        self.image = surf.convert()
        self.rect = self.image.get_rect()
        
    def update(self, pos):
        self.rect.center = pos
        
        
def generate_random_permutations(choices, n, npre=0, npost=0):
    """Generate n permutations of the sequence 'choices'
    
    Before/after the n permutations, npre/npost 'choices'
    are drawn with repetitions
    """
    result = []
    last = max(choices) + 1

    for s in range(npre):
        while True:
            c = random.choice(choices)
            if c != last:
                break
        result.append(c)
        last = c
    
    for s in range(n):
        perm = list(choices)
        while True:
            random.shuffle(perm)
            if perm[0] != last:
                break
        result = result + perm
        last = perm[-1]

    for s in range(npost):
        while True:
            c = random.choice(choices)
            if c != last:
                break
        result.append(c)
        last = c
    
    return result
