import sys,time,os,random,cPickle, math
import traceback

import pygame, thread
from pygame.locals import *

import thread
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


			
											  
							  
											  


	


# GL utilities	

# create a display list for  a sprite   
# for power of two sprites only
def create_sprite_list(texture, width, height):
	newList = glGenLists(1)
	glNewList(newList,GL_COMPILE);
	glBindTexture(GL_TEXTURE_2D, texture)
	glBegin(GL_QUADS)
	glTexCoord2f(0, 0); glVertex2f(-width/2, -height/2)	
	glTexCoord2f(0, 1); glVertex2f(-width/2, height/2)   
	glTexCoord2f(1, 1); glVertex2f( width/2,  height/2)	
	glTexCoord2f(1, 0); glVertex2f(width/2, -height/2)	
	glEnd()
	glEndList()	
	return newList
	

# create a display list for  a sprite   
# for power of two sprites only
def create_best_sprite_list(textureSurface, real_size= False, constant_height=False):

	width = textureSurface.get_width()
	height = textureSurface.get_height()	
	twidth = smallest2power(width)
	theight = smallest2power(height)
	
	textureSurface.convert_alpha()
	
	chunk = pygame.Surface((twidth, theight)).convert_alpha()
	
	chunk.fill((0,0,0,0))   
	
	
	chunk.blit(textureSurface, (0,0))
	
	texture = glGenTextures(1)
	textureData = pygame.image.tostring(chunk, "RGBA", 1)	
	glBindTexture(GL_TEXTURE_2D, texture)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, twidth, theight, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData )   
		
	
	newList = glGenLists(1)
	glNewList(newList,GL_COMPILE);
	glBindTexture(GL_TEXTURE_2D, texture)	
	
	
	if real_size:
		maxl = 1.0
	elif constant_height:
		maxl=max(width, height)
	else:
		maxl = max(width, height)
	
	
	region = (0, 0, width, height)
	
	w,h = 1.0,1.0
	glBindTexture(GL_TEXTURE_2D, texture)	
	glBegin(GL_QUADS)
	# NB odd co-ordinates here are becaause the OpenGL co-ordinates are y reversed compared to pygame surfaces
	
	glTexCoord2f(0, 0); glVertex2f(-w/2, -h/2)	
	glTexCoord2f(0, height/float(theight)); glVertex2f(-w/2, h/2)   
	glTexCoord2f(width/float(twidth), height/float(theight) ); glVertex2f( w/2,  h/2)	
	glTexCoord2f(width/float(twidth), 0); glVertex2f(w/2, -h/2)	
	
	
	# glTexCoord2f(0, 1); glVertex2f((region[0]-width/2)/float(width), -(region[1]-height/2)/float(height))	
	# glTexCoord2f(0, 0); glVertex2f((region[0]-width/2)/float(width), -(region[1]+region[3]-height/2)/float(height))   
	# glTexCoord2f(1, 0); glVertex2f((region[0]+region[2]-width/2)/float(width),  -(region[1]+region[3]-height/2)/float(height))	
	# glTexCoord2f(1, 1); glVertex2f((region[0]+region[2]-width/2)/float(width), -(region[1]-height/2)/float(height))	
	glEnd()
	glEndList()	
	return newList, texture, width, height
	
	
	
	
	
	
   
#recursively split a rectangle into power of 2xpower of 2 chunks
#of maximum size 1024x1024. 
def decompose(x, y, w, h):
	max_tile = 1024
	if w>0 and h>0:
		tile_w = max_tile
		tile_h = max_tile
		
		#find biggest fitting tile
		while tile_w>w:
			tile_w = tile_w/2
			
		while tile_h>h:
			tile_h = tile_h/2
			
			
		# map it like so:
		#
		#  <--tile_w----><w-tile_w>
		#  0000000000000011111111 ^
		#  0000000000000011111111 |
		#  0000000000000011111111 | tile_h
		#  0000000000000011111111 |
		#  0000000000000011111111 |
		#  0000000000000011111111 V
		#  2222222222222233333333 ^				
		#  2222222222222233333333 | h-tile_h
		#  2222222222222233333333 v
		#
		regions = [(x,y,tile_w,tile_h)]		
		regions = regions+decompose(x+tile_w, y, w-tile_w, h-(h-tile_h))
		regions = regions+decompose(x, y+tile_h, w-(w-tile_w), h-tile_h)
		regions = regions+decompose(x+tile_w, y+tile_h, w-tile_w, h-tile_h)
		return regions
	else:
		return []
		
		
# create a sprite list. If real_size is true, return quads in pixel space.
# else quads are scaled to unity
def create_general_sprite_list(textureSurface, real_size= False, constant_height=False):

		   
	width = textureSurface.get_width()
	height = textureSurface.get_height()	
	
	
	
	#split into textures
	textures = []	
	regions = decompose(0,0,width,height)
	
	# load each region into a texture
	for region in regions:

		# chop out the region
		chunk = pygame.Surface((region[2], region[3])).convert_alpha()
		chunk.fill((0,0,0,0))
		rect_region = Rect((region[0], region[1]), (region[2], region[3]))
		chunk.blit(textureSurface, (0,0), rect_region)
		w = chunk.get_width()
		h = chunk.get_height()
		
		
		# bind the image to the texture
		texture = glGenTextures(1)
		textureData = pygame.image.tostring(chunk, "RGBA", 1)	
		glBindTexture(GL_TEXTURE_2D, texture)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData )   
		textures.append(texture)
	

	#create the quads
	newList = glGenLists(1)
	glNewList(newList,GL_COMPILE);
	
	
	
	# set the scaling = -- 1.0 if realsized
	# for 2d sprites, must keep the true pixel size (i.e true size)
	# for 3d sprites, all images will be exactly 1 unit across
	# for constant height sprites, height = 1.0
	
	
	if real_size:
		maxl = 1.0
	elif constant_height:
		maxl=height
	else:
		maxl = max(width, height)
	
	
	i = 0
	
	# this would probably be a bit nicer as a triangle fan.
	# but then it would need to be ordered carefullly...
	for region in regions:
		glBindTexture(GL_TEXTURE_2D, textures[i])	
		glBegin(GL_QUADS)
		
		# NB odd co-ordinates here are becaause the OpenGL co-ordinates are y reversed compared to pygame surfaces
		glTexCoord2f(0, 1); glVertex2f((region[0]-width/2)/float(maxl), -(region[1]-height/2)/float(maxl))	
		glTexCoord2f(0, 0); glVertex2f((region[0]-width/2)/float(maxl), -(region[1]+region[3]-height/2)/float(maxl))   
		glTexCoord2f(1, 0); glVertex2f((region[0]+region[2]-width/2)/float(maxl),  -(region[1]+region[3]-height/2)/float(maxl))	
		glTexCoord2f(1, 1); glVertex2f((region[0]+region[2]-width/2)/float(maxl), -(region[1]-height/2)/float(maxl))	
		
	   
		glEnd()
		i = i + 1
	
	
	glEndList()	
	return newList, textures, width, height
	
	
	
	
	
	
	




	

#delete a call list
def del_sprite_list(list):
	glDeleteLists(list, 1)

	
# delete a texture
def del_texture(texture):
	glDeleteTextures(texture)
  
  
# return the smallest power of 2 > x
def smallest2power(x):
	return int(2**math.ceil(math.log(x)/math.log(2)))

  
  
# load a texture and return it
def load_texture_from_text(font, text, color):
	textSurface = font.render(text, False, color)
	textureSurface = pygame.Surface((smallest2power(textSurface.get_width()), smallest2power(textSurface.get_height()))).convert_alpha()	
	textureSurface.fill((0,0,0,1))
	textureSurface.blit(textSurface, (0,0))
	
	
	textureData = pygame.image.tostring(textureSurface, "RGBA", 1)	
	width = textSurface.get_width()
	height = textSurface.get_height()	
	texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureSurface.get_width(), textureSurface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData )   
	return texture, width, height		
		  


# load a texture and return it
def load_texture(fname):
	textureSurface = pygame.image.load(fname)	
	textureData = pygame.image.tostring(textureSurface, "RGBA", 1)	
	width = textureSurface.get_width()
	height = textureSurface.get_height()	
	texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData )   
	return texture, width, height		
		
		
	
# load a texture from a surface region and return it
def load_texture_from_surface(surface, region):
	textureSurface = pygame.Surface((region[2], region[3]))
	textureSurface.blit(surface, (0,0), Rect(region[0], region[1], region[2], region[3]))
	textureData = pygame.image.tostring(textureSurface, "RGBA", 1)	
	width = textureSurface.get_width()
	height = textureSurface.get_height()	
	texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData )   
	return texture, width, height	 

	
	

	
	
class GLCubeMap:

	# load a complete cube map, create lists 
	def __init__(self, fname):
		textureSurface = pygame.image.load(fname)	
		textureData = pygame.image.tostring(textureSurface, "RGBA", 1)	
		width = textureSurface.get_width()
		height = textureSurface.get_height()   
		
		w3 = width/3
		h4 = height/4
		
		# chop up the texture into the six regions. This only works if those
		# regions are power of 2 sized!
		
		front,w,h = load_texture_from_surface(textureSurface, (w3,0, w3, h4))
		left,w,h = load_texture_from_surface(textureSurface, (0,h4, w3, h4))
		top,w,h = load_texture_from_surface(textureSurface, (w3,h4, w3, h4))
		right,w,h = load_texture_from_surface(textureSurface, (w3*2,h4, w3, h4))
		back,w,h = load_texture_from_surface(textureSurface, (w3,h4*2, w3, h4))
		bottom,w,h = load_texture_from_surface(textureSurface, (w3,h4*3, w3, h4))
		self.textures = (top,left,front,right,bottom,back)
		
		
		#generate the list
		newList = glGenLists(1)
		glNewList(newList,GL_COMPILE);
						   
		# bind each edge to a cube face, with the correct orientation
		# the cube will be unit sized, centered at the origin
		
		glBindTexture(GL_TEXTURE_2D, top)	
		glBegin(GL_QUADS)				
		glTexCoord2f(0, 0); glVertex3f(-0.5, 0.5, -0.5)
		glTexCoord2f(0, 1); glVertex3f(-0.5, 0.5, 0.5)
		glTexCoord2f(1, 1); glVertex3f(0.5, 0.5, 0.5)
		glTexCoord2f(1, 0); glVertex3f(0.5, 0.5, -0.5)			   
		glEnd()
		
		
		glBindTexture(GL_TEXTURE_2D, bottom)	
		glBegin(GL_QUADS)				
		glTexCoord2f(0, 1); glVertex3f(-0.5, -0.5, -0.5)
		glTexCoord2f(0, 0); glVertex3f(-0.5, -0.5, 0.5)
		glTexCoord2f(1, 0); glVertex3f(0.5, -0.5, 0.5)
		glTexCoord2f(1, 1); glVertex3f(0.5, -0.5, -0.5)			   
		glEnd()
		
		
		glBindTexture(GL_TEXTURE_2D, front)	
		glBegin(GL_QUADS)				
		glTexCoord2f(0, 1); glVertex3f(-0.5, -0.5, 0.5)
		glTexCoord2f(0, 0); glVertex3f(-0.5, 0.5, 0.5)
		glTexCoord2f(1, 0); glVertex3f(0.5, 0.5, 0.5)
		glTexCoord2f(1, 1); glVertex3f(0.5, -0.5, 0.5)			   
		glEnd()
		
		glBindTexture(GL_TEXTURE_2D, back)	
		glBegin(GL_QUADS)				
		glTexCoord2f(0, 0); glVertex3f(-0.5, -0.5, -0.5)
		glTexCoord2f(0, 1); glVertex3f(-0.5, 0.5, -0.5)
		glTexCoord2f(1, 1); glVertex3f(0.5, 0.5, -0.5)
		glTexCoord2f(1, 0); glVertex3f(0.5, -0.5, -0.5)			   
		glEnd()
		
		glBindTexture(GL_TEXTURE_2D, left)	
		glBegin(GL_QUADS)				
		glTexCoord2f(0, 0); glVertex3f(-0.5, -0.5, -0.5)
		glTexCoord2f(0, 1); glVertex3f(-0.5, -0.5, 0.5)
		glTexCoord2f(1, 1); glVertex3f(-0.5, 0.5, 0.5)
		glTexCoord2f(1, 0); glVertex3f(-0.5, 0.5, -0.5)			   
		glEnd()
		
		glBindTexture(GL_TEXTURE_2D, right)	
		glBegin(GL_QUADS)				
		glTexCoord2f(0, 0); glVertex3f(0.5, -0.5, -0.5)
		glTexCoord2f(0, 1); glVertex3f(0.5, -0.5, 0.5)
		glTexCoord2f(1, 1); glVertex3f(0.5, 0.5, 0.5)
		glTexCoord2f(1, 0); glVertex3f(0.5, 0.5, -0.5)			   
		glEnd()
				
		glEndList()	
		
		
		
		self.cube = newList
	
		
		
		
		
		
	# free up the texture memory	
	def delete(self):
		for tex in self.textures:
			del_texture(tex)
		del_sprite_list(self.cube)
	
	
	

		
# create a sprite from a surface or a filename (auto-loads the image if it
# is a filename)
class GLSprite:
	def __init__(self, fname=None, surface=None, real_size=False, constant_height=False, split=False):		
	
		if not fname:
			textureSurface = surface
		else:
			textureSurface = pygame.image.load(fname)	
			
		if split:
			self.sprite, self.textures, self.w, self.h = create_general_sprite_list(textureSurface, real_size = real_size, constant_height=constant_height)
		else:
			self.sprite, t, self.w, self.h = create_best_sprite_list(textureSurface, real_size = real_size, constant_height=constant_height)
			self.textures = [t]
				
		
	def draw(self, centre=True):
		pass
		
	def delete(self):
		for tex in self.textures:
			del_texture(tex)
		del_sprite_list(self.sprite)
		
		
class GLButton:
	def __init__(self, fname):
		#(tex,w,h) = load_texture(fname)
	 
		textureSurface = pygame.image.load(fname)	
		w,h = textureSurface.get_width(), textureSurface.get_height()
		h2 = h/2
		topButton = pygame.Surface((w,h2)).convert_alpha()
		bottomButton = pygame.Surface((w,h2)).convert_alpha()
		
		topButton.fill(1,1,1,0)
		bottomButton.fill(1,1,1,0)
		
		topButton.blit(textureSurface, (0,0), Rect(0,0,w,h2))
		bottomButton.blit(textureSurface, (0,0), Rect(0,h/2,w,h2))
		
							
		self.top_button = GLSprite(surface=topButton, real_size=True)
		self.bottom_button = GLSprite(surface=bottomButton, real_size=True)
		
		
				
		
	def draw(self, centre=True):
		pass
		
	def delete(self):
		self.top_button.delete()
		self.bottom_button.delete()

		
		
# enter a 2d ortho drawing mode, with screen co-ordinates w,h
def begin_2d_mode(w,h):
		#switch to 2d
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		glOrtho(0, w, 0,  h, -1, 500)
		
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		
		
# restore the projection and modelview matrices
def end_2d_mode():
		#restore perspective projection
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()
		
	
		
	
# Simple font class. Precompute sprites for each character, and render them as required
class GLFont:
	def __init__(self, font, color):
		self.char_table = {}
		for i in range(32,127):
			c = chr(i)
			s = GLTextSprite(font, c, color)
			self.char_table[c] = s
			
	def render(self, text):		
		for ch in text:
			img = self.char_table.get(ch, None)
			if img:
				glCallList(img.sprite)
				glTranslatef(img.w, 0, 0)
				
		
				
	def newline(self):
		img = self.char_table.get('X', None)
		glTranslatef(0, img.h+2, 0)
		
	def get_size(self, text):
		w = 0
		h = 0
		for ch in text:
			w += self.char_table.get(ch, None).w
			h = self.char_table.get(ch, None).h

		return (w, h)
			
			
	
class GLTextSprite:
	def __init__(self, font, text, color):
		(tex,w,h) = load_texture_from_text(font, text, color)
		self.sprite = create_sprite_list(tex,w,h)
		self.tex = tex
		self.w = w
		self.h = h
		
		
	def draw(self, centre=True):
		pass
		
	def delete(self):
		delete_texture(self.tex)
		delete_sprite_list(self.sprite)
		

