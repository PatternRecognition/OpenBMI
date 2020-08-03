"""PolygonStimulus and helper functions"""

import VisionEgg
import VisionEgg.ParameterTypes as ve_types
from VisionEgg.Core import *
from math import *
import OpenGL.GL as gl
# used to display text
import OpenGL.GLUT as glut
import pylab as py


class Hexagon(Stimulus):             
    """Displays Hexagon"""
    
    #defines the parameters to be used by hexagon
    parameters_and_defaults = {
        'on_or_off':(True,
            ve_types.Boolean,
            "draw stimulus? (Boolean)"),
     
        'radius':(64.0,
            ve_types.Real,
            "units: eye coordinates"),
        'center' : (None,
            ve_types.Sequence2(ve_types.Real),
            "DEPRECATED: don't use"),
        'corners' : (6,
            ve_types.Integer),

        'distance' : (100,
            ve_types.Integer),
        'screenX' : (340, ve_types.Integer),
        'screenY' : (340, ve_types.Integer), 
        'hexNumber' : (0, ve_types.Integer),   
        'characters' : ('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            ve_types.String),
        'sequenceOfLetters' : ('random', ve_types.String),
        'fix_cross':(False, ve_types.Boolean)
            
    }
    __slots__ = (
        '_gave_alpha_warning',
        )
         
    def __init__(self, **kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0  
        # Calculate points
        corners = self.parameters.corners
        r = self.parameters.radius
        offset = 360 / (corners * 2) # If there is two parallell sides, we want them horizontal
        angles = [radians(v) for v in range(offset, 360 + offset , 360 / corners)]
        self.points = [(r * sin(v), r * cos(v)) for v in angles ]

        if len(self.parameters.characters) == 26:
            oldSet = self.parameters.characters
            resultingSet = ""
            for i in range(6):
                #print(oldSet[ (i * 5):(i * 5 + 6) ])
                a = getIndexOfInnermost(i)
                #print(a)
                newString = insertCharacterAtGivenPosition(oldSet[ (i * 5):(i * 5 + 5) ], ' ', getIndexOfInnermost(i))
                resultingSet += newString
#            print(resultingSet)
#            print(len(resultingSet))
            while(len(resultingSet) != 34):
                resultingSet += ' '
            resultingSet += '<'
            resultingSet += '_'
#            print(len(resultingSet))
#            print(resultingSet)

            self.parameters.characters = resultingSet
        if len(self.parameters.characters) == 5:
            oldSet = self.parameters.characters
            oldSet += ' '
            self.parameters.characters = oldSet

        self.hexPositions = circleOfHexagons(self.parameters.distance, (self.parameters.screenY, self.parameters.screenX))
        self.character_positions = circleOfHexagons(self.parameters.radius * 0.35, (0.0, 0.0))
            
            
    def draw(self):
        p = self.parameters
        #if the hexagon is defined to be off, it is drown in white
        #otherwise it is drown in black
        if p.on_or_off:
            colors = [0.0, 0.0, 0.0]
            colors1=[1.0, 1.0, 1.0]
        else:
            colors = [1.0, 1.0, 1.0]
            colors1 = [0.0, 0.0, 0.0]
         
   
        
        #gets the position of the hexagon
        hexPositions = self.hexPositions
        #print(hexPositions)
        j=self.parameters.hexNumber
        #print(self.parameters.hexNumber)
        #print(p.characters)
        
        pos_x = hexPositions[self.parameters.hexNumber][0]
        pos_y = hexPositions[self.parameters.hexNumber][1]
        
        gl.glPushMatrix()
        gl.glLoadIdentity()
        center = (pos_x / 2, pos_y / 2)
        corners = self.parameters.corners
        r = self.parameters.radius        
        offset = 360 / (corners * 2)
        #gives the graphical card colors and draws it to the screen
        gl.glColor3f(colors[0], colors[1], colors[2])
        gl.glTranslate(center[0], center[1], 0.0)        
        angles = [radians(v) for v in range(offset, 360 + offset , 360 / corners)]
        self.points = [(r * sin(v), r * cos(v)) for v in angles ]      
        gl.glBegin(gl.GL_POLYGON)
        for (x, y) in self.points: 
            gl.glVertex3f(x, y, 0.0)        
        gl.glEnd()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glPopMatrix()

        character_positions = circleOfHexagons(p.radius * 0.35, (0.0, 0.0))
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(2.0)
        positionOffset = p.radius * 0.09
        gl.glColor3f(colors1[0], colors1[1], colors1[2])
        gl.glLineWidth(12.0)        
        
        if (self.parameters.sequenceOfLetters != "all"):        
            # put a list of letters inside the hexagons
            gl.glLineWidth(25.0)
            scalingFactor = 0.8
            gl.glPushMatrix()
            gl.glScalef(scalingFactor, scalingFactor, 0.0)
            gl.glTranslatef((center[0] - positionOffset * 3) * (1/scalingFactor), (center[1] - positionOffset * 3) * (1/scalingFactor), 0.0)
            #glut.glutBitmapCharacter(glut.GLUT_BITMAP_TIMES_ROMAN_24, "3")
            #glut.glutBitmapCharacter(glut.GLUT_BITMAP_TIMES_ROMAN_24, ord(p.characters[j + (6 * int(self.parameters.sequenceOfLetters))]))
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

            gl.glEnable(gl.GL_BLEND);
            gl.glEnable(gl.GL_LINE_SMOOTH);
            gl.glLineWidth(2.0);            
            glut.glutStrokeCharacter(glut.GLUT_STROKE_MONO_ROMAN, ord(p.characters[j + (6 * int(self.parameters.sequenceOfLetters))]))
            gl.glPopMatrix()
        elif (self.parameters.sequenceOfLetters == "all"):
            character_positions = self.character_positions
            # put a list of letters inside the hexagons
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glLineWidth(2.0)
            positionOffset = p.radius * 0.09
            gl.glColor3f(colors1[0], colors1[1], colors1[2])
            gl.glLineWidth(15.0)
            
            scalingFactor = 0.25
            gl.glLineWidth(15.0)
                        
            for i in range(6):
                
                gl.glPushMatrix()
                gl.glScalef(scalingFactor, scalingFactor, 0.0)
                gl.glTranslatef((center[0] - positionOffset) * (1/scalingFactor), (center[1] - positionOffset) * (1/scalingFactor), 0.0)
                
                x = character_positions[i][0]
                y = character_positions[i][1]
                gl.glTranslatef(x * (1/scalingFactor), y * (1/scalingFactor), 0.0)
                #glut.glutBitmapCharacter(glut.GLUT_BITMAP_TIMES_ROMAN_24, ord(p.characters[i + 6 * j]))
                #glut.glutStrokeCharacter(glut.GLUT_STROKE_MONO_ROMAN, ord(p.characters[j + (6 * int(self.parameters.sequenceOfLetters))]))
                #glut.glutStrokeCharacter(glut.GLUT_STROKE_ROMAN, ord(p.characters[i + 6 * j]))
                
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);

                gl.glEnable(gl.GL_BLEND);
                gl.glEnable(gl.GL_LINE_SMOOTH);
                gl.glLineWidth(2.0);
                glut.glutStrokeCharacter(glut.GLUT_STROKE_ROMAN, ord(p.characters[i + 6 * j]))

  
                gl.glPopMatrix()
        
def circleOfHexagons(radius, (x, y)):
        """Generate a sequence of hexagon center positions,
        that together forms a circle"""
        dX = radius * sin(radians(30)) + radius
        dY = radius * cos(radians(30))
    
        return [(x, y + dY * 2),
                (x + dX, y + dY),
                (x + dX, y - dY),
                (x, y - dY * 2),
                (x - dX, y - dY),
                (x - dX, y + dY)]
        
def insertCharacterAtGivenPosition(inputString, char, position):
        if position > len(inputString):
            return inputString
        else:
            newString = ""
            for i in range(position):
                newString += inputString[i]
            newString += str(char)
            for i in range(position, len(inputString)):
                newString += inputString[i]    
            return newString

def getIndexOfInnermost(indexOfHexagon):
        return (indexOfHexagon + 3) % 6