
from GraphicComponentUtils import create_line, create_side, get_bounding_width_height

from direct.showbase.DirectObject import DirectObject
import direct.gui.OnscreenText as ost
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import TextNode, PandaNode, NodePath
from pandac.PandaModules import GeomVertexFormat, GeomVertexData, GeomVertexReader, GeomVertexWriter
from pandac.PandaModules import Geom, GeomNode, GeomTrifans, GeomTristrips, GeomLines, PointLight, VBase4
#global render, loader, base, taskMgr

class TextBoard():
    
    def __init__(self, text_color=(1,1,1), max_nr_rows=3, nr_char_per_row=20,
                 frame_color=(0,0,0), frame_padding=0.4, frame_line_width=2,
                 background_color=(1,1,0), background_padding=0.8):
        #print "TextBoard::init"
        # everything that belongs to this TextBoard will be stored under the root node
        self.root_node_path = render.attachNewNode(PandaNode(''))
        # create the text node that will be the TextBoard
        self.text_node = TextNode('')
        self.text_node_path = self.root_node_path.attachNewNode(self.text_node)
        self.set_max_nr_rows(max_nr_rows)
        r,g,b = text_color
        self.set_text_color(r, g, b)
        self.text_node.setAlign(TextNode.ALeft)  # TextNode.ALeft, TextNode.ACenterba
        letter_width, letter_height = self._compute_letter_size()
        self.max_row_length = nr_char_per_row * letter_width
        self.text_node.setWordwrap(self.max_row_length)
        width, height = self._compute_max_text_size()
        self.text_node_path.setPos(0.5*background_padding,-0.01,-letter_height)
        self.background_node_path = self.root_node_path.attachNewNode(PandaNode('background_node'))
        self._create_background(self.background_node_path, width+background_padding, height+background_padding+letter_height)
        self.frame_node_path = self.root_node_path.attachNewNode(PandaNode('frame_node'))
        self._create_frame(self.frame_node_path, width+background_padding, height+background_padding+letter_height)
        r,g,b = frame_color
        self.set_frame_color(r, g, b)
        self.set_frame_line_width(frame_line_width)
        r,g,b = background_color
        self.set_background_color(r, g, b)
    
    def _compute_max_text_size(self):
        self.text_node.setText(' ')
        width = 0
        while (not self.text_node.getWidth() == width) and self.text_node.getWidth() < self.max_row_length:
            width = self.text_node.getWidth()
            self.text_node.appendText(' ')
        while self.text_node.getNumRows() < self.text_node.getMaxRows():
            self.text_node.appendText('\n') 
        return self.text_node.getWidth(), self.text_node.getHeight()  
    
    def _compute_letter_size(self):
        self.text_node.setText('W')
        return self.text_node.getWidth(), self.text_node.getHeight()
        
    def _create_background(self, root_node_path, width, height):
        node, _vdata = create_side((0,0), (width, -height))
        root_node_path.attachNewNode(node)
        
    def _create_frame(self, root_node_path, width, height):
        # top
        node, _vdata = create_line(0, 0, width, 0)
        root_node_path.attachNewNode(node)
        # right side
        node, _vdata = create_line(width, 0, width, -height)
        root_node_path.attachNewNode(node)
        # bottom side
        node, _vdata = create_line(width, -height, 0, -height)
        root_node_path.attachNewNode(node)
        # right side
        node, _vdata = create_line(0, -height, 0, 0)
        root_node_path.attachNewNode(node)
        
    
    def get_node_path(self):
        return self.root_node_path
    
    def set_scale(self, scale):
        self.root_node_path.setScale(scale)
        
    def set_center_pos_xz(self, x, z):
        self.set_center_pos(x, 0, z)
        
    def set_center_pos(self, x, y, z):
        width, height = get_bounding_width_height(self.get_node_path())
        self.get_node_path().setPos(x-width/2.0, y, z+height/2.0)
        
    def set_text(self, text=''):
        self.text_node.clearText()
        self.text_node.setText(text)
            
    def set_text_color(self, r, g, b, alpha=1):
        self.text_node.setTextColor(r,g,b,alpha)
    
    def set_max_nr_rows(self, n):
        self.text_node.setMaxRows(n)
    
    def set_frame_visible(self, visible):
        rgba = self.frame_node_path.getColor()
        r = rgba[0]
        g = rgba[1]
        b = rgba[2]
        if visible:
            alpha = 1
        else:
            alpha = 0
        self.frame_node_path.setColor(r, g, b, alpha)
        
    def set_frame_color(self, r, g, b, alpha=1):
        self.frame_node_path.setColor(r, g, b, alpha)
        
    def set_frame_line_width(self, width):
        self.frame_node_path.setRenderModeThickness(width) 
        
    def set_background_visible(self, visible):
        rgba = self.background_node_path.getColor()
        r = rgba[0]
        g = rgba[1]
        b = rgba[2]
        if visible:
            alpha = 1
        else:
            alpha = 0
        self.background_node_path.setColor(r, g, b, alpha)
        
    def set_background_color(self, r, g, b, alpha=1):
        self.background_node_path.setColor(r,g,b,alpha)
        
        
if __name__ == "__main__":
    from direct.directbase import DirectStart
    board = TextBoard()
    text = "Text"
    text = "The most fundamental way to render text in Panda3D is via the TextNode interface."
    text = "This can also help to make the text easier to read when it is against a similar-colored background. Often, you will want the card to be semitransparent, which you can achieve by specifying an alpha value of 0.2 or 0.3 to the setCardColor() method. "
    board.set_text(text)
    board.set_scale(0.05)
    board.set_background_color(28/255.0, 6/255.0, 121/255.0)
    #board.get_node_path().setPos(0,0,-0.5)
    #board.get_node_path().place()
#    board.get_node_path().showBounds()
#    print board.get_node_path().getBounds()
#    board.get_node_path().showTightBounds()
    board.set_center_pos(0, 0, 0)

    base.disableMouse()
    base.camera.setPos(0,-3,0)

#    plight = PointLight('plight')
#    plight.setColor(VBase4(1, 1, 1, 1))
#    plnp = render.attachNewNode(plight)
#    plnp.setPos(0, -10, 0)
#    render.setLight(plnp)
    
    run()
    
