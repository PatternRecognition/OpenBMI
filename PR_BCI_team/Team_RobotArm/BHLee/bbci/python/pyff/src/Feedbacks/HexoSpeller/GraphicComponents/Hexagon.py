from .. import Utils
#from Utils import degrees_to_radians, rotate_phi_degrees_clockwise
from GraphicComponentUtils import center_node, center_node_on_xyz, get_center_point, create_hexagon, create_side
from math import cos
from direct.showbase.DirectObject import DirectObject
import direct.gui.OnscreenText as ost
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import TextNode, PandaNode, NodePath
from pandac.PandaModules import GeomVertexFormat, GeomVertexData
from pandac.PandaModules import Geom, GeomNode, GeomTrifans, GeomTristrips, GeomVertexWriter, PointLight, VBase4
#from include import TextNode, PointLight, PandaNode
#from include import GeomVertexFormat, GeomVertexData
#from include import Geom, GeomNode, GeomTrifans, GeomTristrips, GeomVertexWriter
#from direct.VBase4_extensions import VBase4_extensions as VBase4
#from direct.NodePath_extensions import NodePath_extensions as NodePath
#global render, loader


class Hexagon():
    
    text_distance = 0.01
    
    def __init__(self, radius=1, width=0.4, color=(1,1,0), hex_index=0):
        """ Creates a hexagon, centered around the origin with radius being the distance to the vertices. """
        #print "Hexagon::init"
        self._create_index_and_pos_lists(hex_index)
        self.front_side_symbols = []
        # everything that belongs to this hexagon will be stored under the root node
        self.root_node_path = render.attachNewNode(PandaNode(''))
        # create the geometry node which is the hexagon surface
        self.hex_node_path = self._create_hexagon(self.root_node_path, radius, width)
        r,g,b = color
        self.set_color(r, g, b)
        # create the text fields
        self.back_side_text = None
        self.front_side_text = list()
        self._create_text_fields(radius, width) 
        # clear all text fields 
        for text_path in self.front_side_text:
            text_path.node().setText('')   
        self.back_side_text.node().setText('')   
        #self.root_node_path.setTwoSided(True)
        
    def _create_index_and_pos_lists(self, hex_index):
        """ idx will refer to the list index of the symbol list that is to be shown on the front side of the hexagon. 
        This list is a preference list, i.e. the first symbol will be placed at the best reachable position and the last
        symbol of the list will be placed at position that is most difficult to reach. Which positions are easy to reach
        depends on the hex_index of this hexagon, because it determines the position of this hexagon within the big hexagon.
        pos will refer to positions on the hexagon. There are six possible starting at the top, then going around clockwise. 
        They are indexed from 0 to 5.
        
        self.idx_to_pos - the mapping from list indices to positions on the hexagon
        self.pos_to_idx - the mapping from positions on the hexagon back to list indices
        
        Since there are less indices than positions on the hexagon (5 symbols, 6 positions), pos_to_idx will contain a None for
        the position that does not correspond to a list index. This position is opposite to the best position. 
        """
        self.hex_index = hex_index
        self.free_spot = (hex_index  + 3) % 6
        self.idx_to_pos = [i%6 for i in range(hex_index, hex_index+6)]
        self.idx_to_pos.remove(self.free_spot)
        self.pos_to_idx = [-1 for _i in range(6)] # create a six element list
        self.pos_to_idx[self.free_spot] = None
        for i, idx in enumerate(self.idx_to_pos):
            self.pos_to_idx[idx] = i
        
    def _create_hexagon(self, root_path, radius, width):
        hex_path = root_path.attachNewNode(PandaNode('hexagon'))
        # create the individual components of the hexagon and put them together
        front_side = create_hexagon(radius)[0]
        node_path = hex_path.attachNewNode(front_side)
        node_path.setY(-width/2.0)
        node_path.setR(30)
        back_side = create_hexagon(radius)[0]
        node_path = hex_path.attachNewNode(back_side)
        node_path.setH(180)
        node_path.setY(width/2.0)
        node_path.setR(30)
        side_dist = cos(Utils.degrees_to_radians(30))*radius
        for phi in range(0,360,60):
            rot_path = hex_path.attachNewNode(PandaNode(''))
            side = create_side((-radius/2.0, width/2.0), (radius/2.0, -width/2.0))[0]
            side_path = rot_path.attachNewNode(side)
            side_path.setP(-90)
            side_path.setZ(side_dist)
            rot_path.setR(phi)
        return hex_path
        
        
    def _create_text_fields(self, radius, width):
        test_letter = 'W'
        # the larger back side text field
        x,z = 0,0
        y = (width/2.0+Hexagon.text_distance)
        text = TextNode('')
        #font = loader.loadFont("cmss12.egg")
        #text.setFont(font)
        text.setGlyphScale(1.1*radius)
        text.setTextColor(0,0,0,1)
        self.back_side_text = self.root_node_path.attachNewNode(text)
        self.back_side_text.node().setText(test_letter)
        self.back_side_text.setH(180)
        center_node_on_xyz(self.back_side_text, x, y, z)
        self.back_side_text_z = self.back_side_text.getZ()
        # the six front side text fields
        self.front_side_text_coordinates = []
        for _i, phi in enumerate(range(0,360,60)):
            text = TextNode('')
            #text.setFont(font)
            text.setGlyphScale(0.45*radius)
            text.setTextColor(0,0,0,1)
            text_path = self.root_node_path.attachNewNode(text)
            self.front_side_text.append(text_path)
            x,z = Utils.rotate_phi_degrees_clockwise(phi, (0,radius/1.6))
            text_path.node().setText(test_letter)
            center_node_on_xyz(text_path, x, -y, z)
            self.front_side_text_coordinates.append((x,-y,z))
            
    
    def set_front_side_symbols(self, symbols):
        """ Sets the given list of symbols on the front side of this hexagon. """
        self.front_side_symbols = symbols
        for idx, symbol in enumerate(symbols):
            if idx < 5:
                self.set_front_side_symbol(idx, symbol)    
        
    def set_front_side_symbol(self, idx, symbol):
        """ Sets the given symbol on the front side of the hexagon. Idx must be smaller than 5 and determines the reachability of the symbol, 
        which depends on the hex_index, i.e. the position, of this hexagon. Idx=0 positions the symbol at the best reachable position, pos=4 
        puts the symbol at worst reachable position. """
        if idx > 4:
            return
        pos = self.idx_to_pos[idx]
        self.front_side_text[pos].node().setText(str(symbol))
        x,y,z = self.front_side_text_coordinates[pos]
        center_node_on_xyz(self.front_side_text[pos], x, y, z)
    
    def get_symbol(self, selected_pos):
        """ Returns the symbol that is currently displayed at the given position pos. Positions are counted from the top clockwise, i.e.
        0 is top, 3 is bottom, 5 is top left, etc. The returned symbol may be None! """
        if selected_pos == self.free_spot:
            symbol = None
        else:
            symbol = self.front_side_symbols[self.pos_to_idx[selected_pos]]
        return symbol
            
        
    def set_back_side_symbol(self, symbol):
        """ Sets the given symbol in large on the back side of this hexagon. """
        self.back_side_text.node().setText(str(symbol)) 
        y = self.back_side_text.getY()   
        center_node_on_xyz(self.back_side_text, 0, 0, 0)
        self.back_side_text.setY(y)
        self.back_side_text.setZ(self.back_side_text_z)
    
    def set_pos(self, x, y, z):
        """ Set the position of the hexagon. """
        self.root_node_path.setPos(x,y,z)
    
    def set_scale(self, val):
        """ Set the scale of this hexagon. """
        self.root_node_path.setScale(val)
        
    def set_front_side_symbols_scale(self, val):
        for text_path in self.front_side_text:
            text_path.node().setGlyphScale(val)
    
    def set_back_side_symbol_scale(self, val):
        self.back_side_text.node().setGlyphScale(val)
            
    def set_color(self, r, g, b, alpha=1):
        self.root_node_path.setColor(r,g,b,alpha)
        
    def set_text_color(self, r, g, b, alpha=1):
        self.set_back_side_text_color(r, g, b, alpha)
        self.set_front_side_text_color(r, g, b, alpha)
        
    def set_back_side_text_color(self, r, g, b, alpha=1):
        self.back_side_text.setColor(r,g,b,alpha)
    
    def set_front_side_text_color(self, r, g, b, alpha=1):
        for text_path in self.front_side_text:
            text_path.setColor(r,g,b,alpha)
    
    def get_node_path(self):
        """ Returns the node path to the top node of the hexagon sub-tree. """
        return self.root_node_path


def rotate(node_path, rotation_per_second, start_angle, task):
    v = 360*rotation_per_second*task.time + start_angle
    v = v%360
    node_path.setH(v)
    
    return Task.cont


if __name__ == "__main__":
    from direct.directbase import DirectStart
    hex_1 = Hexagon(radius=1, hex_index=1)
    hex_1.set_front_side_symbols(["A","B","C","D","E"])
    hex_1.set_back_side_symbol("I")
    #hex_1.get_node_path().showTightBounds()
    #print hex_1.get_node_path().getTightBounds()
    #print get_center_point(hex_1.get_node_path())
    #hex_1.set_pos(-0.4,0,0)
    #hex_1.get_node_path().setH(180)
    
#    hex_2 = Hexagon(radius=0.3)
#    hex_2.set_back_side_symbol("I")
#    hex_2.set_pos(0.4,0,0)
#    hex_2.root_node_path.setH(180)
#    taskMgr.add(rotate, 'rotate', extraArgs=[hex_1.root_node_path,0.1,0], appendTask=True)
#    taskMgr.add(rotate, 'rotate', extraArgs=[hex_2.root_node_path,0.1,0], appendTask=True)

    base.disableMouse()
    base.camera.setPos(0,-5,0)

    plight = PointLight('plight')
    plight.setColor(VBase4(1, 1, 1, 1))
    plnp = render.attachNewNode(plight)
    plnp.setPos(0, -10, 0)
    render.setLight(plnp)
    
    run()
    