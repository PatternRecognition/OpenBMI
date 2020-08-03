

from direct.showbase.DirectObject import DirectObject
import direct.gui.OnscreenText as ost
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import TextNode, PandaNode, NodePath
from pandac.PandaModules import GeomVertexFormat, GeomVertexData, GeomVertexReader, GeomVertexWriter
from pandac.PandaModules import Geom, GeomNode, GeomTrifans, GeomTristrips, GeomLines, PointLight, VBase4
#global render, loader, base, taskMgr

from GraphicComponentUtils import create_line, create_side

class ControlSignalBar():
    
    def __init__(self, bar_height=1, bar_width=0.2, thresholds=(0.3, 0.7), 
                 padding=0.02, frame_line_width=2):
        """ Creates a framed bar. The frame borders will have a distance to the actual bar, which is given by
        padding. """
        #from direct.directbase import DirectStart
        self.padding = padding
        # everything that belongs to this ControlSignalBar will be stored under the root node
        self.root_node_path = render.attachNewNode(PandaNode(''))
        # create the geometry node that is the frame
        self.frame_node_path = self.root_node_path.attachNewNode(PandaNode('frame_node'))
        self._create_frame(bar_height+2*padding, bar_width+2*padding, thresholds, self.frame_node_path)
        self.frame_node_path.setY(-0.01)
        self.frame_node_path.setZ(-padding)
        self.set_frame_line_width(frame_line_width)
        # create the geometry node that is the actual bar
        self.bar_node_path = self.root_node_path.attachNewNode(PandaNode('bar_node'))
        self._create_bar(bar_height, bar_width, self.bar_node_path)
        self.set_frame_color(0, 0, 1)
        self.set_bar_color(0, 1, 0)
        self.root_node_path.setTwoSided(True)
    
    def set_scale(self, scale):
        self.get_node_path().setScale(scale)
    
    def set_pos(self, x, y, z):
        self.get_node_path().setPos(x,y,z)
    
    def set_frame_color(self, r, g, b, alpha=1):
        """ Sets the color of the frame and the thresholds. """
        self.frame_node_path.setColor(r,g,b,alpha)
    
    def set_frame_line_width(self, width):
        self.frame_node_path.setRenderModeThickness(width) 
        
    
    def set_bar_color(self, r, g, b, alpha=1):
        self.bar_node_path.setColor(r,g,b,alpha)
        
    def set_bar_height(self, height):
        """ Sets the height of the bar to the given value. The given height should be between 0 and 1. """        
        reader = GeomVertexReader(self.bar_vdata, 'vertex')        
        writer = GeomVertexWriter(self.bar_vdata, 'vertex')                
        while not reader.isAtEnd():
            v = reader.getData3f()           
            if v[2] > 0:
                writer.setData3f(v[0],v[1],abs(height))
            else:
                writer.setData3f(v) # I have to call the writer setData method in any case 
                                        # so that its counter stays in sync with the reader's
    
    def set_threshold_1(self, t_value):
        self._set_threshold(self.t1_vdata, t_value)
    
    def set_threshold_2(self, t_value):
        self._set_threshold(self.t2_vdata, t_value)
    
    def _set_threshold(self, vdata, t_value):
        reader = GeomVertexReader(vdata, 'vertex')
        writer = GeomVertexWriter(vdata, 'vertex')
        while not reader.isAtEnd():
            v = reader.getData3f()
            writer.setData3f(v[0],v[1],t_value+self.padding)
                
    def get_node_path(self):
        return self.root_node_path
    
        
    def _create_frame(self, height, width, thresholds, root_node_path):
        # create surrounding first
        # bottom line
        node, _vdata = create_line(-width/2.0, 0, width/2.0, 0)
        root_node_path.attachNewNode(node)
        # left line
        node, _vdata = create_line(-width/2.0, 0, -width/2.0, height)
        root_node_path.attachNewNode(node)
        # top line
        node, _vdata = create_line(-width/2.0, height, width/2.0, height)
        root_node_path.attachNewNode(node)
        # right line
        node, _vdata = create_line(width/2.0, 0, width/2.0, height)
        root_node_path.attachNewNode(node)
        # create the threshold lines
        t1, t2 = thresholds
        node, self.t1_vdata = create_line(-width/2.0, t1, width/2.0, t1)
        root_node_path.attachNewNode(node)
        node, self.t2_vdata = create_line(-width/2.0, t2, width/2.0, t2)
        root_node_path.attachNewNode(node)
    
    def _create_bar(self, height, width, root_node_path):
        node, self.bar_vdata = create_side((-width/2.0, height), (width/2.0, 0), False)
        root_node_path.attachNewNode(node)
        


def rotate(node_path, rotation_per_second, start_angle, task):
    v = 360*rotation_per_second*task.time + start_angle
    v = v%360
    node_path.setH(v)
    
    return Task.cont


if __name__ == "__main__":
    bar = ControlSignalBar()
    bar.get_node_path().setPos(0,0,-0.5)
    #taskMgr.add(rotate, 'rotate', extraArgs=[bar.get_node_path(),0.1,0], appendTask=True)
#    taskMgr.add(rotate, 'rotate', extraArgs=[hex_2.root_node_path,0.5,0], appendTask=True)

    base.disableMouse()
    base.camera.setPos(0,-3,0)

#    plight = PointLight('plight')
#    plight.setColor(VBase4(1, 1, 1, 1))
#    plnp = render.attachNewNode(plight)
#    plnp.setPos(0, -10, 0)
#    render.setLight(plnp)
    
    run()
    
    