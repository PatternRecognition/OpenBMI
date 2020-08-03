#import Feedbacks.HexoSpeller.Utils as Utils
from math import sqrt, acos
from GraphicComponentUtils import create_side, create_triangle


from direct.showbase.DirectObject import DirectObject
import direct.gui.OnscreenText as ost
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import TextNode, PandaNode, NodePath
from pandac.PandaModules import GeomVertexFormat, GeomVertexData, GeomVertexReader, GeomVertexWriter
from pandac.PandaModules import Geom, GeomNode, GeomTriangles, GeomTrifans, GeomTristrips, GeomVertexWriter, PointLight, VBase4
#global render, loader, base




class Arrow():
    
    def __init__(self, total_length=1):
        # relative lengths, will be scaled so that the total length is according to 
        shaft_length=0.8
        shaft_width=0.2 
        tip_length=0.2
        tip_width=0.5
        depth=0.1
        self.shaft_length = shaft_length
        self.tip_length = tip_length
        # everything that makes up the arrow will be stored under this dummy root node
        self.root_node_path = render.attachNewNode(PandaNode(''))
        # everything that makes up the shaft will be stored under this node path
        self.shaft_vdata = list()
        self.shaft_node_path = self.root_node_path.attachNewNode(PandaNode('arrow_shaft_root'))
        self._create_shaft(shaft_length, shaft_width, depth, self.shaft_node_path)
        # everything that makes up the tip will be stored under this node path
        self.tip_node_path = self.root_node_path.attachNewNode(PandaNode('arrow_tip'))
        self._create_tip(tip_length, tip_width, depth, self.tip_node_path)
        self.tip_node_path.setZ(shaft_length)
        # set properties that affect the whole arrow
        self.set_color(0, 1, 0)
        self.set_scale(0.5)
        self.root_node_path.setTwoSided(True)
    
    def set_length(self, length):
        """ Sets the length of the arrow. """
        length = length - self.tip_length
        self._set_z_on_shaft(length)
        self.shaft_length = length
        self.tip_node_path.setZ(length)
        
    def get_length(self):
        """ Returns the current length of the arrow. """
        return self.shaft_length + self.tip_length
        
    def set_angle_x_z_plane(self, phi):
        """ Sets the angle of the arrow in the x,z plane, which is the plane of the screen. """
        self.root_node_path.setR(phi)
        
    def set_color(self, r, g, b, alpha=1):
        """ Sets the color of the arrow. """
        self.root_node_path.setColor(r,g,b,alpha)
    
    def set_scale(self, s):
        self.root_node_path.setScale(s)
    
    def get_node_path(self):
        """ Returns the node path to the top node of the arrow sub-tree. """
        return self.root_node_path
    
        
    def _create_shaft(self, length, width, depth, root_node_path):
        """ Creates the shaft sides individually and rotates and shifts them so that
        they form the shaft. """
        # front 
        front_node, vdata = create_side((-width/2.0, length), (width/2.0, 0), static=False)
        front_path = root_node_path.attachNewNode(front_node)
        front_path.setY(-depth/2.0)
        self.shaft_vdata.append(vdata)
        # back 
        back_node, vdata = create_side((-width/2.0, length), (width/2.0, 0), static=False)
        back_path = root_node_path.attachNewNode(back_node)
        back_path.setH(180)
        back_path.setY(depth/2.0)
        self.shaft_vdata.append(vdata)
#        # top
#        top_node, vdata = create_side((-width/2.0, depth/2.0), (width/2.0, -depth/2.0), static=False)
#        top_path = root_node_path.attachNewNode(top_node)
#        top_path.setP(-90)
#        top_path.setZ(length)
#        self.shaft_vdata.append(vdata)
        # bottom
        bottom_node, vdata = create_side((-width/2.0, depth/2.0), (width/2.0, -depth/2.0), static=False)
        bottom_path = root_node_path.attachNewNode(bottom_node)
        bottom_path.setP(90)
        # right
        right_node, vdata = create_side((-depth/2.0, length), (depth/2.0, 0), static=False)
        right_path = root_node_path.attachNewNode(right_node)
        right_path.setH(90)
        right_path.setX(width/2.0)
        self.shaft_vdata.append(vdata)
        # left
        left_node, vdata = create_side((-depth/2.0, length), (depth/2.0, 0), static=False)
        left_path = root_node_path.attachNewNode(left_node)
        left_path.setH(-90)
        left_path.setX(-width/2.0)
        self.shaft_vdata.append(vdata)
        
    def _create_tip(self, length, width, depth, root_node_path):
        # front 
        front_node = create_triangle((-width/2.0,0), (0,length), (width/2.0,0))
        front_path = root_node_path.attachNewNode(front_node)
        front_path.setY(-depth/2.0)
        # back
        back_node = create_triangle((-width/2.0,0), (0,length), (width/2.0,0))
        back_path = root_node_path.attachNewNode(back_node)
        back_path.setH(180)
        back_path.setY(depth/2.0)
        # bottom
        bottom_node, _vdata = create_side((-width/2.0,depth/2.0), (width/2.0,-depth/2.0))
        bottom_path = root_node_path.attachNewNode(bottom_node)
        bottom_path.setP(90)
#        # top right
#        side_length = sqrt(length**2 + (width/2.0)**2)
#        phi = Utils.radians_to_degrees( acos((width/2.0)/side_length) )
#        top_right_node = create_side((-side_length, depth/2.0), (0, -depth/2.0))
#        top_right_path = root_node_path.attachNewNode(top_right_node)
#        #top_right_path.setP(-90)
#        top_right_path.setR(phi)
#        top_right_path.setX(width/2.0)
#        # top left
#        top_left_node = create_side((0, depth/2.0), (side_length, -depth/2.0))
#        top_left_path = root_node_path.attachNewNode(top_left_node)
#        #top_left_path.setP(-90)
#        top_left_path.setR(-phi)
#        top_left_path.setX(-width/2.0)
        
    
    def _set_z_on_shaft(self, z):
        for vdata in self.shaft_vdata:
            reader = GeomVertexReader(vdata, 'vertex')
            writer = GeomVertexWriter(vdata, 'vertex')
            while not reader.isAtEnd():
                v = reader.getData3f()
                if v[2] > 0:
                    writer.setData3f(v[0],v[1],z)
                else:
                    writer.setData3f(v) # I have to call the writer setData method in any case 
                                                # so that its counter stays in sync with the reader's
    
    
def rotate(node_path, rotation_per_second, start_angle, task):
    v = 360*rotation_per_second*task.time + start_angle
    v = v%360
    node_path.setH(v)
    return Task.cont

def scale(arrow, time_to_full_length, start_length, max_length, task):
    if arrow.get_length() > max_length:
        arrow.set_length(start_length)
    else:
        t = task.time % time_to_full_length
        t = t/time_to_full_length
        length_diff = max_length - start_length
        new_length = start_length+t*length_diff
        arrow.set_length(new_length)
    return Task.cont

if __name__ == "__main__":
    from directbase import DirectStart
    print "Arrow::main"
    arrow = Arrow()
    arrow.set_angle_x_z_plane(45)
    #arrow.set_length(0.5)
    base.disableMouse()
    base.camera.setPos(0,-5,.5)
    
    #taskMgr.add(rotate, 'rotate', extraArgs=[arrow.root_node_path,0.05,0], appendTask=True)
    #taskMgr.add(scale, 'scale', extraArgs=[arrow,1.0,0.5,1.0], appendTask=True)

    plight = PointLight('plight')
    plight.setColor(VBase4(1, 1, 1, 1))
    plnp = render.attachNewNode(plight)
    plnp.setPos(0, -10, 0)
    render.setLight(plnp)
    
    run()
    