#from Feedbacks.HexoSpeller.LanguageModel import LanguageModel
#from Feedbacks.HexoSpeller.Utils import Utils
from .. import Utils as Utils
#from Utils import rotate_phi_degrees_counter_clockwise
from pandac.PandaModules import Geom, GeomNode, GeomTrifans, GeomTristrips, \
    GeomLines, GeomVertexFormat, GeomVertexData, GeomVertexReader, GeomVertexWriter, \
    PandaNode, NodePath, GeomTriangles


def create_line(x1, z1, x2, z2):
    format = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData('', format, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    vertex.addData3f(x1, 0, z1)
    vertex.addData3f(x2, 0, z2) 
    for _i in range(2):
        normal.addData3f(0, - 1, 0)
    prim_hint = Geom.UHStatic
    prim = GeomLines(prim_hint)
    prim.addVertices(0, 1)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('')
    node.addGeom(geom)
    return (node, vdata)
    
def create_side(x_z_top_left, x_z_bottom_right, static=True):
    x1, z1 = x_z_top_left
    x2, z2 = x_z_bottom_right
    format = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData('', format, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    vertex.addData3f(x1, 0, z1) # top left
    vertex.addData3f(x2, 0, z1) # top right
    vertex.addData3f(x2, 0, z2) # bottom right
    vertex.addData3f(x1, 0, z2) # bottom left
    for _i in range(4):
        normal.addData3f(0, - 1, 0)
    if static:
        prim_hint = Geom.UHStatic
    else:
        prim_hint = Geom.UHDynamic
    prim = GeomTristrips(prim_hint)
    prim.addVertices(1, 0, 2, 3)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('')
    node.addGeom(geom)
    return (node, vdata)

            
def create_triangle(x_z_left, x_z_top, x_z_right, static=True):
    x1,z1 = x_z_left
    x2,z2 = x_z_top
    x3,z3 = x_z_right
    format = GeomVertexFormat.getV3n3c4t2()
    vdata=GeomVertexData('', format, Geom.UHStatic)
    vertex=GeomVertexWriter(vdata, 'vertex')
    normal=GeomVertexWriter(vdata, 'normal')
    vertex.addData3f(x1, 0, z1) # left
    vertex.addData3f(x2, 0, z2) # top
    vertex.addData3f(x3, 0, z3) # right
    for _i in range(3):
        normal.addData3f(0,-1,0)
    if static:
        prim_hint = Geom.UHStatic
    else:
        prim_hint = Geom.UHDynamic
    prim = GeomTriangles(prim_hint)
    prim.addVertices(0,2,1)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('')
    node.addGeom(geom)
    return node

def create_hexagon(radius):
    """ Creates a hexagon shape that is centered at (0,0,0) with the corners having a distance of radius to the center and
    the normals pointing in direction (0,-1,0). 
    Returns the tuple (PandaNode, GeomVertexData). """
    format = GeomVertexFormat.getV3n3c4t2()
    vdata=GeomVertexData('hexagon', format, Geom.UHStatic)
    vertex=GeomVertexWriter(vdata, 'vertex')
    normal=GeomVertexWriter(vdata, 'normal')
    # create the vertices
    vertex.addData3f(0,0,0)
    normal.addData3f(0,-1,0)
    # add the other vertices
    for phi in range(0,360,60):
        # right-hand-rule (with middle finger pointing upwards): the y-axis points towards the screen,
        # therefore the hexagon will be created in the x,z plane, with x-axis pointing to the right
        # and the z-axis pointing up
        # get the next vertex coordinates by rotating the point (0,0,radius) in the x,z plane
        x,z = Utils.rotate_phi_degrees_counter_clockwise(phi, (0,radius))
        #print (x,z)
        vertex.addData3f(x,0,z) 
        normal.addData3f(0,-1,0) # the normal vector points away from the screen
    # add the vertices to a geometry primitives
    prim = GeomTrifans(Geom.UHStatic)
    for i in range(7):
        prim.addVertex(i)
    prim.addVertex(1)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    hex_node = GeomNode('')
    hex_node.addGeom(geom)
    return hex_node, vdata
    
            
def get_bounding_width_height(node_path):
    (bottom_left, top_right) = node_path.getTightBounds()
    width = top_right[0] - bottom_left[0]
    height = top_right[2] - bottom_left[2]
    return width, height

def get_center_point(node_path):
    """ Computes the center point of the bounding box of the given node_path. 
    Returns the point coordinates as a list [x,y,z]. """
    bottom_left_front, top_right_back = node_path.getTightBounds() # get the bounding box
    c = [0,0,0] # init center point of the bounding box
    for i in range(3):
        c[i] = (top_right_back[i] - bottom_left_front[i]) / 2.0 + bottom_left_front[i]
    return c
    
def center_node(node_path):
    """ Computes the center point of the bounding box and then shifts the node such that the new center will be at (0,0,0). """
    center_node_on_xyz(node_path, 0, 0, 0)
    
def center_node_on_xyz(node_path, x, y, z):
    """ Computes the center point of the bounding box and then shifts the node such that the new center will be at (x,y,z). """
    c = get_center_point(node_path)
    node_path.setPos(node_path.getX()-(c[0]-x),
                     node_path.getY()-(c[1]-y),
                     node_path.getZ()-(c[2]-z))
    
def center_node_on_xz(node_path, x, z):
    center_node_on_xyz(node_path, x, node_path.getY(), z)    