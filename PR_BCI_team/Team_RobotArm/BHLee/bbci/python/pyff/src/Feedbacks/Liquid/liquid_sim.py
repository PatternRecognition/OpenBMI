import liquid



def setBoundaryMatrix(liquid_data, boundary_matrix):
        w = liquid_data.width
        h = liquid_data.height
        z = liquid.new_intArr(w*h)
         
        for i in range(w):
            for j in range(h):
                liquid.intArr_setitem(z, i+j*w,int(boundary_matrix[i,j]))

        liquid.data_set_boundary(liquid_data, z)
    

def createGaussTable(width, height, strength):
    gauss_table = liquid.gen_table(width, height)
    liquid.set_gaussian_table(gauss_table, strength)
    return gauss_table


def createCauchyTable(width, height, strength):
    gauss_table = liquid.gen_table(width, height)
    liquid.set_cauchy_table(gauss_table, strength)
    return gauss_table
    
    
def createLiquidData(width, height):
    data = liquid.new_data(width, height, 5, height-5, 5, width-5)
    return data
    
    
    
def callbackFunc(x,y,dx,dy):
    return (0,0)
    
def createMoleculeSim(repulse_scale, repulse_size, dt, temperature, attraction_scale, attraction_size, min_distance, field_fn, fric_fn):
    sim = liquid.new_molecule_sim(repulse_scale, repulse_size, dt, temperature, attraction_scale, attraction_size, min_distance)
    
    liquid.set_molecule_field(sim, field_fn)
    liquid.set_molecule_friction(sim, fric_fn)
    return sim


def addDroplet(sim, liquid_data , x, y, shape_table):
        pt = liquid.new_lut(x,y,shape_table)        
        mol = liquid.new_molecule(pt)
        liquid.sim_add_molecule(sim, mol)
        liquid.data_add_lut(liquid_data, pt)
        
        return mol
        
        
def updateDynamics(sim, drop_list):
    for drop in drop_list:
        liquid.molecule_update_dynamics(drop, liquid_sim)
    

def getPolygons(liquid_data, x, y, xscale, yscale, height=1, step=5):    
    liquid.data_render(liquid_data, int(height))
    #package into polygons
    sections = liquid.lqSectionArray.frompointer(liquid_data.blob_sections)
    xptrarray = liquid.intArray.frompointer(liquid_data.x_list)
    yptrarray = liquid.intArray.frompointer(liquid_data.y_list)
    
    polys = []
    for i in range(liquid_data.n_sections):
        begin = sections[i].start
        end = sections[i].end
        if begin>=0 and end>=0:
            #append new polygon
            poly = []
            for j in range(begin, end, step):
                poly.append((xptrarray[j]*xscale+x, yptrarray[j]*yscale+y))
            polys.append(poly)

    return polys

    
    
class LiquidSimulator:

    #initialise a new simulator. Width and height are the pixel sizes of the tracing space.
    #mol_sim is a dictionary defining how the molecule simulation parameters
    
    def __init__(self, width=1024, height=1024, repulsion_scale=-5, repulsion_strength=0.2,
    dt=1.0, temperature=0, attraction_scale=-20, attraction_strength=1.9, min_distance=2, field_fn=callbackFunc,
    friction_fn=callbackFunc):
        self.liquid_data = createLiquidData(width=width,height=height)
        self.liquid_sim = createMoleculeSim(repulsion_scale,
        repulsion_strength,
        dt,
        temperature,
        attraction_scale,
        attraction_strength,
        min_distance,
        field_fn,
        friction_fn)
                        
        self.shape_tables = {}
        self.drops = []
        
        
    def getWidth(self):
        return self.liquid_data.width
    
    def getHeight(self):
        return self.liquid_data.height
        
        
    def setBoundary(self, boundary_matrix):
        if boundary_matrix.shape == (self.getWidth(), self.getHeight()):
            setBoundaryMatrix(self.liquid_data, boundary_matrix)
        else:
            print "Bad boundary block size"

                
                
    def setDroplet(self, i, x, y):
        self.drops[i].x = x
        self.drops[i].y = y
        
        
    #set the various properties of the simulation
    def setProperty(self, name, value):    
        if name=='field_fn':
            liquid.set_molecule_field(self.liquid_sim, value)
        
        if name=='friction_fn':
            liquid.set_molecule_friction(self.liquid_sim, value)
                
    
        name_table = {'repulsion_scale' : 'rep_scale',
                              'repulsion_strength' : 'rep_size',
                              'dt' : 'dt',
                              'temperature' : 'temperature',
                              'attraction_scale' : 'att_scale',
                              'attraction_strength' : 'att_size',
                              'min_distance' : 'min_distance'}
                              
        
        if name_table.has_key(name):
            self.liquid_sim.__setattr__(name_table[name], value)
    
    #add a droplet at x,y with a shape table with the given width, height and strength
    def addDroplet(self, x, y, width, height, strength, isGauss=True):
        
        if self.shape_tables.has_key((width,height,strength)):
            shape_table = self.shape_tables[width,height,strength]
        else:
            #memoize
            if isGauss:
                shape_table = createGaussTable(width=width, height=height, strength=strength)
            else:
                shape_table = createCauchyTable(width=width, height=height, strength=strength)
            self.shape_tables[(width,height,strength)] = shape_table
        
        drop = addDroplet(self.liquid_sim, self.liquid_data, x, y, shape_table)
        self.drops.append(drop)
        
        
    def updateDynamics(self):
        liquid.sim_update_dynamics(self.liquid_sim)
                        

    #return a list of polygons (in the form of a list of (x,y) lists)
    def getPolygons(self, x=0, y=0, xscale=1, yscale = 1, height = 5e6, step=20):
        return getPolygons(self.liquid_data, x, y, xscale, yscale, height, step)
        


        







