'''
Created on 24.01.2012

@author: jpascual
'''

from Level import LevelBase

class P300Level(LevelBase):
    '''
    classdocs
    '''
    
    
    def __init__(self, nrealelements):
        '''
        Constructor
        '''
     
        LevelBase.__init__(self, LevelBase.P300, nrealelements)           
  
        self._foo_elems_container = None
            
        ## define shapes in the form ['name', {parameters}]:
        # where parameters can be VisionEgg parameters eg. 'radius', 'size', 'orientation' etc.
        self.shapes = [ ['triangle',     {'size':200, 'innerSize': 60, 'innerColor': (.0,.0,.0)}],
                        ['hourglass',    {'size':100}],
                        ['cross',        {'size':(30., 180.), 'orientation':45.}],
                        ['triangle',     {'size':200, 'innerSize': 60, 'innerColor': (.0,.0,.0), 'orientation':180.}],
                        ['hourglass',    {'size':100, 'orientation':90.}],
                        ['cross',        {'size':(30., 180.)}],
                        ['hourglass',    {'size':100, 'orientation':0.}] ]                 
    
    def get_shapes(self):
        return self.shapes             