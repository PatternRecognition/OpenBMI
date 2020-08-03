## config file should hold all non feedback design parameters etc. like technical buffersizes, verbosity etc.

# specify which audio device is used
# if None, tries to auto-detect ASIO or uses default 
__DEVICE__ = None
#__DEVICE__ = 46 # MOTU ASIO

# buffersize for pyo server; less is more
__BUFFERSIZE__ = 256 # default
__BUFFERSIZE__ = 512
__BUFFERSIZE__ = 1024
#__BUFFERSIZE__ = 256

# verbose
__VERBOSE__ = False # default
__VERBOSE__ = True

# time resolution
__RESOLUTION__ = 1000 # ms
