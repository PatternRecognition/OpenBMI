MATLAB=/home/neuro/software/matlab6.5/extern

CFLAGS=-I$(MATLAB)/include/
CC=gcc
LDFLAGS=-L$(MATLAB)/lib/glnx86
LOADLIBES=-lmat -lmx

objects : acquire_bv.o brainserver.o headerfifo.o udp.o

brainserver.c : brainserver.h headerfifo.h ../winunix/winthreads.h 

acquire_bv.c : brainserver.h

udp.c : udp.h

send_to_udp.c : udp.h

send_to_udp : send_to_udp.o udp.o