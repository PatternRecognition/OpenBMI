\local\swig\swigwin-1.3.31\swig -python liquid.i
gcc -O4 -shared liquid_wrap.c lqrender.c molecular.c -I/python24/include -L/python24/libs -lpython24 -o _liquid.pyd
copy _liquid.pyd .. 
copy liquid.py ..
