GENPY = "liquid.py"
GENWRAP = "liquid_wrap.c"
GENSO = "_liquid.so"

all:
	swig -python liquid.i
	gcc -fPIC -O4 -shared liquid_wrap.c lqrender.c molecular.c -I/usr/include/python2.5 -L/usr/lib -lpython2.5 -o ${GENSO}
	cp ${GENSO} ..
	cp ${GENPY} ..

clean:
	rm ${GENSO} ../${GENSO}
	rm ${GENPY} ../${GENPY}
	rm ${GENWRAP}
