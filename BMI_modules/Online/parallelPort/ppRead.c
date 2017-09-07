#include "mex.h"
#include <stdio.h>
#include "ioPort.c"

#define pAddr prhs[0]
#define pDataOut plhs[0]

/*
  ppWrite uses now the file ioPort.c for the io functionality. In ioPort 
  the parallel port is accesed in windows nt save way.
 
  edit Max Sagebaum           02.11.2007       Added nt save io functions
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ unsigned short portaddr;
  unsigned char value;
  HINSTANCE hLib;
  
  hLib = createLibrary();

  if (nrhs!=1)
	mexErrMsgTxt("Out input argument required");
  if (nlhs>1)
	mexErrMsgTxt("Only one output argument");

  portaddr= (unsigned  short)*mxGetPr(pAddr);
  value = Inp32(portaddr);

 pDataOut= mxCreateDoubleMatrix(1, 1, mxREAL);
 *mxGetPr(pDataOut)= (double)value;
 
 deleteLibrary(hLib);
}
